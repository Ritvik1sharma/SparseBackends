# SKILLS — Debugging the Aliased PHP Path

This file is a working playbook for picking up the aliased-BlockSparse PHP
experiments in this folder. It assumes you've read [README.md](README.md).

## Mental model

The matvec hot-path for projected DMRG with sparse H is:

```
v (dense ψ block)
  → step 1:  L * v             (denseH × denseV, ITensor *)
  → step 2:  H[k] * Hv1        (sparseH × denseV, OUR kernel)
  → step 3:  H[k+1] * Hv2      (sparseH × denseV, OUR kernel)
  → step 4:  R * Hv3           (denseH × denseV, ITensor *)
```

Steps 2 and 3 dispatch into
`SparseBackends.contract_aliased_dense_to_dense!` via the wrapper
`wrapped_contract_aliased`. The kernel:

1. **classify** labels of A, B, C into roles: `shared_prefix`, `c_prefix`,
   `keepA`, `red_dense`, `keepB`.
2. **permute_A** — reorder A's dense tail to `[keepA, red_dense]`. Cheap
   (only `n_templates` template buffers, not `n_blocks`).
3. **permute_B** — reorder B's axes to `[red_dense, keepB, shared_prefix]`
   so the kernel can `reshape(Bsub, K, N)` per block. **Expensive — O(|B|)**.
4. **setup** — compute strides, offsets, allocate Ctgt buffer.
5. **main_loop** — for each block, compute the GEMM offsets inline and call
   `mul!(Cmat, Amat, Bmat, α, 1)`. This is the only place BLAS runs.
6. **permute_back** — if caller's `labelsC` ≠ kernel's `canon = [keepA, keepB, c_prefix]`,
   the kernel writes into a scratch in canon layout, then `permutedims!`
   back into the caller-shaped `C`. **Expensive — O(|C|)**.

`permute_A`, `permute_B`, `permute_back` are diagnosed by:

- Timer entries inside `add.<name>` in the SparseBackends contract
  dispatch breakdown.
- `SB_PERMB_DBG=1` for first-N non-identity `permB` patterns.
- `SB_PERM_PROFILE=1` for the full distribution of `(permB, perm_C, shape)`.

## Standard debugging flow

When aliased is slower than expected:

1. **Run with timers and pull the breakdown**:
   ```bash
   julia --project=. test_sparse_ham/test_pxp_aliased.jl 20 > out.log 2>&1
   awk '/TIMER REPORT: ALIASED_ground/,/TIMER REPORT: DENSE_excited/' out.log
   ```
   Look at `ext_dispatch[wrapped×dense]` and the `add.*` subsections inside
   `contract_aliased_dense_to_dense`. The four big lines are:
   `main_loop` (BLAS — the floor), `permute_B`, `permute_back`,
   `ali_dense_alloc`. Whichever is biggest is your target.

2. **If `permute_B` or `permute_back` dominates**: turn on `SB_PERM_PROFILE=1`,
   read the top patterns. Each row prints `(nkA,nkB,ncp,nsp,nrd) permB perm_C`.
   Look for identity (✓id) vs non-identity. The fraction of identity calls
   tells you how much room is left.

3. **If `main_loop` dominates**: BLAS is the floor; you can't reduce per-block
   compute. The only handle is *fewer/bigger blocks*. Check `n_templates`
   and `n_blocks` in the per-site memory table — if `n_templates ≈ n_blocks`
   there's no aliasing happening, and converting wasn't worth it. Compare
   `blksize` to MPS bd: if `blksize ≪ bd²`, per-block GEMM shape is skinny
   (small K) and BLAS is inefficient. The structural fix is to fuse fewer
   sparse links (leaving bigger templates) — try `SB_FUSE_LINKS=0`.

4. **If `ali_dense_alloc` dominates**: the kernel is allocating output
   buffers. Sum/call should match `eltype × prod(canon_dims)`. The cost is
   roughly `~3 GB/s` so `t ≈ alloc_bytes / 3e9`. The only reductions are
   buffer pooling (ruled out) or producing output in a different
   smaller-memory form (would require restructuring the kernel —
   aliased-aware C output).

5. **Always re-verify correctness**:
   ```bash
   grep -E "E0 =|E1 =|\|ΔE\|" out.log
   ```
   `|ΔE0|` should be ≤ `1e-12`. `|ΔE1|` ≤ `1e-11`. If energy match
   degrades, suspect an axis-ordering bug in the layout machinery
   (`label_key_for_ind` conversion, `permute_back` direction, etc.).

## Diagnostic env-var combos

- **"What patterns are we firing?"** —
  `SB_PERM_PROFILE=1` on a small N (e.g. N=14, bd=40). Cheap, fast.
- **"Why isn't permB identity here?"** —
  `SB_PERMB_DBG=1 SB_PERMB_DBG_MAX=12`. Prints labelsA, labelsB, labelsC
  and the required vs actual permB ordering for the first 12 mismatches.
  Pair with the schema knowledge from the README's "Working hypotheses"
  table.
- **"Is the hint reaching the kernel?"** —
  `SB_HINT_DBG=1` (caller side) shows the hint computed for the first 12
  matvec calls. Then check the kernel's perm_profile for `(perm_C)` —
  it should match the hint after wrapper conversion.
- **"Is the H actually aliased entering DMRG?"** — The test prints
  `[H_aliased storage at DMRG entry]  sites=N  aliased=K  other=...`
  near the top of each run.

## Common pitfalls (and how to spot them)

1. **`SB_FUSE_LINKS` does TWO things.** It (a) fuses multi-strand sparse
   links at H-build time *and* (b) flips the wrapper's `canon_labels`
   heuristic. (a) is the real-data win, (b) is fitted to spin-test
   geometry. **If you change models, the heuristic order is likely
   wrong.** Either decouple the flag, or pass `preferred_output_labels`
   explicitly to override the heuristic.

2. **`to_blocksparse(A::AliasedBlockSparse)` silently destroys
   aliasing.** It writes out `α * template` per block — the full BS
   expansion. The MPO's per-site memory ratio will drop from ~13× → 1×.
   Reachable via `fuse_axes!` on a Wrapped Aliased tensor, and via the
   `:blocksparse` BS fallback in `wrapped_contract_aliased`. Avoid by
   using `_fuse_aliased_strands` for key-rewrite fuse and
   `preserve_bs_output=true` for chained aliased contractions.

3. **`ITensor(::AbstractArray, ::Index...)` deep-copies the array.** It
   uses `NeverAlias` by default. Use `itensor(...)` (lowercase) for
   `AllowAlias` and zero-copy wrap. Already done in
   `wrap_output`, but watch for it if you add new ITensor-wrapping code
   paths.

4. **`permutedims!` on a non-contiguous ITensor allocates.** The
   kernel's `permute_B` writes into `_bdd_permB_buffer` (a reusable
   scratch). Don't construct fresh permuted ITensors in the hot loop.

5. **`Base.summarysize` overstates raw data size.** ITensor metadata,
   index objects, and storage struct overhead add up. For pure
   data-volume comparison use the per-site element counts in the
   `report_aliased_footprint` table (`dense`, `BS`, `aliased` columns).

6. **Excited-state DMRG uses `ProjMPO_MPS`**, not `ProjMPO`. Its `product`
   adds `weight * P.pm[i] * v` terms for each ortho state. Changes to
   `ITensors.contract(P::AbstractProjMPO, v)` apply to both, but the
   *additional* `product(p, v)` calls go through a separate code path
   that doesn't read `SB_PREPERMUTE_ENVS` or any of our flags. If
   excited-state DMRG behaves differently from ground, this is likely
   the reason.

## Performance regression hunt

When something gets slower unexpectedly:

1. **Diff the timer breakdowns** between the slow run and the last-known-fast
   run. Look at `Tot / % measured` and the top 5 `add.*` lines. Whichever
   *fraction* moved is the suspect.
2. **Check `Allocations` columns.** A 10× alloc spike usually means
   something fell off the buffer-pool / `_itensor_from_external_storage`
   path and is doing a full copy each call.
3. **Try the same run twice back-to-back.** If DENSE timing alone moves
   ≥10%, you're in system-noise territory at the resolution you care
   about — re-run with larger workload or compute a median.
4. **Reduce the example.** Drop to N=10 bd=20 nsweeps=3. If the
   regression survives, it's algorithmic. If not, it's load-dependent.

## Adding a new model

To add another projected Hamiltonian (e.g., spin-2, Bose-Hubbard):

1. Copy `test_pxp_aliased.jl` as the template. The structure is:
   - `op` overloads for any non-standard site operators.
   - `<model>_opsum(N)` returns the bare `H` as an OpSum.
   - `<projector>_R1(sites)` returns the constraint MPO P.
   - `sandwich_mpo_aliased(P, H)` and `sandwich_mpo_dense(P, H)` build PHP.
   - `fuse_sparse_links!`, `prepermute_aliased_mpo!` already model-agnostic.
   - `run_dmrg_ground` / `run_dmrg_excited` are reusable.
2. Verify per-site MPO point-equality between aliased and dense via
   `to_dense_itensors(H_aliased[i]) ≈ H_dense[i]`.
3. Run with `SB_PERM_PROFILE=1` to see what shapes occur. The schema
   constants (`PA`, `n_templates`, etc.) and the dominant `(permB, perm_C)`
   patterns will tell you whether the wrapper heuristic and pre-permute
   conventions still apply.
4. If `permB` is mostly non-identity for the new model, the wrapper's
   `SB_FUSE_LINKS` heuristic may need a model-specific override — pass
   `preferred_output_labels` explicitly per call instead.

## Reference timings on this machine (1 BLAS thread, PXP-PHP, N=20, nsweeps=6)

| bd | Variant | ALIASED ground | DENSE ground | ratio |
|---|---|---|---|---|
| 60 | PREV (no auto, no prep) | 15.1 s | 10.5 s | 1.44 |
| 60 | AUTO + PREP | 7.2 s | 4.8 s | 1.49 |
| 80 | AUTO + PREP | 12.5 s | 8.9 s | 1.42 |
| 120 | AUTO + PREP | 30.4 s | 20.0 s | 1.52 |

The ratio is roughly flat across bd ≥ 60 — aliased pays a constant
fraction overhead. The absolute numbers vary ±15% run-to-run from
system load.

## When to stop optimising

Stop when:
- `main_loop` is ≥ 50% of `ext_dispatch[wrapped×dense]`. At that point
  BLAS is the bottleneck and further per-call overhead reductions are
  small.
- Or, when `permute_B + permute_back + ali_dense_alloc` ≤ 30% of the
  matvec budget. Below that you're chasing diminishing returns.
- Or, when aliased ratio ≤ 1.1× dense. At that point the only path
  forward is structural (e.g., aliased-aware Hv output, or a different
  kernel altogether). Worth taking a break and re-scoping.

Past 1.1× ratio, the next real lever is **fewer/bigger blocks**: change
the link-fusion strategy so `blksize` is large enough that per-block
GEMM gets BLAS-efficient. That requires re-thinking which axes go in
the sparse prefix vs dense tail.
