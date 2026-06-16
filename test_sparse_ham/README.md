# test_sparse_ham — Aliased BlockSparse Hamiltonian Experiments

## Target

Demonstrate that storing a projected Hamiltonian `PHP` using
`AliasedBlockSparse` storage (templates shared across blocks via
`(template_id, scalar)` indirection) can match or beat the dense MPO path in
DMRG, while shrinking MPO memory by ~10× or more.

Two test drivers exercise this:

- `test_check_working_aliased.jl` — spin-N chain projected by a 4-site
  conservation MPO. The aliased PHP has up to ~35× compression vs dense
  storage at each site, ~9× vs plain block-sparse.
- `test_pxp_aliased.jl` — PXP model (Rydberg blockade) on S=1 sites,
  projected via the `NotEqlsLoop_R1` constraint MPO (bond=2). Aliased PHP
  is ~13× smaller than dense per-element, ~2× smaller in
  `Base.summarysize` (which includes ITensor metadata overhead).

Each test runs **both ground-state and first-excited DMRG** twice — once
through the dense path (reference correctness check) and once through the
aliased path — and reports timer breakdowns, MPO memory, and `|ΔE|`.

## Quick start

Spin test (N sites passed as the only positional arg):

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
  SB_ALIASED_ENABLE=1 SB_FUSE_LINKS=1 SB_PLAN_B=1 SB_AUTO_DISPATCH=1 SB_PREPERMUTE_H=1 \
  BENCH_MAXDIM=40 BENCH_NSWEEPS=5 \
  julia --project=. test_sparse_ham/test_check_working_aliased.jl 16
```

PXP test:

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
  SB_ALIASED_ENABLE=1 SB_FUSE_LINKS=1 SB_PLAN_B=1 SB_AUTO_DISPATCH=1 SB_PREPERMUTE_H=1 \
  BENCH_MAXDIM=60 BENCH_NSWEEPS=6 \
  julia --project=. test_sparse_ham/test_pxp_aliased.jl 20
```

Each invocation runs ground + excited for both DENSE and ALIASED, and
prints a summary block at the end.

## Pipeline overview

```
                          [build P (sparse MPO)]
                                   │
                          [build H (OpSum MPO)]
                                   │
            ┌──────────────────────┴──────────────────────┐
            │                                             │
   sandwich_mpo_aliased(P, H)               sandwich_mpo_dense(P, H)
      = per-site:                              = ITensor contract
        contract_aliased_itensor(                P'' * H' * P,
          P[i]'', H[i]',                          replaceprime 3→1
          :coo, :dense)
        contract_aliased_itensor(
          P[i], H1,
          :coo, :aliased)
            │                                             │
            ▼                                             ▼
       H_aliased                                       H_dense
            │
   fuse_sparse_links!(H_aliased)              (no fuse needed)
   = collapse multi-strand "Link,l=k" axes
     into one "FusedSparse,Link,bond=k" axis
     via direct key-rewrite (NO data motion,
     templates preserved)
            │
   prepermute_aliased_mpo!(H_aliased)
   = swap each H[k]'s dense tail to
     [right-link, left-link] so the
     kernel's permute_A is identity at
     every matvec call
            │
            ▼
   DMRG (ground + excited)
```

## Environment-variable flags

| Flag | Purpose | Default | Status |
|---|---|---|---|
| `SB_ALIASED_ENABLE` | Hard gate; aliased path is no-op unless this is `1`. | `0` | required |
| `SB_FUSE_LINKS` | Fuse multi-strand sparse "Link" axes per bond into one `FusedSparse` axis. | `0` | recommended |
| `SB_PLAN_B` | In matvec, contract as `it * Hv` instead of `Hv * it`. Output starts with `it`'s uncontracted axes. | `0` | recommended |
| `SB_AUTO_DISPATCH` | Bypass ITensor's `*` for sparseH × denseV; call SparseBackends directly with a schema-driven layout hint. | `0` | helps at small bd; neutral at large bd |
| `SB_PREPERMUTE_H` | One-shot permute of each aliased H[k]'s dense tail at MPO-build time so kernel's `permute_A` is identity. | `0` | small but consistent win |
| `SB_PREPERMUTE_ENVS` | After each `position!`, permute L/R envs so FusedSparse axes come LAST (closer to kernel-preferred B layout). | `0` | **experimental — see TODOs** |
| `SB_PERMB_DBG` | Print up to `SB_PERMB_DBG_MAX` (default 12) examples of non-identity `permB` patterns in the aliased kernel. | `0` | diagnostic |
| `SB_PERM_PROFILE` | Aggregate per-call `(permB, perm_C, shape)` patterns and print sorted count table at end of each DMRG phase. | `0` | diagnostic |
| `SB_HINT_DBG` | Print the first N layout hints computed by `_layout_hint_for_next` (caller-side). | `0` | diagnostic |
| `BENCH_MAXDIM` | DMRG `maxdim`. | depends on test | knob |
| `BENCH_NSWEEPS` | DMRG sweep count. | depends on test | knob |
| `BENCH_WEIGHT` | Weight for excited-state DMRG ortho penalty. | `20.0` | knob |

## Currently achieved

- ✅ Aliased PHP MPO build with ~13× (PXP) / ~35× (spin) raw-element compression vs dense.
- ✅ `_fuse_aliased_strands` key-rewrite fuser — preserves alias structure
  through link fusion (does NOT materialize templates, no data motion).
- ✅ Lean per-block kernel: offset-based slice arithmetic in `add.main_loop`,
  no `ntuple`/`Colon()`/`SubArray` overhead in the hot loop.
- ✅ `itensor()` (lowercase, `AllowAlias`) used in `wrap_output` — skips a
  per-call deep-copy of the output array (~95% drop in `wrap_output` time).
- ✅ Dict → linear-search in `add.classify` and `add.setup`.
- ✅ `SB_PREPERMUTE_H` — kernel's `permute_A` drops to identity check
  (`permute_A` avg 14 µs/call at bd=120 — just the `permB == 1:N` test).
- ✅ `SB_AUTO_DISPATCH` — schema-driven caller-side layout hint via
  `_layout_hint_for_next(it_next, ...)`. Reads `it_next`'s sparse-prefix vs
  dense-tail structure and emits the output's index order so the next
  sparseH call's `permB` is closer to identity.
- ✅ Pattern profiler (`SB_PERM_PROFILE=1`): shows top-N `(permB, perm_C)`
  patterns and their relative frequencies.
- ✅ Excited-state DMRG path (orthogonal-to-ground, `weight=20`) verified
  to converge to same energy as dense within `|ΔE| ≤ 1e-12`.
- ✅ Memory footprint reporter — both `Base.summarysize` and raw-element
  counts (dense / BS / aliased) per site.

## Open TODOs

- ❌ **At bd ≥ 80 ALIASED loses to DENSE on wall-time (~1.4–1.7× slower)**
  despite ~13× MPO memory compression. The aliased path is bottlenecked by
  `add.permute_B` + `add.permute_back` + `ali_dense_alloc`, all of which
  scale linearly with `Hv` size (∼ bd²), while BLAS GEMM cost on
  `M × K × N` per-block grows sub-linearly because BLAS becomes more
  efficient at larger N. Net: overheads grow faster than compute as bd
  grows.

- ❌ **`SB_PREPERMUTE_ENVS=1` has a regression on excited-state DMRG.**
  When enabled, `DENSE_excited` at N=20 bd=120 went from 20 s → 37 s.
  `DENSE_ground` was unaffected. The env-permute helper has an
  `isempty(fused) && return` early-out that should make it a no-op for
  dense MPOs (which have no FusedSparse tags), so the regression isn't
  from the permute itself — likely an interaction with `ProjMPO_MPS`'s
  `product` that does `P.PH * v + Σ weight * P.pm[i] * v`. Needs root-cause
  before turning on.

- ❌ **No buffer pooling for `ali_dense_alloc`.** 3.14 s / 11.6 GiB at
  bd=120. Each call allocates ~7 MB for the kernel's output. User has
  ruled out pooling. Without it this is a hard floor.

- ❌ **`permute_back` still fires on ~50% of calls.** Root cause is
  structural: labels classify differently between adjacent kernel calls
  (e.g., `Site_k'` is `c_prefix` in step k but `keepB` in subsequent
  steps), forcing exactly one of `{permute_back-of-this, permute_B-of-next}`
  to fire per chain link. The hint can choose which one, never zero.

- ❌ **First sparseH step's `permute_B` still non-identity ~half the
  time.** Its input is `Hv1 = L * ψ` via ITensor `*`, which emits output
  in `uncontract(L) ++ uncontract(ψ)` order — structurally cannot be made
  to match the kernel's `[red_dense, keepB, ..., shared_prefix]` order in
  general, because L's FusedSparse axes are forced to the L-section
  (front of output) while we want them at the END. `SB_PREPERMUTE_ENVS`
  was the attempt at addressing this.

- ❌ **`SB_FUSE_LINKS` triggers TWO unrelated behaviours:** (a) the
  build-time multi-strand sparse-link fuse (we want this), and (b) the
  wrapper's `canon_labels` heuristic (was fitted to the spin test, off by
  one swap for PXP). They should be decoupled so the wrapper can default
  to kernel canon (`[keepA, keepB, c_prefix]`) when no explicit hint is
  supplied.

- ❌ **`SB_AUTO_DISPATCH` doesn't cover step 1 (denseH × denseV = L * ψ)
  or step 4 (R × Hv3).** Routing those through SparseBackends with a
  hint would let us control step 2's input layout (potentially
  eliminating the residual `permute_B`).

- ❌ **`fuse_axes!` on `WrappedAliasedBlockSparse` still falls back to
  `to_blocksparse`** ([tensor_wrappers_aliased.jl:106](../SparseBackends/src/tensor_wrappers_aliased.jl#L106)).
  This silently expands aliased storage to BS, losing the memory advantage.
  Not on the hot path today (only invoked during bond canonicalization),
  but a quiet hazard for any future chained-aliased contraction. Need
  native fuse for (tail, tail) and (prefix, prefix) cases; only
  (prefix, tail) genuinely requires materialisation.

## Working hypotheses tested

The diagnosis cycle has been: **measure → hypothesise → instrument → run
→ refute or confirm**. Recorded here so future iterations don't re-derive.

| # | Hypothesis | Outcome | Evidence |
|---|---|---|---|
| H1 | Template-batching by `tidA` (gather Bsubs into one wide GEMM) reduces small-GEMM overhead. | **Refuted.** Gather + scatter dominated (45 s of memory traffic at 75 GB) — worse than the per-block GEMMs it was meant to replace. The structure has unique `(tidA, sp_val)` per block, so reuse is 0. | sparse_h_H1_tb.log |
| H2 | The `SB_FUSE_LINKS` wrapper heuristic emits a `canon_labels` order matching the next call's preferred `B` layout. | **Refuted for PXP-PHP.** Diagnosis via `SB_PERMB_DBG` showed the heuristic was fitted to spin-test geometry; applying it to PXP gave the wrong order. Was patched (changed `[keepA, b_other, c_prefix, b_to_next]` → `[b_other, keepA, b_to_next, c_prefix]`). | sparse_h_H1_lean3.log |
| H3 | The schema-driven caller-side hint (`_layout_hint_for_next`) eliminates `permB` in the next sparseH step. | **Partially confirmed.** Drops average `permB` cost ~30–50% by emitting `shared_prefix` in next-A's prefix order. Doesn't fully eliminate it because the first sparseH in each chain sees output from a non-hint (denseH × denseV) step. | pxp_php_*_auto.log |
| H4 | Pre-permuting H[k]'s dense tail (cheap: only `n_templates × blksize` elements) makes the kernel's `permute_A` identity. | **Confirmed.** `permute_A` avg dropped from ~30 µs/call to ~14 µs/call (just the identity check). Also improved cache locality in `main_loop`. | pxp_php_N20_bd60_auto_prep.log |
| H5 | Aliased's memory compression on H should translate into less data movement at matvec time. | **Refuted at large bd.** What moves in `permute_B`/`permute_back`/`alloc` is `Hv` (dense intermediate), not H. Aliased only saves on H's storage, not on the matvec intermediates. At bd=120, 60% of the kernel's wrapped×dense time is `Hv` data movement. | bd=120 timer reports |
| H6 | At larger bd, BLAS GEMM efficiency improves, so the per-call BLAS overhead becomes negligible. | **Confirmed but uninformative.** Per-GEMM efficiency went from ~3.6 GF/s (bd=60) → ~5.1 GF/s (bd=120). However, memory ops scale linearly with `Hv` size while BLAS scales sub-linearly with throughput improvement — so memory dominates more, not less. | bd=120 main_loop avg |
| H7 | Pre-permuting L/R envs after each `position!` makes step-2 `permB` identity. | **Inconclusive / introduced regression.** `permB` did drop ~50% on the wrapped×dense path, but DENSE_excited at bd=120 went 20 s → 37 s with no clear cause. The helper has an early-return for non-fused tensors so DENSE shouldn't be affected. Needs root-cause before re-enabling. | pxp_php_N20_bd120_auto_prep_envprep.log |
| H8 | The `(1,3,2,2,1)` pattern dominates calls (73%), so hardcoding its `permB`/`perm_C` would help. | **Partially confirmed.** Profiler shows top-7 patterns cover 97% of calls. But the cost of choosing the perm isn't where time goes — APPLYING it is. Hardcoding pattern dispatch saves only the small "perm computation" overhead. | pxp_perm_profile.log |
| H9 | Using `itensor` (lowercase, AllowAlias) instead of `ITensor` saves a per-call copy of the dense output. | **Confirmed.** `wrap_output` dropped from 579 ms (1.39 GiB allocated) to 38 ms (28 MiB) at bd=10 — ~94% time drop. | sparse_h_H1_lean3.log |

## File structure

```
test_sparse_ham/
├── README.md                          ← this file
├── SKILLS.md                          ← how-to-debug guide
├── aliased_helpers.jl                 ← shared utilities included by both test drivers:
│                                         sparse_prefix_inds, _fuse_aliased_strands,
│                                         fuse_sparse_links!, prepermute_aliased_mpo!,
│                                         report_aliased_footprint,
│                                         run_dmrg_ground, run_dmrg_excited
├── test_check_working_aliased.jl      ← spin-N aliased PHP test (AliasedBlockSparse)
├── test_pxp_aliased.jl                ← PXP aliased PHP test (S=1, NotEqlsLoop_R1 projector)
├── measure_env_aliasing.jl            ← diagnostic: env tensor aliasing measurement
└── logs/
    ├── sparse_h_*.log                 ← spin-test results (H1, H2, lean, auto, etc.)
    ├── pxp_php_*.log                  ← PXP-test results (smoke, N20 bd60/80/120, auto/prep/envprep)
    └── back_to_back_*.txt             ← back-to-back run summaries
```

Both test drivers `include("aliased_helpers.jl")` for shared code. Model-specific
logic (OpSum, projector MPO, initial state) stays in each driver. Adding a new
model means copying one test driver and keeping `aliased_helpers.jl` unchanged.

## Cross-references

- Kernel: [SparseBackends/src/tensoralgebra/contract_aliased_dense_to_dense.jl](../SparseBackends/src/tensoralgebra/contract_aliased_dense_to_dense.jl)
- Wrapper: [SparseBackends/src/tensor_wrappers_aliased.jl](../SparseBackends/src/tensor_wrappers_aliased.jl)
- Matvec / hint helper: [ITensorMPS.jl/src/abstractprojmpo/abstractprojmpo.jl](../ITensorMPS.jl/src/abstractprojmpo/abstractprojmpo.jl)
- Dense PXP reference: [test_pxp_psi/test_dense_pxp.jl](../test_pxp_psi/test_dense_pxp.jl)
