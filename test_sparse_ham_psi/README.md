# `test_sparse_ham_psi/` — aliased ψ × aliased PHP (Kitaev ladder)

The "both-aliased" KL experiment: the wavefunction ψ is stored as
`WrappedAliasedBlockSparse` **and** the projected Hamiltonian `PHP` is stored
aliased, run together through Path-B DMRG. This is the union of two
previously-separate, individually-validated lines of work:

| folder | ψ | H | status |
|---|---|---|---|
| `test_aliased_psi/` | **aliased** | dense **bare** H | validated |
| `test_sparse_ham/`  | dense | **aliased PHP** | validated |
| **`test_sparse_ham_psi/` (this)** | **aliased** | **aliased PHP** | NEW — see status below |

ψ is built `contract(P, ψ₀, :coo, :aliased; denseLinksB=0)` (same as
`test_aliased_psi/test_aliased_kl.jl`); the aliased PHP is the per-site sandwich
`P''·H'·P` via `contract_aliased_itensor` (same as
`test_sparse_ham/test_check_working_aliased.jl`). DMRG runs the aliased ψ on the
aliased PHP via Path-B (`BMF_ISO_PATH=0`, `BMF_BOP_PROJECT=1`,
`BMF_MINV_RTOL=1e-2`, the aliased-ψ gate set).

## Files

| file | role |
|---|---|
| `test_sparse_ham_psi_kl.jl` | the both-aliased KL runner. Builds aliased ψ + aliased PHP, runs Path-B DMRG, and (with `--dense-ref`, default on) also builds a **dense PHP + dense ψ** and runs it as an in-script `|ΔE|` correctness reference. Prints alias invariant per sweep, honest BD vs MAXDIM, dedup, footprint, and a VERDICT block. CLI: `--N-plaq --eignv --spin --maxdim --n-sweeps --target-energy --dense-ref`. |
| `SKILLS.md` | operating notes; **read the contention rule before running anything**. |

It reuses the sister helpers (no duplication): `../test_sparse_psi/utils.jl`
(`clean!`) and `../test_sparse_ham/aliased_helpers.jl`
(`fuse_sparse_links!`, `prepermute_aliased_mpo!`, `report_aliased_footprint`).

## Root-cause diagnosis (Confidence: High for mechanism; fix NOT yet verified e2e)

The combination initially crashed in the DMRG matvec with
`AssertionError: ... shared label crosses prefix/dense boundary` (and, in the
default per-step-hint mode, `C sparse prefix must equal A sparse prefix for
dense-tail reduction`). This is **not** a wrong computation or a test-setup
error — it is an **over-promotion in the matvec's output-classification hint**:

- Expected structure (doubled-link convention): ψ site = 3 sparse (site + 2 bond
  links) + 2 dense (multiplicities); H site = 4 sparse (2 sites + 2 MPO links) +
  2 dense. A matvec reduces over **sparse site/link + at most one dense link**,
  with each shared axis classified the **same** on both operands. No crossover
  should occur.
- The **env-build** `_mul_preserve_aliased`
  ([abstractprojmpo.jl:29-31](../../temp/temp/edited_packages/ITensorMPS.jl/src/abstractprojmpo/abstractprojmpo.jl#L29))
  builds its hint as the **union** `dense_inds(A) ∪ dense_inds(B)` when both
  operands are aliased — keeping both operands' dense (multiplicity) axes dense.
- The **matvec** passed `output_inds_hint = dense_inds(v)` — **only ψ's** dense
  axes. When H is also aliased, **H's own dense multiplicity axes are not in the
  hint**, so `output_inds`
  ([tensor_wrappers.jl:967-969](../../temp/temp/edited_packages/SparseBackends/src/tensor_wrappers.jl#L967))
  **fissions them into the sparse prefix**. That multiplicity becomes *prefix* in
  the intermediate while the next aliased operator keeps it *dense* → the
  reduction crosses the prefix/dense boundary, which the aliased×aliased kernel
  rejects ([contract_aliased_shared.jl:791](../../temp/temp/edited_packages/SparseBackends/src/tensoralgebra/contract_aliased_shared.jl#L791)).

This path was never exercised before: aliased H only ran against *dense* ψ (no
hint), and aliased ψ only against *dense* H (one dense operand ⇒ no aliased
prefix/dense classification to conflict).

## The fix

`abstractprojmpo.jl`, the **both-aliased** matvec step (`sparseH_wrapV` branch):
when `it` (H) **and** `Hv` are both aliased, use the **union hint**
`dense_inds(Hv) ∪ dense_inds(it)` — mirroring the already-correct env-build.
Gated by `SB_ALIASED_AA_HINT` (default on) and strictly confined to the
aliased×aliased branch, so the **only-φ-aliased (dense H)**, **only-H-aliased
(dense ψ)**, and **BS** pathways are byte-identical — including the
`test_aliased_psi` aliased-ψ × dense-H benchmark (its H is dense ⇒ the branch is
never entered).

## Status (Confidence-tagged, 2026-06)

**Crashes FIXED — matvec runs end-to-end; structure preserved (Confidence: High).**
The three crash root-causes are resolved and gated to the both-aliased path:
1. **Unfused PHP bonds.** The aliased sandwich `P''·H'·P` leaves multi-strand
   link indices at different prime levels; `SB_FUSE_LINKS` fuses them so each
   H site is the clean **4-sparse (2 sites + 2 fused links) + 2 dense**
   structure. (Default on in the runner.)
2. **Dense matvec env.** `_makeL!/_makeR!` used plain `*`, densifying the env, so
   its multiplicity axes lost their classification and were over-fissioned. Now
   they use `_mul_preserve_aliased` (gated `SB_ALIASED_AA_ENV`, both-aliased
   only) so the env stays aliased. `_reorder_env_for_aliased` skips aliased envs
   (no aliased `ITensors.permute`; the reorder is a dense-kernel perf canon).
3. **Stale matvec hint.** The matvec passed `dense_inds(v)` (original φ ids); now
   it uses the **current operands' union** `dense_inds(Hv) ∪ dense_inds(it)`
   (gated `SB_ALIASED_AA_HINT`, both-aliased only), mirroring the env-builder.

At N=2 the run completes with alias invariant per sweep, honest BD ≤ MAXDIM, and
dedup > 1× all PASS.

**md=16 crossover FIXED; energy correct at md=16 (Confidence: High).** The 4th
root cause was the **md=16 prefix/dense crossover** (`shared label … crosses
prefix/dense boundary`). Root cause (Confirmed via a temporary classifier-input probe, since removed): the Path-B
B_op's pre-H `M⁻¹ᐟ²` apply runs with `fission=false` (deferred-fission), which
passes `template=nothing` ⇒ `output_inds` gets **no hint** ⇒ falls back to "dense
iff in denseA∪denseB"; the `M⁻¹ᐟ²` factor `Linv_R` was built **fully dense**, so
its channel id is in `denseA` ⇒ the output channel was parked in the dense tail
(P=2 non-canonical) ⇒ crossover vs the channel-in-prefix aliased env. **Fix
(case-4 gated, `SB_ALIASED_MINV_WRAP`):** relayout the `M^{±1/2}` factors as
**aliased** carrying φ's {channel→prefix, mult→dense} split
(`wrap_dense_as_aliased_via_template` + threaded `both_aliased`), so the apply is
canonical natively — no hint, no forced fission (deferred-fission 2× kept). Case 2
(dense H) keeps dense factors ⇒ byte-identical. Pure relayout, values bit-identical.

Verified end-to-end **N=2 md=16** (4 sweeps): all four VERDICT gates PASS,
aliased E=−3.6753 vs dense PHP E=−3.7288, **|ΔE|=5.35e-2 (1.43%)**, 5.58× vs-dense
compression, dedup mean 2.67×. Reproduce (from `edited_packages/`):
`julia --project=. test_sparse_ham_psi/test_sparse_ham_psi_kl.jl --N-plaq 2 --maxdim 16 --n-sweeps 4 --dense-ref true`.
The earlier md=8 −3.241 plateau was NOT a bug: it is bit-identical to the
validated bare-H aliased run (the known low-maxdim aliased truncation regime);
|ΔE| shrinks with maxdim (md=16 → 1.43%).

**Regression (still required at larger N / before declaring fully solved):** with
the gates off (or because they are aliased-only), `test_sparse_ham/test_check_working_aliased.jl 4`
(case 3) and `test_aliased_psi/test_aliased_kl.jl --N-plaq 4 --maxdim 20` (case 2)
must be unchanged — the `both_aliased` gate keeps them on the original dense-factor
path. Larger-N (N=4+) convergence of this combination is **not yet run**.

## Cross-references

- `../test_aliased_psi/` — aliased ψ machinery, Path-B, the 5-gate fix.
- `../test_sparse_ham/` — aliased PHP build, matvec kernel, perf notes.
- `SparseBackends/src/tensor_wrappers.jl:929` — `output_inds` (the fission classifier).
- `ITensorMPS.jl/src/abstractprojmpo/abstractprojmpo.jl` — matvec + env-build + the fix.
