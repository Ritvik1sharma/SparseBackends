# `test_aliased_psi/` — DMRG with aliased-BlockSparse psi

This folder drives experiments that store the DMRG wavefunction `psi` (MPS) as
`WrappedAliasedBlockSparse` instead of plain dense or `WrappedBlockSparse`.
`AliasedBlockSparse` (defined in `../SparseBackends/src/aliased/storage.jl`)
stores each logical block as `scalars[i] * templates[alias_ids[i]]`, so when
many block keys share a template the wavefunction collapses to
`O(n_templates × blksize + n_blocks)` memory instead of
`O(n_blocks × blksize)`.

The target is **end-to-end DMRG with aliased psi sites that stay aliased
across sweeps, with energies matching the dense / BlockSparse reference, and
ideally a per-sweep speedup over both.**

---

## ⚠ ARCHIVED: Rayleigh-Ritz local eigensolve (PARKED 2026-06)

A third local-eigensolve path — **generalized Rayleigh-Ritz** — was added and
then **parked**. It solves `H_eff·φ = E·M·φ` WITHOUT ever applying `M^{±1/2}` to
a vector (and WITHOUT `BMF_MINV_RTOL`): it builds a small aliased Krylov subspace
from `H·v` only, forms tiny `k×k` `H_small`/`M_small` via scalar inner products
(M applied with the **raw** `Lgram`/`Rgram`), and solves a null-projected `k×k`
generalized eig. The Ritz vector `Σ cᵢ vᵢ` stays aliased.

**Status — default OFF, do not enable in benchmarks.** Gated entirely behind
`BMF_RAYLEIGH_RITZ=1` (unset ⇒ byte-identical to the prior B_op/A_op behavior).

**What was verified (N=4, md=40):**
- **Correct**: matches the dense generalized-eig oracle at b=1/b=3 to ~1e-11
  (`diag_gram_metric.jl` D4, only runs under `BMF_RAYLEIGH_RITZ=1`); end-to-end
  energy matches B_op to 6 digits (RR −6.417847 vs B_op −6.417844).
- **Stays aliased**: `_ADD_PLUS_DENSE=0`, storage invariant ✓ every sweep, dedup
  preserved — the footprint win is intact.
- **Slower**: ~**2.5×** slower per sweep than the default B_op path (RR 2.18 s vs
  B_op 0.86 s, sweeps 6–10). Cause: the H-matvec — RR does ~25% more matvec calls
  (redundant `HV` recompute + thick-restart rebuilds) **and** ~3× slower per call
  (it feeds H the recast/per-channel-fissioned basis vectors, vs B_op's big-block
  deferred-fission input). The M-side is actually cheaper (raw gram, no `M^{1/2}`).

**Known limitation (pre-existing, NOT an RR bug):** at b=2 on the fresh
non-canonical ψ the aliased `product(PH_ali, ·)` matvec is itself wrong
(`raw==recast`, both ≠ dense `‖Hφ‖²`; `⟨φ|Hφ⟩` right but `⟨φ|H²|φ⟩` wrong) — the
same bond where D2 diverges and D3 crashes. RR inherits the wrong operator there
but is the most graceful of the three (no crash/divergence); it does not affect
end-to-end convergence because the evolving ψ avoids that worst-case config.

**To resume (perf work):** set `BMF_RAYLEIGH_RITZ=1`. Two clear optimizations —
(1) build `H_small` from a Lanczos/Arnoldi recurrence to drop the redundant
H-applies; (2) feed H big-block (deferred-fission) input like B_op to recover the
~3× per-matvec speed. `BMF_RR_RTOL` (default `1e-8`) tunes the `k×k` null cutoff;
a flat energy-vs-rtol band would confirm the rank-deficiency handling is robust.

**Code locations:** `rayleigh_ritz_local_eigsolve` + `solve_small_geneig` in
`../SparseBackends/src/path_b_helpers.jl`; branch in `../ITensorMPS.jl/src/dmrg.jl`
(gated, marked ARCHIVED); oracle in `diag_gram_metric.jl` D4 (gated). A general
aliased `Base.:-` kernel + `ITensors._subtract` hook were also added (in
`../SparseBackends/src/tensor_wrappers_aliased.jl` and `../ITensors.jl/src/itensor.jl`);
these are **not** flag-gated and are kept — they make ITensor subtraction work for
aliased storage (previously it errored), symmetric to `+`, and cannot regress any
path that worked before.

---

## Files

| File | Purpose |
|---|---|
| `test_aliased_kl.jl` | **The canonical aliased-ψ KL runner (single backend).** Mirror of `../test_sparse_psi/test_sparse_kl.jl` with ψ stored aliased. Runs DMRG on the bare H via Path-B (`BMF_ISO_PATH=0`, `BMF_APPLY_MINV=1`), checks the alias invariant each sweep, and reports footprint, honest bond dim, per-site `n_blocks/n_templates`, an iso check, and a regression-to-dense verdict. CLI mirrors the sister runners: `--N-plaq`, `--eignv`, `--spin`, `--maxdim`, `--n-sweeps`, `--target-energy`. |
| `diag_step_by_step.jl` | Side-by-side diagnostic of one DMRG step in dense and aliased: initial psi value diff, matvec value diff, factorize reconstruction error, `L†L` iso check. |
| `diag_gram_metric.jl` | **Decisive Path-B correctness diagnostic.** At bonds 1–3 on the freshly-constructed non-canonical ψ: (A) aliased vs dense gram block-by-block; (B) `<φ\|M\|φ>` via the aliased `M_dot` machinery vs dense reference vs `<Ψ\|Ψ>`; (C) `<φ\|H_eff\|φ>` vs `<Ψ\|H\|Ψ>`; (D) dense ground-truth generalized eig vs actual aliased Path-B `eigsolve`, with (D3) the `A = M⁻¹·H_eff` fix. This is what localized the energy bug to the eigsolve operator. |
| `measure_env_aliasing.jl` | Standalone: compute `H[k] × dag(prime(psi[k]))` densely, then partition the result under all candidate sparse-prefix axis classifications and report `n_unique_slices / n_keys` for each. Used to measure how much template-level dedup the env-build pipeline can achieve. |
| `scratch/` | Superseded scripts kept for reference (not deleted): the old `bench_aliased_vs_bs_vs_dense.jl` / `bench_footprint_honest_bd.jl` head-to-head benches (whose dense baseline was **invalid** — see below) and `test_aliased_php.jl` (the earlier iso-path aliased-only smoke test, replaced by `test_aliased_kl.jl`). |

---

## Baselines — use the sister-folder runners, do NOT recreate them

**Do not write per-benchmark BS/dense runners.** The dense and BS baselines for
the KL model already exist in `../test_sparse_psi/` and enforce the constraint
correctly. To compare aliased against them, run all three with **matching args**
(`--N-plaq`, `--maxdim`, `--n-sweeps`, `--eignv`, `--target-energy`):

| backend | runner | H used | constraint enforcement |
|---|---|---|---|
| dense | `../test_sparse_psi/test_dense_kl.jl` | **projected** `H_dense = densify(P·H·P)` | via the **PHP operator** (a dense ψ has no structural constraint) |
| BS | `../test_sparse_psi/test_sparse_kl.jl` | **bare** `H = MPO(os)` | **structural** — channel sparsity of `ψ = P·ψ₀` keeps it in `image(P)`; relies on `[H,P]=0` |
| aliased | `test_aliased_kl.jl` (this folder) | **bare** `H = MPO(os)` | **structural** — same as BS, plus alias dedup of the channel blocks |

All three share `Random.seed!(42)`, the same `os`/`P_sparse`, and the same psi
construction (`contract(P_sparse, ψ₀, :coo, …)`), so they are directly
comparable. The only run-level differences are intentional and required:
the **Hamiltonian** (projected for dense, bare for sparse/aliased — this is the
constraint-enforcement mechanism, not a free choice) and the **eigsolve path**
(`test_sparse_kl.jl` uses the iso path `BMF_ISO_PATH=1`; `test_aliased_kl.jl`
**must** use Path-B `BMF_ISO_PATH=0` because aliased ψ is structurally non-iso).

> **⚠ Why the old head-to-head bench was retired (constraint not enforced honestly).**
> The old `bench_aliased_vs_bs_vs_dense.jl` ran its *dense* baseline as
> `dmrg(bare H, copy(psi0))` where `psi0 = random_mps(sites)` — an **unprojected**
> random ψ optimized against the **bare** H. That solves the *unconstrained*
> ground-state problem of H, not the constrained one: it neither applies PHP nor
> starts in `image(P)`. Its "dense" energies are therefore **not a valid
> constrained baseline**, and any dense-vs-ALI energy comparison built on it is
> unsound. (Confidence: Confirmed — it is directly visible in the retired
> script, now in `scratch/`.) The honest dense baseline is the PHP runner
> `../test_sparse_psi/test_dense_kl.jl`.

---

## ✅ RESOLVED — aliased Path-B converges at scale with the memory win (Confidence: High, 2026-06)

> **Both goals now hold simultaneously on the correct-energy path.** With the
> 5-gate fix below, `test_aliased_kl.jl` keeps ψ **aliased every sweep** (real
> dedup preserved) AND converges to **near-dense energy**, validated at
> `N_plaq = 2, 4, 12`.
>
> **N=12, md=40, 6 sweeps** (vs dense PHP `test_dense_kl.jl`):
>
> | | dense | aliased (full fix) |
> |---|---|---|
> | final E | −17.1828 | **−17.1605** (gap 0.13%, monotonic, no oscillation) |
> | footprint | 1.312 MiB | **0.196 MiB (6.7× smaller)** |
> | dedup / payload compression | — | 3.69× / **7.90× vs dense** |
> | honest BD | 40 | 40 (bounded) |
>
> Energy converges monotonically (−16.59 → −17.16), sits just above dense
> (variational, physical), and aliasing/dedup are fully intact.
>
> **The fix — 5 env gates (all default-off; turn all on together):**
> 1. `SB_ALIASED_MINV_HINT=1` — classification-preserving `output_inds_hint` for
>    the aliased template in `contract_preserve_bs`, so the `M⁻¹` apply keeps φ's
>    `{N2,P}` split and ψ does **not** collapse to dense (fixes the cross-sweep
>    aliased→dense cascade).
> 2. `SB_ALIASED_NATIVE_FISSION=1` — native aliased fission kernel (dedup-preserving;
>    no dense round-trip).
> 3. `SB_ALIASED_PERCM_CAP=1` — per-cM truncation cap `mult = fld(maxdim/bond_ch_dim)`
>    in `itensor_aliased_factorize`, so the **honest** bond dim (channel × mult)
>    obeys MAXDIM (mirrors the BS `per_cM_cap`, `ops_factorize_qr.jl`).
> 4. `BMF_BOP_PROJECT=1` — solves `B = M⁻¹ᐟ²·H_eff·M⁻¹ᐟ²` with the **standard**
>    inner product and null-zeroed `M⁻¹ᐟ²`, projecting the eigensolve onto
>    range(M). Replaces the fragile `A = M⁻¹·H_eff` + M-inner-product path. Robust
>    to the structurally **rank-deficient** aliased M (template sharing ⇒ non-iso ⇒
>    iso is unreachable without un-deduplicating).
> 5. `BMF_MINV_RTOL=1e-2` — aggressive pseudo-inverse cutoff dropping M's near-null
>    directions. (1e-10 / 1e-6 / 1e-4 all still oscillate; 1e-2 converges.)
>
> **The gram/metric M is constructed CORRECTLY** — verified `|dLgram| = |dRgram| =
> 0` to machine precision at N=2 *and* N=4 (`diag_gram_metric.jl DIAG_GRAM_ONLY=1`).
> The historical instability was **inverting** a correctly-built but rank-deficient
> M, not a kernel bug.
>
> **Residual risk / not-yet-Confirmed:** the 0.13% energy gap (aliased-manifold DOF
> cost; may shrink with more sweeps / rtol tuning); `rtol=1e-2` is hand-tuned, so
> robustness across models/sizes isn't proven. **Next (perf + robustness):** a
> reduced-space / low-rank factored solve — `M = U_r Λ_r U_r'` is already low-rank
> from the eigendecomposition; solve the small `r×r` reduced generalized problem in
> range(M) directly instead of inverting through the metric. That removes the
> hand-tuned rtol and the O(d²) dense `M^{±1/2}` apply (the ~2.7× slowdown vs dense).
>
> Reproduce: `SB_ALIASED_ENABLE=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 SB_ALIASED_MINV_HINT=1 SB_ALIASED_NATIVE_FISSION=1 SB_ALIASED_PERCM_CAP=1 BMF_BOP_PROJECT=1 BMF_MINV_RTOL=1e-2 julia --project=. test_aliased_psi/test_aliased_kl.jl --N-plaq 12 --maxdim 40 --n-sweeps 6`

---

## What works (achieved)

> See "RESOLVED" above for the full validated end-to-end result. The historical
> note below about the "preserved across sweeps" claim was originally verified on
> the **iso path only** (`BMF_ISO_PATH=1`, wrong
> energy for aliased). On the correct-energy Path-B it does **not** hold — ψ
> collapses to dense after sweep 1. The construction-time and per-kernel results
> below are still accurate; the cross-sweep *preservation* claim is path-specific.

End-to-end DMRG with aliased psi **runs to completion**, and at **construction**
the alias structure is present (`psi[i].tensor.data isa
WrappedAliasedBlockSparse`, dedup `n_blocks/n_templates` = 2–4×). On the **iso
path** this is preserved across sweeps; on **Path-B it is NOT** (see KNOWN ISSUE).
The pieces that landed:

1. **Construction.** `contract(P_sparse, psi0, :coo, :aliased; denseLinksB=0)`
   builds an aliased MPS via `wrap_itensor_aliased` + the existing
   `contract_aliased_coo_dense.jl` kernel. Bond canonicalization is native
   (`_canonicalize_bond_aliased!`), and `fuse_axes!` handles prefix-only
   and tail-only axis fuses without demoting to BS.

2. **Aliased factorize (Tier-2 alias-reduced SVD).** In
   `../SparseBackends/src/aliased/factorize.jl`,
   `itensor_aliased_factorize` builds a reduced
   `M_red[a_L, a_R]` matrix sized `n_t_L × n_t_R`, applies per-channel
   whitening, SVDs, then writes `L, R` aliased back with **frozen schema**
   from `M_b / M_b1` (keys / alias_ids / scalars verbatim) and only fresh
   templates. Wired into `replacebond_sparse!`, `orthogonalize!` (both
   sweep directions), and `stable_factorize`.

3. **Aliased VectorInterface.** `eltype`, `_zero_similar`, scalar `*`,
   `Base.:+` (with same-schema fast path AND cross-schema merge that
   preserves aliased storage), Index-permutation alignment in `+`, `norm`,
   `_external_fill!`, `_apply_elementwise!` — implemented in
   `../SparseBackends/src/tensor_wrappers_aliased.jl`.

4. **Native aliased fission in `contract_shared!`** (mirror of the BS
   `_contract_shared_hint!` pattern). When `output_inds_hint` moves
   `keepB` labels into C's prefix, the kernel iterates over moved-key
   values and produces the correctly classified aliased output natively,
   without delegating to BS. Hint is forwarded from
   `wrapped_contract_aliased` → `contract!` → `contract_shared!`.

5. **`recast_aliased_to_template`** (analog of BS `recast_bs_to_template`)
   — permutes a `WrappedAliasedBlockSparse`'s axes via `permutedims` so
   its `inds` tuple matches a template's order, preserving the alias
   schema. Wired into `apply_minv_preserve_bs` and the Path-B `H_op` in
   `ITensorMPS.jl/src/dmrg.jl`.

6. **`is_sparse_mps` recognizes aliased** so the gram-cache and Path-B
   branches in `dmrg.jl` fire for aliased psi too.

7. **Aliased-aware multiply in env-build** (`_makeR!`, `_makeL!`) via
   `_mul_preserve_aliased` — gated to fire only when both operands are
   aliased, otherwise falls through to plain `*`.

---

## Energy correctness — fix identified (HIGH confidence, not yet "Confirmed", 2026-06)

> **Confidence: HIGH (not Confirmed).** Decisive diagnostic + end-to-end runs at
> `N_plaq=2` strongly support the diagnosis and fix, but it is NOT proven
> complete. See "Residual risk" at the end of this section. Keep `BMF_APPLY_MINV`
> as the toggle and `diag_gram_metric.jl` as the regression guard.

**Root cause found, fix verified at small scale.** The aliased Path-B `eigsolve`
was solving the wrong eigenproblem: `H_op` applied `H_eff` under the M-inner
product but never applied `M⁻¹`. The M-self-adjoint operator for the generalized problem
`H_eff·φ = E·M·φ` is `A = M⁻¹·H_eff` (so that `⟨x, A y⟩_M = x†H_eff y` is
Hermitian and Lanczos returns the *generalized* eigenvalues). Omitting `M⁻¹`
made Lanczos solve a different (non-variational) problem → unphysical energy.
When ψ is canonical (`M = I`, the strict-iso BS case) `M⁻¹ = I` and the omission
was harmless — which is why BS "worked" and the bug only surfaced for aliased ψ
(where `M ≠ I` is structurally unavoidable; see hypothesis #5).

The fix (in `ITensorMPS.jl/src/dmrg.jl`, gated by `BMF_APPLY_MINV`, default on)
applies `M⁻¹ = Linv_L² ⊗ Linv_R²` inside `H_op`, reusing the `Linv_L`/`Linv_R`
factors that `build_minv_half_pair_factored` *already returns* but the old code
discarded.

> **⚠ The `dense` columns below are from the retired `bench_aliased_vs_bs_vs_dense.jl`
> and are NOT a valid constrained baseline** — that script ran dense as bare-H
> DMRG from an unprojected random ψ (no PHP, not in `image(P)`; see "Baselines"
> above). The dense-vs-ALI energy comparison must be **re-done** against
> `../test_sparse_psi/test_dense_kl.jl` (PHP) at matching args before any
> "approaches/beats dense" claim is trusted. The `ALI` and `BS` energies
> themselves are valid (they enforce the constraint structurally); only the
> dense reference and the "approaches dense" wording are void. The independent
> correctness evidence below (`diag_gram_metric.jl`, generalized-eigenvalue to
> 10 digits) does NOT depend on the dense DMRG baseline and still holds.

`bench_aliased_vs_bs_vs_dense.jl` (RETIRED, in `scratch/`) at `N_plaq = 2, maxdim = 4, 2 sweeps`:

| backend | final E (before) | final E (after fix) |
|---------|------------------|---------------------|
| dense ⚠ | -3.5259          | -3.5259 (invalid baseline — see warning) |
| BS      | -3.3887          | -3.3887 (bit-identical — fix is a no-op for canonical ψ) |
| **ALI** | **-14.4453** ❌  | **-3.6185** ✅ (Path-B, physical; ALI *below* this "dense" is itself a tell the dense ref was wrong/under-converged) |

Converged over a `maxdim 4→8→16`, 9-sweep schedule (`N_plaq=2`):

| backend | conv E, N=2 (md→16, 9 sw) | conv E, N=4 (md=20, 10 sw) |
|---------|---------------------------|-----------------------------|
| dense ⚠ | -3.7291328058             | -6.4177077771 |
| BS      | -3.6974378382             | -6.3576855008 |
| **ALI** | **-3.7290811501** (Δ≈5e-5) | **-6.4175335807** (Δ≈1.7e-4) |

(dense ⚠ = retired-bench unprojected baseline; re-validate via `test_dense_kl.jl`.)
ALI convergence is monotonic and physical at every sweep. The ALI-vs-dense
closeness is **suggestive but not established** until the PHP dense baseline is
re-run — for the KL +1 sector the unconstrained GS may happen to lie in
`image(P)`, which would explain the apparent agreement, but that is a hypothesis,
not a verified baseline.

Diagnostic evidence (`diag_gram_metric.jl`): the fixed operator `A = M⁻¹·H_eff`
reproduces the dense ground-truth generalized eigenvalue to **10 digits** at
bonds 1–3 (e.g. bond 1: −1.0303737843 both ways; old code gave −0.9237860004).

**The AliasedBS SVD, gram, metric, `H_eff`, and `apply_minv` kernels are all
functionally correct** — the energy discrepancy did NOT stem from a correctness
bug in any of them. Measurements that hold up:
- One-step matvec value matches dense (`Hphi_diff ≈ 1.15e-17`).
- One-step factorize **reconstruction** is exact (`||L*R − phi|| ≈ 1.08e-16`).
- **Gram cache is exact**: `|densify(Lgram_ali) − Lgram_dense| = 0` and same for
  `Rgram` at every bond (`diag_gram_metric.jl` Part A — refutes hypothesis #7).
- **Metric is exact**: `<φ|M|φ>` via the aliased `M_dot` machinery equals
  `<Ψ|Ψ>` exactly at every bond (Part B — refutes hypothesis #6's suspicion that
  `apply_minv` on aliased ψ drifts).
- One-step factorize **isometry fails** (`L†L` off-diagonals ≈ -0.75) — this is
  EXPECTED and structural (hypothesis #5), and Path-B's `M⁻¹` correction is the
  principled answer to it. It is no longer a "bug".

**Residual risk / not-yet-tested (why this is HIGH, not Confirmed):**
- Validated at `N_plaq=2` (6 sites, ≤16 maxdim, 9 sweeps) AND `N_plaq=4`
  (10 sites, maxdim=20, 10 sweeps): both converge monotonically, ALI matches
  dense (Δ≈5e-5 / 1.7e-4) and beats BS, with ALI staying above well-converged
  dense (no variational violation). `N_plaq=16` still untested (README notes it
  takes hours).
- `M⁻¹` is a thresholded pseudo-inverse (`build_half_pair_single` zeroes
  eigvals < `rtol·maxλ`). Near-singular `M` at large bond dim / late sweeps
  could make the correction unstable — not stress-tested. Tested cases had `M`
  full-rank.
- `apply_minv_preserve_bs(dense Mhalf/Linv, aliased φ)` is exact in the tested
  cases, but cross-schema / differing-alias-schema inputs over many sweeps are
  not exhaustively covered.
- ALI tracking *below* dense at intermediate sweeps is expected (dense
  under-converged) but a true variational violation would also look like this —
  watch for ALI dropping below a well-converged dense reference.
- **Further evaluation needed:** yes — confirm at scale, stress near-singular
  `M`, and watch many-sweep stability before marking Confirmed.

### Memory footprint goal

**The guideline:** aliased and BlockSparse psi should have a smaller memory
footprint than dense psi as `N_plaq` grows. This is the primary structural
motivation for these backends: block-sparsity (BS) and alias deduplication (ALI)
mean the wavefunction does not need to store `O(D × blksize)` data per bond when
symmetry or alias compression is present.

**Exception — small-`N_plaq` overhead:** At very small system sizes, fixed
per-block metadata (keys, alias_ids, scalars, wrapper index objects) can
outweigh the data savings, so ALI or BS footprint exceeding dense at small
`N_plaq` is not automatically a regression — it may simply reflect
implementation overhead.

**When ALI/BS footprint exceeds dense — trigger further evaluation:** If
footprint exceeds dense at a scale where overhead should not dominate, the cause
must be investigated before drawing conclusions. Two candidate causes (both
hypotheses until measured):
1. **Pure implementation overhead** — metadata or bookkeeping growing with
   system size independently of tensor data. Not a structural regression, but
   may point to unnecessary allocations.
2. **Datastructure regression** — alias compression or block-sparsity breaking
   down (e.g. one-template SVD collapse where `n_templates → 1`, bond dim
   silently exceeding MAXDIM, unpruned blocks). This is the scenario to fix.

Concretely: at each benchmark run, record:
1. `n_blocks / n_templates` ratio at psi sites (should be > 1 for real dedup).
2. The actual bond dimension at each site vs MAXDIM.
3. MPS footprint for ALI and BS vs dense.

If (3) shows ALI/BS ≥ dense, use (1) and (2) to distinguish overhead from
regression before forming a hypothesis about the fix.

### Performance

With the fix, ALI is competitive on all three axes. `N_plaq=2`, `maxdim 4→8→16`,
9 sweeps (averages exclude sweep 1 / JIT):

`N_plaq=2` (`maxdim 4→8→16`, 9 sweeps):

| backend | avg s/sweep | footprint | notes |
|---------|-------------|-----------|-------|
| dense   | 0.016 | 0.015 MiB | reference |
| BS      | 0.114 | 0.036 MiB | 0.14× dense speed; 0.42× dense footprint |
| **ALI** | **0.015** | **0.011 MiB** | **ties dense speed, 7.38× faster than BS; 1.38× smaller than dense, 3.29× smaller than BS** |

`N_plaq=4` (`maxdim=20`, 10 sweeps):

| backend | avg s/sweep | footprint | notes |
|---------|-------------|-----------|-------|
| dense   | 0.108 | 0.078 MiB | reference |
| BS      | 0.369 | 0.120 MiB | 0.29× dense speed; 0.65× dense footprint |
| **ALI** | **0.121** | **0.088 MiB** | **0.89× dense speed, 3.05× faster than BS; 1.36× smaller than BS (0.89× dense)** |

> **N_plaq=4 footprint note:** ALI (0.088 MiB) marginally exceeds dense (0.078 MiB)
> at this scale. This warrants further evaluation to determine whether it is
> pure implementation overhead or a datastructure regression (see "Memory
> footprint goal" above). Measure `n_blocks / n_templates` at each psi site
> and compare actual bond dimensions against MAXDIM before forming a hypothesis.

So with the fix, ALI is the best of the three on energy (matches dense, beats BS)
and beats BS on both speed and footprint; vs dense it ties/0.89× on speed and is
comparable on footprint at small scale. The old "190× slower / larger than dense"
regime is gone.

Old (broken-run, pre-`M⁻¹`-fix) figures, kept for reference only — NOT
representative. **⚠ Unverified provenance:** the command/path that produced
these is not recorded; they predate the current bench and may be from the iso
path (`BMF_ISO_PATH=1`) rather than Path-B. Do NOT treat them as comparable to
the bench numbers above.
- per-sweep avg: dense 1.288 s, BS 2.436 s, **ALI 246.78 s (~190× slower)**
- footprint at `N_plaq=16`: dense 4.15 MiB, BS 3.12 MiB, **ALI 4.35 MiB**

### Open TODOs

1. **Energy correctness — fix applied, HIGH confidence (not Confirmed).** The
   Path-B operator now applies `A = M⁻¹·H_eff` (gated by `BMF_APPLY_MINV`,
   default on). ALI energy went from -14.45 (unphysical) to -3.6185 (physical,
   beats BS, approaches dense). No per-channel-template / un-dedup workaround
   was needed. **Remaining to upgrade to Confirmed:** validate at scale
   (`N_plaq=4/16`), stress near-singular `M`, confirm many-sweep stability
   (see "Energy correctness → Residual risk").
2. **`build_minv_half_pair_factored` for aliased** — for aliased ψ it takes the
   dense branch (`build_half_pair_single` densifies the small gram and eigen-
   decomposes). `diag_gram_metric.jl` proves the resulting dense `Mhalf`/`Linv`
   applied to aliased φ via `apply_minv_preserve_bs` is **numerically exact**
   (metric and `M⁻¹` reproduce the dense reference to 1e-16 / 10 digits), so a
   dedicated aliased analog is a *performance* optimization, not a correctness
   requirement. Open only as a perf item (keep `Mhalf`/`Linv` aliased to avoid
   the dense×aliased contract per Lanczos iteration).
3. **Performance.** Profile the per-sweep cost on the larger competitive bench
   now that correctness holds. Suspects: per-channel whitening pass, key/lookup
   dict construction per factorize call, slow `permutedims` on aliased used by
   recast / `Base.:+` alignment, and the dense×aliased `apply_minv` per
   Lanczos step (see TODO #2).
4. **Memory footprint regression check (not yet diagnosed).** Confirm ALI and
   BS footprint < dense for all tested `N_plaq`. At `N_plaq=4` ALI (0.088 MiB)
   already marginally exceeds dense (0.078 MiB). **Diagnosis not yet done** —
   do not assume a cause. Start by measuring: print `n_blocks / n_templates` at
   each psi site after the final sweep and confirm the actual bond dimension at
   each site against MAXDIM. Only after that measurement should a hypothesis be
   formed (and tagged with a confidence level per the confidence discipline below).

---

## Working hypotheses tried

Listed in the order they were investigated, with verdict.

1. **`_snap_to_schema` drops data when Hv's key set differs from v's.**
   - Instrumented with `SB_ALIASED_SNAP_DBG=1` to track
     `dropped_keys`, `dropped_norm_frac`.
   - **Verdict: REJECTED.** Diagnostics show `dropped_keys=0`,
     `dropped_norm_frac=0.0` for every snap call. Snap preserves all data.

2. **Native fission kernel `_contract_aliased_prefix_outer_ad!` produces
   wrong templates.**
   - Toggled via `SB_ALIASED_NATIVE_FISSION=0/1` to fall back to a
     BS-delegation path.
   - **Verdict: REJECTED.** Same wrong energy with native fission OFF
     (BS fallback) as with native fission ON.

3. **Axis-permutation in `Base.:+` is the culprit.**
   - Added to fix the case where two aliased operands have identical N,
     N2, P but `dims` differ only in axis order.
   - **Verdict: HELPS but isn't the root cause.** Without it Lanczos
     hits the dense fallback on every `v + α·Hv`; with it the aliased
     path survives, but energy is still wrong.

4. **Factorize doesn't produce a left-isometric L** (i.e. `L†L ≠ I`).
   - Instrumented in `diag_step_by_step.jl` (Step 4). Initial check
     showed `iso_error = 0.77` and `norm(L) = 0.707` (should be
     `sqrt(bond_dim)`).
   - **Verdict: CONFIRMED partial cause.** Added per-channel whitening
     in `_aliased_alias_reduced_factorize` (multiply `M_red` by
     `w_L[a]·w_R[a]` before SVD, divide templates back after). After
     this, `L†L`'s **diagonal** is exactly 1.0, `norm(L) = sqrt(2)` —
     correct. **But off-diagonals remain ≈ -0.75.** The structural cause
     (next hypothesis) prevents full iso.

5. **Alias dedup along the bond-channel axis structurally prevents iso.**
   - For our projector-derived psi, each alias group `a` appears at
     multiple bond-channel values `c`. The map `(s, c) → a` is, e.g.,
     `(1,1)→1, (2,1)→2, (3,1)→3, (1,2)→3, (2,2)→2, (3,2)→1`. Different
     c-columns of L thus pick the **same template basis** but with
     different scalars → they are linear combinations of one another and
     **cannot be made orthogonal** without splitting templates per
     channel (i.e., un-deduplicating, which collapses to BS).
   - **Verdict: CONFIRMED structural.** This is the actual reason
     standard DMRG iso doesn't hold for aliased psi with the current
     single-template-per-block format.

6. **Path-B (M-corrected generalized eigsolve) should sidestep the iso
   requirement.** Standard DMRG assumes `L†L = I`; Path-B replaces this
   with the metric `M = L_gram ⊗ R_gram` and runs Lanczos with the
   M-inner product `<x, y>_M = inner(x, M·y)`. Original psi isn't iso
   either; Path-B works for BS in the same situation.
   - Enabled by extending `is_sparse_mps` to accept aliased and adding
     `recast_aliased_to_template` for the apply step.
   - **Verdict: CORRECT DIRECTION — Path-B *is* the answer, but the
     implementation was incomplete.** The "-14.45" was NOT because the gram
     densified or because the dense Mhalf drifted (that suspicion was WRONG —
     see #7 and `diag_gram_metric.jl` Part B, which shows the dense-Mhalf-on-
     aliased-φ metric is exact). The real omission: `H_op` applied `H_eff`
     under the M-inner product but never applied `M⁻¹`. Lanczos needs the
     M-self-adjoint operator `A = M⁻¹·H_eff` to return the *generalized*
     eigenvalues. **Fix applied (HIGH confidence, not Confirmed)** by applying
     `Linv_L² ⊗ Linv_R²` in `H_op` (`BMF_APPLY_MINV`, default on). ALI energy:
     -14.45 → -3.6185 (physical). Verified by `diag_gram_metric.jl` Part D3:
     the fixed operator reproduces the dense ground-truth generalized eigenvalue
     to 10 digits at bonds 1–3. Residual risk (scale, near-singular `M`,
     many-sweep stability) listed in "Energy correctness" above.

7. **Gram cache values are wrong for aliased psi** (because gram is built
   via `psi[i] * dag_link_primed(psi[i])` which goes through aliased
   kernels).
   - **Verdict: REJECTED.** `diag_gram_metric.jl` Part A builds the gram both
     ways (from aliased ψ and from densified ψ) and compares block-by-block:
     `|densify(Lgram_ali) − Lgram_dense| = 0` and `|dRgram| = 0` at **every**
     bond. The aliased gram kernel is exact. The bug was in the eigsolve
     operator (#6), not the gram.

---

## How to reproduce the current state

Run all three backends with **matching args** to compare. The aliased runner is
in this folder; the BS and dense baselines are the sister-folder runners (do not
recreate them — see "Baselines" above). Annotate every number you record with
the exact command that produced it.

```bash
cd edited_packages

# ── Aliased ψ (this folder), Path-B. --target-energy logs the first sweep ≤ target.
SB_ALIASED_ENABLE=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
  julia --project=. test_aliased_psi/test_aliased_kl.jl \
  --N-plaq 12 --maxdim 40 --n-sweeps 10 --target-energy -6.4

# ── BS baseline (sister folder): bare H, structural constraint, iso path.
BMF_ISO_PATH=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
  julia --project=. test_sparse_psi/test_sparse_kl.jl \
  --N-plaq 12 --maxdim 40 --n-sweeps 10 --target-energy -6.4

# ── Dense baseline (sister folder): PROJECTED Hamiltonian P·H·P (honest constraint).
  julia --project=. test_sparse_psi/test_dense_kl.jl \
  --N-plaq 12 --maxdim 40 --n-sweeps 10 --target-energy -6.4

# ── Diagnostics (aliased only) ──
# Step-by-step (matvec match, factorize reconstruction, iso check)
SB_ALIASED_ENABLE=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
  julia --project=. test_aliased_psi/diag_step_by_step.jl
# Path-B correctness (gram / metric / H_eff / generalized-eig + M⁻¹ fix)
SB_ALIASED_ENABLE=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
  julia --project=. test_aliased_psi/diag_gram_metric.jl
# Achievable env-build dedup under each prefix classification
julia --project=. test_aliased_psi/measure_env_aliasing.jl
```

### Useful environment knobs

- `BMF_APPLY_MINV=1` — apply the M⁻¹ correction (`A = M⁻¹·H_eff`) in the Path-B
  eigsolve operator. **Default on; this is the energy-correctness fix.** Set to
  `0` to reproduce the old (broken) behavior where `H_op` applied `H_eff`
  without `M⁻¹` (ALI → -14.45). No-op for canonical BS ψ (`M = I`).
- `BMF_ISO_PATH=1` — force standard eigsolve (assumes L iso). Default 0 = Path-B.
- `SB_ALIASED_NATIVE_FISSION=1` — use native aliased fission in
  `contract_shared!` (default off; falls back to BS-delegate).
- `SB_ALIASED_SNAP=1` — enable post-matvec snap to v's schema (default off).
- `SB_ALIASED_SNAP_DBG=1` — verbose data-loss tracking on each snap.
- `SB_ALIASED_TRACE=1` / `SB_ALIASED_TRACE_DEEP=1` — verbose tracing of
  matvec storage transitions and cross-schema merge fallbacks.
