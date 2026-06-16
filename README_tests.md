# Sparse DMRG — test file reference

All tests run from the `edited_packages/` directory:
```
julia --project=. <test_file.jl> [args...]
```

`utils.jl` must be on the include path. Currently located at `test_sparse_psi/utils.jl`.
Until the include path is fixed, symlink it: `ln -s test_sparse_psi/utils.jl utils.jl`.

---

## Integration / correctness

### `test_b_phase.jl`
**What:** Full sparse-vs-dense DMRG comparison. Reports energy gap, real bond dims (channel × mult), fidelity, and timing.

```
julia test_b_phase.jl <N> [nsweeps=8] [maxdim=40] [sparse_only=false]
BPHASE_VERBOSE=1 julia test_b_phase.jl 3     # adds inner-product + site-by-site H checks before DMRG
```

Use this to verify that Path B (sparse) gives lower or equal energy to dense and that real bond dims grow.

---

### `test_orthogonalize.jl`
**What:** Checks that `orthogonalize!` preserves ψ ∈ image(P). Four checks: fidelity, ‖ψ‖²/‖Pψ‖² invariance, image leakage ‖ψ−Pψ‖/‖ψ‖, and ⟨ψ|H|ψ⟩ invariance across all orthocenter positions.

```
julia test_orthogonalize.jl <N>
```

---

### `test_sparsity_preserved.jl`
**What:** Verifies DMRG never creates new block-key combinations. Checks per-site link-tuple counts and final-keyset ⊆ initial-keyset.

```
julia test_sparsity_preserved.jl <N> [nsweeps=2] [maxdim=20]
```

---

### `test_psi_properties.jl`
**What:** Post-DMRG properties of the converged sparse ψ: (1) memory compression in cells and real bytes (sparse vs densified), (2) ⟨ψ|Cⱼ|ψ⟩/‖ψ‖² = 1 for every constraint Cⱼ.

```
julia test_psi_properties.jl <N>
```

---

### `test_gram_matrix.jl`
**What:** Path B correctness at one bond. Builds the Gram matrix M explicitly, verifies that M⁻¹H_eff eigsolve gives the same eigenvalue as dense, and cross-checks ⟨ψ|H|ψ⟩ in all four sparse/dense H × sparse/dense ψ combinations.

```
julia test_gram_matrix.jl <N>
```

Use when debugging Path B eigenvalue problems or contraction-kernel bugs.

---

## SVD / factorization

### `test_relax_iso_microbench.jl`
**What:** Probes `itensor_blocksparse_svd_channel_aware` at every bond with and without `relax_iso_cap`. Checks: ‖phi − L·R‖ ≈ 0, block-key subset preservation, isometry error, and a dense-SVD reference.

```
julia test_relax_iso_microbench.jl <N>
```

---

## Performance profiling

### `test_profile.jl`
**What:** Production-regime profiler. Warmup phase (2 sweeps to maxdim=40 so bonds grow), then `n_prof` timed sweeps with per-sweep GC stats. Dumps `PROJMPO_TIMER` and `SparseBackends.TIMER`.

```
julia test_profile.jl <N> [n_prof_sweeps=6]
BENCH_TIMERS=1 julia test_profile.jl 4 3    # also runs dense head-to-head for sparse/dense ratio
DMRG_NO_WARMUP=1 ...                         # skip JIT warmup
```

Look at `gram_envs` fraction in PROJMPO_TIMER — this is the Path B M-build bottleneck.

---

## Kernel-level benchmarks (for optimization work)

### `test_kernel_microbench.jl`
**What:** Isolated BS×dense matvec performance at various plaquette counts and bond dims. Produces sparse/dense throughput ratios.

```
KB_NS="4,6,10" KB_PHYS=3 KB_REPS=100 KB_DPSI="20,40,80" julia test_kernel_microbench.jl
```
Default: `KB_NS=6,10,14`, `KB_PHYS=2`, `KB_REPS=200`, `KB_DPSI=8,20,40,80,160`.

---

### `test_kernel_precision.jl`
**What:** Numerical error of `contract_prefix_outer_bd!` vs a dense×dense reference. Also tests a Kahan-compensated variant and reports error improvement.

```
julia test_kernel_precision.jl    # no args; uses hardcoded synthetic tensors
```

---

### `test_path_b_microbench.jl`
**What:** Timing breakdown of Path B operator variants (Arnoldi/M⁻¹ vs Lanczos/M^(±½)) and eigsolve iterations at specific bonds. Hardcoded at N=2,3.

```
julia test_path_b_microbench.jl
```

---

### `test_recast_microbench.jl`
**What:** Diagnoses whether `recast_bs_to_template` cost is allocation- or compute-dominated. Times baseline, pre-allocated, and GC-disabled variants. Relevant if optimizing the M-build / gram_envs path.

```
julia test_recast_microbench.jl <N> [ITERS=200]
```

---

### `test_linv_compare.jl`
**What:** Per-bond comparison of dense Linv vs BlockSparse Linv to quantify discrepancy. Hardcoded at N=2,3. Relevant while `path_b_helpers.jl` BS-Linv work is in progress.

```
julia test_linv_compare.jl
```

---

## Archived tests

Old or superseded tests live in `auto_archived/` and `test_archive/`. They are not expected to run without further setup but preserve research history.
