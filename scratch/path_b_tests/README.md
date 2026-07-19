# Archived Path-B / sparse-iso test drivers

These test/diagnostic scripts exercise the **removed** aliased-ψ + dense-H methods:
- **Path-B** (`run_mode=:bop_aliased/:bop_densify/:minner/:rr`, `minv_from_p`, `gram_from_h`, `rr_dense_iter`, `minv_rtol`) — variable aliased ψ with a local metric M built from the P†…P envs, M-inversion / Rayleigh-Ritz.
- **Path-A sparse-iso** (`run_mode=:iso` on a `WrappedBlockSparse` ψ).

Both were superseded by the **factor-core** method (ψ = P·core, metric I), which is now the sole aliased-ψ + dense-H path (`dmrg(H, psi0; P=P)` → `dmrg_core_php`). The Path-B kwargs, the `dmrg.jl` Path-B block, and the SparseBackends files `path_b_helpers.jl` / `path_b_utils.jl` / `rayleigh_ritz.jl` (+ `rayleigh_ritz_sweep.jl`) were removed, so these scripts no longer run against the live packages.

Kept here (moved, not deleted) for reference/restore. `test_aliased_kl.jl` provides `build_setup`/`run_sweeps` used by several of the others via bare `include("test_aliased_kl.jl")` (same-dir).

Live tests that remain in the active tree: the dense-ψ + aliased-PHP tests (`test_dense_*.jl`) and the factor-core drivers (`test_aliased_psi/test_factor_core/core_php_*_driver_test.jl`).
