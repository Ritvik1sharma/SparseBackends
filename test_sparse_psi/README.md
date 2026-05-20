# Sparse-ψ DMRG profiling tests

Tests for the Path B sparse DMRG flow (bare H + BS-storage ψ + hint kernel +
allowed_keys filter). All scripts include `utils.jl` from this directory.

## Active scripts

| Script | Purpose |
|---|---|
| `test_profile_n4_bareh.jl` | N=4 plaquettes, S=1 chain, 2 warmup sweeps + 5 profile sweeps at maxdim=40. Reports `ITensorMPS.PROJMPO_TIMER` and `SparseBackends.TIMER`. |
| `test_profile_n16_bareh.jl` | N=16 version, 2 warmup + 8 profile sweeps. |
| `test_dense_n16.jl` | Dense (PHP·ψ) baseline at N=16, for direct sparse-vs-dense comparison. |
| `utils.jl` | Shared helpers (e.g., `clean!`, OpSum builders). |

## How to run

From this directory:

```bash
# Sparse Path B with hint + allowed_keys filter (the production flow)
BMF_USE_HINT=1 julia --project=.. test_profile_n4_bareh.jl   2>&1 | tee /tmp/n4_run.log
BMF_USE_HINT=1 julia --project=.. test_profile_n16_bareh.jl  2>&1 | tee /tmp/n16_run.log

# Baseline (no hint, no allowed_keys): just unset the env var
julia --project=.. test_profile_n4_bareh.jl  2>&1 | tee /tmp/n4_baseline.log
julia --project=.. test_profile_n16_bareh.jl 2>&1 | tee /tmp/n16_baseline.log

# Dense reference
julia --project=.. test_dense_n16.jl 2>&1 | tee /tmp/n16_dense.log
```

The `--project=..` points at `edited_packages/Project.toml` (one level up).

## What to look for in output

**Energy match** — across the three runs at the same N, the converged
`E_after_warm` and final profile energy should agree to ~1e-6.

**Sweep time** — `Profile sweep done in <T>s` line near the end. Divide by
number of profile sweeps for the average.

**Timer breakdown** — two timers printed at end:
- `ITensorMPS.PROJMPO_TIMER` — DMRG-level: `eigsolve`, `replacebond!`, etc.
- `SparseBackends.TIMER` — kernel-level: `khint_fast.*`, `amp.recast2`,
  `apply_minv_preserve_bs`, `wrapped_contract`, etc.

## Diagnostic env flags

| Env var | Effect |
|---|---|
| `BMF_USE_HINT=1` | Enable hint kernel + allowed_keys filter |
| `HINT_DEBUG=1`   | Print per-call hint kernel state (budgeted) |
| `INDEX_DEBUG=1`  | Verbose index debugging from ProjMPO |
| `SB_TRACE=1`     | Trace per-contraction backend dispatch |
| `BSWRAP_DEBUG=1` | Print apply_minv_preserve_bs storage info |

## Current baselines (rough)

| Run | Sweep avg |
|---|---|
| N=4 baseline (Path A.1, no hint) | 10.8 s |
| N=4 hint only | 10.9 s |
| N=4 hint + allowed_keys | **8.28 s** |
| N=16 baseline (Path A.1, no hint) | ~520 s |
| N=16 hint only (sweeps 1-3) | ~240 s |
| N=16 hint + allowed_keys | _running_ |

Note: bond dim for this Hamiltonian only grows to maxlinkdim=4 (not 40);
production-rank stress tests require a different Hamiltonian.
