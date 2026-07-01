# test_aliased_psi — knobs & env vars reference

Scope: `test_aliased_kl.jl` and the code it exercises (SparseBackends, ITensorMPS.jl Path-B).
Defaults shown as `(default X)`. `[dbg]` = print-only, no effect on numerics, default off.
`[+session]` = added in the M-from-P investigation (2026-06); `[+prev]` = added earlier for gram analysis.

## 1. CLI arguments (test_aliased_kl.jl)
| flag | default | meaning |
|---|---|---|
| `--N-plaq` | 12 | number of plaquettes N → 2N+2 sites |
| `--eignv` | true | true → projector sector psign=+1 (P=(Id+YXXY)/2, E≈−17.18); false → psign=−1 (E≈−16.34) |
| `--spin` | 3 | 2 → S=1/2, 3 → S=1 |
| `--n-sweeps` | 10 | DMRG sweeps (sweep 1 = JIT warmup) |
| `--maxdim` | 40 | maxdim cap (honest per-bond dim = channel × mult) |
| `--target-energy` | NaN | log first sweep with E ≤ target (no early-exit) |
| `--run-mode` | bop_aliased | eigensolve pathway (see §2) |

## 2. Core gates & pathway
| var | default | meaning |
|---|---|---|
| `SB_ALIASED_ENABLE` | "0" (test sets 1) | master gate for the aliased backend; nothing runs without it |
| `run_mode` (dmrg kwarg, via `--run-mode`) | :bop_aliased | :iso / :bop_aliased (Path-B, no densify) / :bop_densify (env-dressed densified seed) / :minner |
| `BMF_APPLY_MINV` | legacy | Path-B M⁻¹ apply (now folded into run_mode; older scripts still set it) |
| `BMF_ISO_PATH` | legacy | iso pathway select (superseded by run_mode=:iso) |

## 3. Path-B / M⁻¹ᐟ² numerics
| var | default | meaning |
|---|---|---|
| `BMF_MINV_RTOL` | 1e-1 | pseudo-inverse cutoff for M^{−1/2}; drops eigenvalues < rtol·maxλ. Stable band [1e-3, 7e-1] |
| `BMF_MINV_FLATC` `[+session]` | 0 | force kept M spectrum flat to c=mean(kept λ) → M^{±1/2}=c^{±1/2}·Π (tests the ideal scaled-projector form) |
| `BMF_MINV_DIAG` / `SB_MINV_DIAG` `[dbg]` | 0 | print M conditioning (d, rtol, maxλ, minkeptλ, cond, n_dropped, max 1/√λ) |
| `BMF_RAYLEIGH_RITZ` | 0 | use Rayleigh-Ritz eigensolve variant (parked; ~2.5× slower) |
| `BMF_RR_RTOL` | 1e-8 | RR pseudo-inverse cutoff |
| `RR_KRYLOVDIM` | 8 | RR Krylov dimension |
| `RR_MAXITER` | 100 | RR max iterations |
| `SB_BOND_EIG_DBG` `[dbg]` | 0 | dump per-bond eigensolve info |

## 4. Factorization (SVD/QR)
| var | default | meaning |
|---|---|---|
| `SB_USE_QR` | off | QR-based factorize instead of SVD |
| `SB_USE_OWNED_SVD` | off | owned-SVD path |
| `SB_USE_GROUPED_SVD` / `SB_GROUP_DIAG` | off | grouped SVD + diag |
| `SB_ADAPTIVE_RANK` / `SB_ADAPTIVE_REL` | off / "" | adaptive rank truncation (relative threshold) |
| `SB_BALANCED_OWNERSHIP` | off | balanced block ownership across channels |
| `SB_PARTITION_RR` / `SB_RR_DBG` | off | partition round-robin |
| `SB_FACT_DIAG` / `SB_FACT_KEY_DBG` `[dbg]` | 0 | factorization diagnostics |
| `SPARSE_SVD_DIAG` / `SPARSE_SVD_CAP_DIAG` / `SB_SV_REPORT` `[dbg]` | 0 | SVD spectrum / cap diagnostics |

## 5. Aliased kernel / performance
| var | default | meaning |
|---|---|---|
| `SB_ALIASED_NTHREADS` | 1 | threads for the aliased matvec kernel |
| `SB_KK_NTHREADS` | 1 | KrylovKit orthogonalization threads (pin to 1 for determinism) |
| `SB_ALIASED_MINV_WRAP` | — | wrap M^{−1/2} factors aliased (both-aliased md-crossover fix) |
| `SB_ALIASED_MINV_HINT` / `BMF_USE_HINT` / `SB_ALIASED_MINV_HINT` | — | matvec last-only schema hint for M^{−1/2} apply |
| `SB_ALIASED_KERNEL_POOL` | — | kernel buffer pooling |
| `SB_ALIASED_INTERLEAVE` | — | interleaved kernel scheduling |
| `SB_ALIASED_NATIVE_DOT` / `SB_ALIASED_DOT_CHECK` | — | native aliased inner product (+ correctness check) |
| `SB_ALIASED_AA_ENV` / `SB_ALIASED_AA_HINT` | — | both-aliased (aliased×PHP) env handling |
| `SB_PRECONTRACT_H` / `SB_PREPERMUTE_ENVS` / `SB_NO_ENV_REORDER` | — | matvec operator prep levers |
| `SB_PLAN_B` | — | alternate contraction plan |

## 6. Output layout / schema preservation
| var | default | meaning |
|---|---|---|
| `SB_OUTSTAT_NRED` | — | reduced-next-last output ordering (kills output permute) |
| `SB_OUTSTAT_NATURAL` | — | natural (non-canonical) output ordering |
| `SB_NO_ALIAS_OUTPUT` | — | force dense (non-aliased) output |
| `SB_BONDTYPE` | "?" | bond classification override |
| `BMF_BSWRAP` / `BSWRAP_DEBUG` | — | block-sparse wrap control + debug |

## 7. Gram / M diagnostics `[+prev]` / `[+session]`
| var | default | meaning |
|---|---|---|
| `SB_GRAM_DUMP` `[+prev]` | 0 | dump Lgram/Rgram at ha==1: dims, eigenvalues, numerical rank, SEPARABILITY, BLOCKDIAG, full matrix if d≤16 |
| `SB_GRAM_DUMP_BOND` `[+prev]` | "2" | comma-list of bonds to dump (e.g. "12" or "2,3") |
| (BLOCKDIAG metric inside the dump) `[+session]` | — | off-axis1/off-axis2 fractions → which axis M is block-diagonal in |

## 8. Schema / key diagnostics `[dbg]`
`SB_SCHEMA_DBG` (+ `SB_SCHEMA_DBG_BUDGET`=40), `SB_SCHEMA_KEYS`, `SB_SCHEMA_TEMPLATES`, `SB_SCHEMA_TRACK`,
`SB_PHI_SCHEMA_DUMP` (+`_MAX`), `SB_INSPECT_PHI` (prints φ index classification then exits),
`SB_PHI_Y0_DEDUP`, `SB_RECAST_DBG`, `SB_RECAST_DEDUP_CHK` (+`_MAX`), `SB_SNAP_PHI`, `SB_ALIASED_SNAP` / `SB_ALIASED_SNAP_DBG`(+`_MAX`).

## 9. Trace / index-order diagnostics `[dbg]`
`SB_KEYTRACE` (+`SB_KEYTRACE_BOND`), `SB_STEP` / `SB_STEP_IDX_DBG` (+`_MAX`),
`SB_KRYLOV_IDX_DBG` (+`SB_KRYLOV_IDX_BONDS`="1,3,5", `SB_KRYLOV_IDX_SWEEP`="2"),
`TRACE_BOND` / `TRACE_LABEL` / `TRACE_IMAGE`, `SB_TRACE` / `SB_TRACE_ONE`, `SB_IDX_DBG`, `INDEX_DEBUG`,
`DENSE_INDS_DEBUG`, `DEBUG_HINT`, `SB_ALIASED_DEBUG`(+`_MAX`), `SB_ALIASED_TRACE`, `SB_ALIASED_ALIGN_DBG`/`SB_ALIASED_ALIGN_OUTPUT`,
`SB_IN_MATVEC` / `SB_IN_POSITION` / `SB_IN_WARMUP` (phase latches), `SB_STEP`, `SB_ADD_STACK` / `SB_INPLACE_DBG`,
`SB_DROP_KRYLOV` (+`_DBG`), `SB_DROP_AT_FACT_DBG`, `SB_RR_DBG`, `SB_QR_DIAG`, `SB_RUN_LABEL`.

## 10. Operand / matvec dumps `[dbg]`
`SB_ADD_OPERAND_DUMP`(+`_MAX`), `SB_DENSEDENSE_DUMP`(+`_MAX`), `SB_SPARSEDENSE_DUMP`(+`_MAX`),
`SB_MATVEC_DIAG`(+`SB_MATVEC_DIAG_BUDGET`), `SB_HAMPSI_DIAG`, `SB_ENV_DEBUG`(+`_MAX`).

## 11. Perf / roofline / footprint
`SB_ROOFLINE` (master timing gate), `SB_FLOP_COUNT`, `SB_ENV_FOOTPRINT`, `SB_PERMUTE_PROFILE`, `SB_PERM_CAPTURE`,
`GEMM_DIMS_HIST`, `ALLOC_SAMPLE_RATE`.

## 12. diag_gram_structure.jl / verification scripts
`DIAG_N_PLAQ`, `DIAG_MAXDIM`, `DIAG_NSWEEPS`, `DIAG_GRAM_ONLY` (standalone gram viewer);
`VN`, `VMD`, `VPS`, `VSPIN`, `VSW` (verify_physical_energy.jl params).

---
### Canonical run examples
```
# baseline aliased Path-B, N=12 bd=40, +1 sector
SB_ALIASED_ENABLE=1 julia --project=.. test_aliased_kl.jl --N-plaq 12 --maxdim 40 --n-sweeps 4 --run-mode bop_aliased

# −1 sector
... --eignv false ...

# gram structure dump at bond 12
SB_ALIASED_ENABLE=1 SB_GRAM_DUMP=1 SB_GRAM_DUMP_BOND=12 julia --project=.. test_aliased_kl.jl --N-plaq 12 --maxdim 40 --n-sweeps 4

# ideal flat-c (scaled-projector) test
SB_ALIASED_ENABLE=1 BMF_MINV_FLATC=1 julia --project=.. test_aliased_kl.jl --N-plaq 12 --maxdim 40 --n-sweeps 4
```
