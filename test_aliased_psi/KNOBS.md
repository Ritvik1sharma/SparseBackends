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
| `run_mode` (dmrg kwarg, via `--run-mode`) | :bop_aliased | :iso / :bop_aliased (Path-B, no densify) / :bop_densify (env-dressed densified seed) / :minner |

Removed 2026-06: `BMF_APPLY_MINV`, `BMF_ISO_PATH` — `dmrg.jl` never reads either; pathway selection
is 100% via the `run_mode` kwarg. Scripts that need `:iso` now pass `run_mode=:iso` explicitly
instead of setting `BMF_ISO_PATH=1` (which was a silent no-op).

Removed 2026-06: `SB_ALIASED_ENABLE` — never read in `SparseBackends`/`ITensorMPS.jl`; it was a
test-script-only self-defaulting guard (`get(ENV, "SB_ALIASED_ENABLE", "1")` a few lines before the
check, so it could never actually fire) sitting on top of already-hardcoded `:coo`/`:aliased`
backend selection in each `build_setup`/`contract(...)` call. The data-structure choice (P built
COO, ψ contracted to aliased) was never conditional on this flag — removed the flag, not the choice.

## 3. Path-B / M⁻¹ᐟ² numerics
| var | default | meaning |
|---|---|---|
| `BMF_MINV_RTOL` | 1e-1 | pseudo-inverse cutoff for M^{−1/2}; drops eigenvalues < rtol·maxλ. Stable band [1e-3, 7e-1] |
| `BMF_MINV_DIAG` / `SB_MINV_DIAG` `[dbg]` | 0 | print M conditioning (d, rtol, maxλ, minkeptλ, cond, n_dropped, max 1/√λ) |

**from-P M^{±1/2} (NOT env vars — function kwargs, default off):**
- `dmrg(...; minv_from_p=nothing)` — when `true`, build M^{±1/2}=c^{∓...}·G from geometric
  c=2^⌈env/2⌉ (no eigendecomposition), via `build_half_pair_single_fromP`. Validated correct
  both sectors (Δ≤6e-5 vs eigen baseline). Default `nothing` → eigen path unchanged.
- `build_minv_half_pair_factored(...; p_c=(cL,cR))` — the underlying per-side hook.

Removed 2026-06: `BMF_MINV_FLATC` — the flat-c Step-2a intermediate, superseded by the
`minv_from_p` from-P kwarg above.

Removed 2026-06: `BMF_RAYLEIGH_RITZ`, `BMF_RR_RTOL`, `RR_KRYLOVDIM`, `RR_MAXITER`, `SB_BOND_EIG_DBG` — the
Rayleigh-Ritz variant takes these as plain function args now, not ENV; no `ENV[...]` read of any of
them remains anywhere in the tree.

## 4. Factorization (SVD/QR)
| var | default | meaning |
|---|---|---|
| `SB_USE_OWNED_SVD` | off | owned-SVD factorize variant (per-cM SVD); read in `ITensorMPS.jl/src/mps.jl:1577` and `verify_iso.jl` |
| `SB_USE_QR` | off | QR+GS factorize instead of SVD; read in `mps.jl:1579`, `abstractmps.jl:1664,1724`, `verify_iso.jl` — takes precedence under `SB_USE_OWNED_SVD` |
| `SB_ADAPTIVE_RANK` / `SB_ADAPTIVE_REL` | off / "" | adaptive rank truncation (relative threshold) |
| `SB_BALANCED_OWNERSHIP` | off | balanced block ownership across channels |
| `SB_PARTITION_RR` / `SB_RR_DBG` | off | partition round-robin |
| `SB_FACT_DIAG` `[dbg]` | 0 | factorization diagnostics |
| `SB_FACT_KEY_DBG` `[dbg]` | 0 | factorization key debug; read in `ITensorMPS.jl/src/dmrg.jl:1225` |
| `SPARSE_SVD_DIAG` / `SPARSE_SVD_CAP_DIAG` / `SB_SV_REPORT` `[dbg]` | 0 | SVD spectrum / cap diagnostics |

Removed 2026-06: `SB_USE_GROUPED_SVD` (deprecated; the grouped-SVD call is now hardcoded on).
`SB_GROUP_DIAG` remains (its own diagnostic print, independent of the removed dispatch var).

Correction: `SB_USE_QR`, `SB_USE_OWNED_SVD`, `SB_FACT_KEY_DBG` were wrongly marked dead in an
earlier pass of this cleanup (the audit that flagged them only searched `SparseBackends/`, missing
their real reads in `ITensorMPS.jl`). Restored above.

## 5. Aliased kernel / performance
| var | default | meaning |
|---|---|---|
| `SB_ALIASED_NTHREADS` | 1 | threads for the aliased matvec kernel |
| `SB_KK_NTHREADS` | 1 | KrylovKit orthogonalization threads (pin to 1 for determinism) |
| `SB_ALIASED_MINV_WRAP` | — | wrap M^{−1/2} factors aliased (both-aliased md-crossover fix) |
| `BMF_USE_HINT` | 0 (off) | dense-BS-template output-hint path; gated off, in-kernel hint support not yet implemented — separate from `SB_ALIASED_MINV_HINT` below, don't confuse the two |
| `SB_ALIASED_KERNEL_POOL` | — | kernel buffer pooling |
| `SB_ALIASED_PREALLOC_BUF` | — | task-local scratch-buffer reuse across matvec calls |
| `SB_ALIASED_NATIVE_FISSION` | — | native dedup-preserving output-fission kernel vs BS-delegation fallback |
| `SB_ALIASED_INTERLEAVE` | — | interleaved kernel scheduling |
| `SB_ALIASED_NATIVE_DOT` / `SB_ALIASED_DOT_CHECK` | — | native aliased inner product (+ correctness check) |
| `SB_ALIASED_AA_ENV` / `SB_ALIASED_AA_HINT` | — | both-aliased (aliased×PHP) env handling |
| `SB_PRECONTRACT_H` / `SB_PREPERMUTE_ENVS` / `SB_NO_ENV_REORDER` | — | matvec operator prep levers |
| `SB_PLAN_B` | — | alternate contraction plan |

Retired 2026-06 (hardcoded, no longer ENV-configurable — validated bit-identical when ON, settled):
- `SB_ALIASED_MINV_HINT` — was default-on; now unconditional (`tensor_wrappers.jl`).
- `SB_ALIASED_OUTSTAT` — was default-on; now unconditional (`contract_aliased_dense_shared.jl`).
- `SB_ALIASED_SINGLE_PASS` — was default-on; now unconditional, old two-pass prepass block commented
  out in place (not deleted).
- `SB_ALIASED_LEGACY` — the pre-session A/B regression hook is commented out (dispatch site only;
  `contract_aliased_dense_legacy.jl` itself is left untouched as reference code, still `include`d).

## 6. Output layout / schema preservation
| var | default | meaning |
|---|---|---|
| `SB_OUTSTAT_NRED` | — | reduced-next-last output ordering (kills output permute) |
| `SB_OUTSTAT_NATURAL` | — | natural (non-canonical) output ordering |
| `SB_NO_ALIAS_OUTPUT` | — | force dense (non-aliased) output |
| `SB_BONDTYPE` | "?" | bond classification override |
| `BMF_BSWRAP` / `BSWRAP_DEBUG` | — | block-sparse wrap control + debug |

## 7. Gram / M diagnostics — no flag; `if false` in code
The gram-structure dump (eigenvalues, numerical rank, SEPARABILITY, BLOCKDIAG, full matrix if
d≤16) proved `M=c·Π` (block-diagonal in the variational key, flat `c=2^⌈env/2⌉`). No env var:
the block is guarded by `if false` in `dmrg.jl` (near the `dmrg.gram_envs` timeit). To use it,
change `if false` → `if true` (dumps every bond every ha==1; add a `b==<bond>` guard if noisy).
`SB_GRAM_DUMP` / `SB_GRAM_DUMP_BOND` are removed.

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
julia --project=.. test_aliased_kl.jl --N-plaq 12 --maxdim 40 --n-sweeps 4 --run-mode bop_aliased

# −1 sector
... --eignv false ...

# from-P M^{±1/2} (no eigen), N=12 bd=40, +1 sector — via test_fromp.jl
julia --project=.. test_fromp.jl --N-plaq 12 --maxdim 40 --n-sweeps 10 --psign 1 --from-p true
```
