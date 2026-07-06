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
| `SB_ADAPTIVE_RANK` / `SB_ADAPTIVE_REL` | off / "" | adaptive rank truncation (relative threshold); only consumed by the QR variant (`ops_factorize_qr.jl`) — the default plain channel-aware SVD never reads these |
| `SB_BALANCED_OWNERSHIP` | off | balanced block ownership across channels; QR variant only, same scope as above |
| `SB_USE_QR` | off | QR+GS factorize instead of the default channel-aware SVD; read in `mps.jl:1579`, `abstractmps.jl:1664,1724`, `verify_iso.jl`. Set by `diag_heff_php.jl`, `test_sparse_ham_psi_kl.jl` (the "aliased ψ × aliased PHP" combo tests; `test_native_dot.jl` also set it but that whole script is now retired/commented out — see §12); everything else runs the default SVD |

Removed/retired 2026-06:
- `SB_USE_OWNED_SVD` — never exercised by any script. The dispatch branch (`mps.jl:1577`,
  `verify_iso.jl`) is commented out, and the owned-SVD implementation itself
  (`itensor_blocksparse_svd_owned_channel_aware` in `tensor_wrappers.jl`, and the whole
  `ops_factorize_svd_owned.jl` file it calls into) is commented out / un-included. Uncomment the
  `include` in `SparseBackends.jl`, the wrapper function, and the dispatch branches to re-enable.
- `SB_PARTITION_RR`, `SB_GROUP_DIAG` — both lived exclusively inside
  `blocksparse_svd_channel_aware_fixed`, a function with **zero callers anywhere in the tree**
  (confirmed by full-tree grep); deleted as genuinely dead, matching `SB_USE_GROUPED_SVD`.
- `SPARSE_SVD_DIAG`, `SPARSE_SVD_CAP_DIAG`, `SB_SV_REPORT` — pure debug prints (spectrum / cap
  diagnostics), no numeric effect. All occurrences (in `_right_binned`, the dead
  `_channel_aware_fixed`, the mainline `_channel_aware`, `_left_binned`, and the QR variant)
  converted to `if false` with the body commented out; flip to `true` + uncomment to re-enable.
- `SB_RR_DBG` — Rayleigh-Ritz eigensolve iteration debug print (unrelated to SVD/QR despite living
  in §4 — "RR" here means Rayleigh-Ritz, not round-robin). That whole eigensolve is parked (never
  called in production). `if false` + commented body in `path_b_helpers.jl`.
- `SB_FACT_DIAG` — was gating two unrelated diagnostics under one name: the Tier-1 aliased
  factorize's channel/mult axis-ordering check (`aliased/factorize.jl`) and a φ-operand
  classification dump before eigsolve (`dmrg.jl`, not a factorize step at all). Both converted to
  `if false` + commented body independently.
- `SB_FACT_KEY_DBG` — pre/post-factorize key-diff debug print (`dmrg.jl`). `if false` + commented
  body; the real `replacebond!` call it straddles was left untouched.
- `SB_USE_GROUPED_SVD` (deprecated; the grouped-SVD call is now hardcoded on).

Note: `SB_USE_QR`, `SB_USE_OWNED_SVD`, `SB_FACT_KEY_DBG` were wrongly marked dead in an earlier
pass of this cleanup (the audit that flagged them only searched `SparseBackends/`, missing their
real reads in `ITensorMPS.jl`) — that mistake was caught and corrected before any of the above.

## 5. Aliased kernel / performance
| var | default | meaning |
|---|---|---|
| `SB_ALIASED_NTHREADS` | 1 | threads for the aliased matvec kernel; NOT bit-identical when >1 (FP reduction order changes) — serial-by-default is a correctness/reproducibility choice, not a settled optimization |
| `SB_PREPERMUTE_ENVS` | 0 (off) | permute envs to canonical order before the matvec; a real, still-open A/B lever |
| `SB_NO_ENV_REORDER` | 0 (off) | disables the standalone env-canonicalize step; its own comment says the fire-counter shows it doesn't help — a concluded-negative experiment, still open whether to remove |
| `SB_PLAN_B` | 0 (off) | alternate multiplication order (`it*Hv` vs `Hv*it`) for dense H·v, tied into a permute-cost profiling harness |

Retired 2026-06 (hardcoded, no longer ENV-configurable — validated bit-identical when ON, settled):
- `SB_ALIASED_MINV_HINT` — was default-on; now unconditional (`tensor_wrappers.jl`).
- `SB_ALIASED_OUTSTAT` — was default-on; now unconditional (`contract_aliased_dense_shared.jl`).
- `SB_ALIASED_SINGLE_PASS` — was default-on; now unconditional, old two-pass prepass block commented
  out in place (not deleted).
- `SB_ALIASED_LEGACY` — the pre-session A/B regression hook is commented out (dispatch site only;
  `contract_aliased_dense_legacy.jl` itself is left untouched as reference code, still `include`d).
- `SB_ALIASED_MINV_WRAP` — was default-on for aliased φ; now unconditional (`path_b_helpers.jl`).
- `SB_ALIASED_KERNEL_POOL` — was default-on; now unconditional (`contract_aliased_dense_shared.jl`'s
  `_alloc_ffull`), fresh-allocation fallback commented out in place.
- `SB_ALIASED_PREALLOC_BUF` — was default-on; now unconditional (same file, matvec `pending`
  buffer + GEMM scratch).
- `SB_ALIASED_NATIVE_FISSION` — was default-on; now unconditional (same file's fission dispatch;
  the BS-delegation fallback is still reached, but only via `allowed_keys_C`, not this flag).
- `SB_ALIASED_INTERLEAVE` — was default-on; now unconditional (`tensor_wrappers_aliased.jl`);
  `SB_ALIASED_NTHREADS`, which it used to be conjoined with, is untouched and still live.
- `SB_KK_NTHREADS` — was default-on (=1); `KrylovKit.set_num_threads(...)` now called with the
  literal `1` in `test_aliased_kl.jl`/`test_aliased_pxp.jl` (test-script-level, not core code).
- `BMF_USE_HINT` — **never safe to enable** (in-kernel hint support was never finished; flipping it
  on causes correctness errors). Hardcoded to `false` at all 3 call sites (`tensor_wrappers.jl:512,531`,
  `abstractprojmpo.jl:424`), each marked with an "UNSAFE, kept hardcoded off" comment — do not flip.
- `SB_ALIASED_NATIVE_DOT` / `SB_ALIASED_DOT_CHECK` — both were default-off and never exercised; the
  native-scalar-dot fast path + its correctness-check companion are commented out in place at their
  call site (`tensor_wrappers_aliased.jl`, both-aliased scalar-contraction branch); densify fallback
  right after is the live path.
- `SB_ALIASED_AA_ENV` — was default-on; now unconditional (`abstractprojmpo.jl`, both `_makeL!`/
  `_makeR!` sites) — env stays aliased whenever both ψ and H are aliased.
- `SB_ALIASED_AA_HINT` — was default-on; now unconditional (`abstractprojmpo.jl:468`) — union-of-
  dense-axes hint always computed for both-aliased matvec.
- `SB_PRECONTRACT_H` — was default-off, a pure measurement tool (raw H2-build-cost probe), never
  the live path; removed entirely along with `_h2_eligible`/`_dl`/`_H2_DBG` — `itensor_map` now
  always takes the `append!(itensor_map, P.H[sr])` branch (`abstractprojmpo.jl`).

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
Still live (real experiments, not print-only — SB_SNAP_PHI/SB_ALIASED_SNAP change what's computed
when on; both default off, open correctness questions, not yet resolved):
`SB_SNAP_PHI`, `SB_ALIASED_SNAP`.

- `SB_PHI_SCHEMA_DUMP` / `SB_PHI_SCHEMA_DUMP_MAX` — retired 2026-06. `product(P::AbstractProjMPO,
  v::ITensor; debug::Bool=false)` in `abstractprojmpo.jl` now takes a `debug` keyword (budget
  hardcoded to 1 call); every existing call site omits it and so defaults off. Pass `debug=true`
  at a call site to re-enable the φ-vs-Hv schema dump.

Retired 2026-06 (all were print-only or one-shot inspectors, no numeric effect on the DMRG result):
- `SB_SCHEMA_DBG` / `SB_SCHEMA_DBG_BUDGET` / `SB_SCHEMA_KEYS` / `SB_SCHEMA_TEMPLATES` — merged into
  one `const _SCHEMA_DBG = false` next to `schema_dbg()` in `tensor_wrappers_aliased.jl` (key/
  template detail is no longer separately gated — flip `_SCHEMA_DBG` to `true` to get everything;
  budget hardcoded to `Ref(40)`). `test_aliased_kl.jl`'s own construction-phase dump is gated by a
  separate local `_SCHEMA_DBG_ON` const (same file).
- `SB_SCHEMA_TRACK` — replaced by a local `_SCHEMA_TRACK_ON = false` const in both
  `test_aliased_kl.jl` and `test_aliased_pxp.jl`; flip manually.
- `SB_INSPECT_PHI` — removed entirely (was a one-shot inspector that dumped φ's per-site index
  classification then called `exit(0)`, in `test_aliased_kl.jl`).
- `SB_PHI_Y0_DEDUP` — converted to `if false` in `dmrg.jl` with a comment pointing back to
  [[project_minv_half_destroys_dedup]] (the already-answered question this probed); flip to `true`
  to re-run the dump.
- `SB_RECAST_DBG` — replaced by `const _RECAST_DBG = false` next to `_RECAST_DBG_N` in
  `tensor_wrappers_aliased.jl`.
- `SB_RECAST_DEDUP_CHK` / `SB_RECAST_DEDUP_CHK_MAX` — replaced by a local `_recast_dedup_chk_on =
  false` var in `dmrg.jl` (budget hardcoded to 8).
- `SB_ALIASED_SNAP_DBG` / `SB_ALIASED_SNAP_DBG_MAX` — `_snap_to_schema(...)` now takes a
  `dbg::Bool=false` keyword arg (budget hardcoded to 5 inside); all 4 call sites (`dmrg.jl` ×3,
  `abstractprojmpo.jl` ×1) omit it and so default to off — pass `dbg=true` at a specific call site
  to enable that site's error-tracking dump.

## 9. Trace / index-order diagnostics `[dbg]`
Still live: `SB_DROP_KRYLOV` — **not a debug flag**, a real experimental key-drop toggle (changes
what's computed).

Retired 2026-06 (this round):
- `SB_ALIASED_ALIGN_OUTPUT` — hardened always-on (`tensor_wrappers.jl`): whenever an aliased
  template is available, the contraction always tries to align its output to the template's own
  axis order first (was default-off). The kernel already falls back gracefully
  (`_ALIGN_FALLBACK`) when it can't honor the request, so this is a pure win when it succeeds and
  a no-op otherwise.
- `SB_STEP` — replaced by `SparseBackends.CURRENT_STEP`, a plain runtime `Ref{Union{Nothing,Int}}`
  (NOT an ENV var, explicitly commented as a debugging var at its definition in
  `tensor_wrappers_aliased.jl`). Set by `abstractprojmpo.jl` (`SparseBackends.CURRENT_STEP[] =
  idx`) before each operator in the matvec chain; consulted by trace prints. (Its other former
  consumer, the `STATIC_OUTPUT_PERM` lookup, no longer reads it at all — see `SB_IN_MATVEC` below.)
- `SB_IN_MATVEC` — **fully eliminated, no Ref, no ENV var.** `_static_output_pref(indsC)` (which
  read `bondtype`/`step` from `ENV`/`CURRENT_STEP` and was gated on `SB_IN_MATVEC` because it was
  reached by unrelated non-matvec contractions too) is now `static_output_perm(bondtype::Symbol,
  step::Int)` — a stateless lookup with no gate at all. The matvec loop
  (`abstractprojmpo.jl`, the only caller with the real `bondtype`/`step`) calls it directly and
  threads the *result* (`output_perm::Union{Nothing,Vector{Int}}`) down through
  `SparseBackends.contract_preserve_bs` → the 3 `contract(::WrappedAliasedBlockSparse,...)`
  overloads → `wrapped_contract_aliased`, which applies it in place of the old lookup. No other
  caller can reach this by accident since it's a plain argument, not global state.
  `contract_preserve_bs` also gained a hard `error(...)` (not a silent skip) if `output_perm` is
  given but both operands turn out to be plain dense — that combination would mean a real
  structural bug (Path-B operands are aliased/BS by construction).
- `SB_IN_POSITION` — **fully eliminated, no Ref, no ENV var**, via `in_position::Bool` threaded
  through two groups of call sites (traced call-graph, not assumed):
  - `_env_mul` (`abstractprojmpo.jl`) is called *only* from `_makeL!`/`_makeR!`, which run *only*
    during `position!` — structurally always "position", so its two call-outs
    (`_mul_preserve_aliased`, `add_dense_macs!`) hardcode `in_position=true` directly; no signature
    change needed to `_env_mul`/`_makeL!`/`_makeR!`/`position!` themselves.
  - Everything downstream of that (`_mul_preserve_aliased` → `wrapped_contract_aliased` →
    `contract_aliased_dense_to_dense!` → `add_reshuffle!`/`add_aliased_macs!`; separately
    `contract_preserve_bs`/`wrapped_contract` → `contract_bs_dense_to_dense!`'s permute-profile
    tagging) now takes `in_position::Bool=false` explicitly, since these kernels are shared by both
    the position path (→true) and the matvec/other path (→false default).
  - `add_dense_macs!`/`add_aliased_macs!`/`add_reshuffle!` (`SparseBackends.jl`) take
    `in_position::Bool` directly now; `_flop_in_position()` is deleted.
  - **Known residual gap**: `_env_mul`'s non-`keep` branch falls through to a bare `A * B` operator
    call for env-build contractions that don't need aliased-preserving output. If that reaches
    `contract_bs_dense_to_dense!` (i.e. a BS, non-aliased operand shows up during env-building),
    there's no kwarg channel through plain operator syntax to carry `in_position=true`, so it
    defaults to `false` — mislabeling that one case as "matvec" in the permute-profile/FLOP-counter
    diagnostics only. This does not affect DMRG correctness (energies), only a rarely-exercised
    diagnostic label (`roofline=true` runs only, BS-non-aliased operands during env-building only).
    Not fixed in this round; flag if it needs closing.
- `SB_RUN_LABEL` — genuinely threaded as a real `run_label::String="?"` argument (not a Ref) from
  `dmrg(...)` down through `position!`/`product` → `makeL!`/`makeR!`/`contract` →
  `_record_env_footprint`/`permute_profile_site`, mirroring the existing `debug`/`roofline` kwarg
  pattern exactly. Test scripts that used to set `ENV["SB_RUN_LABEL"]` now pass `run_label="DENSE"`
  etc. directly into their `dmrg(...)` (or `run_dmrg_ground(...)`) call. One exception: the deep
  writer in `contract_bs_dense.jl`'s shared BS dense-dense kernel is reached via generic dispatch
  with no argument-carrying call chain from `dmrg`, so it reads `SparseBackends.CURRENT_RUN_LABEL[]`
  — a small Ref set alongside the threaded argument in `ITensors.contract(P::AbstractProjMPO,...)`,
  same structural reason `GEMM_DIMS_HIST`/the permute-profile path needed one under §11.
- `SB_DROP_KRYLOV_DBG` — converted to `if false` in `dmrg.jl` (the real `SB_DROP_KRYLOV` toggle it
  was nested under is unaffected); flip to `true` to re-enable the seed-vs-φ key-set debug dump.

Removed 2026-06 (confirmed print-only, no numeric effect, none set by any script) — `if false` +
body commented / hardcoded off at each site, kept as reference:
`SB_STEP_IDX_DBG` (+`_MAX`), `SB_KRYLOV_IDX_DBG` (+`SB_KRYLOV_IDX_BONDS`, `SB_KRYLOV_IDX_SWEEP`),
`TRACE_BOND` / `TRACE_LABEL`, `TRACE_IMAGE`, `SB_TRACE_ONE`, `SB_IDX_DBG`, `INDEX_DEBUG`,
`DENSE_INDS_DEBUG`, `DEBUG_HINT`, `SB_ALIASED_DEBUG` (+`_MAX`), `SB_ALIASED_ALIGN_DBG`,
`SB_ADD_STACK`, `SB_INPLACE_DBG`, `SB_DROP_AT_FACT_DBG`, `SB_QR_DIAG`.

Retired 2026-06 (this round):
- `SB_ALIASED_TRACE` — replaced by `SparseBackends.ALIASED_TRACE[]`, a package-level `Ref{Bool}`
  (NOT an ENV var) set once at the top of `dmrg(...)` from its existing `debug::Bool=false` kwarg
  (`dmrg.jl`: `SparseBackends.ALIASED_TRACE[] = debug`). All ~20 trace-print sites across
  `tensor_wrappers_aliased.jl`, `aliased/factorize.jl`, `abstractmps.jl`, `mps.jl`, `dmrg.jl`, and
  `abstractprojmpo.jl` now read this Ref instead of `ENV["SB_ALIASED_TRACE"]`. Each site's own
  fire-count budget (e.g. `< 10`, `< 30`, `< 5`) was already a hardcoded literal, not
  ENV-configurable, and is unchanged. To trace a run: `dmrg(H, psi, sweeps; debug=true, ...)`.
- `SB_IN_WARMUP` — removed entirely (`test_sparse_ham/aliased_helpers.jl`). It only ever gated
  `TRACE_BOND`, which was already retired to a hardcoded `do_trace = false` in `abstractprojmpo.jl`
  in an earlier round — so `SB_IN_WARMUP` had become write-only (no readers left) and was dead.
  Note: `TRACE_BOND`'s hardcoded-off branch still references an undefined `trace_bond_target`
  variable; harmless while `do_trace` stays `false` (branch never executes), but if `do_trace` is
  ever flipped back to `true` to re-enable, `trace_bond_target` needs to be reintroduced as a
  parameter (it is not currently settable as a function argument — `product`/`contract` have no
  bond-target kwarg; would need one added, e.g. `trace_bond::Union{Int,Nothing}=nothing`).

`SB_KEYTRACE` (+`SB_KEYTRACE_BOND`) — the `_keytrace` dump function (dmrg.jl) is kept live/uncommented
so it stays easy to re-enable; only its 5 call sites are commented out (no ENV read anymore — was
already print-only, but the function form makes re-enabling a 1-line uncomment per call site
instead of restoring ENV plumbing).

`SB_TRACE` — kept, but no longer an ENV var: each of its 11 call sites (across the low-level
contract dispatch functions) is now `if false  # SB_TRACE — flip to true here for debug output`,
so re-enabling any one of them is a local one-line edit instead of a process-wide env flag.

## 10. Operand / matvec dumps `[dbg]`
None still live in this section — all retired.

Retired 2026-06 (this round):
- `SB_MATVEC_DIAG` / `SB_MATVEC_DIAG_BUDGET` — removed entirely (`abstractprojmpo.jl`), including
  the now-unused `_MV_DIAG_BUDGET` const. Was a budgeted per-matvec-step size dump
  (block/template counts, compression ratio via `SparseBackends.matvec_size_info`).
- `SB_HAMPSI_DIAG` — removed entirely (`abstractprojmpo.jl`), including the `_v_channel_ids`/
  `_flip_flagged` tracking state. Was a one-shot detector flagging the first matvec step whose
  output had a φ-channel sitting in its dense tail (a "channel demoted to dense" correctness
  drift check).
- `SB_ENV_DEBUG` / `SB_ENV_DEBUG_MAX` — removed (`abstractprojmpo.jl`'s `_makeL!`/`_makeR!`); the
  `_env_dbg`/`_env_dbg2` source booleans are hardcoded `false` (safe pattern — `if`/`end`
  structures left untouched since the prints are interleaved with real env-build control flow,
  not a standalone block). Env-build structure prints are now unreachable dead code kept for
  reference; flip `_env_dbg`/`_env_dbg2` to `true` locally to re-enable.

Retired 2026-06 (earlier round):
- `SB_ADD_OPERAND_DUMP` / `SB_ADD_OPERAND_DUMP_MAX` — replaced by `const _ADD_OPERAND_DUMP = false`
  next to `_dump_add_operands(...)` in `tensor_wrappers_aliased.jl` (budget hardcoded to 4). Flip
  the const to `true` to re-enable.
- `SB_DENSEDENSE_DUMP` / `SB_DENSEDENSE_DUMP_MAX` — the whole dump block (path-based ITensor-pair
  serialization for offline replay) commented out in place in `abstractprojmpo.jl`, not deleted;
  uncomment to re-enable.
- `SB_SPARSEDENSE_DUMP` / `SB_SPARSEDENSE_DUMP_MAX` — same treatment, same file, sparse-H×dense-V
  matvec branch.

## 11. Perf / roofline / footprint

**`roofline::Bool=false`** — `dmrg(...)` kwarg (not an ENV var), consolidates all 6 vars below into
one switch. On every `dmrg(...)` call it flips the enabled state (via `SparseBackends.set_roofline!`)
for: per-phase GEMM/permute timers + CAS redundancy counters, the dense-vs-aliased MAC/step counter,
the per-site env-tensor footprint tracker, the `STATIC_OUTPUT_PERM`-generator hook, the GEMM-call-shape
histogram, and the SparseBackends/ITensorMPS permute-profile TSV writers (fixed path
`roofline_permute_profile.tsv`, see below). It does **not** zero the accumulators itself (so a
per-sweep loop of `dmrg(...)` calls keeps accumulating across the whole run) — call
`SparseBackends.reset_roofline!(true)` / `reset_flops!(true)` yourself once before such a loop to
start from zero, exactly as `test_aliased_kl.jl`/`test_aliased_pxp.jl`/`roofline_driver.jl` now do.
Report functions (`show_roofline()`, `show_cas_stats()`, `report_flops(label)`,
`ITensorMPS.print_env_footprint()`, `show_gemm_dims_hist()`) have no internal gate — they just print
whatever's accumulated (empty if `roofline` was never `true`).

Retired 2026-06:
- `SB_ROOFLINE` — `reset_roofline!(roofline::Bool=false, permute_profile_path::String=
  "roofline_permute_profile.tsv")` now takes real arguments (`contract_aliased_dense_shared.jl`),
  writing into the pre-existing `_RF_ON` Ref; `_roofline_on()` simplified to read that Ref directly.
- `SB_FLOP_COUNT` — `reset_flops!(roofline::Bool=false)` takes a real argument (`SparseBackends.jl`),
  backed by a new `_FLOP_COUNT_ON` Ref (same pattern as `_RF_ON`).
- `SB_ENV_FOOTPRINT` / `SB_PERM_CAPTURE` — genuinely threaded (no Ref at all) as a `roofline::Bool=false`
  kwarg through `position!`→`makeL!`/`makeR!` (gating `_record_env_footprint` calls) and
  `product`→`ITensors.contract(P::AbstractProjMPO,...)` (setting `_capture`), in `abstractprojmpo.jl`
  — mirrors the pre-existing `debug` kwarg threading pattern exactly.
- `GEMM_DIMS_HIST` — folded into `_roofline_on()` (`contract_bs_dense.jl`, `test_sparse_kl.jl`).
- `SB_PERMUTE_PROFILE` — fully retired, all 3 former writers accounted for:
  - `contract_bs_dense.jl`'s `_permute_profile_io_sb`/`_bdd_profile_active` and
    `abstractprojmpo.jl`'s `_permute_profile_io` now gate on `_roofline_on()` /
    `SparseBackends._roofline_on()` and open the fixed `SparseBackends._RF_PERMUTE_PATH[]`
    (default `"roofline_permute_profile.tsv"`, settable via `reset_roofline!`'s second argument)
    instead of a caller-supplied path.
  - The third writer, in `ITensors.jl/NDTensors/src/.../contract.jl`'s `_contract!`, is **removed
    entirely** (per decision) — `NDTensors` cannot depend on `SparseBackends` (the dependency graph
    runs the other way) and there's no call chain from `dmrg()` into that generic-dense-contract
    function to pass an argument either, so it could never participate in the consolidated switch.
    `_ndt_profile_io`/`_NDT_PROFILE_IO`/`_NDT_PROFILE_INIT` and all the inline `_ndt_active`/timing
    instrumentation in `_contract!` are deleted; the function is back to its plain, uninstrumented
    form. The merged permute-profile log no longer includes whatever contractions used to route
    through that one NDTensors fallback path.
- `SB_PERM_CAPTURE` gap fix: 2 additional live read sites in `SparseBackends/src/tensor_wrappers_aliased.jl`
  (`reorder_aliased_by_rank` line ~978, `wrapped_contract_aliased` line ~1302 — both `[PERMCAP...]`/
  `[PERMCAP-PHI...]` print statements) were missed in the initial pass; both now gate on `_roofline_on()`
  too. No more `get(ENV, "SB_PERM_CAPTURE", ...)` reads remain anywhere in the tree.
- `ALLOC_SAMPLE_RATE` — unrelated to the above (only used in the standalone, non-`dmrg()`-path
  `diag_alloc_provenance.jl`); hardcoded as `const SR = 0.01`, same treatment as that script's
  `N`/`MD` from the prior round.

## 12. diag_gram_structure.jl / verification scripts

Retired 2026-06 — all `DIAG_*` / `V*` env vars in this section removed:
- `DIAG_N_PLAQ` — `diag_heff_php.jl` and `diag_gram_metric.jl` now take an ArgParse `--N-plaq`
  CLI flag (default 2, matching the old env default); `diag_gram_structure.jl` likewise gets
  `--N-plaq` (default 1).
- `DIAG_MAXDIM`, `DIAG_NSWEEPS` — `diag_gram_structure.jl` gets `--maxdim` (default 10) and
  `--n-sweeps` (default 2) CLI flags alongside `--N-plaq`.
- `DIAG_GRAM_ONLY` — removed entirely from `diag_gram_metric.jl`; the H-dependent parts (full-dense
  operator build + energy/norm checks) that were skippable now always run. Intended for small N
  only (full H is 2^(2N+2)-dim, OOMs at N≥4) — that constraint is unchanged, just no longer
  bypassable via a flag.
- `VN` / `VMD` / `VSW` / `VPS` / `VSPIN` — none of `diag_alloc_provenance.jl`, `test_native_dot.jl`,
  `verify_physical_energy.jl`, `scratch/roofline_driver.jl` had existing CLI-arg infrastructure, so
  these became hardcoded local consts/`let`-block variables instead (edit the file directly to
  change N/maxdim/sweeps/psign/spin per script — same defaults as before: N=12 everywhere;
  MD=40/40/50/80 resp.; NSW=4/8/4 resp.; psign=−1; spin=3).

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
