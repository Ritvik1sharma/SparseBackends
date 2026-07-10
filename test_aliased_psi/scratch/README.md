# test_aliased_psi/scratch

Superseded / consumed probes. Kept as reproducers; their conclusions are recorded
here and in the memory store. Nothing here is part of the automated test suite.

## Moved 2026-07-07 — superseded by the factorize `cutoff=0` fix

Both were converged-state, **dense-analysis** probes that answered specific
"why is the metric M's tail there" questions. Once the factorize epsilon-floor
fix landed (`_aliased_alias_reduced_factorize` no longer keeps numerical-zero
singular values at `cutoff=0`), the grams drain to a clean scaled projector
`c·Π` regardless of what these measured — so they no longer assert anything the
live `test_gram_structure.jl` doesn't. See memory
`project_factorize_cutoff0_mult_inflation_fix` and
`project_gram_clean_is_right_canonical_gauge`.

### `diag_gram_offblock.jl`
Decomposes the metric `M` on a doubled bond link `(ch, mu ; ch', mu')` and reports
**off-channel weight** `√(Σ_{i≠j}‖M[i,:,j,:]‖² / ‖M‖²)` plus an **eigenvalue-cluster
count** (`#nz-clusters==1 ⇒ CLEAN c·Π`, else `SPREAD`). L and R grams, bonds 5–10,
converged N=12 KL.

**Conclusion:** the tail is **CROSS-channel**, not within-channel. OFF-CHAN measured
`√(1-1/nc)` (large), which *refuted* the earlier wrong claim that `M` was
block-diagonal in the channel index. Cross-channel coupling is the KL signature
(`T_{a,a'}=G_{D(a,a')}`, `D=a⊕a'`), and `M → c·Π` at convergence with a single
nonzero eigenvalue cluster.

### `diag_gauge_flip.jl`
Takes one **fixed physical link** `λ=(k,k+1)` and builds the gram two ways — a
LEFT-canonical view (orthocenter at `k+1`) and a RIGHT-canonical view (orthocenter
at `k`). Same link, opposite gauge.

**Conclusion:** the Lgram-spread / Rgram-clean disparity is **purely the
canonicalization direction** (a gauge op — the SVD ortho), NOT intrinsic to the
physical cut. The same link flips CLEAN↔SPREAD under re-gauge. Actionable
consequence: the right-canonical `apply_half` can be a scalar. (Memory:
`project_gram_clean_is_right_canonical_gauge`.)

### `test_compress.jl`
Validated `compress_aliased_templates!` on the KL aliased single-site self-gram
(`nt 16→4`, value-err ~1e-17). **Merged** into `test_gram_structure.jl` as the
`analyze_aliased` corollary, which now measures *both* sparsity (channel-block
occupancy) and dedup, for KL **and** PXP. Kept here as the original KL-only
reproducer.

## Moved 2026-07-08 — RR / direct-M / from-P-is-KL-only investigation (consumed)

One-off probes from the RR-for-PXP session. All conclusions are in the memory
store; the live equivalents are `test_aliased_pxp.jl` (now parameterized with
`--run-mode`, `--gram-from-h`, `--rr-dense-iter`) and `test_aliased_kl.jl`.

- **`pxp_both_states.jl`** — ground+excited PXP over `(run_mode, rr_dense_iter)`
  configs at fixed bd. **Merged into `test_aliased_pxp.jl`** via the new
  `--run-mode` / `--gram-from-h` / `--rr-dense-iter` args (fixed bd = `--mindim K
  --maxdim K`). See `project_rr_plateau_not_snapping`.
- **`gram_compare.jl`** — covector env-slice M (from H) vs direct ψ†ψ transfer M.
  Conclusion: identical for KL (0.0), mismatch 0.57–0.66 for PXP → covector is
  WRONG for PXP; use `gram_from_h=false`. (`project_aliased_pathb_minv_fix`-adjacent.)
- **`gram_fix_test.jl`** — direct-M correctness check (PXP −8.8 covector → −13.0 direct).
- **`rr_test.jl`** — RR KL+PXP energy/time; RR not a perf win (see `project_rayleigh_ritz_parked`).
- **`rr_profile.jl`** — RR cost breakdown (M-applies ~32% + matvec ~30% + factorize ~29%).
- **`pxp_iso_check.jl`** — confirmed PXP genuinely needs M (not iso; run_mode=:iso unphysical).

## Earlier scratch (pre-existing)
`bench_aliased_vs_bs_vs_dense.jl`, `bench_footprint_honest_bd.jl`,
`test_aliased_php.jl` — prior benchmarking scratch, not part of this consolidation.
