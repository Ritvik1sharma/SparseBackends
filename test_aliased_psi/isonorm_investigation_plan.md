# Aliased-ψ isonorm investigation — plan & state (resumable)

## Goal
Find & fix the iso-normality bug in aliased Path-B DMRG. Symptom: the bond metric
`M = ψ†ψ` is a clean scaled projector `c·Π` (good) on the right-canonical side but
**spread** on the left, AND this is **convergence-dependent** (at 2 sweeps BOTH sides
spread; only near convergence does the right clean up). If isonorm were a pure gauge
property it would hold every sweep — so something violates it mid-sweep.

## Confirmed understanding (do not re-derive)
- **P** = `∏_j ½(I + psign·G_j)`, `G_j = e^{iπSy}⊗e^{iπSx}⊗e^{iπSx}⊗e^{iπSy}` on plaquette
  `(2j-1,2j,2j+1,2j+2)`. Manually verified: spin-1 `e^{iπSa}=I−2Sa²` are Hermitian
  involutions forming the Klein four-group ⇒ all `G_j` commute ⇒ **P is a Hermitian
  projector** `P²=P=P†`. (Confirmed.)
- **ψ = P·core**, stored aliased: alias structure (keys/alias_ids/scalars) ≡ P; templates
  ≡ core slices; template chosen by the FULL sparse key `(s', L-channel, R-channel)`, not
  channel alone; dedup = one core slice under several keys. (User-confirmed.)
- Doubled bond link = (**channel** = P's MPO bond, `nc`=2 odd / 4 even) × (**multiplicity**
  = core's MPS bond, `nm`). So `ψ†ψ = core† P core`.
- **Isonorm = core canonical in its m-bond (⇒ I_mult), P supplies |χ⟩⟨χ| channel factor
  ⇒ ψ†ψ = c·Π**, `Π = |χ⟩⟨χ|_channel ⊗ I_mult` (rank `nm`, `c = 2^env`). `c·Π` does NOT
  force different core slices equal — only makes the P-bond (channel) redundant.
  (Corrected earlier over-claim.)

## Measured facts (Confirmed)
- `diag_gram_offblock.jl` (converged, N=12, psign=-1, md40): `Rgram` = CLEAN `c·Π` every bond;
  `Lgram` spread at interior bonds 6–9, clean at edges 5,10. `OFF-CHAN=√(1−1/nc)`
  identical L & R (the P-alias coupling, not a defect). `within-blk-dev` is NOT a valid
  clean/spread test (large even for clean projector) — use eigenvalue clusters.
- `diag_gauge_flip.jl`: the SAME physical link flips SPREAD (left-canonical) ↔ CLEAN `c·Π`
  (right-canonical). So the disparity is the **canonicalization direction**, not the cut.
- 2-sweep snapshot: BOTH sides spread ⇒ cleanliness is **convergence-dependent** (KEY:
  contradicts pure-gauge; means the invariant/core-canonical is broken mid-sweep).

## Ruled OUT (Confirmed non-bugs)
- Whitening: `WHITEN_DIAG` c-spread L=0,R=0 (exact both directions).
- Channel/P coupling: identical L & R (`OFF-CHAN` equal).
- **Factorize path**: `trace_fact.jl` (SB_FACT_TRACE) shows all 39 factorizations in a
  1-sweep N=6 run route through `itensor_aliased_factorize` from BOTH `orthogonalize!`
  (initial, ortho=right) and `replacebond_sparse!` (sweep, ortho=left then right), with
  **zero non-aliased leaks**. Factorize also freezes M_b/M_b1's schema onto L/R verbatim
  (`_build_aliased_frozen_schema`), so P is preserved structurally by the factorize.
- `_snap_to_schema` does NOT re-derive P — copies v_schema keys/alias_ids/scalars verbatim
  and DROPS matvec-output blocks off the schema (dropped_norm_frac ~1.7e-3), and forces
  dedup by taking one representative block per template.

## Prime hypothesis (Medium confidence)
The `M^{-1/2}` apply is **not acting as the clean projector `c^{-1/2}·Π`**. Since
`ψ∈range(P)` and `range(M)⊆range(P)`, a faithful `M^{-1/2}` + `H_eff` (`[H,P]=0`) keep the
Krylov vectors in `range(P)`, so the snap would drop EXACTLY zero (lossless). The measured
~1.7e-3 drop ⇒ the M-apply (likely the **dense `apply_half` contracting dense `Mhalf` with
aliased φ**, the seed-densification) throws weight onto off-P keys, which the snap discards
→ corrupted core into factorize → spread `M` → perpetuates until convergence smooths it.
Prediction: the from-P projection apply (`M^{-1/2}=c^{-1/2}Π`) would make the drop vanish
and preserve isonorm every sweep.

## Steps
1. [DONE] **Schema-preservation check**: CONFIRMED (a) L/R kh/sh match M_b/M_b1 exactly
   (frozen P ✓), (b) schema kh/sh INVARIANT across the chain & sweeps (c051/7449/2ef5/b6db
   recur by even/odd site) — **P never drifts, never corrupted**. ⇒ the isonorm break is
   PURELY in the template (core) numerics, NOT in P. (SB_FACT_TRACE instrumentation used
   here has been removed; trace_fact.jl deleted.)
2. **M-apply projection check**: instrument the matvec / `apply_half` / `apply_minv` to
   measure the off-P-image fraction of its output (before the snap) at intermediate vs
   converged sweeps. If nonzero & convergence-dependent → confirms the M-apply is the
   isonorm-breaker. Compare dense apply_half vs from-P scalar apply.
3. **Localize** which M-apply step leaks: seed `y0=apply_half(Mhalf,φ)` (dense×aliased),
   the matvec `product(PH,y)` H-env, or the recovery. Use the snap `dropped_norm_frac`
   (enable `_snap_to_schema(...; dbg=true)`) per step.
4. **Fix**: make the leaking step preserve `range(P)` — likely apply `M^{±1/2}` as the
   aliasing-preserving projector (from-P: `M^{-1/2}=c^{-1/2}Π`, `Number×aliased` keeps schema)
   instead of dense contraction. Ties into the parked
   `apply-m-1-2-in-three-eager-twilight.md` plan.
5. **Validate**: `Lgram` clean `c·Π` at EVERY sweep (re-run diag_gram_offblock at 2 & 12
   sweeps); DMRG energy A/B vs baseline, both sectors (psign ±1), 25 sweeps, ΔE ≤ ~1e-5;
   dedup preserved.

## CLEANUP
- [DONE] Removed `SB_FACT_TRACE` env var + `_FACT_TRACE_N` Ref + trace block from
  `SparseBackends/src/aliased/factorize.jl` (back to ALIASED_TRACE-only). Deleted `trace_fact.jl`.
- [PENDING] Any dbg=true snap flags flipped for step 3 — revert after use.

## Tools (kept)
- `diag_gram_offblock.jl` — eigenvalue CLEAN/SPREAD + OFF-CHAN + within-dev per side/parity.
- `diag_gauge_flip.jl` — same-link left vs right gauge flip.
- (temp) `trace_fact.jl` — factorize path/schema trace (delete after step 1).

## Related memories
[[project_gram_clean_is_right_canonical_gauge]], [[project_minv_from_p_works]],
[[project_minv_half_destroys_dedup]], [[project_eigensolve_dedup_loss_is_H_env]],
[[project_gram_scaled_projector_from_p]], [[feedback_no_unnecessary_knobs]].
