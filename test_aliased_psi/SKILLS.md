# Skills — `test_aliased_psi/`

Operating notes for an agent picking up the aliased-psi DMRG line of
work. Pair this with `README.md` (target, completed work, open TODOs,
working hypotheses).

---

## Terminology discipline — "dense" vs "dedup" (READ FIRST)

Two **orthogonal** axes; never conflate them, never blur them under the word "dense":

1. **Density** = *which keys exist* (key-space fill). `dense` = all keys present /
   structural zeros materialized into a full array. `block-sparse` = only a subset
   of keys present. Tracked by `n_keys` / `P` / keyhash.
2. **Template sharing (dedup)** = among the *present* keys, are blocks byte-shared.
   `dedup = n_keys / n_templates`. `dedup=1` = no sharing (each present key its own
   block); `dedup>1` = sharing. **Independent of density** — a tensor can be
   block-sparse AND dedup=1.

Rules:
- `M^{±1/2}`/metric apply **removes template sharing (dedup→1) while PRESERVING the
  key set** (measured: P, n_keys, keyhash unchanged; only n_tmpl rises). Say
  **"loses dedup" / "un-shares templates"** — NOT "densifies."
- Only an explicit `to_dense` (e.g. the dense-post-matvec lever) is a genuine
  **density** change (fills structural zeros into a full ITensor).
- "ψ collapses to dense" in older notes means **dedup→1**, not key-fill.
- Before writing "dense", decide: key-fill or sharing? Use the exact word and quote
  the `n_keys / n_tmpl / dedup` numbers.

---

## Confidence discipline (READ FIRST — applies to every change)

**Always tag every hypothesis, diagnosis, and fix with an explicit confidence
level, and never declare a bug "solved" with certainty.** This is numerically
subtle code: a fix can pass every test you ran and still be incomplete (untested
regime, near-singular metric, many-sweep drift, a different alias schema). State
what you verified, what you did NOT, and what would raise or lower confidence.

Use this scale and record it next to the claim (README hypotheses, commit
message, status report):

- **Confirmed** — proven by a decisive diagnostic AND validated end-to-end;
  no plausible regime left untested. Reserve this; it is rare here.
- **High** — strong decisive evidence (e.g. matches a ground-truth reference to
  many digits) + at least one end-to-end run, but untested regimes remain
  (scale, sweep count, other inputs). Default ceiling for a fresh fix.
- **Medium** — mechanism is sound and one test passes, but the evidence is
  partial or indirect; needs more cases before relying on it.
- **Low / Speculative** — plausible from reading the code; not yet tested.

For every fix also write a short **"Residual risk / not-yet-tested"** line and
mark whether **further evaluation is needed** (and what would settle it). Treat
a fix as provisional until those follow-ups are done — keep its env-gate so it
can be toggled, and keep the diagnostic that would catch a regression.

---

## Result provenance (READ — applies to every number you record)

**Every benchmark/diagnostic number written into the README, SKILLS, a commit
message, or a status report MUST be annotated with the exact command that
produced it** — the script, the env vars, and the parameters (`N_plaq`,
`maxdim`, sweep count, `BMF_*`/`SB_*` flags). A bare number with no command is
not reproducible and not trustworthy: the same script gives different energies
and footprints on the iso path vs Path-B, at different `N_plaq`, or with a
different sweep schedule.

Concretely, record numbers like:

> `N_plaq=12, maxdim=40, 10 sweeps`, produced by
> `SB_ALIASED_ENABLE=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 julia --project=. test_aliased_psi/test_aliased_kl.jl --N-plaq 12 --maxdim 40 --n-sweeps 10`

If you cannot state the command that produced a number, treat it as
**unverified provenance** — flag it as such and do not present it as comparable
to numbers from a known command. (Example of the failure this prevents: the
README's old "190× slower / N_plaq=16 footprint" figures have no recorded
command and may be from a different DMRG path than the current bench — they are
NOT bench-comparable, and that ambiguity is exactly what the annotation rule
exists to avoid.)

**Logging discipline (READ — these runs are long, noisy, and CPU-contended):**
NEVER shape a long run's output inside the run command with `| tail`, `| head`,
`| grep`, or `| cat`. That truncates/discards the rest, and key lines (e.g. the
`[ROOFLINE]` summary, which prints *before* the timer table) get lost — forcing a
costly re-run of the same multi-minute test. Instead, redirect the WHOLE run to a
file under `/tmp/` (`> /tmp/run.log 2>&1`), then `grep`/Read the saved file for
whatever you need. The run happens once; all output is preserved for any later
query. Re-running a test purely because output was truncated wastes a cycle and
pollutes any concurrent timing measurement.

---

## What "aliased" means here

`AliasedBlockSparse{T, N, N2, P, K}` lives in
`../SparseBackends/src/aliased/storage.jl`. Each logical block at key
`keys[i]` has value `scalars[i] * templates[alias_ids[i]]`. So multiple
keys that share an `alias_id` reuse one template buffer — pure structural
deduplication.

Layout convention everywhere in this codebase:

- **Sparse prefix axes (first `P`):** indexed by integer keys
  (`keys :: Vector{NTuple{P,K}}`). Block presence is determined by which
  keys appear.
- **Dense tail axes (last `N2`):** the contents of each block, flat
  column-major into `templates` (length `n_templates × blksize`).
- **Doubled-link convention:** physical bonds appear *twice* — once as a
  sparse-prefix axis (the "channel", small integer dim) and once as a
  dense-tail axis (the "multiplicity", carries the bond growth during
  sweeps). For aliased psi sites, `blksize=1` initially (multiplicity
  axes are dim-1) and grows with the DMRG bond.

`WrappedAliasedBlockSparse` wraps this with ITensor `Index` objects.

---

## The mental model for aliased DMRG

- psi sites have real alias dedup (2–4× in our test setup). They come
  from `contract(P_sparse, psi0, :coo, :aliased)`.
- env tensors (`Lenv`, `Renv`) **do not** get meaningful alias dedup with
  the current single-template-per-block format — see
  `measure_env_aliasing.jl` and the README's "Working hypotheses #5".
  Env-build cleanly going dense is consistent with how BS-DMRG operates;
  don't fight it.
- The factorize at the bond is where alias schema is restored:
  `itensor_aliased_factorize` produces L, R with the **frozen schema**
  inherited from the previous `M_b`, `M_b1` (same keys, alias_ids,
  scalars), and the templates come from the alias-reduced SVD.
- Path-B (M-corrected eigsolve) is the principled answer to non-iso L, and
  **the wrong-energy bug has a HIGH-confidence fix** (2026-06; see README
  "Energy correctness", confidence HIGH/not-Confirmed). The fix: the eigsolve
  operator must be
  `A = M⁻¹·H_eff`, not `H_eff`. Lanczos under the M-inner product only
  returns the *generalized* eigenvalues of `H_eff·φ = E·M·φ` when the
  operator is M-self-adjoint, which `M⁻¹·H_eff` is and `H_eff` alone is
  not. Apply `M⁻¹ = Linv_L² ⊗ Linv_R²` (the `Linv_*` factors that
  `build_minv_half_pair_factored` already returns) inside `H_op`
  (`dmrg.jl`, gated `BMF_APPLY_MINV`, default on).

When working on this code, *don't* try to make `L` strictly left-iso —
that fight is structurally lost (templates shared across bond-channel
values force linear-dependent columns). Path-B + the `M⁻¹` correction is
the right answer; non-iso `L` (off-diagonal `L†L ≈ -0.75`) is EXPECTED and
correctly handled by the metric, not a bug to chase.

---

## Sequencing rules

1. **Construction first; correctness gates everything else.** The first
   step on any debug session: confirm `diag_step_by_step.jl` Steps 1–2
   still match dense to machine precision. If they don't, none of the
   later code matters.

2. **Don't densify silently.** Whenever you hit a path that materializes
   an aliased tensor to dense to "make it work," gate it behind an env
   var and surface it via `SB_ALIASED_TRACE`. Densification cascades:
   one dense intermediate forces every downstream `Aliased+Dense` /
   `Aliased*Dense` into the dense fallback.

3. **Schema-freeze, not schema-derive.** When producing aliased outputs
   (factorize L/R, matvec Hv after snap), keys/alias_ids/scalars come
   from a **template** (typically `M_b` or `v`). Only `templates` are
   freshly computed. Inventing fresh alias schemas mid-pipeline is what
   broke `_aliased_shared_via_bs_fission!`'s trivial-dedup output.

4. **`SB_ALIASED_NATIVE_FISSION=0` is the safe default during debugging.**
   The native path is fast but exercises many edge cases. Until energies
   are confirmed correct via the BS-delegate fallback, debug there first.

5. **Always smoke-test `test_aliased_kl.jl --N-plaq 2 --maxdim 4 --n-sweeps 3`**
   before claiming a fix. Large `N_plaq` takes a long time (JIT alone is ~50s,
   and a converged N=12/md=40/10-sweep run is much longer); N=2 is the fast
   sanity check that the script runs and the alias invariant holds. Compare
   energies against the sister baselines (`../test_sparse_psi/test_dense_kl.jl`
   PHP-dense and `test_sparse_kl.jl` BS) at matching args — never against a
   bare-H unprojected dense run (that was the retired bench's invalid baseline).

---

## Files to know in order

When pulling on a thread, the call stack you'll touch most often:

1. `ITensorMPS.jl/src/dmrg.jl` — the DMRG main loop, esp. the Path-B
   branch (`is_sparse_mps && BMF_ISO_PATH != "1"`), ~lines 685–760. **This is
   where the energy-correctness fix lives:** `H_op` applies `A = M⁻¹·H_eff`
   via `apply_Minv` (`Linv_L²⊗Linv_R²`), gated `BMF_APPLY_MINV`.

2. `ITensorMPS.jl/src/mps.jl` — `replacebond_sparse!`. The aliased branch
   dispatches to `itensor_aliased_factorize`.

3. `ITensorMPS.jl/src/abstractmps.jl` — `orthogonalize!`. Both sweep
   directions have aliased branches.

4. `ITensorMPS.jl/src/abstractprojmpo/abstractprojmpo.jl` — `product`
   (matvec), `_makeR!`, `_makeL!`, `_mul_preserve_aliased`.

5. `SparseBackends/src/aliased/factorize.jl` — `itensor_aliased_factorize`
   (alias-reduced SVD with per-channel whitening). Iso-error is observed
   here.

6. `SparseBackends/src/tensor_wrappers_aliased.jl` — wrapper / dispatch /
   `Base.:+` / `_zero_similar` / `recast_aliased_to_template`.

7. `SparseBackends/src/tensoralgebra/contract_aliased_shared.jl` — multi-
   label `contract_shared!` with native fission detection and the BS-
   delegate fallback `_aliased_shared_via_bs_fission!`.

8. `SparseBackends/src/tensoralgebra/contract_aliased.jl` — single-label
   kernels. `_contract_aliased_prefix_outer_ad!` supports fission.

9. `SparseBackends/src/path_b_helpers.jl` — gram cache, `build_half_pair_*`,
   `build_minv_half_pair_factored` (returns `Mhalf_L, Linv_L, Mhalf_R, Linv_R`),
   `apply_minv_preserve_bs`, `recast_to_template`. All verified correct for
   aliased ψ; the historical "energy wrong" symptom was NOT here — it was the
   caller in `dmrg.jl` discarding `Linv_*` (see #1). Future *perf* work
   (keeping `Mhalf`/`Linv` aliased) does live here.

---

## How to debug an "energy wrong" report

**First: run `diag_gram_metric.jl`.** It already separates the five candidate
layers (gram, metric, `H_eff`, `apply_minv`, eigsolve operator) and, as of
2026-06, established that the gram/metric/`H_eff`/`apply_minv` layers are all
NUMERICALLY EXACT for aliased ψ — the historical "energy wrong" bug lived in
the eigsolve operator (missing `M⁻¹`, now fixed). If a new energy regression
appears, re-run this diagnostic first: parts A/B/C should still be 0 / exact,
and D3 should match D. If D3 ≠ D again, the `M⁻¹` application (`Linv_*` /
`apply_minv_preserve_bs`) or `BMF_APPLY_MINV` gating regressed.

The older bottom-up ladder from `diag_step_by_step.jl` outwards (still useful
for *new* kernel work):

1. **Step 1: initial psi values match dense?** If not, construction
   (`contract(P, psi0, :coo, :aliased)`) is wrong. Check
   `wrap_itensor_aliased` and `contract_aliased_coo_dense.jl`.

2. **Step 2: matvec values match dense?** If not, `product` /
   `contract_aliased*` kernels. This is the most-stable layer; rarely
   the culprit.

3. **Step 3: factorize reconstruction `||L*R − phi||`?** If non-trivially
   large, alias-reduced SVD is wrong. If small (1e-16), move on.

4. **Step 4: `L†L = I` iso check.** Diagonals = 1 is necessary;
   off-diagonals tell you about bond-channel coupling. Off-diagonals
   non-zero means **standard DMRG won't give correct energy**, and you
   need Path-B working.

5. **Gram cache values.** Build gram from aliased psi vs densified psi,
   compare entry-by-entry (`diag_gram_metric.jl` Part A). **Verified EXACT
   (=0) at every bond as of 2026-06** — the gram kernels are NOT the bug.
   Was previously the "most likely remaining culprit"; that guess was wrong.

6. **`build_half_pair_single` on aliased gram.** It densifies internally
   and runs `LinearAlgebra.eigen`. The dense `Mhalf`/`Linv` applied to
   aliased φ via `apply_minv_preserve_bs` is **verified numerically exact**
   (`diag_gram_metric.jl` Part B: `<φ|M|φ>` = `<Ψ|Ψ>` to 1e-16) — no
   classification drift. Correctness-complete; only a *perf* concern (it
   does a dense×aliased contract per Lanczos step).

7. **Eigsolve operator (THE historical bug).** `H_op` must apply
   `A = M⁻¹·H_eff`, not `H_eff`. Check `BMF_APPLY_MINV` is on and that
   `Linv_L`/`Linv_R` are applied (`diag_gram_metric.jl` Part D3 vs D). With
   `H_eff` alone, Lanczos solves the wrong problem and energy is unphysical
   (e.g. -14.45). This is the one that was actually broken.

---

## Reading the trace

Common SB_ALIASED_TRACE lines and what they tell you:

- `[SB_ALIASED_TRACE product matvec #N]  v storage=...  Hv storage=...`
  If both stay `WrappedAliasedBlockSparse` across iterations: matvec is
  preserving alias correctly. Switch to dense → look at the previous few
  lines for `_add_aliased_via_dense` (cross-shape add) or
  `mul_preserve_aliased` results.

- `[DEEP cross-schema merge #N]  A=Aliased{N=4,nb=18,nt=9}  B=...`
  The aliased `+` cross-schema (same dims, different keys) path fired.
  Result is still aliased (with one template per resulting block — trivial
  dedup along this step, but alias structure intact).

- `[DEEP _add_aliased_via_dense #N]  A=...  B=...  → DENSE`
  Aliased + Aliased fell into the dense fallback (different dims). The
  recent fix (`Base.:+` axis-permutation alignment) was specifically to
  catch the case where dims differ only by axis order. If you see this
  fire, the alignment didn't catch it — investigate the inds tuples.

- `[SB_ALIASED_TRACE replacebond! ENTRY b=B]  M[b]=...  M[b+1]=...  phi=...`
  All three storages at the start of `replacebond!`. If `phi=dense` with
  M[b]=aliased, the factorize dispatches to the dense path → produces
  dense L,R → next bond sees `M[b]=dense` and the cascade kills aliased.

---

## ✅ FIXED (validated N≤12; N=32 confirmation pending) — aliased Path-B converges with the memory win (2026-06)

The former headline bug (ψ collapsed to dense on Path-B; then energy oscillated
at N≥4) is **fixed** by a **5-gate** combination. Validated at `N_plaq = 2,4,12`:
ψ stays aliased every sweep, energy converges monotonically to near-dense, and
footprint beats dense. **N=12, md=40:** aliased E=−17.1605 vs dense −17.1828
(gap 0.13%); footprint 0.196 vs 1.312 MiB (**6.7× smaller**); dedup 3.69×.
**N=32/md=80 is the outstanding validation — do not call this "solved" until it
passes** (run launched; see README).

**The fix (all gates default-off until hardened; turn on together):**
`SB_ALIASED_MINV_HINT=1` (classification-preserving hint in `contract_preserve_bs`
→ no dense collapse) · `SB_ALIASED_NATIVE_FISSION=1` (dedup-preserving fission) ·
`SB_ALIASED_PERCM_CAP=1` (per-cM cap `fld(maxdim/bond_ch_dim)` in
`itensor_aliased_factorize` → honest BD ≤ maxdim) · `BMF_BOP_PROJECT=1`
(`B = M⁻¹ᐟ²·H_eff·M⁻¹ᐟ²` null-projected symmetric eigensolve in range(M)) ·
`BMF_MINV_RTOL=1e-2` (aggressive cutoff for the rank-deficient M).

**Two root causes, both understood:**
1. *Collapse to dense* — the Lanczos `v + α·Hv` densified on differing-`{N2,P}`
   operands (`Base.:+` at `:1141`). Fixed by the **hint** keeping the `M⁻¹` apply
   in φ's classification (`SB_ALIASED_MINV_HINT`), so the add stays on the
   same-classification path.
2. *Energy oscillation at N≥4* — the gram M is structurally **rank-deficient**
   (template sharing ⇒ non-iso ⇒ iso unreachable without un-deduplicating). M is
   **constructed correctly** (`|dLgram|=|dRgram|=0` at N=2 AND N=4, via
   `diag_gram_metric.jl DIAG_GRAM_ONLY=1`); the bug was **inverting** it — the
   default `rtol=1e-10` kept near-null λ whose `1/√λ` blew up M⁻¹. Fixed by
   B_op null-projection + aggressive `rtol=1e-2`.

**Lessons for this area:**
- For aliased ψ, `M ≠ I` is **structural** — never try to make L iso (that
  destroys the aliasing). Use B_op (range(M) projection), not strict iso.
- `SB_ALIASED_SNAP=1` does NOT help (only snaps matvec output, not the Krylov
  operand) — leave it off.
- `rtol` for the gram pseudo-inverse needs to be **aggressive** (~1e-2) for
  aliased; the dense-DMRG default (1e-10) is far too loose.
- **Next (perf + robustness):** reduced-space / low-rank factored solve —
  `M = U_r Λ_r U_r'` is already low-rank; solve the small `r×r` reduced
  generalized problem in range(M) directly. Removes the hand-tuned rtol and the
  O(d²) dense `M^{±1/2}` apply (~2.7× slowdown vs dense at N=12).

---

## Memory footprint invariant

**The guideline:** ALI and BS footprint should be less than dense at every
`N_plaq` tested. This is the core structural promise: alias deduplication and
block-sparsity should make the wavefunction *smaller* than an unconstrained
dense MPS as the system grows.

**Exception — small-`N_plaq` overhead:** At very small `N_plaq`, fixed
per-block metadata (keys, alias_ids, scalars, index objects) can outweigh the
data savings, so ALI or BS footprint exceeding dense at small scale is not
automatically a regression. It is expected behavior from implementation
overhead.

**When ALI/BS footprint exceeds dense — trigger further evaluation:** If
footprint exceeds dense at a `N_plaq` large enough that overhead should not
dominate, this requires investigation to distinguish two causes:
1. **Pure implementation overhead** — metadata or bookkeeping structures
   (dicts, index tuples, wrapper objects) that grow with system size
   independently of the actual tensor data. Not a structural regression but
   may point to unnecessary allocations.
2. **Datastructure regression** — the alias compression or block-sparsity is
   breaking down (e.g. one-template SVD collapse, bond dim exceeding MAXDIM,
   blocks not being pruned). This is the scenario to fix.

**Diagnosis-first rule:** Measure before hypothesizing. Print `n_blocks /
n_templates` at each psi site after the final sweep and compare the actual bond
dimension at each site against MAXDIM. Only after that measurement should a
hypothesis about the cause be formed. Any hypothesis must be tagged with a
confidence level (see "Confidence discipline" above) and must NOT be stated as
a definitive fix or confirmed cause until verified by a decisive diagnostic.
Candidate causes are hypotheses until proven.

---

## Things to NOT do (lessons learned)

- **Don't chase the energy bug in the kernels (SVD / gram / factorize / matvec).**
  All of these are verified numerically exact (`diag_gram_metric.jl`). The
  historical wrong energies (-6.44 on the iso path, -14.45 on Path-B) were NOT
  a kernel-value bug. -14.45 was the missing `M⁻¹` in the Path-B operator
  (now fixed); -6.44 came from the iso-path branch (`BMF_ISO_PATH=1`) assuming
  `L†L = I` which is structurally false for aliased ψ — use Path-B, not the iso
  path, for aliased.

- **Don't fix iso by un-deduplicating across bond-channels.** That's
  conceptually identical to BS (the user explicitly rejected this in the
  conversation that led to this work). Path-B + the `M⁻¹` correction is the
  right answer, and it works.

- **Don't densify phi at the eigsolve boundary "as a shortcut."** Path-B uses
  the gram metric to correct for non-iso L; keep phi aliased throughout. (The
  metric/apply machinery is exact on aliased φ — `diag_gram_metric.jl` Part B —
  so there is no correctness reason to densify.)

- **Don't omit `M⁻¹` from the Path-B operator.** The whole point of Path-B is
  the generalized problem `H_eff·φ = E·M·φ`; the M-self-adjoint operator is
  `A = M⁻¹·H_eff`. Applying `H_eff` alone (the original code) silently solves
  the wrong problem and only looks fine for canonical BS ψ where `M = I`.

- **Don't trust `n_templates / n_blocks` as a correctness signal.** The
  alias invariant says "psi sites are still aliased after the sweep" —
  that can be TRUE while the *values* in those templates are silently
  wrong. The bench script's energy column is the ground truth.

- **Don't state a hypothesis as a definitive fix or confirmed cause.** Any
  proposed explanation for a regression (footprint, energy, or otherwise) is a
  hypothesis until a decisive diagnostic confirms it. State it as "Hypothesis
  (Low/Medium/High): …" with a residual-risk note, not as "fix by doing X" or
  "the cause is Y". The confidence discipline (see above) applies to footprint
  investigations exactly as it does to energy bugs — this codebase has too many
  subtle interacting layers to skip that discipline.

---

## What "done" looks like

Status as of 2026-06: **energy correctness fix at HIGH confidence (not yet
Confirmed)**, perf/footprint being re-benchmarked at scale. Do not treat the
energy bug as closed until the README "Residual risk" follow-ups pass.

- ✅ Aliased ψ runs physically on Path-B: `test_aliased_kl.jl --N-plaq 2
  --maxdim 4` produces a physical energy (was -14.45 pre-`M⁻¹`-fix), alias
  invariant holds every sweep, honest BD obeys MAXDIM, dedup 2–4× per site.
- ⚠ **Energy-vs-dense comparison must be REDONE against the honest baseline.**
  The old "ALI -3.6185 approaches dense -3.5259" used the retired bench's
  *invalid* dense run (bare H + unprojected random ψ → unconstrained, and
  under-converged at md=4/2-sweeps). Re-run dense via
  `../test_sparse_psi/test_dense_kl.jl` (PHP) at matching args; ALI should sit
  *above* a well-converged constrained dense (variational), not below it.
  Treat the old dense numbers as void (Confidence: Confirmed — visible in the
  retired script).
- ✅ Iso check: non-trivial off-diagonals (`L†L ≈ -0.75`) **but** Path-B's
  `M⁻¹`-corrected eigsolve delivers correct energy regardless — this is the
  branch that landed (no strict-iso fix needed).
- ⏳ Footprint vs dense/BS at scale: re-measure with `test_aliased_kl.jl` vs the
  sister baselines. Goal: ALI footprint < dense and < BS as `N_plaq` grows
  (small-`N_plaq` overhead exception applies). The `n_blocks/n_templates` dedup
  and honest-BD≤MAXDIM checks in the runner are the regression guards.

Energy correctness holds on Path-B; the open items are (1) re-establishing the
honest dense/BS comparison via the sister runners, and (2) perf (README Open
TODO #2/#3: keep `Mhalf`/`Linv` aliased to drop the dense×aliased `apply_minv`
per Lanczos step).
