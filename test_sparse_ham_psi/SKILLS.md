# SKILLS — `test_sparse_ham_psi/` (aliased ψ × aliased PHP)

Operating notes for the **both-aliased** DMRG line of work: the wavefunction ψ
*and* the projected Hamiltonian `PHP` are BOTH `WrappedAliasedBlockSparse`. Pair
this with `README.md` (target, status, reproduce commands).

This folder is the union of two sister lines:
- `../test_aliased_psi/`  — aliased ψ × **dense (bare) H**
- `../test_sparse_ham/`   — **dense ψ** × aliased PHP

Neither sister ran both backends aliased at once; this folder does.

---

## 0. THE LOCALITY RULE (read first — applies to EVERY change you make)

There are **four** storage configurations that share the same DMRG/contraction
code paths. Only the last is ours:

| # | ψ | H | exercised by | must stay |
|---|------|------|------|------|
| 1 | dense | dense | plain ITensorMPS | **byte-identical** |
| 2 | aliased | dense (bare) | `../test_aliased_psi/test_aliased_kl.jl` | **byte-identical** |
| 3 | dense | aliased (PHP) | `../test_sparse_ham/test_check_working_aliased.jl`, `test_pxp_aliased.jl` | **byte-identical** |
| 4 | **aliased** | **aliased (PHP)** | `test_sparse_ham_psi_kl.jl` (this folder) | the only case we may change |

**RULE: every change you make for case 4 MUST be gated so that cases 1–3 are
byte-identical to before your change.** The DMRG loop, `ProjMPO` env build,
`contract_preserve_bs`, `output_inds`, the aliased kernels, and the Path-B
eigsolve are all SHARED by cases 2, 3, and 4 (and case 1 for the non-aliased
parts). A change that "just helps case 4" but is written into a shared function
without a case-4 gate **will** silently alter cases 2 and 3.

### How to gate for case 4 ("both-aliased")

The reliable, **structural** signal is: *the operands are actually aliased*.
Do not rely on an env-var alone (the runner defaults the `SB_ALIASED_AA_*`
flags ON, and a sister run could set them too). Gate on the storage:

- env build (`_makeL!/_makeR!`):
  `_keep_env = SB_ALIASED_AA_ENV && _is_aliased_itensor(H_site) && _is_aliased_itensor(psi[i])`
  → fires only when **both** H[k] and ψ[k] are aliased ⇒ only case 4.
- matvec hint (`contract(P,v)`):
  `_use_aa_hint = _v_is_aliased && _H_is_aliased && SB_ALIASED_AA_HINT`
  (`_H_is_aliased = any(_is_aliased_itensor, P.H[site_range(P)])`)
  → only case 4 (case 2 has dense H ⇒ `_H_is_aliased=false`; case 3 has dense ψ
  ⇒ `_v_is_aliased=false`).
- any helper that can't see ψ/H directly (e.g. `_reorder_env_for_aliased`)
  takes a `both_aliased::Bool` argument **threaded from the caller's
  `_keep_env`**, and only changes behaviour when it's true.

**Verify locality after any change** by running cases 2 and 3 at fixed args and
confirming the per-sweep energies are unchanged to the last digit:
```bash
# case 2 (aliased ψ × dense H) — must be unchanged:
SB_ALIASED_ENABLE=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
  julia --project=.. ../test_aliased_psi/test_aliased_kl.jl --N-plaq 4 --maxdim 20 --n-sweeps 6
# case 3 (dense ψ × aliased PHP) — must be unchanged:
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 SB_ALIASED_ENABLE=1 SB_FUSE_LINKS=1 \
  SB_PLAN_B=1 SB_AUTO_DISPATCH=1 SB_PREPERMUTE_H=1 BENCH_MAXDIM=40 BENCH_NSWEEPS=5 \
  julia --project=.. ../test_sparse_ham/test_check_working_aliased.jl 4
```
Diagnostics (prints) must be **default-off** so they add nothing to any path
unless explicitly enabled.

---

## 1. Mental model: where the two aliasings meet

- ψ sites are aliased (`contract(P, ψ₀, :coo, :aliased; denseLinksB=0)`), same as
  case 2. Structurally non-iso ⇒ needs Path-B (M-corrected eigsolve).
- H is the aliased PHP, built per-site `P''·H'·P` via
  `contract_aliased_itensor`, same as case 3. **`SB_FUSE_LINKS=1` is REQUIRED**:
  the sandwich leaves multi-strand `Link,l=i` axes at several prime levels; fusing
  collapses each bond to one `FusedSparse` channel so H sites are the clean
  **4 sparse (2 sites + 2 fused links) + 2 dense** shape.
- The new interaction (absent in both sisters): the ProjMPO **environment** and
  every matvec intermediate are now `aliased × aliased`. The env-build `*` would
  densify; the matvec's static `dense_inds(v)` hint is stale. Both needed
  case-4-gated fixes (see §3).

### The doubled-link convention (the crux of every classification bug here)
Each bond carries TWO indices with the **same** `Link,l=i` tag:
- a **sparse CHANNEL** (small dim; the projector channel) → lives in the **prefix**
- a **dense MULTIPLICITY** (grows with maxdim) → lives in the **dense tail**

Convention: **sparse precedes dense.** After `SB_FUSE_LINKS`, H's channel is
renamed `FusedSparse,Link,bond=k` (so for H the split is by tag); ψ/φ keep two
identical `Link,l=i` (so the split is by order/position). The sparse and dense
partners are **distinct Index ids**.

---

## 2. Status (2026-06)

- ✅ **Crashes fixed; runs end-to-end** at small maxdim. The three case-4 fixes
  in §3 take the matvec from "immediate prefix/dense crossover crash" to a
  completed sweep with the alias invariant held, honest BD ≤ maxdim, dedup > 1×.
- ✅ **Operator + matvec proven exact** (`diag_heff_php.jl`): `⟨φ|H_eff|φ⟩` and
  `‖H_eff·φ‖` match the dense-PHP reference to ~1e-16 at bonds 1–3, and the full
  per-bond Path-B (gram M, M⁻¹ᐟ², B_op, recast, eigsolve) reproduces the dense
  generalized eigenvalue to 10 digits. **The Path-B M⁻¹ is correct per-bond — it
  is NOT missing anything.**
- ✅ **==validated reference at md=8**: at N=2 md=8 the energy trajectory is
  **bit-identical** to the validated bare-H aliased run (case 2). So the PHP
  implementation faithfully reproduces the constrained problem (ψ∈image(P) ⇒
  `PHP·φ = H·φ`). The md=8 "wrong energy" is the **known low-maxdim aliased
  truncation regime** (huge `maxtruncerr`), shared with case 2 — not a PHP bug.
- ✅ **RESOLVED — md=16 prefix/dense crossover** (2026-06-09). Root cause (§4):
  the deferred-fission pre-H apply (`fission=false` ⇒ `template=nothing` ⇒ no
  hint) let `output_inds` fall back to "dense iff in denseA∪denseB", and the
  M^{−1/2} factor `Linv_R` was built **fully dense**, so its channel id sat in
  denseA ⇒ the output channel was parked in the dense tail (P=2) ⇒ crossover vs
  the aliased env. **Fix (case-4 gated): relayout the M^{±1/2} factors as
  aliased** carrying φ's {channel→prefix, mult→dense} split
  (`wrap_dense_as_aliased_via_template`, `SB_ALIASED_MINV_WRAP`), so the apply is
  canonical natively — no hint, no forced fission (deferred-fission speedup kept).
  Verified end-to-end at N=2 md=16: all four VERDICT gates PASS, |ΔE| vs dense
  PHP = 5.35e-2 (1.43%), 5.58× vs-dense compression, dedup mean 2.67×.
  Command (from `edited_packages/`):
  `julia --project=. test_sparse_ham_psi/test_sparse_ham_psi_kl.jl --N-plaq 2 --maxdim 16 --n-sweeps 4 --dense-ref true`
  (runner case-4 gates default, incl `SB_ALIASED_MINV_WRAP=1`).

Confidence discipline (inherited from the sisters): tag every hypothesis/fix
Confirmed / High / Medium / Low, state what you did NOT verify, and annotate
every recorded number with the exact command + env flags that produced it.

---

## 3. The case-4 changes that landed (each with its gate)

All in `../ITensorMPS.jl/src/abstractprojmpo/abstractprojmpo.jl`. Each is inert
for cases 1–3.

1. **`SB_FUSE_LINKS`** (set in the runner) — fuse the multi-strand PHP bonds at
   build time. Already shared with case 3; the runner just turns it on.
2. **Aliased env** — `_makeL!/_makeR!` use `_env_mul(A,B,_keep_env)` which routes
   through `_mul_preserve_aliased` (keeps the env aliased, with the union
   `dense_inds` hint) **only when `_keep_env`** (both H and ψ aliased). Otherwise
   plain `*` (cases 1–3 unchanged).
3. **Matvec union hint** — both matvec branches pass
   `output_inds_hint = _use_aa_hint ? _aa_step_hint(Hv,it) : _step_hint`. The
   union hint (current operands' `dense_inds`) replaces the stale `dense_inds(v)`
   **only when `_use_aa_hint`** (case 4). Cases 1–3 keep `_step_hint`.
4. **`_reorder_env_for_aliased(T, both_aliased)`** — skips the dense-kernel
   reorder for the aliased env **only when `both_aliased`** (threaded from
   `_keep_env`). `ITensors.permute` has no aliased method, so the reorder (a perf
   canonicalization) is simply skipped for case 4; cases 1–3 reorder as before.
5. **Aliased M^{±1/2} factor relayout** (`SparseBackends/src/path_b_helpers.jl`)
   — `build_minv_half_pair_factored` now wraps the dense `Mhalf`/`Linv` factors
   as aliased via `wrap_dense_as_aliased_via_template` (φ's id-based
   channel→prefix / mult→dense split). Gated `SB_ALIASED_MINV_WRAP` (default on)
   **AND** a threaded `both_aliased::Bool` (computed in `dmrg.jl` from
   `_is_aliased_itensor(phi) && any aliased PH.H[·]`). Case 2 (dense H) ⇒
   `both_aliased=false` ⇒ factors stay dense ⇒ byte-identical. Pure relayout
   (one template per block, scalar 1), NOT a block-diagonality assumption:
   cross-channel coupling stays in each prefix block's dense tail; values
   bit-identical. This is the md=16 fix — see §2/§4.

Diagnostics (all default-off, no effect on any path unless enabled):
`SB_HAMPSI_DIAG` (per-matvec operand classification + channel→dense "flip"
detector), `SB_MINV_DIAG` (`apply_minv_preserve_bs` stage-by-stage split),
`SB_FACT_DIAG` (factorize/φ classification dump), `SB_BOND_EIG_DBG` (per-bond
Path-B eigenvalue). See `diag_heff_php.jl` for the layered correctness probe.

**Reverted (do NOT re-add without a case-4 gate):** a re-fission path inside
`recast_aliased_to_template` (shared by cases 2,3,4) and a try/catch crossover
dump inside `wrapped_contract_aliased` (shared hot path).

---

## 4. The open md=16 bug — full diagnosis chain

Symptom: `contract_shared! … shared label crosses prefix/dense boundary
(A pos=4 PA=2, B pos=1 PB=3)` at the FIRST matvec step.

Decoded by **index id** (not dim — dims are ambiguous, see below):
- φ (eigsolve operand template) is **canonical**: `P=3`, channel (`Link l=2`,
  dim 2) in prefix, multiplicity (dim 3) in dense.
- the operand `x` actually fed to the matvec is **non-canonical**: `P=2`, the
  **channel pushed into the dense tail**, ordered `[mult, channel]`.
- `x = M⁻¹ᐟ²·y` (Path-B `B_op` / `apply_half`). Stage trace (`SB_MINV_DIAG`):
  the **2nd chained `apply_minv_preserve_bs`** (`Linv_R` applied to
  `y0 = Mhalf_R·φ`) is where `contract_preserve_bs(Minv, y; template=φ)` emits
  `P=2`. The 1st apply (`Mhalf_R·φ`) stays canonical.
- `recast_aliased_to_template` cannot repair it: it only permutes **within** a
  region and bails on any prefix↔dense move (it bails at the cross-region guard,
  not the id guard ⇒ **`x` carries φ's exact bond Index ids**, only the
  prefix/dense split is wrong).
- Then `x` (channel in dense) meets the env (channel correctly in prefix) ⇒ same
  id classified dense on one side, prefix on the other ⇒ crossover crash.
- md dependence: the bond multiplicity must grow enough to produce this
  intermediate; md=8 doesn't, md=16 does.

**What is NOT the cause** (verified, don't re-investigate):
- not the factorize (φ is canonical out of it),
- not the operator/matvec values (`diag_heff_php.jl`: exact),
- not a missing M⁻¹ (the B_op is present and per-bond exact),
- not the env-caching (forced full rebuild → bit-identical energies).

**RESOLVED — the true root** (2026-06-09, found by a temporary probe — since
removed — that dumped the `output_inds` inputs for both calls): the two calls
have **byte-identical operands** (same A=Linv_R inds, same B=y inds). The ONLY
difference is the
**hint**: call 1 (`Mhalf_R·φ`, building y0) runs with `fission=true` ⇒ template
passed ⇒ `hint=dense_inds(φ)`; call 2 (`Linv_R·y0`, the pre-H apply) runs with
`fission=false` (deferred-fission) ⇒ `template=nothing` ⇒ **`hint=nothing`**.
With no hint, `output_inds` uses its fallback ("dense iff in denseA∪denseB").
Because the metric factor `Linv_R` was built **fully dense**, its channel id is
in `denseA` ⇒ the output channel is classified dense ⇒ P=2. It was never the
operands or the ids — it was the dropped hint meeting a dense factor.

**Fix:** make `Linv_R` aliased (channel in its own prefix) so `denseA` excludes
the channel ⇒ the no-hint fallback leaves the output channel in the prefix ⇒
P=3 canonical, deferred-fission preserved. See §3 item 5. The earlier
`recast`-refission attempt was the wrong layer (it fixed the *output* in a
shared function); the right fix is the *operator* relayout, case-4 gated.

### Pitfall: do NOT identify channel-vs-multiplicity by dimension
"smaller dim = channel" is **wrong** — at md where the multiplicity grows to the
channel's dim (e.g. a dim-4 channel with cap `fld(16/4)=4`), dims are equal and
the heuristic misclassifies. The reliable signal is **Index id** (the channel and
multiplicity are distinct ids; φ is the canonical authority and `x` preserves
φ's ids), or the **order convention** (sparse precedes dense) within a single
tensor. Prefer id-matching to φ.

---

## 5. Files

| file | role |
|---|---|
| `test_sparse_ham_psi_kl.jl` | the both-aliased KL runner (aliased ψ + aliased PHP, Path-B, dense-PHP `--dense-ref` ΔE check). Defaults the case-4 gates ON. |
| `diag_heff_php.jl` | layered correctness probe (mirror of `../test_aliased_psi/diag_gram_metric.jl` for PHP): φ sanity, `⟨φ|H_eff|φ⟩` aliased-vs-dense, dense generalized eig vs actual B_op, recast drift. |
| `README.md` | target, status, reproduce commands, diagnosis summary. |

Baselines: do **not** recreate them. Compare against `../test_sparse_psi/test_dense_kl.jl`
(dense PHP), `../test_sparse_psi/test_sparse_kl.jl` (BS), and
`../test_aliased_psi/test_aliased_kl.jl` (aliased ψ × bare H) at matching args.

---

## 6. Experiment hygiene

- **Contention:** before any timed/diagnostic run, `pgrep -af julia` — never run
  while another DMRG/`test_*` Julia process is active (corrupts timings and
  steals cores). Wait for it to finish.
- Run validation jobs as separate background jobs; kill by PID / exact script
  name, never a broad `pkill julia` pattern.
- JIT dominates sweep 1; quote post-JIT (excl-sweep-1) numbers.
- N=2 md=4 is the fast smoke test; md=16 is the current failing point; large N is
  hours.
