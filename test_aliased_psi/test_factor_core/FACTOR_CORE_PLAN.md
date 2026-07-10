# Factor-core DMRG — implementation plan (reconciled)

## 0. Goal
DMRG (later TDVP) where the **stored state is the aliased `ψ = P·core`**, the
**variable is `core`** (the dense template blocks), and the **local metric is I**
— no `M`, no `M`-inversion, anywhere. Distinct from Path-B (variable ψ, metric M,
plateaus −14.42) and PHP+core (variable bare core, state not aliased). Target
**−14.77** on PXP, ψ stays aliased + deduped. **Diagonal-P only** (PXP; P is
physically diagonal, off-diagonal KL-flip out of scope).

## 1. Mathematical foundation (why metric-I is exact and M never appears)
- `[H,P] = 0` (PXP). ⇒ `H·ψ ∈ image(P)`; the bra-P in the operator projects the
  matvec output onto P-allowed configs (nothing illegal appears).
- `M = P†P = c·Π`, a **scaled projector** (`c = 2^env`, the Z₂ count). On
  `image(P)`, `M` acts as the **scalar `c`**. ⇒ the metric-I eigenvector of the
  Hermitian `P†HP` **equals** the generalized (metric-M) eigenvector; only the
  eigenvalue is scaled by `c`. So optimizing `core` with metric I gives the
  **correct state**. `ker(P)` gives eigenvalue 0 and is never excited if we start
  in `image(P)`.
- P is **physically diagonal** (Rf/Rb/Rl nnz all have `s=s'`). ⇒ dedup is over the
  **FSM channel** axis (many `(physical,channel)` keys → one physical template),
  NOT over physical slices. This is the ~3.56× dedup.

## 2. Verified plumbing (env_channel_audit.jl, PXP N=12 b=6)
- **Separability (PASS):** aliased ψ bond = TWO separate `Index`es — P-FSM
  **channel** (dim 2, id `===` P's FSM link) and **core-link** (dim 1, grows with
  maxdim). Same "Link,l=k" tag, different id/dim, **not fused**. FusedSparse
  reorder is a **no-op** for dense-H + aliased-ψ (`abstractprojmpo.jl:992`).
- **Identity (PASS):** env channel `Index` **is** P's FSM bond (Lenv id 36432 ===
  P[6]; Renv id 29439 === P[7]) ⇒ `P·H` closes into the env **automatically on id**.
  plev 0 = ket (in φ), plev 1 = bra (output side the env carries).
- `orthogonalize!` regauges **only** the core-link (id changes); channel id is
  fixed = the exact structural/variational split.

## 3. Retracted concerns (do NOT re-litigate)
- **"Dedup survival" gate — FALSE.** Dedup is structural (P's fixed keys/alias_ids),
  over the channel, and is contracted away in the matvec then restored exactly by
  `ψ=P·core`. It cannot be lost. The old `output_P` leak was bare-H matvec with the
  channel *still present*; the correct `P·H` matvec contracts it.
- **"Template-set completeness" gate — FALSE.** `v_o'` is a plain dense core;
  forming `ψ=P·core'` is deterministic routing valid for *any* core'. Per-site
  physical dim is fixed (3), P's allowed configs fixed ⇒ `slice_to_template` is
  genuinely fixed sweep-to-sweep. Nothing to violate.

## 4. Design decisions (all confirmed)
1. **No new type.** Aliased ψ is a plain `ITensorMPS.MPS`.
2. **Single-P operator** `O[j] = P[j]·H[j]` — P on H's **output/bra** indices only
   (plev 1). NOT `P·H·P` (would double the ket P → the diag_heff_php wrong-energy +
   spurious `c`). ket-P comes from φ.
3. **Dual-operator ProjMPO** `(H_bare, PH)`: `position!`/env growth uses **bare H**
   (aliased ψ bra+ket supply both P's for off-window sites → env carries `P†…P`);
   the matvec uses **PH**.
4. **`O = P·H` built DENSE** (per-site, tiny) so the matvec stays in the optimized
   **aliased×dense** kernel and dedup is provably preserved. *Knob* to try aliased O
   later; decide by output template count, not wall-clock.
5. **Krylov vectors = dense cores; metric I via Frobenius.** Each matvec forms
   `φ = P·core` from the current Krylov core by **deterministic routing** (cheap;
   templates *are* the core + fixed P metadata built once), runs the aliased chain,
   returns dense `v_o'`. Lanczos add/scale/inner are plain dense ops (2-site core is
   small, ≈ 9·χ²). No custom inner, no M. *Alt:* aliased-φ Krylov + multiplicity-
   weighted inner — only if the per-matvec routing shows up hot.
6. **Update in `replacebond!` (`use_core=true`):** plain dense SVD of `v_o'`,
   truncate `maxdim` on the **core** bond, install `core[b],core[b+1]` into ψ by
   P-routing (reattach fixed `keys/alias_ids/scalars/slice_to_template` onto the new
   template buffer; at most a cheap block reorder — never remap the map).
7. **Energy = system energy** `⟨ψ|H|ψ⟩/⟨ψ|ψ⟩` on aliased ψ + bare H (aliased×dense
   → scalars). `c` cancels in the ratio. **M is never materialized.** The DMRG
   eigenvalue λ is internal (scaled by c); we don't convert it.
8. **Truncation:** `maxdim` controls the core bond only; ψ's aliased bond is
   meaningless (P-inflated).

## 5. Per-bond algorithm + kernel map
Precompute once: `H_bare` (MPO), `PH` (MPO of dense `O[j]=P[j]·H[j]`), and the
window `slice_to_template` (+ per-template multiplicity `n_t`).

**Eigensolve — matvec `A(core)`** (order `Lenv*φ` first, per project memory):
| # | op | kernel |
|---|---|---|
| 0 | route core → aliased φ (`P·core`) | not a contraction (template placement) |
| 1 | `t1 = Lenv(dense) * φ(aliased)` | aliased×dense → aliased |
| 2 | `t2 = t1 * O[b](dense)` | aliased×dense → aliased |
| 3 | `t3 = t2 * O[b+1](dense)` | aliased×dense → aliased |
| 4 | `v_o' = t3 * Renv(dense)` | aliased×dense → **dense** (4 FSM bonds closed ⇒ channel-free core) |

**Update (replacebond!, use_core):**
| # | op | kernel |
|---|---|---|
| 5 | plain SVD of `v_o'`, truncate maxdim | dense LA (only all-dense step) |
| 6 | install core[b],core[b+1] into ψ (P-routing) | metadata reattach (+ cheap reorder) |

**Env move (`position!`):** `Lenv·ψ[b]·H_bare[b]·ψ[b]†` → aliased×dense→aliased
twice, then bra-close aliased×aliased→dense. Reused verbatim from `bop_aliased`.

No `dense×dense` *contraction* in the hot path; only the small-core SVD + dense
Lanczos arithmetic are all-dense.

## 6. Implementation work items (ordered)
- **[DONE] Step 1** — `read_core`/`write_core!`/`slice_to_template`/
  `populate_slice_map!` + `slice_to_template` storage field. (`factor_core.jl`,
  `storage.jl`)
- **[DONE, likely UNUSED] Step 2** — `core_canonical` aliased factorize. Superseded:
  `v_o'` is dense, so we plain-SVD it (no aliased factorize). Keep for reference.
- **[BUILD] Step 3a — operator `PH`.** Build `O[j]=P[j]·H[j]` dense, P on H output.
  Prime bookkeeping: follow the diag_heff_php contract pattern but **single-sided**;
  the output physical must land at the plev matching the env **bra**-channel
  (plev 1) and unprime to core plev 0. *(Confidence: medium — exact plev to confirm
  in 3c.)*
- **[BUILD] Step 3b — dual-operator ProjMPO.** `CoreProjMPO{H_bare, PH, LR, …}`;
  `position!`/env growth on `H_bare`; `product` = `Lenv·PH[b]·PH[b+1]·Renv` with the
  core-routing wrapper (form φ in, dense `v_o'` out). Reuse env code paths.
- **[BUILD] Step 3c — one-shot matvec check** (before any run-mode wiring):
  at one bond assert (i) `v_o'` is channel-free (no P-FSM index survives),
  (ii) `A` Hermitian in the core metric, (iii) `⟨core|A|core⟩` == dense `P†HP`
  reference. This nails the plev detail and the channel-closing empirically.
- **[BUILD] Step 4 — `replacebond!` use_core path.** dense SVD + maxdim on core bond
  + P-routing install; keep `slice_to_template` fixed.
- **[BUILD] Step 5 — `run_mode=:core_php` in dmrg.jl.** Thread `use_core` through
  the eigensolve (KrylovKit on dense cores) + replacebond!; report system energy
  each sweep.
- **[VALIDATE] PXP N=12 run** → target −14.77 (beats Path-B/RR −14.42); confirm ψ
  stays aliased + deduped (dedup ratio > 1); energy stable both sectors.

## 7. Files
- `SparseBackends/src/aliased/factor_core.jl` — core I/O (+ operator build?)
- `SparseBackends/src/aliased/storage.jl` — slice_to_template field (done)
- `ITensorMPS.jl/src/…/dmrg.jl` — `:core_php` run mode, replacebond! use_core
- `ITensorMPS.jl/src/abstractprojmpo/…` — CoreProjMPO (dual operator)
- `test_aliased_psi/test_factor_core/` — env_channel_audit.jl (done), one-shot
  matvec check, PXP validation.

## 8. Open items to confirm before/while building
- **Krylov representation:** dense-core (decision 5) vs aliased-φ+multiplicity —
  confirm dense-core as the first cut.
- **Exact plev** on `PH`'s output leg — resolved empirically in step 3c.
- **Step-4 output form:** whether step-4 kernel emits dense vs aliased-dedup-1 —
  densify if needed (cheap, channel-free).
- **`O` backend knob** (dense now / aliased later) — parameterize from the start?
