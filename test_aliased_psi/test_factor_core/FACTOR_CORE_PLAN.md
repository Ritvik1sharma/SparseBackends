# Factor-core DMRG — implementation plan (reconciled)

## 0. Goal
DMRG (later TDVP) where the **stored state is the aliased `ψ = P·core`**, the
**variable is `core`** (the dense template blocks), and the **local metric is I**
— no `M`, no `M`-inversion, anywhere. Distinct from Path-B (variable ψ, metric M,
plateaus −14.42) and PHP+core (variable bare core, state not aliased). Target
**−14.77** on PXP, ψ stays aliased + deduped. **Both PXP (diagonal P) AND KL
(off-diagonal / Z₂-flip P) must be supported** — read_core/write_core! key the core
on the pre-P `rv` slice (stored `slice_to_template`), which is correct for flip P.

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
5. **Krylov vectors = the 2-site CHANNEL-FREE core** (dense: `s_b, s_{b+1}` + the two
   core-links), NOT the aliased φ. There is **NO per-matvec lift and NO write** — the
   bare core is contracted straight through `Lenv·core·PH[b]·PH[b+1]·Renv`, and the P
   structure lives entirely in the envs (built from aliased ψ) and PH: PH's channel
   links merge Lenv/Renv's channel links, so a channel-free core maps to a channel-
   free core. Metric is plain Frobenius on that dense core = metric I (no M).
   Start vector each bond = `read_core(ψ[b])·read_core(ψ[b+1])` (per-site read_core,
   merged over the shared core-link — read is per site, KL-safe via the `rv` map).
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
Precompute ONCE, before any DMRG sweep: `H_bare` (MPO, for env growth) and the
**aliased `PH`** (`PH[j] = P[j]·H[j]`, P applied locally on H's output at sites j and
j+1; `build_ph_output`). `PH` is aliased and carries P's FSM **channel** links.

`Lenv`/`Renv` are grown by `position!` from **aliased ψ = P·core (bra + ket)** and
**bare H** — so the envs carry `P†…P` for the off-window sites and expose P-FSM
**channel** links at the window boundary.

**Eigensolve vector = the 2-site CHANNEL-FREE core** `c(s_b, s_{b+1}, cl_L, cl_R)`
(`cl` = core-link). Start each bond from `read_core(ψ[b])·read_core(ψ[b+1])`.

**Matvec `A(c)` — NO lift, NO write, NO aliased-φ:**
| # | op | note |
|---|---|---|
| 1 | `t1 = Lenv · c` | contracts `cl_L`; Lenv's boundary **channel** link stays open |
| 2 | `t2 = t1 · PH[b]` | PH[b]'s channel links merge Lenv's channel link |
| 3 | `t3 = t2 · PH[b+1]` | PH[b+1] channels chain b→b+1 |
| 4 | `v_o = t3 · Renv` | Renv's channel link merges PH[b+1]'s; **all channels closed ⇒ channel-free dense core** |

The P structure is supplied entirely by `Lenv/Renv` + `PH`; the bare core `c` only
carries site + core-link legs, so `c → v_o` are both channel-free. Contraction order
`Lenv·c` first (project memory).

**Update (writeback):** SVD `v_o` (channel-free 2-site core), truncate `maxdim` on the
core bond → `c[b], c[b+1]`, then `write_core!` each into ψ **per site** (single-site,
KL-safe via the stored `rv` map; keys/alias_ids/scalars fixed). No merged write.

**Env move (`position!`):** grow `Lenv/Renv` from aliased ψ + bare H (`bop_aliased`
paths). No `dense×dense` contraction in the hot path.

## 6. Implementation work items (ordered)
- **[DONE] Step 1** — `read_core`/`write_core!`/`slice_to_template`/
  `populate_slice_map!` + `slice_to_template` storage field. (`factor_core.jl`,
  `storage.jl`)
- **[DONE, likely UNUSED] Step 2** — `core_canonical` aliased factorize. Superseded:
  `v_o'` is dense, so we plain-SVD it (no aliased factorize). Keep for reference.
- **[DONE ✓] Step 3a — operator `PH`** — `build_ph_output(P,H)` in
  `ITensorMPS/src/abstractprojmpo/core_projmpo.jl`. Recipe: `prime(P;tags="Link")`
  (FSM→1), per site `PH[j] = contract(prime(Pl[j];tags="Site"), H[j], :coo, :dense,
  :aliased)` then `replaceprime(2=>1;Site)`. **Aliased** output (COO+complex-P
  capable ⇒ KL), built once. Uses the public 3-backend `contract` (NOT the internal
  `contract_aliased_itensor`).
- **[DONE ✓] Step 3c — one-shot matvec check** — `ph_matvec_oneshot_test.jl` (PXP).
- **[DONE ✓] Step 3b — dual-operator `CoreProjMPO`** (core_projmpo.jl). `{Hbare, PH}`;
  `position!`/env growth on bare H; `product(cpm, c) = Lenv·c·PH[b]·PH[b+1]·Renv`
  applied to the **channel-free core** `c` (channels close via env + PH).
- **[DONE ✓] Step 4 — writeback** — `_core_writeback!` (factor_core_dmrg.jl): plain
  SVD of the channel-free `v_o`, `maxdim`/`mindim`/`cutoff` on the core bond, then
  `_core_rebuild` (resize-capable per-site write_core!) into ψ[b], ψ[b+1]. Keys/
  alias_ids/scalars + `slice_to_template` fixed.
- **[DONE ✓] Step 5 — `dmrg_core_php`** (factor_core_dmrg.jl): self-contained driver
  (separate from dmrg.jl per user); eigsolve on channel-free cores + writeback +
  per-sweep system energy. `dmrg(...; run_mode=:core_php, P=P)` gate: PENDING.
- **[VALIDATE — GATED, not run]** PXP N=12 bd40 → target −14.77 (regression gate),
  then **KL** N=12 bd40; confirm ψ stays aliased + deduped, energy stable.

## 7. Files
- `SparseBackends/src/aliased/factor_core.jl` — core I/O (read_core/write_core!/
  slice_to_template/populate_slice_map!); write_core! is **single-site only**.
- `SparseBackends/src/aliased/storage.jl` — `slice_to_template` field (done)
- `SparseBackends/src/tensoralgebra/contract_aliased_coo_dense.jl` — persists the
  pre-P `rv → tid` map into `slice_to_template` (KL routing).
- `SparseBackends/src/tensor_wrappers.jl` — public 3-backend `contract(A,B,Ab,Bb,Cb)`.
- `ITensorMPS.jl/src/abstractprojmpo/core_projmpo.jl` — `build_ph_output` + `CoreProjMPO`.
- `ITensorMPS.jl/src/factor_core_dmrg.jl` — **self-contained** `dmrg_core_php` driver
  (separate from dmrg.jl); `_core_read_window`, `_core_writeback!`, `_core_rebuild`,
  `_core_canonicalize!`, `core_php_energy`.
- `ITensorMPS.jl/src/dmrg.jl` — `run_mode=:core_php` gate delegating to `dmrg_core_php` (PENDING).
- `test_aliased_psi/test_factor_core/` — env_channel_audit.jl, ph_matvec_oneshot_test.jl,
  core_php_driver_test.jl (PXP), + a KL round-trip + KL driver test (TODO).

## 8. Open items / status
- **Krylov representation — RESOLVED:** the 2-site **channel-free core** (decision 5),
  contracted straight through `Lenv·c·PH·Renv` (no lift, no write). Rejected dead-ends
  logged so they are not re-tried: per-matvec P-contraction lift, aliased-φ vector,
  merged 2-site write, `_core_place_window`/`_core_window_map` helpers.
- **Exact plev** on `PH`'s output leg — resolved empirically (step 3c).
- **KL correctness — the open risk:** read_core/write_core! are single-site and key on
  the stored pre-P `rv` map (KL-intended) but **never run on KL**. Settle with a KL
  round-trip (write_core→read_core identity) before trusting the full KL DMRG.
- **Gate:** `dmrg(...; run_mode=:core_php, P=P)` delegation into `dmrg_core_php`.
