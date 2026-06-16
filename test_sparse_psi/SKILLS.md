# SKILLS — what you need to know to work on block-sparse ψ DMRG

A primer on the concepts, abstractions, and pitfalls of the
block-sparse MPS ψ path. Read this before touching the factorisation,
gram cache, or eigsolve wrappers — most of the "why is this
complicated?" answers live here.

## 1. The physics setup

We solve `H |ψ⟩ = E |ψ⟩` for `ψ ∈ image(P)`, where `P` is a global
projector that commutes with `H`:

- `H` is a local lattice Hamiltonian (S=1 spin chain Sx/Sy/Sz terms;
  PXP-style Hamiltonian; or anything with `[H, P] = 0`).
- `P = ∏_j (I + C_j)/2` for the (I+C)/2 case. Each `C_j` is a unitary
  on a local 4-site plaquette (`exp(iπSy)·exp(iπSx)·exp(iπSx)·exp(iπSy)`).
  `(I + C_j)/2` projects onto `C_j = +1` eigenspace.
- Or `P = R1` for PXP — bond-dim-2 MPO enforcing "no two adjacent
  state-1 sites".

Initial state: `ψ₀ = P · ψ_random` (so `ψ₀ ∈ image(P)`). DMRG runs on
the **bare H** for sparse-ψ (since `[H, P] = 0` keeps ψ in image(P)
automatically), or on **PHP** for the dense baseline (since `ψ_dense`
isn't structurally constrained).

## 2. The block-sparse MPS

Each MPS site tensor is a `WrappedBlockSparse` wrapping a
`NewBlockSparseSorted{T, N, N2, P, K}`. The storage layout:

- **`dims::NTuple{N, Int}`** — one scalar dim per axis. All channels
  on a given axis share the same multiplicity dim.
- **`blksize::Int`** — one scalar block size = product of dense (mult +
  physical) dim axes. All stored blocks have the same shape.
- **`keys::Vector{NTuple{P, K}}`** — sorted, deduped channel keys
  (`P` = number of sparse-prefix axes).
- **`ids::Vector{Int}`** — index into `data` for each stored block.
- **`data::Vector{T}`** — concatenated flat block payload.

Each "channel" at a bond is a discrete label (small integer). For
the (I + C)/2 projector, channels track which MPO-factor (I-track or
C-track) was contributed by each crossing projector. Bond dim
patterns are 2-2-4-2-4-2-4-2-2 across the chain (2 inside single-
factor regions, 4 inside two-factor-overlap regions).

**Uniform-multiplicity caveat**: every channel on a given axis gets
the same `n_new_d` slots. A channel whose natural rank is 3 still
allocates 10 slots if `n_new_d = 10`, padded with zeros. This is the
root of the "share-the-budget" tension — see TODO (2) in `README.md`.

## 3. The DMRG inner loop in this codebase

The standard ITensorMPS DMRG sweep, with one branch for sparse ψ:

```
for each bond b:
    PH = position!(PH, ψ, b)           # build effective H_eff at bond b
    φ  = ψ[b] * ψ[b+1]                  # 2-site tensor
    if is_sparse_mps(ψ) and BMF_ISO_PATH != "1":
        # Path B: M-corrected eigsolve
        Lgram, Rgram  = gram_cache[b]
        Mhalf, Linv   = build_minv_half_pair_factored(Lgram, Rgram; phi)
        H_op = v -> PH · v                              (with BS recast)
        M_op = v -> apply(Mhalf, Mhalf, Mhalf, Mhalf, v)
        vals, vecs    = geneigsolve((H_op, M_op), φ, 1, :SR; …)
    else:
        # Path A: standard eigsolve, ψ assumed iso
        vals, vecs    = eigsolve(PH, φ, 1, :SR; …)
    ψ_new             = vecs[1]
    L, R, spec        = factorise(ψ_new, ψ[b], ψ[b+1];
                                  ortho, maxdim, mindim, cutoff)
    ψ[b], ψ[b+1]      = L, R
    update_gram_cache!(b)
```

The two big choice points are (a) the factorisation
(`itensor_blocksparse_*_channel_aware`) and (b) the eigsolve (standard
`eigsolve` vs `geneigsolve`).

## 4. The factorisation — three variants, all in `SparseBackends/`

### `itensor_blocksparse_svd_channel_aware` (default, no flag)

Per-channel SVD: each `cM` does an independent SVD on its slab of φ.
Maintains template-based block-key inheritance from `M_b`/`M_b1`.

**Bug for multi-factor projectors**: at bonds where two `cM`s share
`(lk, rk)` cells (the (2,4,2) / (4,2,4) patterns), each `cM`'s SVD
captures the *full* `φ[lk, rk]`. Reconstruction sums both → `2·φ` at
shared cells. Energies diverge wildly. **Do not use as-is for the
(I+C)/2 projector**.

### `itensor_blocksparse_qr_channel_aware` (`SB_USE_QR=1`)

QR + sequential Gram–Schmidt **within row groups**. Each shared
`(lk, rk)` cell is owned by exactly one `cM` (primary ownership). GS
across `cM`s in the same row group enforces cross-channel iso
(`L'L = I`).

Cross-term R blocks at `(prev_cM, rk_in_cur_cM)` capture the off-
diagonal coupling that GS extracts. **This is the production path.**

Key implementation files:

- `SparseBackends/src/ops_factorize_qr.jl` — the kernel
  (`blocksparse_qr_channel_aware`).
- `SparseBackends/src/tensor_wrappers.jl` —
  `itensor_blocksparse_qr_channel_aware` (the wrapper that builds
  templates from `M_b`/`M_b1` and dispatches).

### `itensor_blocksparse_svd_owned_channel_aware` (`SB_USE_OWNED_SVD=1`)

Per-channel SVD with primary-ownership but **no GS**. The owned-cells
trick fixes the double-count bug from the default SVD. Reconstruction
is lossless. But ψ ends up non-iso → requires Path B M-correction →
currently broken (see TODO 1).

## 5. The per-cM cap — what it does and doesn't do

At `ops_factorize_qr.jl:263`:

```julia
chi_cap = min(chi_rank, per_cM_cap)
```

where `per_cM_cap = floor(maxdim / bond_sparse_dim)`. **This caps each
channel's chi during the per-cM SVD** — not after.

- It is the *only* cap that actually fires. The downstream cap at
  line ~477 (`min(n_new_d_natural, mult_cap)`) is redundant: by the
  time we get there, every `chi_M ≤ per_cM_cap = mult_cap`, so the
  `min` is a no-op.
- It is **required** for uniform-multiplicity storage: total bond dim
  `= bond_sparse_dim × n_new_d ≤ maxdim` is only achievable if every
  channel takes the same share.
- It is **wasteful** when natural ranks differ across channels (the
  dominant channel has its useful SVs truncated; weak channels keep
  noise SVs).
- The historical `relax_iso_cap` flag was a no-op (only flipped the
  redundant downstream cap) and was removed during the cleanup pass.

## 6. The two eigsolve paths

### Path A — strict iso ψ, standard `eigsolve`

When ψ stays inside the iso constraint (`L'L = I` across channels),
the gram metric `M = ψ†ψ = I`. Standard `eigsolve(PH, φ)` solves the
right problem. No M-correction needed.

This is what `SB_USE_QR=1 BMF_ISO_PATH=1` gives. Krylov defaults
(`krylovdim = 3, maxiter = 1`) work fine.

### Path B — non-iso ψ, generalised eigsolve

When the per-cM cap doesn't bind (large `maxdim` relative to natural
rank), ψ may be non-iso. Then `M ≠ I` and `eigsolve(PH, φ)` solves
the wrong problem.

The fix: `geneigsolve((H_op, M_op), φ, 1, :SR; ishermitian=true,
isposdef=true)` — KrylovKit's generalised Krylov–Schur. M·v is built
via four `apply_minv_preserve_bs(Mhalf, …)` calls (since `M = Mhalf²
= (Mhalf_L · Mhalf_L) · (Mhalf_R · Mhalf_R)`).

**Caveat**: `geneigsolve` assumes `M` is positive *definite*. When `M`
is rank-deficient (which happens with `SB_USE_OWNED_SVD=1`),
`geneigsolve` returns ghost eigenvalues from `null(M)`. The
textbook fix is the symmetric-whitening B-op form
(`B = M^{-1/2} H M^{-1/2}` with `Linv` projecting out `null(M)`); it
is implemented in `path_b_helpers.jl` but not currently wired in
(TODO 1).

In the current setup `[H, P] = 0` holds to ~1e-15 and the per-cM cap
typically doesn't bind at md ≤ 80, so Path A is sufficient and Path B
is mostly a placeholder for the rank-relaxation regime.

## 7. The gram cache

`SparseBackends/src/path_b_helpers.jl`:

- `init_gram_cache(ψ)` — full left + right sweep, builds `L[b]` and
  `R[b]` for every bond at start of DMRG.
- `update_left!(cache, ψ, i)` / `update_right!(cache, ψ, i)` — called
  after each `replacebond!` to keep the cache current; `O(1)` work
  per call.
- `get_left_gram(cache, b)` / `get_right_gram(cache, b)` — O(1)
  lookups during the inner eigsolve.

The cache is what makes Path B viable — without it, computing `M`
fresh at every bond would cost `O(N)` per eigsolve, total `O(N²)` per
sweep.

## 8. Diagnostic env vars

| flag | effect |
|------|--------|
| `BMF_ISO_PATH=1` | Path A (strict iso + standard eigsolve). |
| `SB_USE_QR=1` | QR + GS factorisation (the production path). |
| `SB_USE_OWNED_SVD=1` | Per-cM SVD with primary ownership, no GS. |
| `SB_USE_GROUPED_SVD=1` | Joint SVD across all cMs in a row group. |
| `SB_BALANCED_OWNERSHIP=1` | Least-loaded `cM` owns each shared `(lk, rk)` cell (vs. first-cM-wins default). |
| `SB_ADAPTIVE_RANK=1` | Drop the `chi_pre_max` floor on `n_new_d` (let bonds shrink at low-rank). |
| `SB_ADAPTIVE_REL=1e-8` | SV cutoff threshold (`drop SV < adaptive_rel * sv_max`). |
| `SB_QR_DIAG=1` | Verbose per-bond QR factorisation diagnostics. |
| `SB_SV_REPORT=1` | Per-bond per-cM SV report (smallest kept, largest dropped). |
| `BMF_ARNOLDI=1` | Use Arnoldi instead of Lanczos in Path B (`ishermitian=false`). Diagnostic only. |

## 9. Common pitfalls

- **Don't use the default channel-aware SVD on multi-factor
  projectors.** Use `SB_USE_QR=1`. The cross-term R-block bug in the
  default path is real and gives wildly wrong energies (e.g., -102
  instead of -6.4 at N=4).
- **The `relax_iso_cap` flag is removed.** Don't write code that
  passes it. The per-cM cap at line 263 is the only knob; if you want
  to relax it, edit there directly (and accept the storage padding cost
  or refactor `NewBlockSparseSorted` to sectored).
- **Don't pass `mindim = 1` and expect the bond dim to grow.** DMRG
  will shrink to natural rank when allowed. Use `mindim = maxdim` if
  you want to pin the bond dim (the reference PXP test does this).
- **JIT dominates short runs.** The runners report `total (incl
  sweep 1)` and `total (excl sweep 1)` — quote the latter for any
  performance number. Sweep 1 at md = 80 can be 60+ seconds even when
  steady-state per-sweep is 1 s.
- **Don't compare timings across runs with other Julia processes
  active.** Initial PXP md = 80 numbers were 50 % off because another
  bench was running concurrently. Always check `pgrep -af julia`
  before kicking off benchmark runs.
- **`copy(::ExternalStorage)` had to be added** for excited-state
  DMRG. If you see `MethodError: no method matching copy(::ITensors.
  ExternalStorage{…})` it means `deepcopy(::MPS)` is being called and
  the patch in `ITensors.jl/src/external_storage.jl` was reverted.
- **`position!(::ProjMPO_MPS)` needed a `debug` kwarg** for the same
  reason; see `ITensorMPS.jl/src/abstractprojmpo/projmpo_mps.jl`.

## 10. Where to look first if a result looks wrong

1. **Energy way off** (e.g., -102 instead of -6.4): probably the
   factorisation has a recon bug. Run `specific_tests/verify_iso.jl` —
   it isolates the bond where `‖L·R − φ‖` blows up.
2. **Energy slightly off vs dense** (e.g., 0.5 % gap): per-cM cap is
   binding. Run with `SB_SV_REPORT=1` and look for `largest_dropped
   > smallest_kept` patterns.
3. **Speed regression vs old numbers**: check `pgrep -af julia` for
   contention; re-run.
4. **Memory blow-up**: check `honest_maxlinkdim` vs `maxdim`. If
   `honest = bond_sparse_dim × maxdim` (e.g., 160 at md = 40 with 4
   channels), some channel is being padded with zeros — usually
   means `SB_USE_GROUPED_SVD` is on or `n_new_d` is being floored.

## Cross-references

- `README.md` (this directory) — quick benchmark numbers + TODO, and
  the KL vs PXP benchmark breakdown. Both benchmarks now live here;
  the runners are `test_sparse_kl.jl` / `test_dense_kl.jl` (Kitaev
  ladder, (I+C)/2 projector) and `test_sparse_pxp.jl` /
  `test_dense_pxp.jl` (PXP / Rydberg blockade, R1 constraint).
- `specific_tests/` — preserved diagnostics & regression tests
  (`verify_iso.jl`, the `diag_*.jl` channel-balance probes, the PXP
  reference test `test_pxp_check*.jl`, and the experimental Path-B
  sweep drivers).
- `../SparseBackends/src/ops_factorize_qr.jl` — the kernel; lines 263
  (per-cM cap) and 287–304 (cross-term R scatter) are the hot spots.
- `../ITensorMPS.jl/src/dmrg.jl` lines 660–730 — the Path A vs Path B
  branch.
- `../ITensorMPS.jl/src/mps.jl` line ~1545 — dispatch among QR / SVD /
  SVD-owned at `replacebond!`.
