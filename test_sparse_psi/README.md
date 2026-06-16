# `test_sparse_psi/` — Block-sparse ψ DMRG experiments

This directory holds the experiments that drive and benchmark the
**block-sparse MPS ψ** path through DMRG, as implemented in
`SparseBackends/` and called from the patched `ITensorMPS.jl/`.

It covers **two benchmark models** that share the exact same
sparse-ψ machinery (factorisation, eigsolve paths, gram cache). The
only thing that changes between them is the projector `P` and the
Hamiltonian `H`:

| benchmark | `P` (projector) | `H` |
|-----------|-----------------|-----|
| **KL** (Kitaev-ladder) | `P = ∏_j (I + C_j)/2` — each `C_j` a unitary on a 4-site plaquette; bond dim 2/4 channel pattern | S=1 spin-chain Sx/Sy/Sz terms |
| **PXP** (Rydberg blockade) | `P = R1 = NotEqlsLoop_R1` — "no two adjacent state-1 sites" (bond-dim-2 MPO) | PXP Hamiltonian (Xp / Px / LP / RP on a spin-1 chain) |

The basic premise (both models): for a Hamiltonian that commutes with
a projector `P`, we carry ψ as a block-sparse MPS whose channel
structure is inherited from `P`'s MPO decomposition. DMRG then
optimises ψ inside `image(P)` without ever materialising the full
Hilbert space.

**This is the "sparse ψ, dense H" configuration.** ψ is block-sparse;
`H` is a *bare, dense MPO*. We run DMRG on the bare `H` and rely on
`[H, P] = 0` to keep ψ inside `image(P)` automatically. Compared with
the dense `PHP` baseline at the same `maxdim`:

- ψ_sparse has fewer stored non-zero blocks (channels labelled by which
  MPO-factor each crossing projector contributed).
- The H·ψ matvec uses block-sparse routines that skip zero channels.
- Memory and per-sweep wall-time are lower at moderate-to-large `maxdim`.

> Each benchmark keeps its **own runner files** — they are *not* merged
> into one parametric file. The shared scaffolding (`utils.jl`, the
> sweep driver, the docs) is the only thing held in common.

## Layout

### Core runners (main directory)

| file | benchmark | role |
|------|-----------|------|
| `test_sparse_kl.jl`  | KL  | **sparse-ψ runner.** Builds `H` (bare, dense MPO) and `ψ_sp = P · ψ₀` (block-sparse), runs DMRG on bare `H`. CLI: `--N-plaq`, `--eignv`, `--maxdim`, `--n-sweeps`, `--target-energy`. |
| `test_dense_kl.jl`   | KL  | **dense baseline.** Builds `H_dense = densify(P·H·P)` and `ψ_dense = densify(P·ψ₀)`, runs DMRG on the projected Hamiltonian. Same CLI plus `--spin`. |
| `test_sparse_pxp.jl` | PXP | **sparse-ψ runner (Path A).** Builds PXP `H`, `R1` projector, `ψ_sp = R1 · ψ_random`. Ground-state DMRG, then excited-state DMRG with `weight = 20` and the ground state as the orthogonal-state penalty. CLI: `--N`, `--maxdim`, `--mindim`, `--n-sweeps`, `--target-energy`. |
| `test_dense_pxp.jl`  | PXP | **dense baseline.** Builds PHP via dense `contract(P'', H', :coo, :dense)`. Same CLI. |
| `run_sweep.sh`       | both | consolidated sweep driver — dense + sparse, ground (+ excited for PXP), across a `maxdim` sweep, for either or both benchmarks. `BENCH=kl\|pxp\|both`, `MDS=...`, `OUT=...`. |
| `make_plots.py`      | both | reads the sweep summary numbers and produces the per-model comparison plots (footprint / runtime / energy error). |
| `utils.jl`           | both | shared helpers: `clean!(MPO)` (round near-zero / round to {±1, ±0.5, ±i}), `mps_memory_bytes`, `mpo_memory_bytes`. |
| `Manifest.toml`      | — | pins the Julia environment (resolved against the parent `--project=..`). |

### `specific_tests/` — preserved diagnostics & regression tests

One-off diagnostics, correctness checks, and experimental sweep
drivers from the development history. **Not part of the main runner
set**, but kept so future sessions can re-run them if a regression is
suspected. They `include("../utils.jl")` (one level up).

| file | benchmark | purpose |
|------|-----------|---------|
| `verify_iso.jl` | KL | factorisation correctness: per bond, runs `itensor_blocksparse_*_channel_aware`, checks `‖L·R − ψ‖/‖ψ‖` (recon) and `‖L'L − I‖` (cross-channel iso). **First thing to run if an energy looks wrong.** |
| `test_qr_channel.jl` | KL | QR-channel factorisation deep-dive harness. |
| `test_sparse_php.jl` | KL | sparse `PHP` construction probe. |
| `probe_svd_channels.jl` | KL | per-channel SV inspection. |
| `diag_P_structure.jl`, `diag_P_tensor_values.jl`, `diag_channel_balance.jl`, `diag_psi_sp_keys.jl` | KL | channel-imbalance investigation (block norms, P-MPO tensor dumps, channel balance, ψ_sp keys). |
| `test_pxp.jl` | PXP | diagnostic timing harness (`DMRG_DIAG=1` for fast iteration). |
| `test_pxp_check.jl` | PXP | the original reference test (provided by the user) — cross-check energies. |
| `test_pxp_check_runnable.jl` | PXP | standalone wrapper supplying the helpers (`itensor_from_nonzeros`, `bind_to_idx`) the reference test needs; `include`s `test_pxp_check.jl`. |
| `run_md40.sh`, `run_n32_md80_fixup.sh`, `run_pathA_nocap.sh`, `run_pathB.sh`, `run_pathB_grouped.sh` | KL | experimental / Path-B sweep drivers from the factorisation work. Superseded by `../run_sweep.sh` for routine use. |

## Recommended environment flags

For the **production sparse path** (what all the headline results use):

```bash
BMF_ISO_PATH=1               # use standard eigsolve (psi kept strictly iso)
SB_USE_QR=1                  # channel-aware QR + sequential Gram–Schmidt
SB_BALANCED_OWNERSHIP=1      # least-loaded cM owns each shared (lk,rk) cell
SB_ADAPTIVE_RANK=1           # drop the chi_pre_max floor on n_new_d
SB_ADAPTIVE_REL=1e-8         # SV cutoff (optional; this is the default)
```

For the **dense baseline**, no flags are needed beyond the script's
defaults. `run_sweep.sh` sets the sparse flags for you.

## Headline benchmarks

### KL — clean, single-process runs

Post-JIT (excluding sweep 1), N=32 chain (2N+2 = 66 sites), +1
projector sector, 5 production sweeps, production flags above. Data in
`/home/ritvik/temp/results/md_sweep_clean/`.

| `maxdim` | sparse avg/sweep | dense avg/sweep | speedup |
|----------|------------------|------------------|---------|
| 20       | 1.49 s           | 4.89 s           | **3.3×** |
| 40       | 3.39 s           | 13.96 s          | **4.1×** |
| 80       | 23.24 s          | 70.27 s          | **3.0×** |

| `maxdim` | sparse footprint | dense footprint |
|----------|------------------|------------------|
| 40       | 3.06 MiB         | 4.27 MiB (28 % bigger) |
| 80       | 12.93 MiB        | 16.42 MiB (27 % bigger) |

Ground-state energies match dense to ≤ 1 × 10⁻⁴ at md = 40 in the +1
sector, and to ≤ 5 × 10⁻⁵ at md = 80.

### PXP — clean, N=100

Path A sparse vs dense PHP. Numbers from
`/home/ritvik/temp/results/md_sweep_clean/`.

**Ground state**

| `maxdim` | sparse excl-JIT (s) | dense excl-JIT (s) | speedup | sparse footprint | dense footprint |
|----------|---------------------|---------------------|---------|------------------|------------------|
| 20       | 14.20               | 21.27               | 1.50×   | 1.36 MiB         | 0.93 MiB         |
| 40       | 24.90               | 54.20               | 2.18×   | 2.70 MiB         | 3.49 MiB         |
| 80       | 124.41              | 191.09              | 1.54×   | 9.85 MiB         | 13.65 MiB        |

Sparse ground-state energies match dense to ≤ 3 × 10⁻⁷ at all md.

**Excited state** (orthogonal to ground, weight = 20)

| `maxdim` | sparse excited (s) | dense excited (s) | sparse E_ex | dense E_ex |
|----------|--------------------|--------------------|--------------|--------------|
| 20       | 24.20              | 30.34              | −120.020389 | −120.000741 |
| 40       | 42.02              | 70.62              | −120.014331 | −120.018582 |
| 80       | 178.60             | 205.84             | −120.011148 | −120.018763 |

**Sparse excited-state energy is non-monotonic in `maxdim`** (best at
md = 20, worse at md = 40/80), while dense is monotonic. Suspected
cause: the per-cM cap forces a basis split mis-aligned with the
excited state's structure when the cap doesn't bind. See TODO 3.

## Reproducing

```bash
cd test_sparse_psi

# --- single configuration ---
# KL sparse:
BMF_ISO_PATH=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
  julia --project=.. test_sparse_kl.jl --N-plaq 32 --eignv true --maxdim 40 --n-sweeps 6
# KL dense:
julia --project=.. test_dense_kl.jl --N-plaq 32 --eignv true --maxdim 40 --n-sweeps 6

# PXP sparse:
BMF_ISO_PATH=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
  julia --project=.. test_sparse_pxp.jl --N 100 --maxdim 40 --mindim 40 --n-sweeps 10
# PXP dense:
julia --project=.. test_dense_pxp.jl --N 100 --maxdim 40 --mindim 40 --n-sweeps 10

# --- full sweep (both benchmarks, dense + sparse, all md) ---
OUT=/home/ritvik/temp/results/md_sweep_clean ./run_sweep.sh        # both
BENCH=kl  ./run_sweep.sh                                           # KL only
BENCH=pxp MDS="40" ./run_sweep.sh                                  # PXP, single md

# --- plots ---
python3 make_plots.py        # outputs in the results dir
```

## Status — what works today

- **Path A (`BMF_ISO_PATH=1`, `SB_USE_QR=1`)**: complete and validated
  for both KL and PXP. Strict iso ψ + standard `eigsolve`. The headline
  numbers come from this path. `specific_tests/verify_iso.jl` shows
  recon ≤ 1 × 10⁻¹⁵ and cross-channel iso ≤ 1 × 10⁻¹⁵ at every bond.
- **Channel-aware QR + Gram–Schmidt**
  (`SparseBackends/src/ops_factorize_qr.jl`): union–find row grouping
  across cMs with overlapping `lk` support; primary ownership of shared
  `(lk, rk)` cells with optional least-loaded balancing; cross-term R
  blocks added (not assigned) — the fix that made multi-factor bonds
  reconstruct losslessly.
- **Adaptive per-cM rank** (`SB_ADAPTIVE_RANK=1`): drops the
  `chi_pre_max` floor at low-rank bonds so bond dims shrink when global
  SV truncation says they should.
- **Excited-state DMRG (PXP)** required two ITensorMPS patches, both
  now in place: `position!(::ProjMPO_MPS, …; debug)` kwarg, and
  `Base.copy(::ITensors.ExternalStorage)` (so `deepcopy(::MPS)` works).
- **Path B (`BMF_ISO_PATH=0`)** refactored to
  `geneigsolve((H_op, M_op), φ; …)`. Correct when paired with the QR
  factorisation (M ≈ I). The old `InnerProductVec`-with-`krylovdim≥30`
  path is removed.

## Status — what does *not* work yet

- **Path B + truly non-iso ψ.** With QR-with-GS, ψ stays iso in the
  regime we run (the per-cM cap never binds at md ≤ 80), so the
  M-correction does no work. Two attempts to feed it a genuinely
  non-iso ψ failed:
  - `SB_USE_OWNED_SVD=1` (per-cM SVD, primary-ownership, no GS):
    `geneigsolve` returns spurious ghost eigenvalues because the gram
    metric `M` is rank-deficient. The B_op-with-Linv form
    (`M^{-1/2} H M^{-1/2}` with explicit null-space projection) is the
    textbook fix but is not re-wired in the post-refactor code.
  - `SB_USE_GROUPED_SVD=1` (joint SVD across all cMs in a row group)
    works numerically but **collapses the channel structure** — storage
    blows up (`honest_maxlinkdim = bond_sparse_dim × n_new_d`).
- **Per-cM cap is too rigid.** Even when one channel needs rank 20 and
  another rank 3, the cap is `floor(maxdim / bond_sparse_dim)`, forcing
  both to the same per-channel slice. PXP surfaces this most sharply
  (Fibonacci-constrained subspace → natural rank often exceeds the cap).

## TODO

1. **Restore B_op + Linv for Path B.** The Linv pseudo-inverse from
   `build_minv_half_pair_factored` already filters `null(M)`; re-wire the
   eigsolve to use it instead of `geneigsolve`, to actually exercise a
   non-iso ψ from `SB_USE_OWNED_SVD=1`.
2. **Sectored storage for `NewBlockSparseSorted`.** Replace the scalar
   `dims::NTuple{N,Int}` / `blksize::Int` with per-channel sectored
   versions so different channels carry different multiplicity dims
   without zero-padding the loser. Only path to closing the energy gap
   without inflating memory.
3. **PXP excited-state non-monotonicity** in sparse: at md = 80 the
   sparse excited energy is *worse* than at md = 20. Suspected cause:
   per-cM cap forces a basis split mis-aligned with the excited state's
   structure when the cap doesn't bind. Likely benefits from item (2).
4. **Sectored-storage Path B B_op + Linv combo** would unlock the
   "share-the-budget" cap variant: each channel keeps its natural rank,
   storage stays compact, M absorbs cross-channel non-iso during eigsolve.
5. **Optimise `kdb.dispatch.permute_and_validate!`.** At md = 80 it
   allocates 6.67 GiB per ~85 s of work (27 % of all kernel allocations).
6. **Optimise output buffer allocation (`alloc_C`).** 4.37 GiB per
   md = 80 run; a thread-local recycled buffer pool would remove most of it.

## Cross-references

- `SparseBackends/src/ops_factorize_qr.jl` — the factorisation kernel
  (lines ~263 per-cM cap, ~287–304 cross-term R scatter).
- `SparseBackends/src/ops_factorize_svd_owned.jl` — the experimental
  no-GS variant (broken without the Path B fix).
- `SparseBackends/src/path_b_helpers.jl` — M^{1/2}, Linv, gram cache.
- `ITensorMPS.jl/src/dmrg.jl` (~660–730) — the Path-A/Path-B branch.
- `ITensorMPS.jl/src/mps.jl` (~1545) — `replacebond!` dispatch to the
  QR / SVD / SVD-owned factorisation kernels.
- `SKILLS.md` (this directory) — conceptual primer; read before touching
  the factorisation, gram cache, or eigsolve wrappers.
- `/home/ritvik/temp/results/md_sweep_clean/` — canonical benchmark data.
