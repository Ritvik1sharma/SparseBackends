# Microbench of the BS×dense matvec kernel that dominates DMRG, using the
# same H_new pipeline (sandwich_mpo via :coo,:dense and :coo,:blocksparse) as
# test_check_working.jl. We extract ONE tensor pair (H_new[j], Hv) and time
# repeated H_sp * Hv vs H_dn * Hv at various bond dims / projector sizes.
#
# Vary thread count via -t auto vs -t 1.
#
# Env:
#   KB_NS    = "6,10,14"     # plaquette counts (drives H_new bond dim)
#   KB_PHYS  = "2"           # 2 = S=1/2, 3 = S=1
#   KB_REPS  = "200"         # timed reps per cell (5 extra warmup)
#   KB_J     = "mid"         # which site of the MPO to use ("mid" or an int)

using SparseBackends, Random
using ITensors, ITensorMPS
using Printf
using TimerOutputs: reset_timer!, print_timer

const _Ns    = parse.(Int, split(get(ENV, "KB_NS",    "6,10,14"), ','))
const _PHYS  = parse(Int, get(ENV, "KB_PHYS", "2"))
const _REPS  = parse(Int, get(ENV, "KB_REPS", "200"))
const _WARM  = 5
const _J_REQ = get(ENV, "KB_J", "mid")
# D(psi) — bond dim of the *wavefunction*; what `maxdim` controls in DMRG.
# Independent of D(H[j]), which is fixed by the projector.
const _D_PSIS = parse.(Int, split(get(ENV, "KB_DPSI", "8,20,40,80,160"), ','))

# Lifted from test_check_working.jl: build the sparse and dense projected MPOs.
function _build_projected_H(N::Int, spin::Int; lambda = 0.0, spin_sector = 1.0)
  states = 2*N + 2
  sites = if spin == 2
    siteinds("S=1/2", states)
  elseif spin == 3
    siteinds("S=1", states)
  else
    error("spin must be 2 or 3")
  end

  os = OpSum()
  for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
  for j in 1:N
    os += "Sx", 2*j-1, "Sx", 2*j+2
    os += "Sy", 2*j,   "Sy", 2*j+1
  end

  # Same construction test_check_working.jl uses: 4-site localized plaquette
  # projectors as the SAME `ConsOps` vector for both chain methods.
  ConsOps = MPO[]
  for j in 1:N
    coeff = 0.5
    temp = OpSum()
    temp += coeff,                "Id",            2*j-1, "Id",            2*j, "Id",            2*j+1, "Id",            2*j+2
    temp += spin_sector*coeff,    "exp(i*pi*Sy)",  2*j-1, "exp(i*pi*Sx)",  2*j, "exp(i*pi*Sx)",  2*j+1, "exp(i*pi*Sy)",  2*j+2
    push!(ConsOps, MPO(temp, sites, [2*j-1, 2*j, 2*j+1, 2*j+2]))
  end

  # ---- Sparse projector chain via :coo, :coo (what `multiplyVecMPOtoMPO` does)
  P_sparse = ConsOps[1]
  for j in 2:N
    Bp = prime(ConsOps[j], "Site")
    P_sparse = contract(P_sparse, Bp, :coo, :coo)
    P_sparse = replaceprime(P_sparse, 2 => 1)
  end

  # ---- Dense projector chain via `is_ctn_compression=true` (what `multiplydense` does)
  P_dense = ConsOps[1]
  for j in 2:N
    Bp = prime(ConsOps[j], "Site")
    P_dense = contract(P_dense, Bp; is_ctn_compression=true)
    P_dense = replaceprime(P_dense, 2 => 1)
  end

  H = MPO(os, sites)

  # ---- Sparse sandwich → WrappedBlockSparse-storage MPO (`sandwich_mpo`)
  H1   = contract(P_sparse'', H', :coo, :dense)
  H_sp = contract(P_sparse, H1, :coo, :blocksparse)
  H_sp = replaceprime(H_sp, 3 => 1)

  # ---- Dense sandwich (`sandwich_mpo_dense`)
  H1d  = contract(P_dense'', H'; is_ctn_compression=true)
  H_dn = contract(P_dense, H1d; is_ctn_compression=true)
  H_dn = replaceprime(H_dn, 3 => 1)

  return sites, H_sp, H_dn
end

# Build a 2-site MPS-like wavefunction Hv for site b, with indices matching
# H_sp[b]'s site indices (so H_sp[b] * Hv contracts on one site label).
function _make_Hv(H::MPO, b::Int, D_psi::Int; rng = MersenneTwister(0))
  is = collect(inds(H[b]))
  site_un = nothing
  for I in is
    hastags(I, "Site") || continue
    plev(I) == 0 || continue
    site_un = I
    break
  end
  isnothing(site_un) && error("could not find unprimed site index in H[$b]")
  lleft  = Index(D_psi, "Link,kbHvL")
  lright = Index(D_psi, "Link,kbHvR")
  return randomITensor(rng, ComplexF64, lleft, site_un, lright)
end

function bench_one(N::Int, phys::Int, D_psi::Int; reps = _REPS)
  sites, H_sp, H_dn = _build_projected_H(N, phys)
  L = length(H_sp)
  jstr = _J_REQ
  j = jstr == "mid" ? max(2, L ÷ 2) : parse(Int, jstr)
  H_sp_j = H_sp[j]
  H_dn_j = H_dn[j]
  Hv = _make_Hv(H_sp, j, D_psi)

  # Validate that both H[j] tensors share an index with Hv
  c_sp = ITensors.commoninds(H_sp_j, Hv)
  c_dn = ITensors.commoninds(H_dn_j, Hv)
  @assert !isempty(c_sp) "H_sp[$j] shares no indices with synthetic Hv"
  @assert !isempty(c_dn) "H_dn[$j] shares no indices with synthetic Hv"

  # Sanity: storage types
  H_sp_storage = ITensors.has_external_storage(H_sp_j) ? "WrappedBlockSparse" : "dense"

  # Warmup
  for _ in 1:_WARM
    H_sp_j * Hv
    H_dn_j * Hv
  end

  GC.gc()
  reset_timer!(SparseBackends.TIMER)
  t_sp = @elapsed for _ in 1:reps; H_sp_j * Hv; end
  GC.gc()
  t_dn = @elapsed for _ in 1:reps; H_dn_j * Hv; end

  # Estimate bond dim of H[j] from its indices for reporting
  Dleft  = maximum([dim(I) for I in inds(H_sp_j) if hastags(I, "Link")]; init = 1)
  return (
    N = N, phys = phys, j = j, D_H = Dleft, D_psi = D_psi,
    storage = H_sp_storage,
    sparse_us = 1e6 * t_sp / reps,
    dense_us  = 1e6 * t_dn / reps,
    ratio     = t_sp / t_dn,
  )
end

function main()
  println("# kernel microbench (sandwich_mpo vs sandwich_mpo_dense)")
  println("# threads = $(Threads.nthreads()), phys = $_PHYS, reps = $_REPS")
  println("# N    D(H[j])  D(psi)    sparse_us       dense_us    sparse/dense")
  for N in _Ns, D_psi in _D_PSIS
    r = bench_one(N, _PHYS, D_psi)
    @printf "  %3d    %4d   %5d    %10.1f      %10.1f      %6.3f%s\n" r.N r.D_H r.D_psi r.sparse_us r.dense_us r.ratio (r.ratio < 1 ? "  ← SPARSE WINS" : "")
    flush(stdout)
    # Per-cell SparseBackends timer dump (where the sparse-side time went)
    println("--- SparseBackends.TIMER for N=$(r.N), D(psi)=$(r.D_psi) ---")
    print_timer(SparseBackends.TIMER; sortby=:firstexec)
    println()
  end
end

main()
