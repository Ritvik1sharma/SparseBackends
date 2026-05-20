using SparseBackends, Random
using ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer

# Diagnostic timing harness. Toggle with env vars:
#   DMRG_DIAG=1 → small run (nsweeps=3, maxdim=20) for fast iteration on timing data
#   DMRG_DIAG unset/0 → full run (nsweeps=25, maxdim=40)
const _DIAG = get(ENV, "DMRG_DIAG", "0") == "1"

function run_dmrg_with_timers(label, H, psi0; kwargs...)
  do_warmup = get(ENV, "DMRG_NO_WARMUP", "0") != "1"
  if do_warmup
    println("[warmup pass: 1 JIT sweep, results discarded]")
    kw = NamedTuple(kwargs)
    md = haskey(kw, :maxdim) ? kw.maxdim : [20]
    mn = haskey(kw, :mindim) ? kw.mindim : md
    co = haskey(kw, :cutoff) ? kw.cutoff : 1e-10
    try
      dmrg(H, deepcopy(psi0);
           nsweeps = 1, maxdim = md, mindim = mn, cutoff = co,
           outputlevel = 0, use_early_exit = false)
    catch e
      println("  (warmup failed: ", sprint(showerror, e), " — continuing without warmup)")
    end
  end
  reset_timer!(SparseBackends.TIMER)
  reset_timer!(ITensorMPS.PROJMPO_TIMER)
  GC.gc(); GC.gc()
  wall = @elapsed result = dmrg(H, psi0; kwargs...)
  global LAST_DMRG_WALL = wall
  println("\n========== TIMER REPORT: $label  (wall = $(round(wall; digits=3)) s, JIT excluded) ==========")
  println("\n--- ProjMPO matvec breakdown (sparse vs dense H[j]) ---")
  print_timer(ITensorMPS.PROJMPO_TIMER; sortby=:firstexec)
  println("\n--- SparseBackends contract dispatch breakdown ---")
  print_timer(SparseBackends.TIMER; sortby=:firstexec)
  println("==========\n")
  return result
end

LAST_DMRG_WALL = 0.0

# --- Pull in the same helpers the KL benchmark uses (sandwich_mpo etc.) ---
# include("temp/edited_packages/utils.jl")

# --- Replacement for utils/cotenn_utils.jl (which is missing in this tree) ---
# `itensor_from_nonzeros(dims, coords; left=false)` returns a plain Array{Float64}
# of size `dims` with 1.0 at every (0-indexed) coord. `bind_to_idx(A, inds...)`
# wraps it as an ITensor with the given Index objects (column-major, so dims
# must be passed in the same order as the indices).
function itensor_from_nonzeros(dims::NTuple{N,Int}, coords::Vector; left::Bool=false) where {N}
  A = zeros(Float64, dims...)
  for c in coords
    @assert length(c) == N
    A[(c .+ 1)...] = 1.0
  end
  return A
end
bind_to_idx(A::AbstractArray, idxs::Index...) = ITensor(A, idxs...)

function align_links_pxp(t1::ITensor, t2::ITensor, label::String)
  links1 = filter(i -> hastags(i, "Link"), inds(t1))
  links2 = filter(i -> hastags(i, "Link"), inds(t2))
  old_inds = Index{Int64}[]
  new_inds = Index{Int64}[]
  for l2 in links2
    base_tag = tags(l2)
    matches = filter(l1 -> tags(l1) == base_tag && dim(l1) == dim(l2) && l1 ∉ new_inds, links1)
    if isempty(matches)
      println("  [$label] no matching link for tag $base_tag (dim=$(dim(l2)))")
      return nothing
    end
    push!(old_inds, l2); push!(new_inds, first(matches))
  end
  return replaceinds(t2, old_inds, new_inds)
end

function sandwich_mpo(P::MPO, H::MPO)
  H1 = contract(P'', H', :coo, :dense)
  H_eff = contract(P, H1, :coo, :blocksparse)
  H_eff = replaceprime(H_eff, 3 => 1)
  return H_eff
end

function sandwich_mpo_dense(P::MPO, H::MPO)
  H1 = contract(P'', H'; is_ctn_compression=true)
  H_eff = contract(P, H1; is_ctn_compression=true)
  H_eff = replaceprime(H_eff, 3 => 1)
  return H_eff
end


length(ARGS) < 1 && error("Usage: julia test_pxp.jl <N> [nsweeps] [maxdim]")

let
  N = parse(Int, ARGS[1])

  ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
  ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
  ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
  ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]

  sites = siteinds("S=1", N)

  HamiltonianTerms = OpSum()
  HamiltonianTerms += 1, "Xp", 1
  for j in 0:N-2
    HamiltonianTerms += 1, "Px", j+1, "LP", j+2
  end
  for j in 0:N-2
    HamiltonianTerms += 1, "RP", j+1, "Xp", j+2
  end
  HamiltonianTerms += 1, "Px", N

  H = MPO(HamiltonianTerms, sites)

  # Hand-written R1 constraint MPO, ported from user's NotEqlsLoop_R1.
  function NotEqlsLoop_R1(sites)
    N = length(sites)
    R1_first = itensor_from_nonzeros((3, 3, 2), [(0,0,0), (1,1,1), (2,2,0)])
    R1_bulk  = itensor_from_nonzeros((3, 3, 2, 2),
        [(0,0,0,0), (0,0,1,0), (1,1,0,1), (1,1,1,0), (2,2,0,0)])
    R1_last  = itensor_from_nonzeros((3, 3, 2),
        [(0,0,0), (0,0,1), (1,1,0), (1,1,1), (2,2,0)]; left=true)
    bonds = [Index(2, "Link,l=$(i)") for i in 1:N-1]
    Wvec = Vector{ITensor}(undef, N)
    Wvec[1] = bind_to_idx(R1_first, sites[1], sites[1]', bonds[1])
    for j in 2:N-1
      Wvec[j] = bind_to_idx(R1_bulk, sites[j], sites[j]', bonds[j-1], bonds[j])
    end
    Wvec[N] = bind_to_idx(R1_last, sites[N], sites[N]', bonds[N-1])
    return MPO(Wvec)
  end

  P = NotEqlsLoop_R1(sites)

  # PHP via the two backends.
  H_new  = sandwich_mpo(P, copy(H))           # sparse path  (:coo / :blocksparse)
  H_new2 = sandwich_mpo_dense(P, copy(H))     # dense path   (is_ctn_compression=true)

  # Sanity check: the two PHPs should match value-wise after link relabeling.
  for i in 1:length(H_new)
    t1 = ITensors.has_external_storage(H_new[i])  ? SparseBackends.to_dense_itensors(H_new[i])  : H_new[i]
    t2 = ITensors.has_external_storage(H_new2[i]) ? SparseBackends.to_dense_itensors(H_new2[i]) : H_new2[i]
    t2_aligned = align_links_pxp(t1, t2, "H[$i]")
    if !isnothing(t2_aligned)
      isapprox(t1, t2_aligned) ? println("  H[$i] ✓ match") : println("  H[$i] ✗ values differ")
    end
  end

  println("\nMemory:")
  mpo_memory_bytes(H_new)
  mpo_memory_bytes(H_new2)
  mpo_memory_bytes(H)
  mpo_memory_bytes(P)

  # Project a random MPS into the constrained subspace for both runs.
  Random.seed!(42)
  psi_old = random_mps(sites)
  psi0 = replaceprime(contract(P, copy(psi_old), :coo, :dense), 1 => 0)
  normalize!(psi0)
  psi1, psi2 = copy(psi0), copy(psi0)

  _md_default = _DIAG ? 20 : 40
  _ns_default = _DIAG ? 3  : 25
  _md = parse(Int, get(ENV, "BENCH_MAXDIM",  string(_md_default)))
  _ns = parse(Int, get(ENV, "BENCH_NSWEEPS", string(_ns_default)))
  maxdim = [_md]; mindim = [_md]
  cutoff = 1e-12
  target_energy = nothing
  last_sweep_energy = nothing
  tensor_tracker = Any[]

  println("\n[diag=$_DIAG] PXP DENSE DMRG (H_new2) — N=$N, nsweeps=$_ns, maxdim=$_md ...")
  t = @elapsed begin
    energy_d, psi_d, sweeps_d, terr_d = run_dmrg_with_timers("PXP DENSE H_new2",
        H_new2, psi1; nsweeps=_ns, maxdim, mindim, cutoff, target_energy,
        use_early_exit=false, last_sweep_energy=last_sweep_energy,
        outputlevel=1, tensor_tracker=tensor_tracker, only_store=true)
  end
  E_d = inner(psi_d', H, psi_d)
  println("[DENSE]  Energy: $E_d  sweeps=$sweeps_d  terr=$terr_d  wall=$t s")
  dense_wall              = t
  dense_wall_jit_excluded = LAST_DMRG_WALL

  println("\n[diag=$_DIAG] PXP SPARSE DMRG (H_new) — N=$N, nsweeps=$_ns, maxdim=$_md ...")
  if get(ENV, "SB_CPFX_STATS", "0") == "1"
    SparseBackends.reset_cpfx_stats!()
  end
  t = @elapsed begin
    energy_s, psi_s, sweeps_s, terr_s = run_dmrg_with_timers("PXP SPARSE H_new",
        H_new, psi2; nsweeps=_ns, maxdim, mindim, cutoff, target_energy,
        use_early_exit=false, last_sweep_energy=last_sweep_energy,
        outputlevel=1, tensor_tracker=tensor_tracker)
  end
  if get(ENV, "SB_CPFX_STATS", "0") == "1"
    SparseBackends.print_cpfx_stats()
  end
  E_s = inner(psi_s', H, psi_s)
  println("[SPARSE] Energy: $E_s  sweeps=$sweeps_s  terr=$terr_s  wall=$t s")
  sparse_wall              = t
  sparse_wall_jit_excluded = LAST_DMRG_WALL

  println("\n========== PXP HEAD-TO-HEAD WALL TIME ==========")
  println("  Includes-warmup totals (for reference):")
  println("    DENSE  total: $(round(dense_wall;  digits=3)) s")
  println("    SPARSE total: $(round(sparse_wall; digits=3)) s")
  println("    ratio = $(round(sparse_wall/dense_wall; digits=3))")
  println("\n  JIT-EXCLUDED (post-warmup) totals — the fair comparison:")
  println("    DENSE  measured: $(round(dense_wall_jit_excluded;  digits=3)) s")
  println("    SPARSE measured: $(round(sparse_wall_jit_excluded; digits=3)) s")
  println("    ratio sparse/dense = $(round(sparse_wall_jit_excluded/dense_wall_jit_excluded; digits=3))  (want < 1)")
end
nothing
