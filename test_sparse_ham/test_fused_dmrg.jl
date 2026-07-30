# test_fused_dmrg.jl — validate run_mode=:fused on the aliased-PHP + dense-ψ path.
# Builds the PXP aliased P†HP operator + dense ψ, runs ground-state DMRG in BOTH
# run_mode=:standard and run_mode=:fused, and checks the energies agree (the partial-
# fused matvec is bit-identical; :fused should match :standard to ~1e-12).
#   Usage: julia --project=. test_sparse_ham/test_fused_dmrg.jl <N>

using SparseBackends, ITensors, ITensorMPS, Random, Printf, LinearAlgebra
include("aliased_helpers.jl")

# PXP model definitions (copied from test_pxp_aliased.jl).
ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]
function itensor_from_nonzeros(dims::NTuple{N,Int}, coords::Vector; left::Bool=false) where {N}
    A = zeros(Float64, dims...); for c in coords; A[(c .+ 1)...] = 1.0; end; return A
end
bind_to_idx(A::AbstractArray, idxs::Index...) = ITensor(A, idxs...)
function NotEqlsLoop_R1(sites)
    N = length(sites)
    R1_first = itensor_from_nonzeros((3,3,2), [(0,0,0),(1,1,1),(2,2,0)])
    R1_bulk  = itensor_from_nonzeros((3,3,2,2), [(0,0,0,0),(0,0,1,0),(1,1,0,1),(1,1,1,0),(2,2,0,0)])
    R1_last  = itensor_from_nonzeros((3,3,2), [(0,0,0),(0,0,1),(1,1,0),(1,1,1),(2,2,0)]; left=true)
    bonds = [Index(2, "Link,l=$(i)") for i in 1:N-1]; Wvec = Vector{ITensor}(undef, N)
    Wvec[1] = bind_to_idx(R1_first, sites[1], sites[1]', bonds[1])
    for j in 2:N-1; Wvec[j] = bind_to_idx(R1_bulk, sites[j], sites[j]', bonds[j-1], bonds[j]); end
    Wvec[N] = bind_to_idx(R1_last, sites[N], sites[N]', bonds[N-1]); return MPO(Wvec)
end
function pxp_opsum(N::Int)
    os = OpSum(); os += 1, "Xp", 1
    for j in 0:N-2; os += 1, "Px", j+1, "LP", j+2; os += 1, "RP", j+1, "Xp", j+2; end
    os += 1, "Px", N; return os
end

function sandwich_mpo_aliased(P::MPO, H::MPO)
    new_H = MPO(length(H))
    for i in 1:length(H)
        H1      = SparseBackends.contract_aliased_itensor(P[i]'', H[i]', :coo, :dense)
        H_eff_i = SparseBackends.contract_aliased_itensor(P[i],   H1,    :coo, :aliased)
        new_H[i] = replaceprime(H_eff_i, 3 => 1)
    end
    return new_H
end

length(ARGS) < 1 && error("Usage: julia test_sparse_ham/test_fused_dmrg.jl <N>")

let
    Random.seed!(42)
    N     = parse(Int, ARGS[1])
    sites = siteinds("S=1", N)
    H_raw = MPO(pxp_opsum(N), sites)
    P     = NotEqlsLoop_R1(sites)

    H_aliased = sandwich_mpo_aliased(P, copy(H_raw))
    fuse_sparse_links!(H_aliased)
    prepermute_aliased_mpo!(H_aliased)

    psi_raw  = random_mps(sites)
    psi_init = replaceprime(contract(P, psi_raw; cutoff=1e-12), 1 => 0)
    normalize!(psi_init)

    md = parse(Int, get(ENV, "BENCH_MAXDIM", "40"))
    ns = parse(Int, get(ENV, "BENCH_NSWEEPS", "6"))
    maxdim = [md]; mindim = [md]; cutoff = 1e-10
    @printf("\n[test_fused_dmrg] PXP N=%d  maxdim=%d  nsweeps=%d\n", N, md, ns)

    println("\n────────── GROUND run_mode = :standard ──────────")
    (Es, psi0s), ts = run_dmrg_ground("STD", H_aliased, deepcopy(psi_init);
        nsweeps=ns, maxdim, mindim, cutoff, outputlevel=1, run_mode=:standard)

    println("\n────────── GROUND run_mode = :fused ──────────")
    (Ef, _), tf = run_dmrg_ground("FUSED", H_aliased, deepcopy(psi_init);
        nsweeps=ns, maxdim, mindim, cutoff, outputlevel=1, run_mode=:fused)

    # ── First excited (project out the :standard ground state) ──
    Random.seed!(43)
    psi_raw2  = random_mps(sites)
    psi_init2 = replaceprime(contract(P, psi_raw2; cutoff=1e-12), 1 => 0); normalize!(psi_init2)
    W = 20.0
    println("\n────────── EXCITED run_mode = :standard ──────────")
    (E1s, _), t1s = run_dmrg_excited("STD_EXC", H_aliased, [psi0s], deepcopy(psi_init2);
        nsweeps=ns, maxdim, mindim, cutoff, weight=W, outputlevel=1, run_mode=:standard)
    println("\n────────── EXCITED run_mode = :fused ──────────")
    (E1f, _), t1f = run_dmrg_excited("FUSED_EXC", H_aliased, [psi0s], deepcopy(psi_init2);
        nsweeps=ns, maxdim, mindim, cutoff, weight=W, outputlevel=1, run_mode=:fused)

    println("\n========== RESULT ==========")
    @printf("  GROUND   :standard E = %.12f  (%.3f s)\n", real(Es), ts)
    @printf("  GROUND   :fused    E = %.12f  (%.3f s)\n", real(Ef), tf)
    @printf("    |ΔE0| = %.3e   %s   wall :fused/:standard = %.3f\n",
            abs(Es - Ef), abs(Es - Ef) < 1e-9 ? "PASS ✓" : "CHECK", tf/ts)
    @printf("  EXCITED  :standard E = %.12f  (%.3f s)\n", real(E1s), t1s)
    @printf("  EXCITED  :fused    E = %.12f  (%.3f s)\n", real(E1f), t1f)
    @printf("    |ΔE1| = %.3e   %s   wall :fused/:standard = %.3f\n",
            abs(E1s - E1f), abs(E1s - E1f) < 1e-9 ? "PASS ✓" : "CHECK", t1f/t1s)
end
nothing
