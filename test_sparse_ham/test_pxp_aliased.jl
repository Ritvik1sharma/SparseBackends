# test_pxp_aliased.jl
#
# Aliased counterpart to test_pxp_psi/test_dense_pxp.jl. Same PXP Hamiltonian
# (S=1 sites with custom Xp/Px/LP/RP operators), same constraint MPO P from
# `NotEqlsLoop_R1` (bond dim 2). The PHP sandwich is routed through
# SparseBackends.contract_aliased_itensor with the :coo backend for P so the
# effective Hamiltonian's intermediates and output are stored as
# AliasedBlockSparse. Mirrors the structure of test_check_working_aliased.jl.
#
# After sandwich we fuse multi-strand sparse links so each bond carries one
# fused-sparse link instead of multiple Link strands — same trick the spin
# test uses to give the kernel the right canonical layout.
#
# Then we run DMRG ground state + first-excited state on both
#   - the dense reference H_dense  (= test_dense_pxp.jl's H_eff)
#   - the aliased H_aliased
# and compare energies + per-site MPO footprint.
#
# Usage:
#   SB_FUSE_LINKS=1 SB_PLAN_B=1 \
#     BENCH_MAXDIM=40 BENCH_NSWEEPS=6 \
#     julia --project=. test_sparse_ham/test_pxp_aliased.jl <N>

using SparseBackends, Random
using ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using LinearAlgebra: BLAS
BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_BLAS_THREADS", "1")))
println("[BLAS threads pinned to ", BLAS.get_num_threads(), "]")

include("aliased_helpers.jl")

# ─────────────────────────────────────────────────────────────────────────────
# S=1 site operators used by the PXP construction.
# Identical to test_pxp_psi/test_dense_pxp.jl.
# ─────────────────────────────────────────────────────────────────────────────
ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]

# ─────────────────────────────────────────────────────────────────────────────
# Constraint MPO `P` for the PXP blockade (bond dim 2). Same as test_dense_pxp.
# ─────────────────────────────────────────────────────────────────────────────
function itensor_from_nonzeros(dims::NTuple{N,Int}, coords::Vector; left::Bool=false) where {N}
    A = zeros(Float64, dims...)
    for c in coords
        @assert length(c) == N
        A[(c .+ 1)...] = 1.0
    end
    return A
end

bind_to_idx(A::AbstractArray, idxs::Index...) = ITensor(A, idxs...)

function NotEqlsLoop_R1(sites)
    N        = length(sites)
    R1_first = itensor_from_nonzeros((3, 3, 2), [(0,0,0), (1,1,1), (2,2,0)])
    R1_bulk  = itensor_from_nonzeros((3, 3, 2, 2),
                   [(0,0,0,0), (0,0,1,0), (1,1,0,1), (1,1,1,0), (2,2,0,0)])
    R1_last  = itensor_from_nonzeros((3, 3, 2),
                   [(0,0,0), (0,0,1), (1,1,0), (1,1,1), (2,2,0)]; left=true)
    bonds    = [Index(2, "Link,l=$(i)") for i in 1:N-1]
    Wvec     = Vector{ITensor}(undef, N)
    Wvec[1]  = bind_to_idx(R1_first, sites[1], sites[1]', bonds[1])
    for j in 2:N-1
        Wvec[j] = bind_to_idx(R1_bulk, sites[j], sites[j]', bonds[j-1], bonds[j])
    end
    Wvec[N] = bind_to_idx(R1_last, sites[N], sites[N]', bonds[N-1])
    return MPO(Wvec)
end

# ─────────────────────────────────────────────────────────────────────────────
# PXP H via OpSum on S=1 sites (matches test_dense_pxp.jl exactly).
# ─────────────────────────────────────────────────────────────────────────────
function pxp_opsum(N::Int)
    os = OpSum()
    os += 1, "Xp", 1
    for j in 0:N-2
        os += 1, "Px", j+1, "LP", j+2
        os += 1, "RP", j+1, "Xp", j+2
    end
    os += 1, "Px", N
    return os
end

# ─────────────────────────────────────────────────────────────────────────────
# PHP sandwich through the aliased path.
# ─────────────────────────────────────────────────────────────────────────────
function sandwich_mpo_aliased(P::MPO, H::MPO)
    new_H = MPO(length(H))
    for i in 1:length(H)
        H1      = SparseBackends.contract_aliased_itensor(P[i]'', H[i]', :coo, :dense)
        H_eff_i = SparseBackends.contract_aliased_itensor(P[i],   H1,    :coo, :aliased)
        new_H[i] = replaceprime(H_eff_i, 3 => 1)
    end
    return new_H
end

function sandwich_mpo_dense(P::MPO, H::MPO)
    H_eff = contract(contract(P'', H'; cutoff=1e-12), P; cutoff=1e-32)
    return replaceprime(H_eff, 3 => 1)
end

# ─────────────────────────────────────────────────────────────────────────────
length(ARGS) < 1 && error("Usage: julia test_pxp_aliased.jl <N>")

let
    Random.seed!(42)
    N     = parse(Int, ARGS[1])
    sites = siteinds("S=1", N)

    println("\n[Building PXP H and constraint MPO P, N=$N]")
    H_raw = MPO(pxp_opsum(N), sites)
    P     = NotEqlsLoop_R1(sites)

    println("\n[Building H_aliased via sandwich_mpo_aliased]")
    H_aliased = sandwich_mpo_aliased(P, copy(H_raw))

    # Link fusion always applied for the aliased path (was gated by SB_FUSE_LINKS).
    println("\n[fusing multi-strand sparse links in H_aliased]")
    length(H_aliased) >= 3 && println("  pre-fuse  H[3] inds: ", inds(H_aliased[3]))
    fuse_sparse_links!(H_aliased)
    length(H_aliased) >= 3 && println("  post-fuse H[3] inds: ", inds(H_aliased[3]))

    if get(ENV, "SB_PREPERMUTE_H", "0") == "1"
        println("\n[prepermuting aliased H tails]")
        length(H_aliased) >= 3 && println("  pre-perm  H[3] inds: ", inds(H_aliased[3]))
        prepermute_aliased_mpo!(H_aliased)
        length(H_aliased) >= 3 && println("  post-perm H[3] inds: ", inds(H_aliased[3]))
    end

    println("\n[Building H_dense via sandwich_mpo_dense]")
    H_dense = sandwich_mpo_dense(P, copy(H_raw))

    # ── Memory footprint ──────────────────────────────────────────────────────
    println("\n── Memory footprint of PHP ──")
    dense_bytes   = sum(Base.summarysize(W) for W in H_dense)
    aliased_bytes = sum(Base.summarysize(W) for W in H_aliased)
    println("DENSE   PHP:  total MPO memory: $(round(dense_bytes/1e6;   digits=4)) MB   ($dense_bytes bytes)")
    println("ALIASED PHP:  total MPO memory: $(round(aliased_bytes/1e6; digits=4)) MB   ($aliased_bytes bytes)")
    println("  dense/aliased Base.summarysize ratio = $(round(dense_bytes/aliased_bytes; digits=3))")
    report_aliased_footprint(H_aliased, "ALIASED PHP:")

    # ── Pre-flight storage check ───────────────────────────────────────────────
    n_ali = 0; n_other = 0
    for i in 1:length(H_aliased)
        T = H_aliased[i]
        (ITensors.has_external_storage(T) &&
         T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse) ? (n_ali += 1) : (n_other += 1)
    end
    println("\n[H_aliased storage at DMRG entry]  sites=$(length(H_aliased))  aliased=$n_ali  other=$n_other")

    # ── Initial state: P-projected random MPS ────────────────────────────────
    psi_raw  = random_mps(sites)
    psi_init = replaceprime(contract(P, psi_raw; cutoff=1e-12), 1 => 0)
    normalize!(psi_init)

    _md = parse(Int, get(ENV, "BENCH_MAXDIM", "40"))
    _ns = parse(Int, get(ENV, "BENCH_NSWEEPS", "6"))
    maxdim = [_md]; mindim = [_md]; cutoff = 1e-10
    excited_weight = parse(Float64, get(ENV, "BENCH_WEIGHT", "20.0"))
    println("\n[Run params]  maxdim=$_md  nsweeps=$_ns  excited_weight=$excited_weight")

    # ── Ground state ──────────────────────────────────────────────────────────
    println("\n────────── Ground state DMRG ──────────")
    println("[DENSE  ground]")
    (E0_d, psi0_d), t_g_dense = run_dmrg_ground("DENSE_ground",
        H_dense, deepcopy(psi_init); nsweeps=_ns, maxdim, mindim, cutoff, outputlevel=1, run_label="DENSE_G")

    println("[ALIASED ground]")
    (E0_a, psi0_a), t_g_ali = run_dmrg_ground("ALIASED_ground",
        H_aliased, deepcopy(psi_init); nsweeps=_ns, maxdim, mindim, cutoff, outputlevel=1, run_label="ALIASED_G")

    # ── First excited state ───────────────────────────────────────────────────
    Random.seed!(43)
    psi_raw2  = random_mps(sites)
    psi_init2 = replaceprime(contract(P, psi_raw2; cutoff=1e-12), 1 => 0)
    normalize!(psi_init2)

    println("\n────────── First excited DMRG ──────────")
    println("[DENSE  excited]")
    (E1_d, psi1_d), t_e_dense = run_dmrg_excited("DENSE_excited",
        H_dense, [psi0_d], deepcopy(psi_init2);
        nsweeps=_ns, maxdim, mindim, cutoff, weight=excited_weight, outputlevel=1, run_label="DENSE_E")

    println("[ALIASED excited]")
    (E1_a, psi1_a), t_e_ali = run_dmrg_excited("ALIASED_excited",
        H_aliased, [psi0_a], deepcopy(psi_init2);
        nsweeps=_ns, maxdim, mindim, cutoff, weight=excited_weight, outputlevel=1, run_label="ALIASED_E")

    # ── Summary ───────────────────────────────────────────────────────────────
    println("\n========== SUMMARY (PXP PHP, N=$N, maxdim=$_md, nsweeps=$_ns) ==========")
    println("MPO memory:")
    println("  DENSE   PHP = $(round(dense_bytes/1e6;   digits=4)) MB")
    println("  ALIASED PHP = $(round(aliased_bytes/1e6; digits=4)) MB")
    println("  dense/aliased = $(round(dense_bytes/aliased_bytes; digits=3))×")
    println()
    println("Ground state:")
    println("  DENSE   E0 = $E0_d   wall = $(round(t_g_dense; digits=3)) s")
    println("  ALIASED E0 = $E0_a   wall = $(round(t_g_ali;   digits=3)) s")
    println("  |ΔE0|     = $(abs(E0_d - E0_a))")
    println("  ratio aliased/dense = $(round(t_g_ali/t_g_dense; digits=3))")
    println()
    println("First excited:")
    println("  DENSE   E1 = $E1_d   wall = $(round(t_e_dense; digits=3)) s")
    println("  ALIASED E1 = $E1_a   wall = $(round(t_e_ali;   digits=3)) s")
    println("  |ΔE1|     = $(abs(E1_d - E1_a))")
    println("  ratio aliased/dense = $(round(t_e_ali/t_e_dense; digits=3))")
    println()
    println("Gap (E1 − E0):")
    println("  DENSE   gap = $(E1_d - E0_d)")
    println("  ALIASED gap = $(E1_a - E0_a)")
end
nothing
