# pxp.jl — the PXP model: operators and initial state.
#
# Copied verbatim from the validated driver test_sparse_ham/test_pxp_aliased.jl,
# so the benchmark measures exactly the operator that test compares. The sandwich
# itself lives in php.jl.
#
# Duplicated (not shared) with experiments/manual_tests/pxp.jl on purpose: each
# tree constructs its operators from scratch under its own ITensors fork.

ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]

function pxp_from_nonzeros(dims::NTuple{N,Int}, coords::Vector) where {N}
    A = zeros(Float64, dims...)
    for c in coords
        @assert length(c) == N
        A[(c .+ 1)...] = 1.0
    end
    return A
end

"""Blockade constraint MPO (bond dimension 2)."""
function NotEqlsLoop_R1(sites)
    N = length(sites)
    R1_first = pxp_from_nonzeros((3, 3, 2), [(0,0,0), (1,1,1), (2,2,0)])
    R1_bulk  = pxp_from_nonzeros((3, 3, 2, 2),
                   [(0,0,0,0), (0,0,1,0), (1,1,0,1), (1,1,1,0), (2,2,0,0)])
    R1_last  = pxp_from_nonzeros((3, 3, 2),
                   [(0,0,0), (0,0,1), (1,1,0), (1,1,1), (2,2,0)])
    bonds = [Index(2, "Link,l=$(i)") for i in 1:(N - 1)]
    W = Vector{ITensor}(undef, N)
    W[1] = ITensor(R1_first, sites[1], sites[1]', bonds[1])
    for j in 2:(N - 1)
        W[j] = ITensor(R1_bulk, sites[j], sites[j]', bonds[j - 1], bonds[j])
    end
    W[N] = ITensor(R1_last, sites[N], sites[N]', bonds[N - 1])
    return MPO(W)
end

function pxp_opsum(N::Int)
    os = OpSum()
    os += 1, "Xp", 1
    for j in 0:(N - 2)
        os += 1, "Px", j + 1, "LP", j + 2
        os += 1, "RP", j + 1, "Xp", j + 2
    end
    os += 1, "Px", N
    return os
end

"""
    pxp_operators(nsites; pad_h_chi = 0)

Return `(sites, H_raw, P)`. Unlike KL there is a single P (the blockade
constraint), so nothing needs merging before the sandwich.
"""
function pxp_operators(nsites::Int; pad_h_chi::Int = 0)
    sites = siteinds("S=1", nsites)
    os    = pxp_opsum(nsites)
    H     = pad_h_chi > 0 ? first(pad_hamiltonian(os, sites, pad_h_chi)) : MPO(os, sites)
    return sites, H, NotEqlsLoop_R1(sites)
end

"""Initial state: random MPS projected once by P."""
function pxp_psi0(sites, P::MPO, seed::Int)
    Random.seed!(seed)
    psi = replaceprime(contract(P, random_mps(sites); cutoff = 1e-12), 1 => 0)
    normalize!(psi)
    return psi
end
