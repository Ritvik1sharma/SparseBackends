# models.jl — KL and PXP model builders for the SparseBackends benchmark grid.
#
# The operator definitions here are copied verbatim from the validated test
# drivers so the benchmark measures exactly the operators those tests compare:
#   KL  : test_sparse_ham/test_check_working_aliased.jl
#   PXP : test_sparse_ham/test_pxp_aliased.jl
#
# Both models expose the same pair of PHP builders:
#   *_php_dense(...)    -> dense PHP MPO           (variant :sb_dense)
#   *_php_aliased(...)  -> AliasedBlockSparse PHP  (variants :sb_aliased, :sb_fused)
# plus a matching initial state builder, so bench_common.jl can drive either
# model through one code path.

using SparseBackends
using ITensors, ITensorMPS
using Random

# fuse_sparse_links! / prepermute_aliased_mpo! / report_aliased_footprint /
# sparse_prefix_inds. Reused rather than duplicated — this is the same file the
# test drivers include.
include(joinpath(@__DIR__, "..", "test_sparse_ham", "aliased_helpers.jl"))

# inflate_mpo_bonds / pad_hamiltonian — zero-pads the bare H's bond dimension for
# the padded-H cost experiment. No-op unless a config sets pad_h_chi.
include(joinpath(@__DIR__, "..", "..", "experiments", "pad_h_utils.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Snap near-integer/near-half entries to exact values (KL plaquette cleanup)."""
function clean_mpo!(op::MPO; tol = 1e-12)
    for j in 1:length(op)
        T = op[j]
        A = array(T)
        for i in eachindex(A)
            v = A[i]
            if     abs(v)         < tol; A[i] =  0.0
            elseif abs(v - 1.0)   < tol; A[i] =  1.0
            elseif abs(v + 1.0)   < tol; A[i] = -1.0
            elseif abs(v - 0.5)   < tol; A[i] =  0.5
            elseif abs(v + 0.5)   < tol; A[i] = -0.5
            elseif abs(v - 1.0im) < tol; A[i] =  1.0im
            elseif abs(v + 1.0im) < tol; A[i] = -1.0im
            elseif abs(v - 0.5im) < tol; A[i] =  0.5im
            elseif abs(v + 0.5im) < tol; A[i] = -0.5im
            elseif abs(v + 2.0)   < tol; A[i] = -2.0
            elseif abs(v - 2.0)   < tol; A[i] =  2.0
            end
        end
        op[j] = ITensor(A, inds(T)...)
    end
    return op
end

# ─────────────────────────────────────────────────────────────────────────────
# KL (Kitaev ladder)
# ─────────────────────────────────────────────────────────────────────────────

"""
    kl_operators(nplaq, spin, psign)

Return `(sites, H, ConsOps1, ConsOps2)`:
  H         — the raw (unprojected) Hamiltonian MPO, used for the reported energy
  ConsOps1  — per-plaquette constraint MPOs built on their 4-site support (cleaned);
              these are what get merged into the single P used by the sandwich
  ConsOps2  — the same plaquettes as full-lattice MPOs, used to project psi0
"""
function kl_operators(nplaq::Int, spin::Int, psign::Float64; pad_h_chi::Int = 0)
    states = 2 * nplaq + 2
    sites = spin == 3 ? siteinds("S=1",   states) :
            spin == 2 ? siteinds("S=1/2", states) :
            error("kl_operators: unsupported spin=$spin (want 2 => S=1/2, 3 => S=1)")

    os = OpSum()
    for j in 1:(nplaq + 1)
        os += "Sz", 2j - 1, "Sz", 2j
    end
    for j in 1:nplaq
        os += "Sx", 2j - 1, "Sx", 2j + 2
        os += "Sy", 2j,     "Sy", 2j + 1
    end

    os2 = OpSum[]
    for j in 1:nplaq
        coeff = 0.5
        t = OpSum()
        t += coeff, "Id", 2j - 1, "Id", 2j, "Id", 2j + 1, "Id", 2j + 2
        t += psign * coeff, "exp(i*pi*Sy)", 2j - 1, "exp(i*pi*Sx)", 2j,
                            "exp(i*pi*Sx)", 2j + 1, "exp(i*pi*Sy)", 2j + 2
        push!(os2, t)
    end

    ConsOps1 = MPO[]
    ConsOps2 = MPO[]
    for j in 1:nplaq
        op = MPO(os2[j], sites, [2j - 1, 2j, 2j + 1, 2j + 2])
        clean_mpo!(op; tol = 1e-12)
        push!(ConsOps1, op)
        push!(ConsOps2, MPO(os2[j], sites))
    end

    H = pad_h_chi > 0 ? first(pad_hamiltonian(os, sites, pad_h_chi)) : MPO(os, sites)
    return sites, H, ConsOps1, ConsOps2
end

# Merge the per-plaquette constraint MPOs into one P.
# Sparse (COO) merge — feeds the aliased sandwich.
function kl_merge_sparse(ConsOps::Vector{MPO})
    result = ConsOps[1]
    for j in 2:length(ConsOps)
        Bp = prime(ConsOps[j], "Site")
        result = replaceprime(contract(result, Bp, :coo, :coo), 2 => 1)
    end
    return result
end

# Dense merge — feeds the dense sandwich.
function kl_merge_dense(ConsOps::Vector{MPO})
    result = ConsOps[1]
    for j in 2:length(ConsOps)
        Bp = prime(ConsOps[j], "Site")
        result = replaceprime(contract(result, Bp; is_ctn_compression = true), 2 => 1)
    end
    return result
end

"""Dense P†HP (the :sb_dense variant's operator)."""
function kl_php_dense(ConsOps1::Vector{MPO}, H::MPO)
    P = kl_merge_dense(ConsOps1)
    H1    = contract(P'', H'; is_ctn_compression = true)
    H_eff = contract(P, H1;   is_ctn_compression = true)
    return replaceprime(H_eff, 3 => 1)
end

"""
Aliased P†HP (the :sb_aliased / :sb_fused operator), including the two
transforms the aliased matvec relies on: sparse-link fusion and the one-shot
dense-tail prepermute. Both are unconditional in the test drivers.
"""
function kl_php_aliased(ConsOps1::Vector{MPO}, H::MPO)
    P = kl_merge_sparse(ConsOps1)
    H1    = contract(P'', H', :coo, :dense; Cbackend = :aliased)
    H_eff = contract(P, H1,   :coo, :aliased; Cbackend = :aliased)
    Hphp  = replaceprime(H_eff, 3 => 1)
    fuse_sparse_links!(Hphp)
    prepermute_aliased_mpo!(Hphp)
    return Hphp
end

"""Initial state: random MPS projected by each full-lattice plaquette MPO."""
function kl_psi0(sites, ConsOps2::Vector{MPO}, seed::Int)
    Random.seed!(seed)
    psi = random_mps(sites)
    for j in 1:length(ConsOps2)
        psi = replaceprime(ConsOps2[j] * psi, 1 => 0)
        normalize!(psi)
    end
    return psi
end

# ─────────────────────────────────────────────────────────────────────────────
# PXP
# ─────────────────────────────────────────────────────────────────────────────

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

"""Return `(sites, H_raw, P)`."""
function pxp_operators(nsites::Int; pad_h_chi::Int = 0)
    sites = siteinds("S=1", nsites)
    os    = pxp_opsum(nsites)
    H     = pad_h_chi > 0 ? first(pad_hamiltonian(os, sites, pad_h_chi)) : MPO(os, sites)
    return sites, H, NotEqlsLoop_R1(sites)
end

"""
    pxp_php_dense(P, H; exact=false)

Dense P†HP for PXP.

DEFAULT IS NOW `exact=true`, and that is a deliberate change from the original
driver. `test_pxp_aliased.jl` used an SVD zip-up with cutoff 1e-12 / 1e-32, which
left the PXP dense reference TRUNCATED (measured chi_PHP = 12) while
`sb_aliased`/`sb_fused` carry the EXACT operator (and `kl_php_dense` uses
`is_ctn_compression=true`, also exact). That made the dense denominator cheaper
than the numerators — an unfair comparison that flattered dense by ~15-25% on PXP
and is why sb_dense/orig_dense came out at 0.85-0.94 there.

`exact=true` uses the KL discipline (no SVD), giving chi_PHP = chi_P^2*chi_H = 16
and matching what `orig_dense` already produced at cutoff 0.0. It is also required
for the padded-H experiment: zero-padded bond slots have zero singular values, so
any cutoff>0 SVD deletes the padding and chi_PHP silently stays unpadded.

`exact=false` reproduces the original driver, kept only for re-deriving the older
PXP rows. Rows produced with the two settings are NOT comparable.
"""
function pxp_php_dense(P::MPO, H::MPO; exact::Bool = true)
    if exact
        H1    = contract(P'', H'; is_ctn_compression = true)
        H_eff = contract(P, H1;   is_ctn_compression = true)
        return replaceprime(H_eff, 3 => 1)
    end
    H_eff = contract(contract(P'', H'; cutoff = 1e-12), P; cutoff = 1e-32)
    return replaceprime(H_eff, 3 => 1)
end

function pxp_php_aliased(P::MPO, H::MPO)
    new_H = MPO(length(H))
    for i in 1:length(H)
        H1  = SparseBackends.contract_aliased_itensor(P[i]'', H[i]', :coo, :dense)
        Hi  = SparseBackends.contract_aliased_itensor(P[i],   H1,    :coo, :aliased)
        new_H[i] = replaceprime(Hi, 3 => 1)
    end
    fuse_sparse_links!(new_H)
    prepermute_aliased_mpo!(new_H)
    return new_H
end

"""Initial state: random MPS projected once by P."""
function pxp_psi0(sites, P::MPO, seed::Int)
    Random.seed!(seed)
    psi = replaceprime(contract(P, random_mps(sites); cutoff = 1e-12), 1 => 0)
    normalize!(psi)
    return psi
end
