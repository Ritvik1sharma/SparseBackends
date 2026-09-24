# kl.jl — the KL (Kitaev ladder) model: operators and initial state.
#
# Copied verbatim from the validated driver
# test_sparse_ham/test_check_working_aliased.jl, so the benchmark measures exactly
# the operator that test compares. The sandwich itself lives in php.jl.
#
# Duplicated (not shared) with experiments/manual_tests/kl.jl on purpose: each
# tree constructs its operators from scratch under its own ITensors fork, which
# is what makes the fork comparison independent rather than an artefact of shared
# code.

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

"""
    kl_operators(nplaq, spin, psign; pad_h_chi = 0)

Return `(sites, H, ConsOps1, ConsOps2)` for a `2*nplaq + 2` site ladder.

  H         raw (unprojected) Hamiltonian MPO, used for the reported energy
  ConsOps1  per-plaquette constraint MPOs on their 4-site support, CLEANED —
            these are merged into the single P used by the sandwich
  ConsOps2  the SAME plaquettes as full-lattice MPOs, NOT cleaned — used only to
            project psi0

The two forms are the same operator built two ways, and the split is inherited
from the driver, not a design choice: psi0 ends up using the uncleaned
full-lattice form applied sequentially while the operator uses the cleaned local
form merged. See the "psi0 divergence probe" entry in OPEN_QUESTIONS.md.
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

"""
Initial state: random MPS projected by each full-lattice plaquette MPO.

NOTE the plaquettes are applied SEQUENTIALLY here with a renormalisation between
each, whereas the sandwich merges them into one P first. Algebraically the same
projector, numerically not. Kept because it is what the validated driver does, so
psi0 is bit-identical to the reference.
"""
function kl_psi0(sites, ConsOps2::Vector{MPO}, seed::Int)
    Random.seed!(seed)
    psi = random_mps(sites)
    for j in 1:length(ConsOps2)
        psi = replaceprime(ConsOps2[j] * psi, 1 => 0)
        normalize!(psi)
    end
    return psi
end
