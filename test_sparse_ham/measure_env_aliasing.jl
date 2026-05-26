# measure_env_aliasing.jl
#
# Empirical question: does aliasing structure survive the env-build chain?
#
# DMRG's matvec uses left/right envs that are built by repeatedly contracting:
#   L[k+1] = L[k] * H[k+1] * psi[k+1] * dag(prime(psi[k+1]))
# Currently we collapse to dense at each step. This script forces aliased
# output at every contraction (via preserve_bs_output=true) and reports
# n_blocks / n_templates / compression_ratio at each step.

using SparseBackends, Random
using ITensors, ITensorMPS

length(ARGS) < 1 && error("Usage: julia measure_env_aliasing.jl <N_plaq>")

const _ALIASED_ENABLE = get(ENV, "SB_ALIASED_ENABLE", "0") == "1"
if !_ALIASED_ENABLE
    println("[SB_ALIASED_ENABLE != 1] Aliased path is gated off — set SB_ALIASED_ENABLE=1 to run.")
    exit(0)
end

function clean!(op::MPO; tol=1e-12)
    for j in 1:length(op)
        T = op[j]
        A = array(T)
        for i in eachindex(A)
            if abs(A[i]) < tol
                A[i] = 0.0
            elseif abs(A[i] - 1.0) < tol
                A[i] = 1.0
            elseif abs(A[i] + 1.0) < tol
                A[i] = -1.0
            elseif abs(A[i] - 0.5) < tol
                A[i] = 0.5
            elseif abs(A[i] + 0.5) < tol
                A[i] = -0.5
            elseif abs(A[i] - 1.0im) < tol
                A[i] = 1.0im
            elseif abs(A[i] + 1.0im) < tol
                A[i] = -1.0im
            elseif abs(A[i] - 0.5im) < tol
                A[i] = 0.5im
            elseif abs(A[i] + 0.5im) < tol
                A[i] = -0.5im
            end
        end
        op[j] = ITensor(A, inds(T)...)
    end
    return op
end

function mulMPO(A::MPO, B::MPO)
    Bp = prime(B, "Site")
    C = contract(A, Bp, :coo, :coo)
    return replaceprime(C, 2 => 1)
end

function multiplyVec(vec::Vector{MPO})
    result = vec[1]
    for j in 2:length(vec); result = mulMPO(result, vec[j]); end
    return result
end

function sandwich_mpo_aliased(P::MPO, H::MPO)
    new_H = MPO(length(H))
    for i in 1:length(H)
        H1 = SparseBackends.contract_aliased_itensor(P[i]'', H[i]', :coo, :dense)
        H_eff_i = SparseBackends.contract_aliased_itensor(P[i], H1, :coo, :aliased)
        new_H[i] = replaceprime(H_eff_i, 3 => 1)
    end
    return new_H
end

function alias_stats(label::String, T::ITensor)
    if ITensors.has_external_storage(T) && T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse
        ali = T.tensor.data.aliased
        nb = length(ali.keys)
        nt = ali.n_templates
        cr = nb == 0 ? 1.0 : nb / nt
        bks = ali.blksize
        # Data-element accounting (ignoring index overhead):
        dense_elems   = prod(ali.dims)
        bs_elems      = nb * bks
        aliased_elems = nt * bks + nb        # templates + per-block scalar/id
        vs_dense = dense_elems / max(aliased_elems, 1)
        vs_bs    = bs_elems    / max(aliased_elems, 1)
        bytes = Base.summarysize(T)
        println(rpad(label, 32),
                " nb=", lpad(nb,4),
                " nt=", lpad(nt,4),
                " comp=", lpad(round(cr;digits=2),5),
                "  dense_el=", lpad(dense_elems,9),
                " bs_el=", lpad(bs_elems,8),
                " ali_el=", lpad(aliased_elems,8),
                "  vs_dense=", lpad(round(vs_dense;digits=2),6),
                " vs_BS=", round(vs_bs;digits=2))
        return cr
    else
        # Dense or other — compute prod(dims) for reference
        bytes = Base.summarysize(T)
        dims = collect(ITensors.dim.(ITensors.inds(T)))
        dense_elems = isempty(dims) ? 1 : prod(dims)
        println(rpad(label, 32), " (dense)             dense_el=", lpad(dense_elems,9))
        return -1.0
    end
end

# Wrap an ITensor as :aliased external storage if it has external storage,
# otherwise as :dense.
backend_of(T::ITensor) = ITensors.has_external_storage(T) ? :aliased : :dense

# Contract two ITensors preserving aliased structure (one or both can be aliased).
function ali_contract(A::ITensor, B::ITensor)
    ba = backend_of(A); bb = backend_of(B)
    if ba === :dense && bb === :dense
        return A * B
    end
    return SparseBackends.contract_aliased_itensor(A, B, ba, bb;
                                                   preserve_bs_output=true)
end

let
    spin = 3
    N = parse(Int, ARGS[1])
    states = 2*N + 2
    sites = siteinds("S=1", states)

    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    os2 = OpSum[]
    for j in 1:N
        c = 0.5
        temp = OpSum()
        temp += c, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        temp += c, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                  "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, temp)
    end

    ConsOps = MPO[]
    for j in 1:N
        op = MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])
        clean!(op; tol=1e-12)
        push!(ConsOps, op)
    end
    P = multiplyVec(ConsOps)

    H_raw = MPO(os, sites)
    H = sandwich_mpo_aliased(P, copy(H_raw))
    L = length(H)
    println("\n[H_aliased per-site stats]")
    for i in 1:L
        alias_stats("H[$i]", H[i])
    end

    Random.seed!(42)
    psi = random_mps(sites; linkdims=2)
    for j in 1:N
        psi = replaceprime(ConsOps[j] * psi, 1 => 0)
        normalize!(psi)
    end
    # Set link dimensions by acting on a maxdim sweep
    md = parse(Int, get(ENV, "BENCH_MAXDIM", "20"))
    println("\n[env build, max_link_dim ~ $md]")

    # Build L envs left → right.
    # Convention: L[1] = a scalar-ish ITensor; L[k+1] = L[k] * psi[k] * H[k] * dag(prime(psi[k]))
    # Use an MPS that has been put into mixed canonical form to keep bond dims sane.
    ITensorMPS.orthogonalize!(psi, L)  # right canonical

    # Build right env one site at a time
    R = ITensor(1.0)
    println("\n--- Building R envs (right → left) ---")
    println(rpad("step",6), rpad("after-op",26), " stats")
    for k in L:-1:1
        # R ← R * H[k]
        R1 = ali_contract(R, H[k])
        alias_stats("R*H[$k]", R1)

        # R ← R1 * psi[k]
        R2 = ali_contract(R1, psi[k])
        alias_stats("(R*H)*psi[$k]", R2)

        # R ← R2 * dag(prime(psi[k]))
        R3 = ali_contract(R2, dag(prime(psi[k])))
        alias_stats("(R*H*psi)*dag(psi')[$k]", R3)
        println("")

        R = R3
    end
end
nothing
