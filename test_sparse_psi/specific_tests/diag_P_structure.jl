# Inspect P_sparse structure: bond dims, channel labels, per-channel block norms.
using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra
using Random
using Printf
include("../utils.jl")

function build_setup(N::Int, psign::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
    cs = 0.5 * psign
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += cs,  "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]
    for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    return ConsOps1, P_sparse, sites
end

function per_bond_block_norms(mpo, label::String)
    println("=== $label ===")
    println("  N_sites = $(length(mpo))")
    for i in 1:length(mpo)
        T = mpo[i]
        # find link inds
        link_inds = filter(I -> ITensors.hastags(I, "Link"), collect(ITensors.inds(T)))
        dims_str = join([string(ITensors.dim(I)) for I in link_inds], ",")
        nblocks = "n/a"
        keys_str = "n/a"
        if ITensors.has_external_storage(T)
            s = ITensors.get_external_storage(T)
            if s isa SparseBackends.WrappedBlockSparse
                bs = s.blocksparse
                nblocks = string(length(bs.keys))
                # show unique values per axis
                P = length(bs.keys) > 0 ? length(bs.keys[1]) : 0
                axis_unique = String[]
                for a in 1:P
                    vals = sort(unique(k[a] for k in bs.keys))
                    push!(axis_unique, "[" * join(vals, ",") * "]")
                end
                keys_str = join(axis_unique, " x ")
            end
        end
        @printf("  site %2d  link_dims=[%s]  bs_nblocks=%s  bs_per_axis_keys=%s\n",
                i, dims_str, nblocks, keys_str)
    end
end

let
    N_plaq = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
    psign  = +1
    println("Building setup for N=$N_plaq, psign=$psign ...")
    ConsOps1, P_sparse, sites = build_setup(N_plaq, psign)

    println("\n--- Each per-plaquette projector P_j ---")
    for j in 1:length(ConsOps1)
        per_bond_block_norms(ConsOps1[j], "ConsOps1[$j] = (I+C_$j)/2")
    end

    println("\n--- Full P_sparse = product of all P_j ---")
    per_bond_block_norms(P_sparse, "P_sparse")
end
nothing
