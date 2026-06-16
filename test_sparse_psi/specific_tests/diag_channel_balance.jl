# Diagnose where the channel imbalance enters:
#   Test 1: per-channel block-norm² of psi_sp = P · psi_0 (before any DMRG)
#   Test 2: same, of phi = psi[b] * psi[b+1] before eigsolve at bond 5
#   Test 3: same, of eigsolve output phi (after H_eff)
#   Test 4: same, of L, R from QR factorization
#
# Channel-imbalance source detected as the first test where norms differ.
ENV["BMF_ISO_PATH"] = "1"
using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra: norm
using Random
using Printf
include("../utils.jl")

function build_setup(N::Int, psign::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
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
    H        = MPO(os, sites)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return H, psi_sp
end

# For each BS tensor T in the MPS, compute per-channel block norm² grouped by
# the channel key of the bond between T and the next site.
function per_channel_block_norms(T::ITensor, bond_axis_idx::Int)
    !ITensors.has_external_storage(T) && return Dict{Int, Float64}()
    s = ITensors.get_external_storage(T)
    s isa SparseBackends.WrappedBlockSparse || return Dict{Int, Float64}()
    bs = s.blocksparse
    P  = length(bs.keys) > 0 ? length(bs.keys[1]) : 0
    bond_axis_idx <= P || return Dict{Int, Float64}()
    out = Dict{Int, Float64}()
    blksize = bs.blksize
    for (id, k) in enumerate(bs.keys)
        ch = k[bond_axis_idx]
        off = (id - 1) * blksize
        s_norm² = 0.0
        for i in 1:blksize
            s_norm² += abs2(bs.data[off + i])
        end
        out[ch] = get(out, ch, 0.0) + s_norm²
    end
    return out
end

function find_bond_axis(T::ITensor, bond_index)
    !ITensors.has_external_storage(T) && return 0
    s = ITensors.get_external_storage(T)
    s isa SparseBackends.WrappedBlockSparse || return 0
    for (i, idx) in enumerate(s.inds)
        idx == bond_index && return i
    end
    return 0
end

function print_per_bond(psi, label::String; bond_range = 1:length(psi)-1)
    println("=== $label ===")
    for b in bond_range
        T = psi[b]
        bi = commonind(psi[b], psi[b+1])
        bi === nothing && continue
        axis = find_bond_axis(T, bi)
        axis == 0 && continue
        chan_norms = per_channel_block_norms(T, axis)
        isempty(chan_norms) && continue
        ks = sort(collect(keys(chan_norms)))
        total = sum(values(chan_norms))
        @printf("  bond %d  n_channels=%d  total=%.4e\n", b, length(ks), total)
        for k in ks
            v = chan_norms[k]
            @printf("    cM=%d  ‖blk‖²=%.6e  fraction=%.4f\n", k, v, v/max(total,1e-30))
        end
    end
end

let
    N_plaq = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
    psign  = +1
    println("Building setup for N=$N_plaq plaquettes, psign=$psign ...")
    H, psi_sp = build_setup(N_plaq, psign)
    println("System: $(length(psi_sp)) sites")

    println("\n--- TEST 1: per-channel block norms of psi_sp BEFORE any DMRG ---")
    print_per_bond(psi_sp, "psi_sp (raw, P·psi_0)")

    println("\n--- Orthogonalize to bond 1, repeat ---")
    psi_sp = orthogonalize!(psi_sp, 1)
    print_per_bond(psi_sp, "psi_sp (orthogonalized to b=1)")
end
nothing
