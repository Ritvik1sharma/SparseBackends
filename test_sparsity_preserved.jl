# Verifies that DMRG preserves the block-sparse key structure of ψ.
# Usage: julia test_sparsity_preserved.jl <N> [nsweeps] [maxdim]
# Two complementary checks:
#   (1) Per-site link-tuple count is unchanged before/after DMRG
#   (2) Final key set ⊆ initial key set at every site (strict subset check)
using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

length(ARGS) < 1 && error("Usage: julia test_sparsity_preserved.jl <N_plaq> [nsweeps] [maxdim]")
const N_PLAQ   = parse(Int, ARGS[1])
const N_SWEEPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
const MAXDIM   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20

function build_setup(N::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += 0.5, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return H_sparse, psi_sp
end

# Raw key set off a BS tensor.
bs_keyset(T::ITensor) = Set(collect(ITensors.get_external_storage(T).blocksparse.keys))

# Sparse axis positions (first P axes by convention).
function sparse_positions(w::SparseBackends.WrappedBlockSparse)
    collect(1:(length(w.inds) - length(SparseBackends.dense_inds(w))))
end

# Per-site block signature: map site-value → count of link-key tuples.
function site_block_signature(T::ITensor)
    @assert ITensors.has_external_storage(T)
    w       = ITensors.get_external_storage(T)::SparseBackends.WrappedBlockSparse
    sp_pos  = sparse_positions(w)
    site_p  = findfirst(i -> ITensors.hastags(w.inds[i], "Site"), sp_pos)
    isnothing(site_p) && return Dict(0 => Set{Tuple}())
    site_p  = sp_pos[site_p]
    link_p  = [p for p in sp_pos if p != site_p]
    sig     = Dict{Int, Set{Tuple}}()
    for key in w.blocksparse.keys
        s   = key[site_p]
        lks = ntuple(i -> key[link_p[i]], length(link_p))
        push!(get!(sig, s, Set{Tuple}()), lks)
    end
    return sig
end

describe_T(T, name) = begin
    if ITensors.has_external_storage(T)
        w = ITensors.get_external_storage(T)
        println("  $name: BS($(length(w.blocksparse.keys)) blk)")
    else
        println("  $name: Dense")
    end
end

let
    println("=== Sparsity preservation across DMRG (N=$N_PLAQ, nsweeps=$N_SWEEPS, maxdim=$MAXDIM) ===")
    H_sp, psi = build_setup(N_PLAQ)
    L = length(psi)

    # Capture signatures before DMRG.
    sig_init    = [site_block_signature(psi[i]) for i in 1:L]
    initial_keys = [bs_keyset(T) for T in psi]
    println("Initial per-site |keyset|: ", length.(initial_keys))
    for i in 1:L; describe_T(psi[i], "psi[$i]"); end

    # Run DMRG.
    sweeps = Sweeps(N_SWEEPS)
    setmaxdim!(sweeps, [min(10*k, MAXDIM) for k in 1:N_SWEEPS]...)
    setmindim!(sweeps, 1)
    setcutoff!(sweeps, 1e-10)
    E, psi_out = dmrg(H_sp, psi, sweeps; outputlevel=0, use_early_exit=false)
    println("\nDMRG done. E = $E")

    final_keys = [bs_keyset(T) for T in psi_out]
    sig_now    = [site_block_signature(psi_out[i]) for i in 1:L]
    println("Final per-site |keyset|:   ", length.(final_keys))
    for i in 1:L; describe_T(psi_out[i], "psi_out[$i]"); end

    # --- Check 1: Per-site link-tuple count ---
    println("\n--- Check 1: per-site link-tuple count (before vs after) ---")
    ok1 = true
    for i in 1:L
        for s in sort(collect(union(keys(sig_init[i]), keys(sig_now[i]))))
            n_i = length(get(sig_init[i], s, Set{Tuple}()))
            n_n = length(get(sig_now[i],  s, Set{Tuple}()))
            flag = n_i == n_n ? "✓" : "✗"
            ok1 &= (n_i == n_n)
            println("  psi[$i] site=$s: init=$n_i  now=$n_n  $flag")
        end
    end
    println("Check 1: ", ok1 ? "PASS ✓" : "FAIL ✗")

    # --- Check 2: Keyset subset (final ⊆ initial) ---
    println("\n--- Check 2: keyset subset check (final ⊆ initial per site) ---")
    ok2 = true
    for i in 1:L
        ok = issubset(final_keys[i], initial_keys[i])
        extra   = length(setdiff(final_keys[i], initial_keys[i]))
        missing_ = length(setdiff(initial_keys[i], final_keys[i]))
        flag = ok ? "✓" : "✗"
        ok2 &= ok
        println("  site $i: init=$(length(initial_keys[i]))  final=$(length(final_keys[i]))  $flag  (extra=$extra, missing=$missing_)")
    end
    println("Check 2: ", ok2 ? "PASS ✓" : "FAIL ✗")

    println("\nOverall: ", (ok1 && ok2) ? "PASS ✓ — block-key structure preserved" : "FAIL ✗ — sparsity pattern altered")
end
nothing
