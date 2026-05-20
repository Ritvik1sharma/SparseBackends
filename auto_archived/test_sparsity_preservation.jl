# Verifies that the SPARSITY PATTERN (set of sparse-key tuples) of each
# psi tensor is preserved across a full DMRG pass.
#
# Initial psi has block-keys determined by the P_sparse channel structure.
# After every SVD / orthogonalize step, the new psi should live in the SAME
# restricted-block subspace — i.e., the sparse-key SET (as a per-site signature)
# stays the same.  Bond multiplicities can grow with maxdim, but the set of
# allowed (sparse_axis_value) tuples must not change.
using SparseBackends, ITensors, ITensorMPS
using Random

include("utils.jl")

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
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H = MPO(os, sites)
    H1 = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0 = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    # Dense reference: convert H_sparse to dense ITensors element-wise.
    H_dense = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in H_sparse])
    return H_sparse, H_dense, psi_sp
end

# Identify which axis positions are sparse-axis (vs dense multiplicity).
# Returns a vector of "sparse position" indices into w.inds.
function sparse_positions(w::SparseBackends.WrappedBlockSparse)
    N  = length(w.inds)
    N2 = length(SparseBackends.dense_inds(w))
    return collect(1:(N - N2))   # convention: first P axes are sparse
end

# Per-site sparsity signature:
# Records (site_dim, set_of_link_sparse_tuples_per_site_value).
# A "link sparse tuple" excludes the site axis itself and any dense (multiplicity)
# axes — keeping only the link sparse axes' values.
function site_block_signature(T::ITensor)
    @assert ITensors.has_external_storage(T) "expected BS tensor"
    w = ITensors.get_external_storage(T)::SparseBackends.WrappedBlockSparse
    sp_pos = sparse_positions(w)
    # Find the site axis position (the one with "Site" in its tag)
    site_pos = findfirst(i -> ITensors.hastags(w.inds[i], "Site"), sp_pos)
    isnothing(site_pos) && return Dict(0 => Set{Tuple}())
    site_pos = sp_pos[site_pos]
    # The link sparse axes are all sparse positions except the site one
    link_pos = [p for p in sp_pos if p != site_pos]
    # Walk all blocks of w.blocksparse and group by site key
    sig = Dict{Int, Set{Tuple}}()
    for key in w.blocksparse.keys
        s   = key[site_pos]
        lks = ntuple(i -> key[link_pos[i]], length(link_pos))
        push!(get!(sig, s, Set{Tuple}()), lks)
    end
    return sig
end

# Compare two signatures site-by-site. Returns (ok::Bool, report::String).
function compare_signatures(sig_init::Dict, sig_now::Dict, label::String)
    all_sites = sort(collect(union(keys(sig_init), keys(sig_now))))
    ok = true
    lines = String[]
    for s in all_sites
        i_keys = get(sig_init, s, Set{Tuple}())
        n_keys = get(sig_now,  s, Set{Tuple}())
        # Compare just the COUNT of link-key tuples at this site value
        # (label permutations are allowed since indices are freshly created)
        if length(i_keys) != length(n_keys)
            ok = false
            push!(lines, "  $label site=$s: init=$(length(i_keys))  now=$(length(n_keys))  ✗")
        else
            push!(lines, "  $label site=$s: init=$(length(i_keys))  now=$(length(n_keys))  ✓")
        end
    end
    return ok, join(lines, "\n")
end

describe_T(T, name) = begin
    if ITensors.has_external_storage(T)
        w = ITensors.get_external_storage(T)
        println("  $name: BS($(length(w.blocksparse.keys)) blk, $(length(w.blocksparse.data))/$(prod(w.blocksparse.dims)))")
    else
        println("  $name: Dense")
    end
end

let
    H_sparse, H_dense, psi = build_setup(2)
    N = length(psi)

    println("=== Initial psi ===")
    sig_init = [site_block_signature(psi[i]) for i in 1:N]
    for i in 1:N; describe_T(psi[i], "psi[$i]"); end

    # Total expected block count per site (across all site values)
    expected_counts = [sum(length(v) for (_, v) in sig_init[i]; init=0) for i in 1:N]
    println("\nExpected per-site link-tuple count: ", expected_counts)

    println("\n=== Running 1 DMRG sweep ===")
    E, psi_after, _, _ = dmrg(H_sparse, psi;
        nsweeps=1, maxdim=[20], mindim=[1], cutoff=1e-12,
        target_energy=nothing, use_early_exit=false, outputlevel=0)
    println("Sparse DMRG  E = ", E)

    println("\n=== Post-sweep psi ===")
    for i in 1:N; describe_T(psi_after[i], "psi[$i]"); end
    sig_now = [site_block_signature(psi_after[i]) for i in 1:N]

    println("\n=== Sparsity-pattern check (per-site link-tuple count) ===")
    all_ok = true
    for i in 1:N
        ok, rpt = compare_signatures(sig_init[i], sig_now[i], "psi[$i]")
        all_ok &= ok
        println(rpt)
    end
    println("\nOverall: ", all_ok ? "PASS ✓ — block-key counts preserved per site" : "FAIL ✗ — sparsity pattern altered")

    # Dense reference energy
    println("\n=== Dense DMRG (reference energy) ===")
    psi_d = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi])
    E_d, _, _, _ = dmrg(H_dense, psi_d;
        nsweeps=1, maxdim=[20], mindim=[1], cutoff=1e-12,
        target_energy=nothing, use_early_exit=false, outputlevel=0)
    println("Dense  DMRG  E = ", E_d)
    println("|E_sparse - E_dense| = ", abs(E - E_d))
end
nothing
