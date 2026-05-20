# Check that channel-aware SVD preserves block-key structure of M[b], M[b+1]
# AND reconstructs phi correctly (vs dense SVD reference).
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
include("utils.jl")

function build_setup(N::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
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
    psi0 = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return psi_sp
end

function bs_info(T)
    !ITensors.has_external_storage(T) && return "Dense"
    w = ITensors.get_external_storage(T)
    "BS($(length(w.blocksparse.keys)) blk, $(length(w.blocksparse.data))/$(prod(w.blocksparse.dims)))"
end

# Set of sparse-axis-key tuples for a BS tensor
function block_keys(T)
    w = ITensors.get_external_storage(T)::SparseBackends.WrappedBlockSparse
    return Set(Tuple(k) for k in w.blocksparse.keys)
end

# Density-test: ||phi - L*R|| via dense conversion
densify(T) = ITensors.has_external_storage(T) ? SparseBackends.to_dense_itensors_unfused(T) : T
recon_err(phi::ITensor, L::ITensor, R::ITensor) = norm(densify(phi) - densify(L * R))

function check_one_bond(psi, b::Int)
    println("\n========== Bond $b (psi[$b] ↔ psi[$(b+1)]) ==========")
    println("  psi[$b]:   $(bs_info(psi[b]))")
    println("  psi[$(b+1)]: $(bs_info(psi[b+1]))")

    keys_b_init  = block_keys(psi[b])
    keys_b1_init = block_keys(psi[b+1])

    phi = psi[b] * psi[b+1]
    println("  phi:      $(bs_info(phi))")

    L, R, spec = SparseBackends.itensor_blocksparse_svd_channel_aware(
        phi, psi[b], psi[b+1];
        ortho="left",
        maxdim=typemax(Int), mindim=1, cutoff=0.0)
    println("  L:        $(bs_info(L))")
    println("  R:        $(bs_info(R))")

    # 1. Verify L*R ≈ phi
    err = recon_err(phi, L, R)
    println("  ||phi - L*R||  = $err  ", err < 1e-10 ? "✓" : "✗")

    # 2. Compare against dense SVD reference
    phi_d   = densify(phi)
    # Build the same indsMb (left side of phi inherited from psi[b])
    shared  = commoninds(psi[b], psi[b+1])
    indsMb  = [I for I in inds(phi) if !(I in shared)]
    # Restrict indsMb to phi's actual inds intersected with M_b's non-shared inds
    indsMb  = [I for I in inds(phi) if I in inds(psi[b]) && !(I in shared)]
    Ud, Sd, Vd, _, _, _ = ITensors.svd(phi_d, indsMb;
        lefttags = TagSet("Link,l=$b"),
        righttags = TagSet("Link,l=$b"))
    Ld = Ud
    Rd = Sd * Vd
    err_d = norm(densify(phi) - densify(Ld * Rd))
    println("  ||phi - Ld*Rd|| (dense ref) = $err_d")

    # 3. Block-key set equality with original psi[b], psi[b+1]
    keys_L  = block_keys(L)
    keys_R  = block_keys(R)
    same_L  = keys_L == keys_b_init
    same_R  = keys_R == keys_b1_init
    println("  L block keys == psi[$b] keys:   ", same_L ? "✓ ($(length(keys_L)) keys)" : "✗ (L=$(length(keys_L)), psi[$b]=$(length(keys_b_init)))")
    println("  R block keys == psi[$(b+1)] keys: ", same_R ? "✓ ($(length(keys_R)) keys)" : "✗ (R=$(length(keys_R)), psi[$(b+1)]=$(length(keys_b1_init)))")

    return err < 1e-10 && same_L && same_R
end

let
    psi = build_setup(2)
    println("Initial psi: ", [bs_info(psi[i]) for i in 1:length(psi)])

    ok = true
    for b in 1:(length(psi)-1)
        ok &= check_one_bond(psi, b)
    end
    println("\n===== Overall: ", ok ? "PASS ✓" : "FAIL ✗", " =====")
end
nothing
