# Find the remaining all-sparse DMRG bug by isolating product(PH, phi).
# Build PH (ProjMPO over H_sparse) at a bond, compute product(PH, phi)
# both fully-sparse and via densified reference. Compare.
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
    ConsOps2 = [MPO(os2[j], sites) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end

    H = MPO(os, sites)
    H1 = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)

    psi0 = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return psi_sp, H_sparse, H, sites
end

# Densify a sparse ITensor preserving Index identity
densify(T::ITensor) = ITensors.has_external_storage(T) ?
    SparseBackends.to_dense_itensors_unfused(T) : T

# Densify an MPS/MPO
function densify(M::Union{MPS,MPO})
    Md = typeof(M)(length(M))
    for i in 1:length(M); Md[i] = densify(M[i]); end
    return Md
end

length(ARGS) < 1 && error("Usage: julia test_eigsolve_kernel.jl <N_plaq>")

let
    N = parse(Int, ARGS[1])
    psi_sp, H_sparse, H, sites = build_setup(N)
    println("="^78)
    println("Isolate the contraction kernel that breaks all-sparse DMRG")
    println("="^78)

    println("\n--- Initial ⟨psi_sp|H_sparse|psi_sp⟩ (both sparse) ---")
    # This is just sparse * sparse via MPO * MPS contractions
    E_sp_sp = inner(psi_sp', H_sparse, psi_sp)
    println("  E = $E_sp_sp")

    println("\n--- Same but densify H_sparse first ---")
    H_dense_eq = densify(H_sparse)
    E_sp_dn = inner(psi_sp', H_dense_eq, psi_sp)
    println("  E = $E_sp_dn")

    println("\n--- Same but densify psi_sp first ---")
    psi_dense_eq = densify(psi_sp)
    E_dn_sp = inner(psi_dense_eq', H_sparse, psi_dense_eq)
    println("  E = $E_dn_sp")

    println("\n--- Same densified both ---")
    E_dn_dn = inner(psi_dense_eq', H_dense_eq, psi_dense_eq)
    println("  E = $E_dn_dn")

    println("\nAll four should agree (within rounding). Disagreements localize the bug.")

    # Simulate one DMRG sweep step manually to see when sparse vs dense diverge
    println("\n--- Simulated sweep: orthogonalize psi to bond 1, then product(PH, phi) at each bond ---")
    function orth_via_sparse_svd!(psi::MPS, j::Int)
        # Manually move orth center to position j using sparse SVD (bin_by_right=true)
        for b in 1:j-1
            if ITensors.has_external_storage(psi[b])
                linds = uniqueinds(psi[b], psi[b+1])
                ltags = TagSet("Link,l=$b")
                L, R, _ = SparseBackends.itensor_blocksparse_svd(
                    psi[b], linds; ortho="left",
                    tags = ltags, maxdim=typemax(Int), mindim=1, cutoff=0.0,
                    bin_by_right = true)
                psi[b]   = L
                psi[b+1] = R * psi[b+1]
            end
        end
    end

    function rq_at_bond(H_mpo, psi::MPS, b::Int)
        PH = ProjMPO(H_mpo); PH.nsite = 2
        try
            position!(PH, psi, b)
            phi = psi[b] * psi[b+1]
            Hphi = product(PH, phi)
            denom = ITensors.scalar(ITensors.dag(phi) * phi)
            num   = ITensors.scalar(ITensors.dag(phi) * Hphi)
            return real(num/denom)
        catch e
            return "THREW: $(sprint(showerror, e))"
        end
    end

    function dump_psi_blocks(psi::MPS, label::String)
        println("  --- $label ---")
        for i in 1:length(psi)
            T = psi[i]
            if ITensors.has_external_storage(T)
                w = ITensors.get_external_storage(T)
                if w isa SparseBackends.WrappedBlockSparse
                    println("    psi[$i] inds_dims=$(map(ITensors.dim, ITensors.inds(T)))  P=$(SparseBackends._P(w.blocksparse))  N2=$(SparseBackends._N2(w.blocksparse))  storage_dims=$(w.blocksparse.dims)  n_blocks=$(length(w.blocksparse.keys))")
                    println("      first 6 keys: $(w.blocksparse.keys[1:min(6,end)])")
                end
            else
                println("    psi[$i] DENSE  dims=$(map(ITensors.dim, ITensors.inds(T)))")
            end
        end
    end

    psi_sp_w = copy(psi_sp)
    psi_dn_w = densify(psi_sp)

    dump_psi_blocks(psi_sp_w, "psi_sp initial")

    println("\nBefore any orthogonalize:")
    for b in 1:length(psi_sp_w)-1
        r_sp = rq_at_bond(H_sparse,    psi_sp_w, b)
        r_dn = rq_at_bond(H_dense_eq,  psi_dn_w, b)
        println("  bond $b: RQ_sparse=$r_sp  RQ_dense=$r_dn  diff=$(abs(r_sp - r_dn))")
    end

    # Now do one sparse SVD step at bond 1 (move to position 2)
    psi_sp_w = copy(psi_sp)
    orth_via_sparse_svd!(psi_sp_w, 2)
    dump_psi_blocks(psi_sp_w, "psi_sp after orth_to_2")
    psi_dn_w = copy(psi_dn_w)
    orthogonalize!(psi_dn_w, 2)

    println("\nAfter orthogonalize to bond 2:")
    for b in 1:length(psi_sp_w)-1
        r_sp = rq_at_bond(H_sparse,    psi_sp_w, b)
        r_dn = rq_at_bond(H_dense_eq,  psi_dn_w, b)
        println("  bond $b: RQ_sparse=$r_sp  RQ_dense=$r_dn  diff=$(abs(r_sp - r_dn))")
    end

    # Do another step
    orth_via_sparse_svd!(psi_sp_w, 4)
    orthogonalize!(psi_dn_w, 4)
    println("\nAfter orthogonalize to bond 4:")
    for b in 1:length(psi_sp_w)-1
        r_sp = rq_at_bond(H_sparse,    psi_sp_w, b)
        r_dn = rq_at_bond(H_dense_eq,  psi_dn_w, b)
        println("  bond $b: RQ_sparse=$r_sp  RQ_dense=$r_dn  diff=$(abs(r_sp - r_dn))")
    end
end
nothing
