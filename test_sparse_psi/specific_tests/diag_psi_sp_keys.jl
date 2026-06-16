# Compare psi_sp's BS bond keys to what we'd expect from literal (I,C) tracks.
# At site 1: bond_right of psi_sp should have just 2 channels, labeled by (P_1's bond track).
# If keys really are (I=1, C=2), per-channel block norms should be equal by unitarity.
using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra: norm
using Random
using Printf
include("../utils.jl")

let
    Random.seed!(42)
    N = 4
    sites = siteinds("S=1", 2*N+2)
    cs = 0.5
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
    psi0   = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)

    println("== psi0 type at site 1 ==")
    s = ITensors.has_external_storage(psi0[1]) ? typeof(ITensors.get_external_storage(psi0[1])) : "no-ext-storage"
    println("  ", s, "  inds=", inds(psi0[1]), "  link dim=", [dim(i) for i in inds(psi0[1])])

    println("\n== psi_sp at site 1: BS storage details ==")
    T1 = psi_sp[1]
    println("inds = ", inds(T1))
    println("link dims = ", [dim(i) for i in inds(T1) if hastags(i, "Link")])
    if ITensors.has_external_storage(T1)
        w = ITensors.get_external_storage(T1)
        if w isa SparseBackends.WrappedBlockSparse
            bs = w.blocksparse
            println("dims = ", bs.dims, "  blksize = ", bs.blksize, "  n_blocks = ", length(bs.keys))
            println("inds (storage order) = ", w.inds)
            println("\nPer-block content (key, norm²):")
            for (id, k) in enumerate(bs.keys)
                off = (id - 1) * bs.blksize
                n2 = sum(abs2, view(bs.data, off+1:off+bs.blksize))
                @printf("  key=%s  norm²=%.6e\n", k, n2)
            end
        end
    end

    # Now also check raw site-1 contraction without any block-sparse stuff:
    # P_1[1] * psi_0[1] should give a 2-channel tensor.
    println("\n== Direct check: contract P_1[site1] with psi_0[site1] ==")
    T_P1_site1 = ConsOps1[1][1]
    T_psi0_site1 = psi0[1]
    contracted = T_P1_site1 * T_psi0_site1
    println("contracted inds = ", inds(contracted))
    # find the link from P_1 (dim 2)
    P_link = filter(I -> dim(I) == 2 && hastags(I, "Link"), collect(inds(contracted)))
    psi_link = filter(I -> hastags(I, "Link") && !(I in P_link), collect(inds(contracted)))
    println("P_1 link found = ", P_link)
    println("psi_0 link found = ", psi_link)
    # Compute norm² for each track of P-link
    if !isempty(P_link)
        pl = first(P_link)
        for k in 1:dim(pl)
            slice = contracted * onehot(pl => k)
            n2 = norm(slice)^2
            @printf("  P-track=%d (1=I, 2=C):  ‖slice‖²=%.6e\n", k, n2)
        end
    end
end
nothing
