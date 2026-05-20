# Inspect what's happening inside the channel-aware SVD at bond 3.
# Specifically: are channel assignments truly unique per phi block, or are
# there ambiguous ones being silently truncated?
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

let
    psi = build_setup(2)
    println("psi[3] block keys (sparse layout):")
    w3 = ITensors.get_external_storage(psi[3])
    sp3_pos = let ds = SparseBackends.dense_inds(w3); [i for i in 1:length(w3.inds) if !(w3.inds[i] in ds)]; end
    sp3 = [w3.inds[i] for i in sp3_pos]
    println("  psi[3] sparse axes: ", [(ITensors.dim(I), string(ITensors.tags(I))) for I in sp3])
    for k in w3.blocksparse.keys
        println("  ", k)
    end

    println("\npsi[4] block keys (sparse layout):")
    w4 = ITensors.get_external_storage(psi[4])
    sp4_pos = let ds = SparseBackends.dense_inds(w4); [i for i in 1:length(w4.inds) if !(w4.inds[i] in ds)]; end
    sp4 = [w4.inds[i] for i in sp4_pos]
    println("  psi[4] sparse axes: ", [(ITensors.dim(I), string(ITensors.tags(I))) for I in sp4])
    for k in w4.blocksparse.keys
        println("  ", k)
    end

    # Form phi and inspect
    phi = psi[3] * psi[4]
    wphi = ITensors.get_external_storage(phi)
    sp_phi_pos = let ds = SparseBackends.dense_inds(wphi); [i for i in 1:length(wphi.inds) if !(wphi.inds[i] in ds)]; end
    sp_phi = [wphi.inds[i] for i in sp_phi_pos]
    println("\nphi sparse axes: ", [(ITensors.dim(I), string(ITensors.tags(I))) for I in sp_phi])
    println("phi has $(length(wphi.blocksparse.keys)) blocks")
    for k in wphi.blocksparse.keys
        println("  ", k)
    end
end
nothing
