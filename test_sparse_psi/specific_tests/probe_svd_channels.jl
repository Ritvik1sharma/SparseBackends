# Diagnostic probe: for each bond of psi_sp, snapshot phi = psi[b]*psi[b+1] and
# run blocksparse_svd_channel_aware with SPARSE_SVD_DIAG=1 to see per-channel
# m_c, n_c, aggregation counts. Tells us how skewed the channel structure is
# at each bond and whether the iso_cap binds for this projector geometry.
#
# Usage:  SPARSE_SVD_DIAG=1 julia --project=.. probe_svd_channels.jl [N]

using SparseBackends, ITensors, ITensorMPS
using Random
include("../utils.jl")

ENV["SPARSE_SVD_DIAG"] = "1"  # force-on for this run

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
    H        = MPO(os, sites)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return H, psi_sp
end

let
    N_plaq = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
    println("=== Building setup for N=$N_plaq plaquettes ===")
    H, psi = build_setup(N_plaq)
    println("System: $(length(psi)) sites; orthogonalizing to bond 1...")
    psi = orthogonalize!(psi, 1)

    println("\n=== Probing per-bond channel structure (ortho=left, maxdim=40) ===")
    for b in 1:length(psi)-1
        phi = psi[b] * psi[b+1]
        if !ITensors.has_external_storage(phi)
            println("[bond $b] phi is plain dense — skipping (no BS channel structure to probe)")
            continue
        end
        println("\n--- bond $b ---")
        try
            L, R, spec = SparseBackends.itensor_blocksparse_svd_channel_aware(
                phi, psi[b], psi[b+1];
                ortho="left", maxdim=40, mindim=1, cutoff=0.0,
            )
            psi[b]   = L
            psi[b+1] = R
        catch e
            println("[bond $b] channel-aware SVD failed: ", sprint(showerror, e))
        end
    end
end
nothing
