# Full-byte memory comparison (sparse ψ vs dense ψ) via Base.summarysize,
# which walks the whole object graph and includes Index metadata, block-key
# arrays, the data Vector, etc.
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
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return H_sparse, psi_sp
end

densify_mps(M) = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in M])

human_bytes(n) = n < 1024 ? "$(n) B" :
                 n < 1024^2 ? "$(round(n/1024; digits=1)) KiB" :
                 "$(round(n/1024^2; digits=2)) MiB"

let
    N = 3
    println("=== Byte-level memory: sparse ψ vs densified ψ (N=$N, after DMRG) ===")
    H_sp, psi_sp = build_setup(N)
    sweeps = Sweeps(2); setmaxdim!(sweeps, 10, 20); setmindim!(sweeps, 1); setcutoff!(sweeps, 1e-10)
    E, psi = dmrg(H_sp, psi_sp, sweeps; outputlevel=0)
    println("DMRG E = $E\n")

    psi_d = densify_mps(psi)

    # Per-tensor and total summarysize.
    println("Per-site bytes (sparse / dense / ratio):")
    tot_sp = 0; tot_d = 0
    for i in 1:length(psi)
        bs = Base.summarysize(psi[i])
        bd = Base.summarysize(psi_d[i])
        tot_sp += bs; tot_d += bd
        println("  site $i:  sparse=$(human_bytes(bs))  dense=$(human_bytes(bd))  ratio=$(round(bs/bd; digits=4))")
    end
    println("\nTotals:")
    println("  sparse ψ:  $(human_bytes(tot_sp))  ($tot_sp B)")
    println("  dense  ψ:  $(human_bytes(tot_d))   ($tot_d B)")
    println("  overall compression ratio (sparse/dense): $(round(tot_sp/tot_d; digits=4))")
    println("  fold reduction: $(round(tot_d/tot_sp; digits=2))x")
end
nothing
