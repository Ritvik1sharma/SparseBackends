using SparseBackends, ITensors, ITensorMPS, Random
include("utils.jl")

function run_test()
    Random.seed!(42)
    N=2; sites = siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N; os += "Sx", 2*j-1, "Sx", 2*j+2; os += "Sy", 2*j, "Sy", 2*j+1; end
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += 0.5, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j, "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    ConsOps2 = [MPO(os2[j], sites) for j in 1:N]
    function mulMPO(A,B); Bp = prime(B,"Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H = MPO(os, sites)
    H1 = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0 = random_mps(sites)
    for j in 1:N; psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0); normalize!(psi0); end
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)

    E, psi_out, _, terr = dmrg(H_sparse, psi_sp; nsweeps=15, maxdim=[300], mindim=[1], cutoff=1e-12,
        target_energy=nothing, use_early_exit=false, outputlevel=0)
    println("Sparse with maxdim=300, 15 sweeps:")
    println("  E_dmrg = ", real(E))
    E_H_check = real(inner(psi_out', H, psi_out))
    println("  E from inner(psi',H,psi) = ", E_H_check)
    println("  ‖psi‖² = ", real(inner(psi_out, psi_out)))
    println("Expected dense ≈ -3.7291478821")
end
run_test()
