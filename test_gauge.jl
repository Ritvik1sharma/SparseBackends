# Check whether the new bin_by_right_link_only SVD preserves the state's gauge
# invariance. Energy ⟨psi|H|psi⟩ should be unchanged after orthogonalize.
using SparseBackends, ITensors, ITensorMPS, Random
include("utils.jl")

function run_test(N::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
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

    println("Initial: norm² = ", inner(psi_sp, psi_sp), "  ⟨H⟩ = ", inner(psi_sp', H, psi_sp))

    for j in [1, 2, 3, 4, 5, 6]
        psi = copy(psi_sp)
        orthogonalize!(psi, j)
        types = [ITensors.has_external_storage(psi[i]) ? "BS" : "D" for i in 1:length(psi)]
        n2 = real(inner(psi, psi))
        Hexp = real(inner(psi', H, psi))
        println("orth_to_$j: norm² = ", round(n2; digits=8),
                "  ⟨H⟩ = ", round(Hexp; digits=8),
                "  isortho=", isortho(psi),
                "  oc=", isortho(psi) ? orthocenter(psi) : "?",
                "  types=", types)
    end
end
length(ARGS) < 1 && error("Usage: julia test_gauge.jl <N_plaq>")
run_test(parse(Int, ARGS[1]))
