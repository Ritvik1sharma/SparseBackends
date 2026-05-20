# Test the opt-in preserve_bs_output pathway: same contraction, two storage
# outcomes — default returns dense ITensor, opt-in returns BS-stored ITensor.
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

function gram_right_env(psi::MPS, b::Int)
    N = length(psi)
    Mright = ITensor(1.0)
    for i in N:-1:(b+2)
        T  = psi[i]
        link_inds = filter(I -> ITensors.hastags(I, "Link"), collect(inds(T)))
        Td = dag(T)
        for I in link_inds; Td = prime(Td, I); end
        Mright = Mright * T * Td
    end
    return Mright
end

length(ARGS) < 1 && error("Usage: julia test_bs_preserve.jl <N_plaq>")
const _N_BSP = parse(Int, ARGS[1])

let
    psi = build_setup(_N_BSP)
    psi_o = ITensorMPS.orthogonalize(psi, 1)
    b = 1
    M = gram_right_env(psi_o, b)
    phi = psi_o[1] * psi_o[2]

    println("=== M and phi storage types ===")
    println("  M has external storage? ", ITensors.has_external_storage(M))
    println("  phi has external storage? ", ITensors.has_external_storage(phi))
    if ITensors.has_external_storage(M)
        println("  M storage type: ", typeof(ITensors.get_external_storage(M)))
    end
    if ITensors.has_external_storage(phi)
        println("  phi storage type: ", typeof(ITensors.get_external_storage(phi)))
    end

    println("\n=== Default contraction M*phi ===")
    Mphi_default = M * phi
    println("  Mphi_default has external storage? ", ITensors.has_external_storage(Mphi_default))
    println("  Mphi_default storage type: ", typeof(ITensors.storage(Mphi_default)))

    println("\n=== preserve_bs_output contraction ===")
    Mphi_bs = SparseBackends.contract_preserve_bs(M, phi)
    println("  Mphi_bs has external storage? ", ITensors.has_external_storage(Mphi_bs))
    if ITensors.has_external_storage(Mphi_bs)
        println("  Mphi_bs storage type: ", typeof(ITensors.get_external_storage(Mphi_bs)))
    end

    println("\n=== preserve_bs_output + template (phi as template, after replaceprime) ===")
    # First, apply replaceprime to bring Link bond primes back to 0 so output inds
    # match phi's inds.
    Mphi_default_rp = replaceprime(M * phi, 1 => 0; tags="Link")
    Mphi_bs_template = SparseBackends.contract_preserve_bs(M, phi)
    # Manually apply replaceprime to the BS result via external storage path.
    Mphi_bs_template_rp = replaceprime(Mphi_bs_template, 1 => 0; tags="Link")
    # Now recast to template = phi.
    Mphi_bs_recast = if ITensors.has_external_storage(Mphi_bs_template_rp)
        Tw_phi = ITensors.get_external_storage(phi)
        Cw_rp  = ITensors.get_external_storage(Mphi_bs_template_rp)
        Cw_recast = SparseBackends.recast_bs_to_template(Cw_rp, Tw_phi)
        ITensors._itensor_from_external_storage(Cw_recast)
    else
        Mphi_bs_template_rp
    end
    println("  Mphi_bs_recast storage: ", typeof(ITensors.get_external_storage(Mphi_bs_recast)))
    println("  phi storage:            ", typeof(ITensors.get_external_storage(phi)))
    println("  Types identical? ", typeof(ITensors.get_external_storage(Mphi_bs_recast)) == typeof(ITensors.get_external_storage(phi)))

    println("\n=== Numerical agreement ===")
    Mphi_default_d = ITensors.has_external_storage(Mphi_default) ?
        SparseBackends.to_dense_itensors_unfused(Mphi_default) : Mphi_default
    Mphi_bs_d = SparseBackends.to_dense_itensors_unfused(Mphi_bs)
    common = collect(inds(Mphi_default_d))
    Mphi_bs_d_p = permute(Mphi_bs_d, common...; allow_alias=true)
    diff = norm(Array(Mphi_default_d, common...) - Array(Mphi_bs_d_p, common...))
    println("  ||default - bs_preserved|| = $diff")
end
nothing
