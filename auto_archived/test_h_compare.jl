using SparseBackends, Random, ITensors, ITensorMPS
include("utils.jl")

function sandwich_mpo(P::MPO, H::MPO)
    H1   = contract(P'', H', :coo, :dense)
    H_eff = contract(P, H1, :coo, :blocksparse)
    replaceprime(H_eff, 3 => 1)
end

function sandwich_mpo_dense(P::MPO, H::MPO)
    H1   = contract(P'', H'; is_ctn_compression=true)
    H_eff = contract(P, H1; is_ctn_compression=true)
    replaceprime(H_eff, 3 => 1)
end

function mulMPO(A::MPO, B::MPO)
    Bp = prime(B, "Site")
    C  = contract(A, Bp, :coo, :coo)
    replaceprime(C, 2 => 1)
end

function multiplyVecMPOtoMPO(vec::Vector{MPO})
    result = vec[1]
    for j in 2:length(vec)
        result = mulMPO(result, vec[j])
    end
    result
end

function multiplydense(vec::Vector{MPO})
    result = vec[1]
    for j in 2:length(vec)
        Bp = prime(vec[j], "Site")
        result = replaceprime(contract(result, Bp; is_ctn_compression=true), 2 => 1)
    end
    result
end

function clean!(op::MPO; tol=1e-12)
    for j in 1:length(op)
        T = op[j]
        A = array(T)
        for i in eachindex(A)
            abs(A[i])         < tol && (A[i] = 0.0)
            abs(A[i] - 1.0)  < tol && (A[i] = 1.0)
            abs(A[i] + 1.0)  < tol && (A[i] = -1.0)
            abs(A[i] - 0.5)  < tol && (A[i] = 0.5)
            abs(A[i] + 0.5)  < tol && (A[i] = -0.5)
        end
        op[j] = ITensor(A, inds(T)...)
    end
    op
end

length(ARGS) < 1 && error("Usage: julia test_h_compare.jl <N_plaq>")

let
    Random.seed!(42)
    N          = parse(Int, ARGS[1])
    spin       = 3
    spin_sector = 1.0
    states     = 2 * N + 2
    sites      = siteinds("S=1", states)

    os = OpSum()
    for j in 1:N+1
        os += "Sz", 2*j-1, "Sz", 2*j
    end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end

    os2 = OpSum[]
    for j in 1:N
        coeff = 0.5
        temp  = OpSum()
        temp += coeff, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        temp += spin_sector*coeff, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j, "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, temp)
    end

    ConsOps1 = [begin
        op = MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])
        clean!(op)
    end for j in 1:N]

    ConsOps2      = [MPO(os2[j], sites) for j in 1:N]
    ConsOpsCombined  = multiplyVecMPOtoMPO(ConsOps1)
    ConsOpsCombined2 = multiplydense(ConsOps1)

    H       = MPO(os, sites)
    H_sparse = sandwich_mpo(ConsOpsCombined, copy(H))
    H_dense  = sandwich_mpo_dense(ConsOpsCombined2, copy(H))

    # Compare H_sparse and H_dense site by site
    println("=== Comparing H_sparse vs H_dense ===")
    for j in 1:length(H_sparse)
        hs = H_sparse[j]
        hd = H_dense[j]
        if ITensors.has_external_storage(hs)
            hs_dense = SparseBackends.to_dense_itensors(hs)
        else
            hs_dense = hs
        end
        # Check if inds match
        if issetequal(inds(hs_dense), inds(hd))
            diff = norm(hs_dense - hd)
            println("Site $j: |H_sparse - H_dense| = $diff")
        else
            println("Site $j: inds mismatch!")
            println("  H_sparse inds: ", inds(hs_dense))
            println("  H_dense  inds: ", inds(hd))
        end
    end

    # Test inner products with initial state
    psi_old = random_mps(sites)
    psi0    = copy(psi_old)
    for j in 1:N
        psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0)
        normalize!(psi0)
    end

    E_sparse_init = inner(psi0', H_sparse, psi0)
    E_dense_init  = inner(psi0', H_dense,  psi0)
    E_orig_init   = inner(psi0', H,        psi0)
    println("\n=== Initial state energies ===")
    println("inner(psi0, H_sparse, psi0) = $E_sparse_init")
    println("inner(psi0, H_dense,  psi0) = $E_dense_init")
    println("inner(psi0, H,        psi0) = $E_orig_init")

    psi_sp = replaceprime(contract(ConsOpsCombined, copy(psi0), :coo, :dense), 1 => 0)
    println("\nNorm of psi_sp: ", norm(psi_sp))
    E_sp_with_H = inner(psi_sp', H, psi_sp)
    println("inner(psi_sp, H, psi_sp) = $E_sp_with_H")

    E_sp_with_Hs = inner(psi_sp', H_sparse, psi_sp)
    println("inner(psi_sp, H_sparse, psi_sp) = $E_sp_with_Hs")
end
nothing
