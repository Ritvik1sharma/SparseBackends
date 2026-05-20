# Audit bond dimensions and storage layout to clarify the memory comparison.
using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

let
    Random.seed!(42)
    N = 2
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
    ConsOps2 = [MPO(os2[j], sites) for j in 1:N]

    println("--- Individual ConsOps (per-plaquette projector, sparse-built) ---")
    for j in 1:N
        bd = [dim(linkind(ConsOps1[j], i)) for i in 1:length(ConsOps1[j])-1]
        println("  ConsOps1[$j] (sparse) link dims: $bd")
    end
    for j in 1:N
        bd = [dim(linkind(ConsOps2[j], i)) for i in 1:length(ConsOps2[j])-1]
        println("  ConsOps2[$j] (dense)  link dims: $bd")
    end

    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end

    function multiplydense(vec)
        r = vec[1]
        for j in 2:length(vec)
            Bp = prime(vec[j], "Site")
            r = replaceprime(contract(r, Bp; is_ctn_compression=true), 2 => 1)
        end
        r
    end
    P_dense = multiplydense(ConsOps1)

    println("\n--- Combined projector P = ConsOps1[1] * ConsOps1[2] ---")
    println("  P_sparse link dims (per-leg): ", [dim(linkind(P_sparse, i)) for i in 1:length(P_sparse)-1])
    println("  P_dense  link dims (per-leg): ", [dim(linkind(P_dense, i)) for i in 1:length(P_dense)-1])

    println("\n--- P_sparse per-tensor full inds (showing doubled-link structure) ---")
    for i in 1:length(P_sparse)
        all_inds = ITensors.inds(P_sparse[i])
        println("  P_sparse[$i] inds=$([dim(I) for I in all_inds])")
    end

    psi0 = random_mps(sites)
    println("\n--- Starting psi0 ---")
    println("  link dims: ", [dim(linkind(psi0, i)) for i in 1:length(psi0)-1])

    for j in 1:N; psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0); normalize!(psi0); end
    psi_dense = psi0

    Random.seed!(42)
    psi0 = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)

    println("\n--- After projection ---")
    println("  psi_dense link dims: ", [dim(linkind(psi_dense, i)) for i in 1:length(psi_dense)-1])
    println("  psi_sp    link dims: ", [dim(linkind(psi_sp, i)) for i in 1:length(psi_sp)-1])
    println("  (note: linkind returns ONE link; sparse psi has DOUBLED links per bond)")

    println("\n--- psi_sp per-tensor full inds (doubled-link visible) ---")
    for i in 1:length(psi_sp)
        all_inds = ITensors.inds(psi_sp[i])
        all_dims = [dim(I) for I in all_inds]
        all_tags = [repr(tags(I)) for I in all_inds]
        println("  psi_sp[$i] inds_dims=$all_dims  tags=$all_tags")
    end

    println("\n--- psi_dense per-tensor full inds ---")
    for i in 1:length(psi_dense)
        all_inds = ITensors.inds(psi_dense[i])
        all_dims = [dim(I) for I in all_inds]
        all_tags = [repr(tags(I)) for I in all_inds]
        println("  psi_dense[$i] inds_dims=$all_dims  tags=$all_tags")
    end

    # Total effective BD per bond: product of all link dims sharing that bond.
    println("\n--- Effective bond dimension per link (totaled across doubled axes) ---")
    function effective_bond_dims(psi::MPS)
        bds = Int[]
        for b in 1:length(psi)-1
            inds_b = ITensors.inds(psi[b])
            inds_bp1 = ITensors.inds(psi[b+1])
            common = collect(commoninds(psi[b], psi[b+1]))
            push!(bds, prod(dim(I) for I in common))
        end
        return bds
    end
    println("  psi_dense effective BD per bond: ", effective_bond_dims(psi_dense))
    println("  psi_sp    effective BD per bond: ", effective_bond_dims(psi_sp))
end
nothing
