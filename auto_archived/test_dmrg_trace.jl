# Track psi types through a single DMRG sweep
using SparseBackends, ITensors, ITensorMPS
using Random
using KrylovKit: eigsolve

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
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H = MPO(os, sites)
    H1 = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0 = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return H_sparse, psi_sp
end

stor(T) = ITensors.has_external_storage(T) ? "BS" : "D"
function describe(psi, label)
    sparsity_info = []
    total_data = 0
    total_full = 0
    for i in 1:length(psi)
        T = psi[i]
        if ITensors.has_external_storage(T)
            w = ITensors.get_external_storage(T)
            if w isa SparseBackends.WrappedBlockSparse
                n_blocks = length(w.blocksparse.keys)
                n_data = length(w.blocksparse.data)
                full = prod(w.blocksparse.dims)
                push!(sparsity_info, "BS($n_blocks blk, $n_data/$full)")
                total_data += n_data
                total_full += full
            end
        else
            sz = prod(map(ITensors.dim, ITensors.inds(T)))
            push!(sparsity_info, "D($sz)")
            total_data += sz
            total_full += sz
        end
    end
    ratio = total_full > 0 ? total_data/total_full : 0.0
    bytes = sum(Base.summarysize(psi[i]) for i in 1:length(psi))
    println("  [$label] storage = $sparsity_info  total_stored=$total_data sparsity=$(round(ratio*100;digits=1))%  bytes=$(round(bytes/1024;digits=2))KB")
end

let
    H_sparse, psi = build_setup(2)
    println("Initial:")
    describe(psi, "init")

    println("\nManually orthogonalize to position 1:")
    psi = ITensorMPS.orthogonalize(psi, 1)
    describe(psi, "after ortho_to_1")

    println("\nRun 1 DMRG sweep:")

    let psi = orthogonalize(psi, 1)
        PH_init = ProjMPO(H_sparse)
        PH_init = position!(PH_init, psi, 1)
        phi1    = psi[1] * psi[2]
        println("  bond 1 phi inds: ", inds(phi1))
        vals, vecs = eigsolve(PH_init, phi1, 1, :SR;
                            ishermitian=true,
                            tol=1e-14,
                            krylovdim=3,
                            maxiter=1,
                            verbosity=0)
        println("  eigsolve at bond 1 → energy = ", vals[1])

        spec = replacebond!(psi, 1, vecs[1];
                            ortho="left",
                            maxdim=81,
                            mindim=81,
                            cutoff=1e-12,
                            normalize=true)
        println("  replacebond! at bond 1 done; truncerr=", spec.truncerr,
                "  new psi[1] inds=", inds(psi[1]),
                "  new psi[2] inds=", inds(psi[2]))

        # Fresh PH on each side, positioned at every bond, identical psi
        describe(psi, "after manual orth_to_2")
    end


    E, psi, _, _ = dmrg(H_sparse, psi;
        nsweeps=1, maxdim=[20], mindim=[1], cutoff=1e-12,
        target_energy=nothing, use_early_exit=false, outputlevel=0)
    println("  E after sweep 1 = $E")
    describe(psi, "after sweep 1")
end
nothing
