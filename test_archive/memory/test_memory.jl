# Compare memory footprint of psi as dense MPS vs BlockSparse-storage MPS
# at the initial constraint-projected state.
using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

function build_psi(N::Int)
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
    ConsOps2 = [MPO(os2[j], sites) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end

    psi0 = random_mps(sites)
    for j in 1:N; psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0); normalize!(psi0); end
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)

    # Dense reference: build psi_dense via standard contraction + truncation
    psi_dense = copy(psi0)
    for j in 1:N
        psi_dense = replaceprime(ConsOps2[j] * psi_dense, 1 => 0)
        normalize!(psi_dense)
    end
    return psi_sp, psi_dense, P_sparse
end

function tensor_bytes(T::ITensor)
    return Base.summarysize(T)
end

function mps_bytes(M::MPS)
    return sum(tensor_bytes(M[i]) for i in 1:length(M))
end

function effective_storage_dim(T::ITensor)
    # For BlockSparse: number of stored values + overhead per block
    if ITensors.has_external_storage(T)
        w = ITensors.get_external_storage(T)
        if w isa SparseBackends.WrappedBlockSparse
            n_blocks = length(w.blocksparse.keys)
            n_data   = length(w.blocksparse.data)
            full_size = prod(w.blocksparse.dims)
            return (storage="BlockSparse", n_blocks=n_blocks, stored=n_data, full=full_size, ratio=n_data/full_size)
        end
    end
    sz = prod(size(T))
    return (storage="Dense", n_blocks=1, stored=sz, full=sz, ratio=1.0)
end

println("=" ^ 70)
println("Memory comparison: dense MPS vs BlockSparse-storage MPS")
println("(constraint-projected initial state, before DMRG)")
println("=" ^ 70)

for N in [2, 3, 4]
    println("\n--- N=$N plaquettes ($(2*N+2) sites, S=1) ---")
    psi_sp, psi_dense, _ = build_psi(N)
    sp_bytes  = mps_bytes(psi_sp)
    dn_bytes  = mps_bytes(psi_dense)
    println("  link dims sparse: ", [dim(linkind(psi_sp, i)) for i in 1:length(psi_sp)-1])
    println("  link dims dense : ", [dim(linkind(psi_dense, i)) for i in 1:length(psi_dense)-1])
    println("  total bytes  sparse psi : $(round(sp_bytes/1024, digits=2)) KB")
    println("  total bytes  dense  psi : $(round(dn_bytes/1024, digits=2)) KB")
    println("  ratio (sparse/dense)    : $(round(sp_bytes/dn_bytes; digits=3))")
    println("  per-tensor breakdown:")
    for i in 1:length(psi_sp)
        info_sp = effective_storage_dim(psi_sp[i])
        info_dn = effective_storage_dim(psi_dense[i])
        full_sp = map(ITensors.dim, ITensors.inds(psi_sp[i]))
        full_dn = map(ITensors.dim, ITensors.inds(psi_dense[i]))
        println("    site $i: sparse $(info_sp.storage)  dims=$full_sp  blocks=$(info_sp.n_blocks)  stored/full=$(info_sp.stored)/$(info_sp.full) ($(round(info_sp.ratio*100;digits=1))%)")
        println("           : dense  Dense       dims=$full_dn  stored/full=$(info_dn.stored)/$(info_dn.full)")
    end
end

# Per-sweep tracking during DMRG
println("\n" * "=" ^ 70)
println("Per-sweep memory tracking (N=2, maxdim=20, 5 sweeps)")
println("=" ^ 70)
let
    Random.seed!(42)
    N = 2
    sites = siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    H = MPO(os, sites)
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
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    psi0 = random_mps(sites)
    for j in 1:N; psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0); normalize!(psi0); end
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    psi_dense = copy(psi0)
    for j in 1:N
        psi_dense = replaceprime(ConsOps2[j] * psi_dense, 1 => 0)
        normalize!(psi_dense)
    end
    H1 = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    println("\n[Sparse-psi DMRG, dense fallback at SVD]")
    psi_track = copy(psi_sp)
    println("  initial bytes: $(round(mps_bytes(psi_track)/1024, digits=2)) KB")
    for sw in 1:5
        E, psi_track, sweeps, terr = dmrg(H_sparse, psi_track;
            nsweeps=1, maxdim=[20], mindim=[20], cutoff=1e-12,
            target_energy=nothing, use_early_exit=false, outputlevel=0)
        bytes_after = mps_bytes(psi_track)
        types = [ITensors.has_external_storage(psi_track[i]) ? "BS" : "D" for i in 1:length(psi_track)]
        println("  sweep $sw: E=$(round(real(E); sigdigits=8))  bytes=$(round(bytes_after/1024, digits=2)) KB  types=$types")
    end

    println("\n[Dense-psi DMRG]")
    psi_track = copy(psi_dense)
    println("  initial bytes: $(round(mps_bytes(psi_track)/1024, digits=2)) KB")
    for sw in 1:5
        E, psi_track, sweeps, terr = dmrg(H, psi_track;
            nsweeps=1, maxdim=[20], mindim=[20], cutoff=1e-12,
            target_energy=nothing, use_early_exit=false, outputlevel=0)
        bytes_after = mps_bytes(psi_track)
        println("  sweep $sw: E=$(round(real(E); sigdigits=8))  bytes=$(round(bytes_after/1024, digits=2)) KB")
    end
end
nothing
