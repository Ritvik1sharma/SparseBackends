# FAIR memory comparison: same physical state, two different storage formats.
# Both psi_sp and psi_dense are P * psi0_random (single projection).
using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

function build_states(N::Int)
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
    # Sparse: single projection via :coo, :dense
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    # Dense: single projection via apply (default densitymatrix algorithm, with default truncation)
    psi_dense = copy(psi0)
    for j in 1:N
        psi_dense = replaceprime(ConsOps2[j] * psi_dense, 1 => 0)
    end
    return psi_sp, psi_dense
end

function tensor_bytes(T)
    return Base.summarysize(T)
end
mps_bytes(M::MPS) = sum(tensor_bytes(M[i]) for i in 1:length(M))

function effective_bd(psi::MPS)
    return [prod(dim(I) for I in commoninds(psi[b], psi[b+1])) for b in 1:length(psi)-1]
end

function tensor_storage_info(T::ITensor)
    if ITensors.has_external_storage(T)
        w = ITensors.get_external_storage(T)
        if w isa SparseBackends.WrappedBlockSparse
            n_blocks = length(w.blocksparse.keys)
            n_data   = length(w.blocksparse.data)
            full     = prod(w.blocksparse.dims)
            return (kind="BS", dims=w.blocksparse.dims, n_blocks=n_blocks, stored=n_data, full=full)
        end
    end
    sz = prod(map(ITensors.dim, ITensors.inds(T)))
    return (kind="D", dims=tuple(map(ITensors.dim, ITensors.inds(T))...), n_blocks=1, stored=sz, full=sz)
end

println("=" ^ 78)
println("FAIR memory comparison: psi_sp = contract(P, psi0, :coo, :dense)")
println("                        psi_dense = apply(ConsOps2..., psi0)")
println("(same operation, two different storage formats)")
println("=" ^ 78)

for N in [2, 3, 4]
    println("\n--- N=$N plaquettes ($(2*N+2) sites) ---")
    psi_sp, psi_dense = build_states(N)
    bd_sp = effective_bd(psi_sp)
    bd_dn = effective_bd(psi_dense)
    println("  effective BD (sparse): $bd_sp")
    println("  effective BD (dense):  $bd_dn")
    sp_total_data = sum(tensor_storage_info(psi_sp[i]).stored   for i in 1:length(psi_sp))
    dn_total_data = sum(tensor_storage_info(psi_dense[i]).stored for i in 1:length(psi_dense))
    sp_full_dim   = sum(tensor_storage_info(psi_sp[i]).full     for i in 1:length(psi_sp))
    println("  sparse total stored values: $sp_total_data  (out of $sp_full_dim possible = $(round(100*sp_total_data/sp_full_dim;digits=1))%)")
    println("  dense  total stored values: $dn_total_data")
    println("  sparse data-only ratio (sp_data / dn_data): $(round(sp_total_data/dn_total_data; digits=2))")
    sp_bytes = mps_bytes(psi_sp)
    dn_bytes = mps_bytes(psi_dense)
    println("  total bytes (sparse): $(round(sp_bytes/1024; digits=2)) KB  ← incl. block keys/ids overhead")
    println("  total bytes (dense):  $(round(dn_bytes/1024; digits=2)) KB")
    println("  bytes ratio (sparse/dense): $(round(sp_bytes/dn_bytes; digits=2))")
    println("\n  per-tensor:")
    for i in 1:length(psi_sp)
        si = tensor_storage_info(psi_sp[i])
        di = tensor_storage_info(psi_dense[i])
        println("    site $i: sparse dims=$(si.dims) blocks=$(si.n_blocks) stored=$(si.stored)/$(si.full)  | dense dims=$(di.dims) stored=$(di.stored)")
    end
end
nothing
