# How does sparse-vs-dense memory ratio scale with bond dimension?
# Build a psi at varying BDs (by projecting random MPS with different linkdims)
# and compare BlockSparse vs Dense byte count.
using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

function build_states(N::Int, link_bd::Int)
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

    psi0 = randomMPS(sites; linkdims=link_bd)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    psi_dense = copy(psi0)
    for j in 1:N
        psi_dense = replaceprime(ConsOps2[j] * psi_dense, 1 => 0)
    end
    return psi_sp, psi_dense
end

mps_bytes(M::MPS) = sum(Base.summarysize(M[i]) for i in 1:length(M))

function tensor_info(T)
    if ITensors.has_external_storage(T)
        w = ITensors.get_external_storage(T)
        if w isa SparseBackends.WrappedBlockSparse
            return (kind="BS", n_blocks=length(w.blocksparse.keys),
                    stored=length(w.blocksparse.data),
                    full=prod(w.blocksparse.dims),
                    blksize=w.blocksparse.blksize)
        end
    end
    sz = prod(map(ITensors.dim, ITensors.inds(T)))
    return (kind="D", n_blocks=1, stored=sz, full=sz, blksize=sz)
end

effective_bd(psi::MPS) = [prod(dim(I) for I in commoninds(psi[b], psi[b+1])) for b in 1:length(psi)-1]

println("=" ^ 78)
println("Sparse vs Dense memory scaling with bond dimension (N=2, 6 sites)")
println("=" ^ 78)

println("\nP*psi0 with varying psi0_linkdim:")
println("psi0_bd | eff. BD per bond  | sparse data | dense data | sp_data ratio | sparse KB | dense KB | bytes ratio | avg blksize")

for psi0_bd in [1, 4, 8, 16, 32, 64]
    psi_sp, psi_dense = build_states(2, psi0_bd)
    bd_sp = effective_bd(psi_sp)
    sp_data = sum(tensor_info(psi_sp[i]).stored for i in 1:length(psi_sp))
    dn_data = sum(tensor_info(psi_dense[i]).stored for i in 1:length(psi_dense))
    sp_bytes = mps_bytes(psi_sp)
    dn_bytes = mps_bytes(psi_dense)
    total_blocks = sum(tensor_info(psi_sp[i]).n_blocks for i in 1:length(psi_sp))
    avg_blksize = sp_data / total_blocks
    println(
        "$(lpad(psi0_bd,7)) | $(lpad(string(bd_sp),17)) | $(lpad(sp_data,11)) | $(lpad(dn_data,10)) | $(lpad(round(sp_data/dn_data;digits=2),13)) | $(lpad(round(sp_bytes/1024;digits=2),9)) | $(lpad(round(dn_bytes/1024;digits=2),8)) | $(lpad(round(sp_bytes/dn_bytes;digits=2),11)) | $(round(avg_blksize;digits=1))"
    )
end

println("\n--- Larger N=3, varying psi0_bd ---")
println("psi0_bd | sparse data | dense data | sp_data ratio | sparse KB | dense KB | bytes ratio | avg blksize")
for psi0_bd in [1, 4, 16, 64]
    psi_sp, psi_dense = build_states(3, psi0_bd)
    sp_data = sum(tensor_info(psi_sp[i]).stored for i in 1:length(psi_sp))
    dn_data = sum(tensor_info(psi_dense[i]).stored for i in 1:length(psi_dense))
    sp_bytes = mps_bytes(psi_sp)
    dn_bytes = mps_bytes(psi_dense)
    total_blocks = sum(tensor_info(psi_sp[i]).n_blocks for i in 1:length(psi_sp))
    avg_blksize = sp_data / total_blocks
    println("$(lpad(psi0_bd,7)) | $(lpad(sp_data,11)) | $(lpad(dn_data,10)) | $(lpad(round(sp_data/dn_data;digits=2),13)) | $(lpad(round(sp_bytes/1024;digits=2),9)) | $(lpad(round(dn_bytes/1024;digits=2),8)) | $(lpad(round(sp_bytes/dn_bytes;digits=2),11)) | $(round(avg_blksize;digits=1))")
end
nothing
