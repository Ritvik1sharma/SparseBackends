# Microbenchmark: is recast_bs_to_template dominated by ALLOCATION (GC pressure)
# or by COMPUTE (densify+permute+scatter)?
#
# If alloc-dominated → path 1 (pre-allocated buffers) will recover most of the
# 30% recast cost in the sweep profile.
# If compute-dominated → path 1 is a small win; need path 2 (in-kernel filter)
# or live with it.
#
# Method: build a realistic (N=4 mid-bond, ~maxdim=40) BS×Dense recast scenario,
# warm up, then time recast_bs_to_template in three regimes:
#   (a) baseline: as-is (allocates each call)
#   (b) alloc-free upper bound: time only the to_dense+permutedims+scatter loop
#       with PRE-ALLOCATED output buffers — measures the irreducible compute.
#   (c) gc-disabled: same as (a) but with GC.enable(false) around the loop —
#       isolates allocation cost from GC-pause cost.

using SparseBackends, ITensors, ITensorMPS
using Random, Printf

include("utils.jl")

# ---- Build a realistic scenario: N=4 maxdim=40, mid-bond ----
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
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return H_sparse, psi_sp
end

length(ARGS) < 1 && error("Usage: julia test_recast_microbench.jl <N_plaq> [ITERS]")
N_plaq = parse(Int, ARGS[1])
println("Building N=$N_plaq setup ...")
H_sp, psi = build_setup(N_plaq)

# Grow bond dims to ~40 by running a couple of cheap dense sweeps first?
# Simpler: just orthogonalize + grab a mid-bond phi. The setup already has
# some non-trivial bond dim from the projector construction.
psi = ITensorMPS.orthogonalize(psi, 1)

# Pick a representative mid bond
b = N_plaq  # mid bond: for 2N+2 sites there are 2N+1 bonds; N_plaq is a reasonable mid
println("Using bond b=$b")

cache = SparseBackends.init_gram_cache(psi)
Lgram = SparseBackends.get_left_gram(cache, b)
phi   = psi[b] * psi[b+1]

println("phi inds: ", inds(phi))
println("phi nnz : ", length(ITensors.get_external_storage(phi).blocksparse.data))

# Build a BS Linv that we will repeatedly contract with phi to feed recast.
_, Linv_L_bs = SparseBackends.build_minv_half_pair_bs(Lgram; phi_template=phi)
println("Linv_L_bs inds: ", inds(Linv_L_bs))

# Produce the C tensor (Linv_L * phi) ONCE to get a representative Cw/Tw pair.
# This is the actual output the kernel produces, before recast.
Linv_w = ITensors.get_external_storage(Linv_L_bs)
phi_w  = ITensors.get_external_storage(phi)
# Use the same code path the production hits: contract(Aw, Bw; preserve_bs_output=true)
Cw_proto = SparseBackends.contract(Linv_w, phi_w; preserve_bs_output=true)
@assert Cw_proto isa SparseBackends.WrappedBlockSparse
Tw = phi_w  # template — the recast aligns to phi's keys
println("Cw blocks: ", length(Cw_proto.blocksparse.keys),
        "  Tw blocks: ", length(Tw.blocksparse.keys),
        "  Tw blksize: ", Tw.blocksparse.blksize,
        "  payload total: ", length(Tw.blocksparse.data))

# Convenience: a closure that rebuilds Cw fresh each iter (so we time JUST recast,
# not the contract). We'll also test re-running the full chain for context.
function fresh_Cw()
    SparseBackends.contract(Linv_w, phi_w; preserve_bs_output=true)::SparseBackends.WrappedBlockSparse
end

# ---- (a) baseline: recast_bs_to_template, allocating ----
function bench_recast_baseline(Cw, Tw, iters)
    # Warm
    SparseBackends.recast_bs_to_template(Cw, Tw)
    GC.gc()
    t0 = time_ns(); allocs0 = Base.gc_num()
    for _ in 1:iters
        Cw_fresh = fresh_Cw()
        SparseBackends.recast_bs_to_template(Cw_fresh, Tw)
    end
    t1 = time_ns(); allocs1 = Base.gc_num()
    elapsed = (t1 - t0)/1e9
    gc_diff = Base.GC_Diff(allocs1, allocs0)
    return (elapsed=elapsed,
            bytes=gc_diff.allocd,
            gc_time=gc_diff.total_time/1e9,
            num_gc=gc_diff.pause)
end

# Also time JUST recast on the same Cw repeatedly (no contract). The recast
# mutates nothing externally; it returns a NEW WrappedBlockSparse.
function bench_recast_only(Cw, Tw, iters)
    SparseBackends.recast_bs_to_template(Cw, Tw)
    GC.gc()
    t0 = time_ns(); allocs0 = Base.gc_num()
    for _ in 1:iters
        SparseBackends.recast_bs_to_template(Cw, Tw)
    end
    t1 = time_ns(); allocs1 = Base.gc_num()
    elapsed = (t1 - t0)/1e9
    gc_diff = Base.GC_Diff(allocs1, allocs0)
    return (elapsed=elapsed,
            bytes=gc_diff.allocd,
            gc_time=gc_diff.total_time/1e9,
            num_gc=gc_diff.pause)
end

# ---- (b) alloc-free upper bound: pre-allocated buffers reused ----
# Replicates the body of recast_bs_to_template but with the three big buffers
# (data_dense, data_perm, new_data) allocated ONCE outside the loop.
function bench_recast_prealloc(Cw0, Tw, iters)
    c_inds = collect(Cw0.inds); t_inds = collect(Tw.inds)
    N = length(t_inds)
    perm = ntuple(i -> findfirst(==(t_inds[i]), c_inds), N)
    bs_t = Tw.blocksparse
    P_t = length(bs_t.keys[1])         # number of sparse axes
    N2t = N - P_t
    dims_t = bs_t.dims
    blksize_t = bs_t.blksize
    suffix_dims = ntuple(i -> dims_t[P_t + i], N2t)
    suffix_CI = CartesianIndices(suffix_dims)
    suffix_LI = LinearIndices(suffix_dims)

    # Pre-allocate: dense + permuted dense + output payload
    TC = eltype(Cw0.blocksparse.data)
    dense_dims_C = ntuple(i -> ITensors.dim(c_inds[i]), N)
    dense_dims_T = ntuple(i -> ITensors.dim(t_inds[i]), N)
    data_dense_buf = Array{TC}(undef, dense_dims_C)
    data_perm_buf  = Array{TC}(undef, dense_dims_T)
    new_data_buf   = Vector{TC}(undef, length(bs_t.data))
    new_keys_buf   = copy(bs_t.keys)
    new_ids_buf    = copy(bs_t.ids)

    # Inline in-place densify (mimic to_dense but write into data_dense_buf).
    bs_c = Cw0.blocksparse
    P_c  = length(bs_c.keys) == 0 ? 0 : length(bs_c.keys[1])
    N2c  = N - P_c
    blksize_c = bs_c.blksize
    suffix_dims_c = ntuple(i -> dense_dims_C[P_c + i], N2c)
    suffix_CI_c = CartesianIndices(suffix_dims_c)
    suffix_LI_c = LinearIndices(suffix_dims_c)

    function densify_into!(buf, bs)
        fill!(buf, zero(TC))
        @inbounds for i in eachindex(bs.keys)
            prefix = bs.keys[i]
            bid    = bs.ids[i]
            base   = (bid - 1) * blksize_c
            for sCI in suffix_CI_c
                suffix = Tuple(sCI)::NTuple{N2c,Int}
                full = SparseBackends._full_index(prefix, suffix, Val(N))
                lin = suffix_LI_c[sCI]
                buf[full...] = bs.data[base + lin]
            end
        end
    end

    function recast_into!(Cw)
        densify_into!(data_dense_buf, Cw.blocksparse)
        permutedims!(data_perm_buf, data_dense_buf, perm)
        @inbounds for i in eachindex(new_keys_buf)
            prefix = new_keys_buf[i]
            bid    = new_ids_buf[i]
            base   = (bid - 1) * blksize_t
            for sCI in suffix_CI
                suffix = Tuple(sCI)::NTuple{N2t,Int}
                full = SparseBackends._full_index(prefix, suffix, Val(N))
                lin = suffix_LI[sCI]
                new_data_buf[base + lin] = data_perm_buf[full...]
            end
        end
        return new_data_buf
    end

    # Warm
    recast_into!(Cw0)
    GC.gc()
    t0 = time_ns(); allocs0 = Base.gc_num()
    for _ in 1:iters
        recast_into!(Cw0)
    end
    t1 = time_ns(); allocs1 = Base.gc_num()
    elapsed = (t1 - t0)/1e9
    gc_diff = Base.GC_Diff(allocs1, allocs0)
    return (elapsed=elapsed,
            bytes=gc_diff.allocd,
            gc_time=gc_diff.total_time/1e9,
            num_gc=gc_diff.pause)
end

ITERS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500

println("\n--- baseline (recast + fresh contract per call) ---")
b0 = bench_recast_baseline(Cw_proto, Tw, ITERS)
@printf("  iters=%d  total=%.3fs  per=%.3fµs  alloc=%.2f MB  gc=%.3fs (%d pauses)\n",
        ITERS, b0.elapsed, 1e6*b0.elapsed/ITERS, b0.bytes/2^20, b0.gc_time, b0.num_gc)

println("\n--- recast-only (allocates each call, same Cw) ---")
b1 = bench_recast_only(Cw_proto, Tw, ITERS)
@printf("  iters=%d  total=%.3fs  per=%.3fµs  alloc=%.2f MB  gc=%.3fs (%d pauses)\n",
        ITERS, b1.elapsed, 1e6*b1.elapsed/ITERS, b1.bytes/2^20, b1.gc_time, b1.num_gc)

println("\n--- pre-allocated buffers (alloc-free upper bound) ---")
try
    b2 = bench_recast_prealloc(Cw_proto, Tw, ITERS)
    @printf("  iters=%d  total=%.3fs  per=%.3fµs  alloc=%.2f MB  gc=%.3fs (%d pauses)\n",
            ITERS, b2.elapsed, 1e6*b2.elapsed/ITERS, b2.bytes/2^20, b2.gc_time, b2.num_gc)
    @printf("\n  ALLOC-DOMINATED FRACTION = 1 - prealloc/alloc = %.1f%%\n",
            100*(1 - b2.elapsed/b1.elapsed))
catch e
    println("  prealloc variant failed: ", e)
    println("  (likely to_dense! is missing — fall back to estimating from gc_time)")
    @printf("\n  GC time fraction of recast-only = %.1f%%  (lower-bound on alloc-dominated %%)\n",
            100*b1.gc_time/b1.elapsed)
end

nothing
