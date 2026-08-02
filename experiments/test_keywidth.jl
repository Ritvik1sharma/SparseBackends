# test_keywidth.jl — does narrowing AliasedBlockSparse key tuples from
# NTuple{P,Int64} to NTuple{P,UInt8} change DMRG runtime or memory?
#
# This tests the REAL operator through the REAL kernel. No library edits are
# needed: `AliasedBlockSparse{T,N,N2,P,K,AI}` is already parameterised on the key
# type K, the contract kernels are generic in K
# (contract_aliased_dense_to_dense.jl takes `A::AliasedBlockSparse{TA,NA,NA2,PA}`
# with K unbound), and `WrappedAliasedBlockSparse.aliased` is declared with K
# unbound too — so a narrowed storage drops straight into the wrapper.
#
# What it does, per config:
#   1. build the aliased PHP exactly as bench_common.build_problem does
#   2. deep-copy it and rebuild every site's storage with UInt8 keys
#   3. ASSERT the two operators are numerically identical (element-wise over all
#      stored blocks) — a timing comparison between different operators is
#      meaningless
#   4. report Base.summarysize for both
#   5. run the SAME timed DMRG on each, warmup excluded, and compare
#
# Usage:  julia --project=SparseBackends SparseBackends/experiments/test_keywidth.jl <config_id> [seed]

using LinearAlgebra: BLAS
BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_BLAS_THREADS", string(Sys.CPU_THREADS))))

include(joinpath(@__DIR__, "bench_common.jl"))

using SparseBackends: AliasedBlockSparse, WrappedAliasedBlockSparse

# ── Rebuild one AliasedBlockSparse with a narrower key element type ──────────
# Everything except the key vector's eltype is copied verbatim: same templates,
# same alias_ids, same scalars, same order. The keys are 1-based prefix indices
# bounded by the prefix dims, so the conversion is lossless iff every prefix dim
# fits in K -- asserted, not assumed.
function narrow_keys(a::AliasedBlockSparse{T,N,N2,P,Kold,AI}, ::Type{Knew}) where {T,N,N2,P,Kold,AI,Knew}
    pmax = P == 0 ? 1 : maximum(a.dims[1:P])
    pmax <= typemax(Knew) || error(
        "narrow_keys: prefix dim $pmax exceeds $Knew capacity $(typemax(Knew))")
    newkeys = Vector{NTuple{P,Knew}}(undef, length(a.keys))
    @inbounds for i in eachindex(a.keys)
        newkeys[i] = ntuple(j -> Knew(a.keys[i][j]), Val(P))
    end
    b = AliasedBlockSparse{T,N,N2,P,Knew,AI}(
        a.dims, a.blksize, copy(a.templates), a.n_templates,
        newkeys, copy(a.alias_ids), copy(a.scalars))
    # post-construction hints (empty for a Hamiltonian, copied for safety)
    b.slice_to_template = copy(a.slice_to_template)
    b.window_slice_map  = copy(a.window_slice_map)
    return b
end

"""Return a copy of MPO `H` with every aliased site's keys narrowed to `Knew`."""
function narrow_mpo_keys(H::MPO, ::Type{Knew}) where {Knew}
    H2 = deepcopy(H)
    nsites_narrowed = 0
    for i in 1:length(H2)
        Tn = H2[i]
        (ITensors.has_external_storage(Tn) &&
         Tn.tensor.data isa WrappedAliasedBlockSparse) || continue
        w = Tn.tensor.data
        w.aliased = narrow_keys(w.aliased, Knew)
        nsites_narrowed += 1
    end
    return H2, nsites_narrowed
end

"""Element-wise identity check over the stored representation of two MPOs."""
function assert_same_operator(H1::MPO, H2::MPO)
    length(H1) == length(H2) || error("length mismatch")
    for i in 1:length(H1)
        d1 = H1[i].tensor.data; d2 = H2[i].tensor.data
        (d1 isa WrappedAliasedBlockSparse && d2 isa WrappedAliasedBlockSparse) || continue
        a1 = d1.aliased; a2 = d2.aliased
        a1.dims        == a2.dims        || error("site $i: dims differ")
        a1.blksize     == a2.blksize     || error("site $i: blksize differs")
        a1.n_templates == a2.n_templates || error("site $i: n_templates differ")
        a1.templates   == a2.templates   || error("site $i: templates differ")
        a1.alias_ids   == a2.alias_ids   || error("site $i: alias_ids differ")
        a1.scalars     == a2.scalars     || error("site $i: scalars differ")
        length(a1.keys) == length(a2.keys) || error("site $i: n_blocks differ")
        for j in eachindex(a1.keys)
            Tuple(Int.(a1.keys[j])) == Tuple(Int.(a2.keys[j])) ||
                error("site $i block $j: key differs")
        end
    end
    return true
end

function aliased_bytes(H::MPO)
    tot = 0
    for i in 1:length(H)
        d = H[i].tensor.data
        d isa WrappedAliasedBlockSparse || continue
        a = d.aliased
        tot += Base.summarysize(a.keys) + Base.summarysize(a.alias_ids) +
               Base.summarysize(a.scalars) + Base.summarysize(a.templates)
    end
    return tot
end

function main()
    config_id = get(ARGS, 1, "pxp_bd40")
    seed      = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
    cfg       = config_or_die(config_id)

    println("="^78)
    println("[keywidth] config=$config_id seed=$seed  BLAS=$(BLAS.get_num_threads())")
    println("="^78)

    _, H_i64, psi0, sites, build_s, _ = build_problem(cfg, :sb_aliased, seed)
    H_u8, nnar = narrow_mpo_keys(H_i64, UInt8)
    println("[keywidth] narrowed $nnar aliased sites; build was $(round(build_s, digits=3)) s")

    assert_same_operator(H_i64, H_u8)
    println("[keywidth] operators verified element-wise identical ✓")

    b64 = aliased_bytes(H_i64); b8 = aliased_bytes(H_u8)
    ss64 = mpo_bytes(H_i64);    ss8 = mpo_bytes(H_u8)
    @printf("[keywidth] aliased arrays : Int64 %.1f KB   UInt8 %.1f KB   (%.2fx smaller)\n",
            b64/1e3, b8/1e3, b64/max(b8,1))
    @printf("[keywidth] whole-MPO size : Int64 %.1f KB   UInt8 %.1f KB   (%.2fx smaller)\n",
            ss64/1e3, ss8/1e3, ss64/max(ss8,1))

    md = [cfg.maxdim]
    common = (nsweeps = cfg.nsweeps, maxdim = md, mindim = md,
              cutoff = cfg.cutoff, run_mode = :standard)

    # Alternate the order across two rounds so a warm-cache / node-drift bias
    # cannot be mistaken for a key-width effect.
    results = Dict{String,Vector{Float64}}("int64" => Float64[], "uint8" => Float64[])
    energies = Dict{String,Vector{Float64}}("int64" => Float64[], "uint8" => Float64[])
    for round in 1:2
        order = round == 1 ? (("int64", H_i64), ("uint8", H_u8)) :
                             (("uint8", H_u8), ("int64", H_i64))
        for (name, Hx) in order
            E, _, _, rec = timed_dmrg_ground("$config_id/$name/r$round",
                                             Hx, deepcopy(psi0); common...)
            push!(results[name],  rec["total_seconds"])
            push!(energies[name], real(E))
        end
    end

    println("\n", "="^78)
    println("[keywidth] RESULT  ($config_id, seed $seed, $(cfg.nsweeps) sweeps)")
    println("="^78)
    for name in ("int64", "uint8")
        @printf("  %-6s  runs = %s s\n", name,
                join(map(x -> string(round(x, digits=2)), results[name]), ", "))
    end
    t64 = minimum(results["int64"]); t8 = minimum(results["uint8"])
    @printf("  best-of-2:  int64 %.2f s   uint8 %.2f s   speedup %.4fx\n", t64, t8, t64/t8)
    @printf("  energies :  int64 %.10f   uint8 %.10f   |diff| %.3e\n",
            energies["int64"][1], energies["uint8"][1],
            abs(energies["int64"][1] - energies["uint8"][1]))
    println("="^78)
end

main()
