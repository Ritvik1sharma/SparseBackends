# aliased_helpers.jl
#
# Shared utilities for aliased BlockSparse PHP tests.
# Include from a test driver with:  include("aliased_helpers.jl")
#
# Provides:
#   sparse_prefix_inds          — extract sparse-prefix Index set from an aliased ITensor
#   _materialize_combiner_dense — build a dense combiner tensor (fallback for fuse)
#   _fuse_aliased_strands       — key-rewrite fuse of sparse-prefix strands (no data motion)
#   fuse_sparse_links!          — apply fuse to all bonds of an aliased MPO
#   prepermute_aliased_mpo!     — one-shot permute of dense tails so kernel permute_A = identity
#   report_aliased_footprint    — per-site memory table (dense / BS / aliased) for an aliased MPO
#   run_dmrg_ground             — warmup + timed ground-state DMRG + timer reports
#   run_dmrg_excited            — warmup + timed excited-state DMRG + timer reports

using SparseBackends
using ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer

# ─────────────────────────────────────────────────────────────────────────────
# Sparse-prefix index extraction
# ─────────────────────────────────────────────────────────────────────────────
function sparse_prefix_inds(T::ITensor)
    if ITensors.has_external_storage(T) && T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse
        w  = T.tensor.data
        PA = SparseBackends._abs_head_len(w)
        return Set(w.inds[1:PA])
    end
    return Set(inds(T))
end

# ─────────────────────────────────────────────────────────────────────────────
# Dense combiner materialisation (fallback for _fuse_aliased_strands when
# the tensor is not WrappedAliasedBlockSparse)
# ─────────────────────────────────────────────────────────────────────────────
function _materialize_combiner_dense(strands::Vector{<:Index}, fused::Index)
    dims_in = Tuple(ITensors.dim(i) for i in strands)
    Nin     = length(strands)
    dout    = ITensors.dim(fused)
    @assert prod(dims_in) == dout
    data    = zeros(ComplexF64, dims_in..., dout)
    strides = ones(Int, Nin)
    for d in 2:Nin; strides[d] = strides[d-1] * dims_in[d-1]; end
    for cart in CartesianIndices(dims_in)
        lin = 1
        for d in 1:Nin
            lin += (cart[d] - 1) * strides[d]
        end
        data[cart, lin] = 1.0 + 0.0im
    end
    return ITensors.ITensor(data, strands..., fused)
end

# ─────────────────────────────────────────────────────────────────────────────
# Direct key-rewrite fuse: collapse `strand_list` (sparse-prefix axes) into a
# single `fused` axis via column-major encoding. No GEMM, no template
# materialisation — alias structure is preserved because only the key tuples
# shrink; underlying block data is unchanged.
# ─────────────────────────────────────────────────────────────────────────────
function _fuse_aliased_strands(H_site::ITensor, strand_list::Vector{<:Index},
                               fused::Index)
    if !(ITensors.has_external_storage(H_site) &&
         H_site.tensor.data isa SparseBackends.WrappedAliasedBlockSparse)
        cmb = _materialize_combiner_dense(strand_list, fused)
        return SparseBackends.contract_aliased_itensor(H_site, cmb, :dense, :dense;
                                                       preserve_bs_output=false)
    end
    w  = H_site.tensor.data
    A  = w.aliased
    PA = SparseBackends._abs_head_len(w)
    old_inds = w.inds

    strand_pos = Int[]
    for s in strand_list
        p = findfirst(I -> I == s, old_inds)
        p === nothing && error("strand $s not in inds")
        p > PA        && error("strand $s is not in sparse prefix (pos=$p, PA=$PA)")
        push!(strand_pos, p)
    end
    sort!(strand_pos)
    nstr = length(strand_pos)

    strand_dims   = [ITensors.dim(old_inds[p]) for p in strand_pos]
    strand_stride = Vector{Int}(undef, nstr)
    let s = 1
        for t in 1:nstr
            strand_stride[t] = s
            s *= strand_dims[t]
        end
    end

    fused_pos_in_new = strand_pos[1]
    new_P = PA - (nstr - 1)
    new_N = length(old_inds) - (nstr - 1)
    N2    = new_N - new_P

    keep_old_positions = Int[]
    for p in 1:length(old_inds)
        (p in strand_pos) && continue
        push!(keep_old_positions, p)
    end

    new_inds = Vector{ITensors.Index}(undef, new_N)
    nki = 0
    for newpos in 1:new_N
        if newpos == fused_pos_in_new
            new_inds[newpos] = fused
        else
            nki += 1
            new_inds[newpos] = old_inds[keep_old_positions[nki]]
        end
    end
    new_inds_t = Tuple(new_inds)

    Kt   = eltype(eltype(A.keys))
    nb   = length(A.keys)
    new_keys = Vector{NTuple{new_P, Kt}}(undef, nb)
    keep_prefix_old = Int[p for p in 1:PA if !(p in strand_pos)]
    for i in 1:nb
        oldk      = A.keys[i]
        fused_val = 1
        for t in 1:nstr
            fused_val += (oldk[strand_pos[t]] - 1) * strand_stride[t]
        end
        nk = ntuple(new_P) do j
            if j == fused_pos_in_new
                Kt(fused_val)
            else
                kj = j < fused_pos_in_new ? j : j - 1
                Kt(oldk[keep_prefix_old[kj]])
            end
        end
        new_keys[i] = nk
    end

    new_dims    = ntuple(i -> ITensors.dim(new_inds[i]), new_N)
    Tel         = eltype(A.templates)
    new_aliased = SparseBackends.AliasedBlockSparse{Tel, new_N, N2, new_P, Kt}(
        new_dims, A.blksize, copy(A.templates), A.n_templates,
        new_keys, copy(A.alias_ids), copy(A.scalars))
    new_wrap    = SparseBackends.WrappedAliasedBlockSparse{Tel, new_N, N2, new_P}(
        new_aliased, new_inds_t)
    return ITensors._itensor_from_external_storage(new_wrap)
end

# ─────────────────────────────────────────────────────────────────────────────
# Fuse multi-strand sparse links across all bonds of an aliased MPO.
# Groups by tag-string; only fuses groups with ≥ 2 strands that are sparse-
# prefix on BOTH H[k] and H[k+1].
# ─────────────────────────────────────────────────────────────────────────────
function fuse_sparse_links!(H::MPO)
    L = length(H)
    for k in 1:(L-1)
        common = commoninds(H[k], H[k+1])
        sp_k   = sparse_prefix_inds(H[k])
        sp_k1  = sparse_prefix_inds(H[k+1])
        by_tag = Dict{String, Vector{Index}}()
        for I in common
            (I in sp_k) && (I in sp_k1) || continue
            t = string(tags(I))
            push!(get!(by_tag, t, Index[]), I)
        end
        for (_, strand_list) in by_tag
            length(strand_list) <= 1 && continue
            cmb_raw = combiner(strand_list...; tags="Link,FusedSparse,bond=$(k)")
            fused   = ITensors.combinedind(cmb_raw)
            H[k]    = _fuse_aliased_strands(H[k],   strand_list, fused)
            H[k+1]  = _fuse_aliased_strands(H[k+1], strand_list, fused)
        end
    end
    return H
end

# ─────────────────────────────────────────────────────────────────────────────
# Pre-permute each aliased H[k]'s dense tail so the kernel's `permute_A` is
# identity at every matvec call. For a left→right (Plan B) sweep the preferred
# tail order is [right-link, left-link]: right link (higher l=NN) sits at
# the keepA slot (PA+1), left link at the red_dense slot. Permuting only
# touches n_templates blocks — O(n_templates · blksize) — not n_blocks.
# ─────────────────────────────────────────────────────────────────────────────
function prepermute_aliased_mpo!(H::MPO)
    function _link_pos(I)
        m = match(r"l=(\d+)", string(ITensors.tags(I)))
        m === nothing && return -1
        return parse(Int, m.captures[1])
    end
    L = length(H)
    n_permuted = 0
    for k in 1:L
        T = H[k]
        (ITensors.has_external_storage(T) &&
         T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse) || continue
        w  = T.tensor.data
        PA = SparseBackends._abs_head_len(w)
        N  = ndims(w.aliased)
        N2 = N - PA
        N2 < 2 && continue
        tail_inds     = collect(w.inds[PA+1:N])
        idx_pos       = [(i, _link_pos(tail_inds[i])) for i in 1:N2]
        sort!(idx_pos, by = x -> -x[2])
        new_tail_order = [x[1] for x in idx_pos]
        new_tail_order == collect(1:N2) && continue
        perm        = vcat(collect(1:PA), new_tail_order .+ PA)
        new_aliased = permutedims(w.aliased, perm)
        new_inds    = Tuple(w.inds[i] for i in perm)
        T_el        = eltype(new_aliased)
        new_w       = SparseBackends.WrappedAliasedBlockSparse{T_el, N, N2, PA}(new_aliased, new_inds)
        H[k]        = ITensors._itensor_from_external_storage(new_w)
        n_permuted += 1
    end
    println("  prepermute_aliased_mpo!: permuted $n_permuted of $L sites")
    return H
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-site memory footprint reporter for an aliased MPO.
# Prints dense / BS / aliased element counts and compression ratios.
# ─────────────────────────────────────────────────────────────────────────────
function report_aliased_footprint(H::MPO, label::String)
    total_bytes = sum(Base.summarysize(W) for W in H)
    println("$label total MPO memory: $(round(total_bytes/1e6; digits=4)) MB")
    println("  ", rpad("site", 5), rpad("nb", 6), rpad("ntmpl", 7),
            rpad("blksize", 9), rpad("dense", 10), rpad("BS", 10),
            rpad("aliased", 10), rpad("vs-dense", 10), "vs-BS")
    tot_d = 0; tot_b = 0; tot_a = 0
    for i in 1:length(H)
        T = H[i]
        ITensors.has_external_storage(T) &&
            T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse || continue
        ali  = T.tensor.data.aliased
        nb   = length(ali.keys); nt = ali.n_templates; bksz = ali.blksize
        ds   = prod(ali.dims);   bs = nb * bksz;       as   = nt * bksz + nb
        tot_d += ds; tot_b += bs; tot_a += as
        println("  ", rpad(string(i), 5), rpad(string(nb), 6), rpad(string(nt), 7),
                       rpad(string(bksz), 9), rpad(string(ds), 10),
                       rpad(string(bs), 10), rpad(string(as), 10),
                       rpad(string(round(ds/max(as,1); digits=2)), 10),
                       round(bs/max(as,1); digits=2))
    end
    println("  ", rpad("TOT", 5), rpad("", 6), rpad("", 7), rpad("", 9),
                  rpad(string(tot_d), 10), rpad(string(tot_b), 10),
                  rpad(string(tot_a), 10),
                  rpad(string(round(tot_d/max(tot_a,1); digits=2)), 10),
                  round(tot_b/max(tot_a,1); digits=2))
end

# ─────────────────────────────────────────────────────────────────────────────
# DMRG runners — ground state and excited state are separate functions.
#
# Both follow the same pattern:
#   1. Warmup JIT sweep (discarded, error caught silently).
#   2. Reset SparseBackends.TIMER and ITensorMPS.PROJMPO_TIMER.
#   3. GC twice to reduce allocation noise.
#   4. Timed production run.
#   5. Print ProjMPO and SparseBackends timer reports.
#   6. Optionally print SB_PERM_PROFILE / SB_ALIAS_STATS diagnostics.
#
# Return: (dmrg_result_tuple, wall_seconds_excluding_jit)
# ─────────────────────────────────────────────────────────────────────────────
function run_dmrg_ground(label, H, psi0; nsweeps, maxdim, mindim, cutoff, kwargs...)
    println("[warmup pass: 1 JIT sweep, results discarded]")
    ENV["SB_IN_WARMUP"] = "1"   # gate TRACE_BOND so the latch fires on the real run, not warmup
    try
        dmrg(H, deepcopy(psi0); nsweeps=1, maxdim=10, mindim=10, cutoff=1e-12,
             outputlevel=0, use_early_exit=false)
    catch e
        println("  warmup failed: ", sprint(showerror, e))
    finally
        ENV["SB_IN_WARMUP"] = "0"
    end
    reset_timer!(SparseBackends.TIMER)
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    get(ENV, "SB_FLOP_COUNT", "0") == "1" && SparseBackends.reset_flops!()
    GC.gc(); GC.gc()
    wall = @elapsed begin
        result = dmrg(H, psi0; nsweeps, maxdim, mindim, cutoff, use_early_exit=false, kwargs...)
    end
    get(ENV, "SB_FLOP_COUNT", "0") == "1" && SparseBackends.report_flops(label)
    println("\n========== TIMER REPORT: $label  (wall = $(round(wall; digits=3)) s, JIT excluded) ==========")
    println("\n--- ProjMPO matvec breakdown ---")
    print_timer(ITensorMPS.PROJMPO_TIMER; sortby=:firstexec)
    println("\n--- SparseBackends contract dispatch breakdown ---")
    print_timer(SparseBackends.TIMER; sortby=:firstexec)
    get(ENV, "SB_PERM_PROFILE", "0") == "1" && (SparseBackends._report_perm_profile(); SparseBackends._reset_perm_profile!())
    get(ENV, "SB_ALIAS_STATS",  "0") == "1" && (SparseBackends._report_alias_stats();  SparseBackends._reset_alias_stats!())
    println("==========\n")
    return result, wall
end

function run_dmrg_excited(label, H, Ms::Vector{MPS}, psi0; nsweeps, maxdim, mindim,
                          cutoff, weight, kwargs...)
    println("[warmup pass: 1 JIT sweep, results discarded]")
    try
        dmrg(H, Ms, deepcopy(psi0); nsweeps=1, maxdim=10, mindim=10, cutoff=1e-12,
             weight=weight, outputlevel=0, use_early_exit=false)
    catch e
        println("  warmup failed: ", sprint(showerror, e))
    end
    reset_timer!(SparseBackends.TIMER)
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    GC.gc(); GC.gc()
    wall = @elapsed begin
        result = dmrg(H, Ms, psi0; nsweeps, maxdim, mindim, cutoff,
                      weight, use_early_exit=false, kwargs...)
    end
    println("\n========== TIMER REPORT: $label  (wall = $(round(wall; digits=3)) s, JIT excluded) ==========")
    println("\n--- ProjMPO matvec breakdown ---")
    print_timer(ITensorMPS.PROJMPO_TIMER; sortby=:firstexec)
    println("\n--- SparseBackends contract dispatch breakdown ---")
    print_timer(SparseBackends.TIMER; sortby=:firstexec)
    get(ENV, "SB_PERM_PROFILE", "0") == "1" && (SparseBackends._report_perm_profile(); SparseBackends._reset_perm_profile!())
    get(ENV, "SB_ALIAS_STATS",  "0") == "1" && (SparseBackends._report_alias_stats();  SparseBackends._reset_alias_stats!())
    println("==========\n")
    return result, wall
end
