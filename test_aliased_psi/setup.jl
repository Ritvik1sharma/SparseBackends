# ── Roofline hardware ceilings (only computed when SB_ROOFLINE=1) ────────────
# Peak ZGEMM GFLOP/s and STREAM-triad bandwidth via Julia microbenchmarks, so the
# kernel's achieved GFLOP/s + arithmetic intensity (from show_roofline) can be
# placed against the hardware roofline. (No HW counters — WSL2 has no uncore IMC PMU.)
function peak_zgemm_gflops(n=2048, reps=5)
    A = rand(ComplexF64, n, n); B = rand(ComplexF64, n, n); C = zeros(ComplexF64, n, n)
    mul!(C, A, B)  # warmup
    t = @elapsed (for _ in 1:reps; mul!(C, A, B); end)
    return 8.0 * n^3 * reps / t / 1e9
end

function stream_triad_gbs(N=50_000_000, reps=5)
    a = rand(N); b = rand(N); c = rand(N); s = 3.0
    @. a = b + s*c  # warmup
    t = @elapsed (for _ in 1:reps; @. a = b + s*c; end)
    return 3.0 * N * 8 * reps / t / 1e9   # 2 read + 1 write, 8 B/Float64
end

function print_roofline_ceilings()
    println("\n========== roofline ceilings (microbenchmarks) ==========")
    println("BLAS threads = ", BLAS.get_num_threads())
    pk = peak_zgemm_gflops(); bw = stream_triad_gbs()
    @printf("peak ZGEMM          = %.1f GFLOP/s\n", pk)
    @printf("STREAM-triad BW     = %.1f GB/s\n", bw)
    @printf("roofline ridge (AI*) = %.3f FLOP/byte  (peak/BW)\n", pk / bw)
end

mps_footprint_bytes(psi) = Base.summarysize(psi)


# Reported link dim: dim of the single shared link Index between neighbours.
reported_linkdims(psi) =
    [ITensors.dim(commonind(psi[i], psi[i+1])) for i in 1:length(psi)-1]

# HONEST per-bond dim: ∏ of ALL shared-index dims between psi[i], psi[i+1]
# (channel × multiplicity under the doubled-link convention) — the true rank of
# the bipartition. `dim(commonind)` returns only ONE shared index (the channel).
function honest_linkdims(psi)
    [(cis = commoninds(psi[i], psi[i+1]); isempty(cis) ? 0 : prod(ITensors.dim, cis))
     for i in 1:length(psi)-1]
end

# Per-site storage breakdown for aliased (and BS / dense, for robustness).
# Returns rows + totals + the per-site alias dedup ratios (nb/nt).
function inspect_storage(psi)
    payload_total = 0   # numeric data (templates / bs.data / dense)
    keys_total    = 0
    site_total    = 0
    full_total    = 0   # if stored fully dense
    bs_total      = 0   # if stored as plain BS (n_blocks * blksize)
    ratios        = Float64[]
    rows = String[]
    for (i, T) in enumerate(psi)
        s  = try ITensors.get_external_storage(T) catch _ nothing end
        ss = Base.summarysize(T)
        site_total += ss
        if s isa SparseBackends.WrappedAliasedBlockSparse
            ali    = s.aliased
            nb     = length(ali.keys)
            nt     = ali.n_templates
            blksz  = ali.blksize
            data_b = Base.summarysize(ali.templates)
            keys_b = Base.summarysize(ali.keys) + Base.summarysize(ali.alias_ids) + Base.summarysize(ali.scalars)
            full_b = prod(ali.dims) * sizeof(eltype(ali.templates))
            bs_b   = nb * blksz * sizeof(eltype(ali.templates))
            ratio  = nb / max(nt, 1)
            payload_total += data_b; keys_total += keys_b; full_total += full_b; bs_total += bs_b
            push!(ratios, ratio)
            push!(rows, @sprintf("site %2d [ALI] nb=%d ntmpl=%d blksize=%d  dedup(nb/nt)=%.2fx  templates=%.2fKiB  keys+ids+scalars=%.2fKiB  if_dense=%.2fKiB  total=%.2fKiB",
                                 i, nb, nt, blksz, ratio, data_b/1024, keys_b/1024, full_b/1024, ss/1024))
        elseif s isa SparseBackends.WrappedBlockSparse
            bs     = s.blocksparse
            nb     = length(bs.keys)
            data_b = Base.summarysize(bs.data)
            keys_b = Base.summarysize(bs.keys) + Base.summarysize(bs.ids)
            full_b = prod(bs.dims) * sizeof(eltype(bs.data))
            payload_total += data_b; keys_total += keys_b; full_total += full_b; bs_total += data_b
            push!(rows, @sprintf("site %2d [BS]  nblocks=%d blksize=%d  data=%.2fKiB  keys+ids=%.2fKiB  if_dense=%.2fKiB  total=%.2fKiB",
                                 i, nb, bs.blksize, data_b/1024, keys_b/1024, full_b/1024, ss/1024))
        else
            a = ITensors.array(T)
            data_b = Base.summarysize(a)
            payload_total += data_b; full_total += data_b; bs_total += data_b
            push!(rows, @sprintf("site %2d [dense] data=%.2fKiB  total=%.2fKiB", i, data_b/1024, ss/1024))
        end
    end
    return rows, payload_total, keys_total, site_total, full_total, bs_total, ratios
end


# ── Iso check ─────────────────────────────────────────────────────────────────

# Iso check (sparse-aware). G = T * dag(prime(T, link)) on the link indices;
# densify only that small (link × link') result. For aliased ψ the off-diagonals
# are EXPECTED to be nonzero (structural; handled by Path-B's M⁻¹).
function _link_gram(T::ITensors.ITensor, link_inds)
    Tp = prime(T, link_inds)
    G  = T * dag(Tp)
    Gd = ITensors.has_external_storage(G) ? SparseBackends.to_dense_itensors_unfused(G) : G
    link_pr = [prime(I) for I in link_inds]
    G_arr = Array(Gd, link_inds..., link_pr...)
    n = prod(ITensors.dim, link_inds)
    return reshape(G_arr, n, n), n
end

function iso_violations(psi)
    N = length(psi)
    rows = NamedTuple{(:site, :left_iso_err, :right_iso_err, :right_link_dim, :left_link_dim), Tuple{Int, Float64, Float64, Int, Int}}[]
    for i in 1:N
        T = psi[i]
        le = NaN; r_dim = 0
        if i < N
            ri = commoninds(psi[i], psi[i+1])
            if !isempty(ri)
                G, n = _link_gram(T, ri); r_dim = n; le = norm(G - eye(n)) / sqrt(n)
            end
        end
        re = NaN; l_dim = 0
        if i > 1
            li = commoninds(psi[i], psi[i-1])
            if !isempty(li)
                G, n = _link_gram(T, li); l_dim = n; re = norm(G - eye(n)) / sqrt(n)
            end
        end
        push!(rows, (site=i, left_iso_err=le, right_iso_err=re, right_link_dim=r_dim, left_link_dim=l_dim))
    end
    return rows
end

function print_iso(label, psi)
    println("  iso check ($label):  [aliased ψ is structurally non-iso → off-diagonals expected; Path-B handles it]")
    println("    site | left-iso(L†L=I rt)  right-iso(R R†=I lt) | rt-dim   lt-dim")
    for r in iso_violations(psi)
        lstr = isnan(r.left_iso_err)  ? "   -  " : @sprintf("%.2e", r.left_iso_err)
        rstr = isnan(r.right_iso_err) ? "   -  " : @sprintf("%.2e", r.right_iso_err)
        println(@sprintf("    %4d | %s            %s        | %5d    %5d", r.site, lstr, rstr, r.right_link_dim, r.left_link_dim))
    end
end

# ── Schema-invariance check (see _SCHEMA_TRACK_ON in the calling script) ──

# ψ_aliased ≡ P·ψ_dense, so the alias SCHEMA — keys (which prefix blocks are
# nonzero, set by P's sparsity), the key→template partition (alias_ids, set by
# P's structure), and scalars (set by P's values) — encodes the constraint P,
# which never changes. Across the whole DMRG flow only the *templates* (the
# ψ_dense slices) may grow; keys / partition / scalars MUST stay invariant.
function _schema_fingerprint(psi)
    fps = Vector{Any}(undef, length(psi))
    for (i, T) in enumerate(psi)
        s = try ITensors.get_external_storage(T) catch _ nothing end
        if s isa SparseBackends.WrappedAliasedBlockSparse
            a = s.aliased
            groups = Dict{Int,Vector{Int}}()
            for (j, aid) in enumerate(a.alias_ids); push!(get!(groups, aid, Int[]), j); end
            partition = Set(Set(a.keys[j] for j in g) for g in values(groups))
            sc = sort([(round(real(z), digits=10), round(imag(z), digits=10)) for z in a.scalars])
            fps[i] = (nkeys=length(a.keys), nt=a.n_templates,
                      keys=Set(a.keys), scalars=sc, partition=partition)
        else
            fps[i] = nothing
        end
    end
    return fps
end

function _compare_schema(init, cur; label="")
    drift = String[]
    for i in eachindex(init)
        ii, cc = init[i], cur[i]
        if ii === nothing || cc === nothing
            ii === cc || push!(drift, "site $i storage-kind changed ($(ii===nothing ? "→aliased" : "aliased→dense/other"))")
            continue
        end
        ii.nkeys      != cc.nkeys     && push!(drift, "site $i nkeys $(ii.nkeys)→$(cc.nkeys)")
        ii.nt         != cc.nt        && push!(drift, "site $i n_templates $(ii.nt)→$(cc.nt)")
        ii.keys       != cc.keys      && push!(drift, "site $i KEYS set changed")
        ii.scalars    != cc.scalars   && push!(drift, "site $i SCALARS multiset changed")
        ii.partition  != cc.partition && push!(drift, "site $i key→template PARTITION changed")
    end
    if isempty(drift)
        println("  [$label] SCHEMA INVARIANT ✓ (keys / partition / scalars / n_templates unchanged at all sites)")
    else
        println("  [$label] ⚠ SCHEMA DRIFT ($(length(drift))):")
        for d in drift; println("        $d"); end
    end
    return isempty(drift)
end


function check_aliased_invariant(psi; label="")
    bad = Int[]
    for (i, T) in enumerate(psi)
        if !(ITensors.has_external_storage(T) && T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse)
            push!(bad, i)
        end
    end
    if !isempty(bad)
        @warn "[$label] psi sites NOT aliased: $bad (alias invariant broken)"
    else
        println("  [$label] psi storage invariant ✓ (all $(length(psi)) sites aliased)")
    end
    return isempty(bad)
end

function report_state(label, psi, maxdim, E=nothing; verbose=false)
    mb   = round(mps_footprint_bytes(psi) / 2^20, digits=3)
    rep  = reported_linkdims(psi)
    hon  = honest_linkdims(psi)
    mx   = isempty(rep) ? 0 : maximum(rep)
    mxh  = isempty(hon) ? 0 : maximum(hon)
    println("  $label: footprint=$(mb) MiB  reported_maxlinkdim=$mx  honest_maxlinkdim=$mxh  (MAXDIM=$maxdim)")
    println("    linkdims(reported, single shared idx)=$rep")
    println("    linkdims(honest, ∏all shared)=$hon" * (E === nothing ? "" : "  E=$E"))
    over = findall(h -> h > maxdim, hon)
    if isempty(over)
        println("    ✓ honest bond dim obeys MAXDIM at every bond.")
    else
        println("    ✗ honest bond dim EXCEEDS MAXDIM at bonds $over — rank not honestly bounded.")
    end
    if verbose
        rows, payload, keys_b, site_total, full, bs_eq, ratios = inspect_storage(psi)
        for r in rows; println("    $r"); end
        non_data = site_total - payload
        min_r  = isempty(ratios) ? 0.0 : minimum(ratios)
        mean_r = isempty(ratios) ? 0.0 : sum(ratios)/length(ratios)
        println("    --- MPS totals ---")
        println("    data(numeric)=$(round(payload/1024,digits=2)) KiB   keys+ids+scalars=$(round(keys_b/1024,digits=2)) KiB   non-data=$(round(non_data/1024,digits=2)) KiB")
        println("    sum-of-sites=$(round(site_total/1024,digits=2)) KiB   if_BS=$(round(bs_eq/1024,digits=2)) KiB   if_dense=$(round(full/1024,digits=2)) KiB")
        @printf("    vs-dense payload compression = %.2fx   vs-BS = %.2fx\n", full/max(payload,1), bs_eq/max(payload,1))
        @printf("    alias dedup (nb/nt): min=%.2fx  mean=%.2fx\n", min_r, mean_r)
    end
end
