# PXP / Rydberg-blockade DMRG — ALIASED-ψ runner (Path-B, bare H).
#
# Hamiltonian: H = Xp_1 + Σ_j (Px_j ⊗ LP_{j+1} + RP_j ⊗ Xp_{j+1}) + Px_N
# Projector:   NotEqlsLoop_R1 — no two adjacent state-1 (Rydberg-excited) sites.
#
# Constraint enforcement:
#   ψ = P·ψ₀ lives in image(P) and carries aliased storage (WrappedAliasedBlockSparse).
#   DMRG runs on the BARE H; [H,P] = 0 keeps ψ in image(P) structurally.
#   DO NOT compare against the dense PXP runner (test_dense_pxp.jl) which uses
#   the projected Hamiltonian PHP — that is a different linear problem.
#   The aliased runner is the apples-to-apples match for test_sparse_pxp.jl
#   (Path-A BS runner) and this test (Path-B aliased runner).
#
# Path: aliased ψ is structurally NON-iso (template sharing ⇒ off-diagonal L†L).
#   Path-B (BMF_ISO_PATH=0, BMF_APPLY_MINV=1) is required for correct energies.
#
# Example:
#   SB_ALIASED_ENABLE=1 julia --project=.. test_aliased_pxp.jl --N 12 --maxdim 40 --n-sweeps 10
ENV["SB_ALIASED_ENABLE"] = get(ENV, "SB_ALIASED_ENABLE", "1")
ENV["BMF_ISO_PATH"]      = "0"
ENV["BMF_APPLY_MINV"]    = get(ENV, "BMF_APPLY_MINV", "1")

using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using LinearAlgebra: I as eye, norm
using Random
using ArgParse
using Printf

import KrylovKit
KrylovKit.set_num_threads(parse(Int, get(ENV, "SB_KK_NTHREADS", "1")))
println("[KrylovKit threads = ", KrylovKit.get_num_threads(),
        "   Julia threads = ", Threads.nthreads(), "]")

include("../test_sparse_psi/utils.jl")

const _ALIASED_ENABLE = get(ENV, "SB_ALIASED_ENABLE", "0") == "1"
if !_ALIASED_ENABLE
    println("[SB_ALIASED_ENABLE != 1] Aliased path is gated off — set SB_ALIASED_ENABLE=1 to run.")
    exit(0)
end

# ── PXP site operators on S=1 (states 0, 1, 2; "1" = Rydberg-excited) ────────
ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]

function parse_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N"
            help = "Chain length (number of sites). PXP is single-site, so total dim = N."
            arg_type = Int
            default = 12
        "--n-sweeps"
            help = "Total DMRG sweeps at maxdim. Sweep 1 is JIT warmup, excluded from post-JIT totals."
            arg_type = Int
            default = 10
        "--maxdim"
            help = "DMRG maxdim cap."
            arg_type = Int
            default = 40
        "--mindim"
            arg_type = Int
            default = 1
        "--target-energy"
            help = "If set, log the first sweep at which E ≤ target (does not early-exit)."
            arg_type = Float64
            default = NaN
        "--no-excited"
            help = "Skip the excited-state search (useful for quick benchmarking)."
            action = :store_true
    end
    return parse_args(s)
end

# ── Rydberg / Fibonacci constraint MPO: no two adjacent state-1 sites ─────────
function itensor_from_nonzeros(dims::NTuple{N,Int}, coords::Vector; left::Bool=false) where {N}
    A = zeros(Float64, dims...)
    for c in coords
        @assert length(c) == N
        A[(c .+ 1)...] = 1.0
    end
    return A
end
bind_to_idx(A::AbstractArray, idxs::Index...) = ITensor(A, idxs...)

function NotEqlsLoop_R1(sites)
    N = length(sites)
    R1_first = itensor_from_nonzeros((3, 3, 2), [(0,0,0), (1,1,1), (2,2,0)])
    R1_bulk  = itensor_from_nonzeros((3, 3, 2, 2),
        [(0,0,0,0), (0,0,1,0), (1,1,0,1), (1,1,1,0), (2,2,0,0)])
    R1_last  = itensor_from_nonzeros((3, 3, 2),
        [(0,0,0), (0,0,1), (1,1,0), (1,1,1), (2,2,0)]; left=true)
    bonds = [Index(2, "Link,l=$(i)") for i in 1:N-1]
    Wvec = Vector{ITensor}(undef, N)
    Wvec[1] = bind_to_idx(R1_first, sites[1], sites[1]', bonds[1])
    for j in 2:N-1
        Wvec[j] = bind_to_idx(R1_bulk, sites[j], sites[j]', bonds[j-1], bonds[j])
    end
    Wvec[N] = bind_to_idx(R1_last, sites[N], sites[N]', bonds[N-1])
    return MPO(Wvec)
end

# ── Build: bare H + aliased ψ = P·ψ₀ ─────────────────────────────────────────
function build_setup(N::Int)
    Random.seed!(42)
    sites = siteinds("S=1", N)

    HT = OpSum()
    HT += 1, "Xp", 1
    for j in 0:N-2
        HT += 1, "Px", j+1, "LP", j+2
        HT += 1, "RP", j+1, "Xp", j+2
    end
    HT += 1, "Px", N
    H = MPO(HT, sites)

    P_sparse = NotEqlsLoop_R1(sites)
    psi0     = random_mps(sites)
    psi_ali  = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)
    return H, psi_ali
end

# ── Storage / footprint helpers ────────────────────────────────────────────────
mps_footprint_bytes(psi) = Base.summarysize(psi)

reported_linkdims(psi) =
    [ITensors.dim(commonind(psi[i], psi[i+1])) for i in 1:length(psi)-1]

function honest_linkdims(psi)
    [(cis = commoninds(psi[i], psi[i+1]); isempty(cis) ? 0 : prod(ITensors.dim, cis))
     for i in 1:length(psi)-1]
end

function inspect_storage(psi)
    payload_total = 0; keys_total = 0; site_total = 0; full_total = 0; bs_total = 0
    ratios = Float64[]
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

function report_state(label, psi, maxdim, E=nothing; verbose=false)
    mb  = round(mps_footprint_bytes(psi) / 2^20, digits=3)
    rep = reported_linkdims(psi)
    hon = honest_linkdims(psi)
    mx  = isempty(rep) ? 0 : maximum(rep)
    mxh = isempty(hon) ? 0 : maximum(hon)
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

# ── Iso check ─────────────────────────────────────────────────────────────────
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
    rows = NamedTuple{(:site, :left_iso_err, :right_iso_err, :right_link_dim, :left_link_dim), Tuple{Int,Float64,Float64,Int,Int}}[]
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
        println(@sprintf("    %4d | %s            %s        | %5d    %5d",
                         r.site, lstr, rstr, r.right_link_dim, r.left_link_dim))
    end
end

# ── Schema invariance ─────────────────────────────────────────────────────────
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
            ii === cc || push!(drift, "site $i storage-kind changed")
            continue
        end
        ii.nkeys     != cc.nkeys     && push!(drift, "site $i nkeys $(ii.nkeys)→$(cc.nkeys)")
        ii.nt        != cc.nt        && push!(drift, "site $i n_templates $(ii.nt)→$(cc.nt)")
        ii.keys      != cc.keys      && push!(drift, "site $i KEYS set changed")
        ii.scalars   != cc.scalars   && push!(drift, "site $i SCALARS multiset changed")
        ii.partition != cc.partition && push!(drift, "site $i key→template PARTITION changed")
    end
    if isempty(drift)
        println("  [$label] SCHEMA INVARIANT ✓ (keys / partition / scalars / n_templates unchanged at all sites)")
    else
        println("  [$label] ⚠ SCHEMA DRIFT ($(length(drift))):")
        for d in drift; println("        $d"); end
    end
    return isempty(drift)
end

const _INIT_SCHEMA = Ref{Any}(nothing)

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

# ── Sweep runner ──────────────────────────────────────────────────────────────
function run_sweeps(H, psi0, n_sweeps::Int, maxdim::Int;
                    cutoff=1e-10, mindim=1, target_E=NaN, label="ALI",
                    orthogonal_states=nothing, weight=20.0)
    psi = psi0
    E = NaN
    cum = 0.0; cum_excl1 = 0.0
    target_reached_sweep = 0; target_reached_cum = NaN; target_reached_cum_excl1 = NaN
    for i in 1:n_sweeps
        sw = Sweeps(1); setmaxdim!(sw, maxdim); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        t = if orthogonal_states === nothing
            @elapsed (E, psi, _esw, terr) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false)
        else
            @elapsed (E, psi, _esw, terr) = dmrg(H, orthogonal_states, psi, sw;
                outputlevel=0, use_early_exit=false, weight=weight)
        end
        cum += t; if i > 1; cum_excl1 += t; end
        @printf("  [%s sweep %2d] t=%8.3fs  E=%.12f  maxtruncerr=%.3e\n", label, i, t, E, terr)
        check_aliased_invariant(psi; label="after sweep $i")
        if get(ENV, "SB_SCHEMA_TRACK", "0") == "1" && _INIT_SCHEMA[] !== nothing
            _compare_schema(_INIT_SCHEMA[], _schema_fingerprint(psi); label="after sweep $i vs init")
        end
        flush(stdout)
        if target_reached_sweep == 0 && !isnan(target_E) && E <= target_E
            target_reached_sweep = i; target_reached_cum = cum; target_reached_cum_excl1 = cum_excl1
        end
    end
    return (; E, psi, total=cum, total_excl1=cum_excl1,
            target_reached_sweep, target_reached_cum, target_reached_cum_excl1)
end

# ── Main ──────────────────────────────────────────────────────────────────────
let
    parsed_args = parse_command_line()
    N          = parsed_args["N"]
    n_sweeps   = parsed_args["n-sweeps"]
    maxdim     = parsed_args["maxdim"]
    mindim     = parsed_args["mindim"]
    target_E   = parsed_args["target-energy"]
    no_excited = parsed_args["no-excited"]

    println("=== PXP benchmark — ALIASED ψ (Path-B) ===")
    println("BMF_ISO_PATH=", ENV["BMF_ISO_PATH"], "  BMF_APPLY_MINV=", ENV["BMF_APPLY_MINV"],
            "  (Path-B: A = M⁻¹·H_eff — required for non-iso aliased ψ)")
    println("N=$N  n_sweeps=$n_sweeps  maxdim=$maxdim  mindim=$mindim  target_E=$(isnan(target_E) ? "—" : target_E)")
    _percm = get(ENV, "SB_ALIASED_PERCM_CAP", "0")
    println("SB_ALIASED_PERCM_CAP=$_percm  → ", _percm == "1" ?
            "CAPPED (honest_bd≤maxdim — starved)" :
            "UNCAPPED (honest_bd=channel×maxdim) [DEFAULT]")
    println("ψ = ALIASED (P·ψ₀ via NotEqlsLoop_R1); DMRG on BARE H (constraint enforced structurally).")

    println("\nBuilding setup for N=$N sites ...")
    t_setup = @elapsed (H, psi_ali) = build_setup(N)
    println("Setup time: $(round(t_setup, digits=1))s.  System: $(length(psi_ali)) sites.")
    report_state("initial psi_ali", psi_ali, maxdim; verbose=true)

    if get(ENV, "SB_SCHEMA_TRACK", "0") == "1"
        _INIT_SCHEMA[] = _schema_fingerprint(psi_ali)
        println("  [init] captured alias schema fingerprint for invariance tracking")
    end
    check_aliased_invariant(psi_ali; label="initial")
    flush(stdout)

    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)
    SparseBackends.reset_cas_stats!()

    println("\n=== GROUND STATE ($n_sweeps sweeps at maxdim=$maxdim, mindim=$mindim; sweep 1 = JIT) ===")
    res_gs = run_sweeps(H, psi_ali, n_sweeps, maxdim; mindim=mindim, target_E=target_E)
    E_gs = res_gs.E; psi_gs = res_gs.psi
    avg_excl1 = n_sweeps > 1 ? res_gs.total_excl1 / (n_sweeps - 1) : NaN
    @printf("[ground]    total=%.3fs  excl1=%.3fs  avg/sw=%.3fs  E=%.12f\n",
            res_gs.total, res_gs.total_excl1, avg_excl1, E_gs)
    report_state("final psi_gs", psi_gs, maxdim, E_gs; verbose=true)
    print_iso("final gs", psi_gs)

    E_ex = NaN; psi_ex = nothing; t_ex = NaN
    if !no_excited
        println("\n=== EXCITED STATE (orthogonal to gs; $n_sweeps sweeps; weight=20) ===")
        psi_init = deepcopy(psi_ali)
        res_ex = run_sweeps(H, psi_init, n_sweeps, maxdim; mindim=mindim, label="EX",
                            orthogonal_states=[psi_gs], weight=20.0)
        E_ex = res_ex.E; psi_ex = res_ex.psi; t_ex = res_ex.total
        avg_ex = n_sweeps > 1 ? res_ex.total_excl1 / (n_sweeps - 1) : NaN
        @printf("[excited]   total=%.3fs  excl1=%.3fs  avg/sw=%.3fs  E=%.12f\n",
                res_ex.total, res_ex.total_excl1, avg_ex, E_ex)
        report_state("final psi_ex", psi_ex, maxdim, E_ex; verbose=false)
        print_iso("final ex", psi_ex)
    end

    println("\n=========== SUMMARY (aliased PXP  N=$N  md=$maxdim) ===========")
    @printf("ground E:   %.12f\n", E_gs)
    if !no_excited
        @printf("excited E:  %.12f\n", E_ex)
        @printf("gap (E_ex - E_gs): %.12f\n", E_ex - E_gs)
    end
    if !isnan(target_E)
        if res_gs.target_reached_sweep > 0
            @printf("reached target E=%.12f by gs sweep %d  (cum=%.3fs  cum_excl1=%.3fs)\n",
                    target_E, res_gs.target_reached_sweep, res_gs.target_reached_cum,
                    res_gs.target_reached_cum_excl1)
        else
            @printf("did NOT reach target E=%.12f within %d sweeps (final %.12f)\n",
                    target_E, n_sweeps, E_gs)
        end
    end

    # ── Regression verdict ─────────────────────────────────────────────────
    hon = honest_linkdims(psi_gs)
    _, _, _, _, _, _, ratios = inspect_storage(psi_gs)
    honest_ok = all(h -> h <= maxdim, hon)
    dedup_ok  = !isempty(ratios) && minimum(ratios) > 1.0 + 1e-9
    println("\n=========== REGRESSION VERDICT (aliased vs dense) ===========")
    @printf("  [%s] honest bond dim obeys MAXDIM (no silent rank inflation)\n", honest_ok ? "PASS" : "FAIL")
    @printf("  [%s] alias dedup > 1x at every site (n_templates not collapsed to 1)\n", dedup_ok ? "PASS" : "FAIL")
    if honest_ok && dedup_ok
        println("  → Structure preserved: honest BD ≤ MAXDIM AND alias dedup intact.")
        println("    Compare footprint against test_sparse_pxp.jl (BS Path-A) at same --N/--maxdim/--n-sweeps.")
    else
        println("  → POSSIBLE REGRESSION TO DENSE:")
        !honest_ok && println("    - honest bond dim exceeds MAXDIM → truncation not bounding the true rank.")
        !dedup_ok  && println("    - some site has n_templates == n_blocks → one-template collapse (dedup lost).")
    end

    if get(ENV, "SB_CAS_STATS", "0") == "1"
        println("\n========== CAS redundancy stats =========="); SparseBackends.show_cas_stats()
    end
    println("\n========== ITensorMPS.PROJMPO_TIMER ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
    println("\n========== SparseBackends.TIMER ==========")
    print_timer(SparseBackends.TIMER)
end
nothing
