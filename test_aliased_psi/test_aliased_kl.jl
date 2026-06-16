# KL benchmark — ALIASED-ψ runner (single backend).
#
# Mirror of ../test_sparse_psi/test_sparse_kl.jl, but ψ is carried as
# WrappedAliasedBlockSparse (alias structure frozen across sweeps; only template
# numeric data updates). This is the canonical aliased KL test — same model,
# same CLI, same instrumentation as the sister sparse/dense KL runners, so the
# three are directly comparable when run with matching args.
#
# Constraint enforcement (see ../test_sparse_psi/README.md):
#   - Aliased/BS ψ run DMRG on the BARE H. The projector constraint is enforced
#     STRUCTURALLY: ψ = P·ψ₀ lives in image(P) and the channel sparsity is
#     preserved across sweeps, so we rely on [H,P]=0 to stay in image(P).
#   - The DENSE baseline (../test_sparse_psi/test_dense_kl.jl) does NOT have a
#     structural constraint, so it must run DMRG on the PROJECTED Hamiltonian
#     H_dense = densify(P·H·P). Do NOT compare against a bare-H dense run from an
#     unprojected ψ — that solves the unconstrained problem and is not a valid
#     constrained baseline.
#
# Pathway: aliased ψ is structurally NON-iso (templates shared across
# bond-channel values ⇒ off-diagonal L†L coupling). The reliable path is
# Path-B (M-corrected generalized eigsolve): BMF_ISO_PATH=0 with the M⁻¹
# operator fix BMF_APPLY_MINV=1. The iso path (BMF_ISO_PATH=1, used by the
# sister sparse KL runner where strict-cap SVD keeps ψ iso) is NOT valid for
# aliased ψ and gives unphysical energies — do not use it here.
#
# Gated: requires SB_ALIASED_ENABLE=1.
#
# Example:
#   SB_ALIASED_ENABLE=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
#     julia --project=.. test_aliased_kl.jl --N-plaq 12 --maxdim 40 --n-sweeps 10
ENV["SB_ALIASED_ENABLE"] = get(ENV, "SB_ALIASED_ENABLE", "1")
ENV["BMF_ISO_PATH"]      = "0"   # Path-B (M-corrected eigsolve) — required for aliased ψ.
ENV["BMF_APPLY_MINV"]    = get(ENV, "BMF_APPLY_MINV", "1")  # apply A = M⁻¹·H_eff (energy-correctness fix).

using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using LinearAlgebra: I as eye, norm
using Random
using ArgParse
using Printf

# KrylovKit threading is a SEPARATE knob from the aliased-kernel threading
# (SB_ALIASED_NTHREADS). KrylovKit's __init__ defaults its count to Threads.nthreads(),
# so launching `julia --threads=N` would silently turn on its threaded orthogonalization
# (and confound a kernel-threading A/B). Pin it explicitly here; default 1 = serial,
# matching every prior single-Julia-thread run. Set SB_KK_NTHREADS>1 to opt in.
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

function parse_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N-plaq"
            help = "Number of plaquettes (N) → system size = 2N+2 sites"
            arg_type = Int
            default = 12
        "--eignv"
            help = "Whether to use eigenvalue +1/-1 (true → +1, false → -1)."
            arg_type = Bool
            default = true
        "--spin"
            help = "Local Hilbert space: 2 → S=1/2, 3 → S=1."
            arg_type = Int
            default = 3
        "--n-sweeps"
            help = "Total DMRG sweeps at maxdim. Sweep 1 is the JIT warmup and is excluded from post-JIT totals."
            arg_type = Int
            default = 10
        "--maxdim"
            help = "DMRG maxdim cap (honest per-bond total dim target)."
            arg_type = Int
            default = 40
        "--target-energy"
            help = "If set, log the first sweep at which E ≤ target (does not early-exit)."
            arg_type = Float64
            default = NaN
    end
    return parse_args(s)
end

# ── Setup: same KL model + projector as test_sparse_kl.jl / test_dense_kl.jl.
# ψ is built ALIASED (`:coo, :aliased`); DMRG runs on the BARE H. ────────────
function build_setup(N::Int, psign::Int, spin::Int)
    Random.seed!(42)
    if spin == 2
        sites = siteinds("S=1/2", 2*N+2)
    elseif spin == 3
        sites = siteinds("S=1", 2*N+2)
    else
        error("Unsupported spin: $spin")
    end
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    cs = 0.5 * psign
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += cs,  "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    psi0     = random_mps(sites)
    # KEY: ψ is ALIASED (vs BS in test_sparse_kl.jl). denseLinksB=0 keeps site +
    # both bond axes in the sparse prefix. DMRG runs on the bare H (above).
    psi_ali  = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)
    return H, psi_ali
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

# ── Schema-invariance check (SB_SCHEMA_TRACK=1) ────────────────────────────
# ψ_aliased ≡ P·ψ_dense, so the alias SCHEMA — keys (which prefix blocks are
# nonzero, set by P's sparsity), the key→template partition (alias_ids, set by
# P's structure), and scalars (set by P's values) — encodes the constraint P,
# which never changes. Across the whole DMRG flow only the *templates* (the
# ψ_dense slices) may grow; keys / partition / scalars MUST stay invariant. Any
# drift = the schema-freeze is broken = a bug.
# Per-site fingerprint: (Set(keys), n_templates, sorted scalars, key→template
# partition as a relabel-invariant Set-of-Sets-of-keys).
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

function run_sweeps(H, psi0, n_sweeps::Int, maxdim::Int; cutoff=1e-10, mindim=1, target_E=NaN, label="ALI")
    psi = psi0
    # SB_SCHEMA_DBG: print the P-classification (sparse keys vs dense tail) of the
    # freshly-constructed aliased ψ at each site — the reference schema that the
    # DMRG operations (orthogonalize/eigsolve/replacebond/add) should preserve.
    if get(ENV, "SB_SCHEMA_DBG", "0") == "1"
        for k in 1:length(psi); SparseBackends.schema_dbg("CONSTRUCT site $k", psi[k]); end
    end
    E = NaN
    cum = 0.0; cum_excl1 = 0.0
    target_reached_sweep = 0; target_reached_cum = NaN; target_reached_cum_excl1 = NaN
    for i in 1:n_sweeps
        sw = Sweeps(1); setmaxdim!(sw, maxdim); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        t = @elapsed (E, psi, _esw, terr) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false)
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

let
    parsed_args = parse_command_line()
    N_plaq   = parsed_args["N-plaq"]
    psign    = parsed_args["eignv"] ? +1 : -1
    spin     = parsed_args["spin"]
    n_sweeps = parsed_args["n-sweeps"]
    maxdim   = parsed_args["maxdim"]
    target_E = parsed_args["target-energy"]

    println("=== KL benchmark — ALIASED ψ (Path-B) ===")
    println("BMF_ISO_PATH=", ENV["BMF_ISO_PATH"], "  BMF_APPLY_MINV=", ENV["BMF_APPLY_MINV"],
            "  (Path-B: A = M⁻¹·H_eff — the valid path for non-iso aliased ψ)")
    println("Projector sign = $psign  (P = ∏(I", psign > 0 ? "+" : "-", "C)/2)")
    println("N_plaq=$N_plaq  spin=$spin  n_sweeps=$n_sweeps  maxdim=$maxdim  target_E=$(isnan(target_E) ? "—" : target_E)")
    _percm = get(ENV, "SB_ALIASED_PERCM_CAP", "0")
    println("SB_ALIASED_PERCM_CAP=$_percm  → ", _percm == "1" ?
            "CAPPED (mult=fld(maxdim,channel), honest_bd≤maxdim — starved, worse energy/maxdim)" :
            "UNCAPPED (mult=maxdim, honest_bd=channel×maxdim — the benchmarked regime) [DEFAULT]")
    println("ψ = ALIASED (P·ψ₀); DMRG on BARE H (constraint enforced structurally by channel sparsity).")
    println("Building setup for N=$N_plaq plaquettes ...")
    t_setup = @elapsed (H, psi_ali) = build_setup(N_plaq, psign, spin)
    println("Setup time: $(round(t_setup, digits=1))s.  System: $(length(psi_ali)) sites.")
    report_state("initial psi_ali", psi_ali, maxdim; verbose=true)
    if get(ENV, "SB_INSPECT_PHI", "0") == "1"
        println("\n========== φ index classification (per site) ==========")
        for (i, T) in enumerate(psi_ali)
            s = try ITensors.get_external_storage(T) catch _; nothing end
            s isa SparseBackends.WrappedAliasedBlockSparse || continue
            ali = s.aliased
            NN = ndims(T); P = typeof(ali).parameters[4]; N2 = typeof(ali).parameters[3]
            idxinfo = [(string(ITensors.tags(I)), ITensors.dim(I), ITensors.plev(I)) for I in s.inds]
            println("site $i: N=$NN P(prefix)=$P N2(densetail)=$N2 dims=$(ali.dims) blksize=$(ali.blksize)")
            println("   prefix axes 1:$P = ", idxinfo[1:P])
            println("   dense  axes $(P+1):$NN = ", idxinfo[P+1:NN])
        end
        println("======================================================\n")
        exit(0)
    end
    if get(ENV, "SB_SCHEMA_TRACK", "0") == "1"
        _INIT_SCHEMA[] = _schema_fingerprint(psi_ali)
        println("  [init] captured alias schema fingerprint (keys/partition/scalars) for invariance tracking")
    end
    check_aliased_invariant(psi_ali; label="initial")
    flush(stdout)

    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)
    SparseBackends.reset_cas_stats!()
    println("\n=== RUN ($n_sweeps sweeps at maxdim=$maxdim; sweep 1 = JIT) ===")
    res = run_sweeps(H, psi_ali, n_sweeps, maxdim; target_E=target_E)
    E_prof = res.E; psi_prof = res.psi

    avg_excl1 = n_sweeps > 1 ? res.total_excl1 / (n_sweeps - 1) : NaN
    println("\n=========== SUMMARY (aliased KL  N=$N_plaq  md=$maxdim) ===========")
    @printf("total time (incl sweep 1, JIT): %8.3fs\n", res.total)
    @printf("total time (excl sweep 1):      %8.3fs\n", res.total_excl1)
    @printf("avg per sweep (excl sweep 1):   %8.3fs\n", avg_excl1)
    @printf("final E: %.12f\n", E_prof)
    if !isnan(target_E)
        if res.target_reached_sweep > 0
            @printf("reached target E=%.12f by sweep %d  (cum=%.3fs  cum_excl1=%.3fs)\n",
                    target_E, res.target_reached_sweep, res.target_reached_cum, res.target_reached_cum_excl1)
        else
            @printf("did NOT reach target E=%.12f within %d sweeps (final %.12f)\n",
                    target_E, n_sweeps, E_prof)
        end
    end

    println("\n--- final state ---")
    report_state("final psi_ali", psi_prof, maxdim, E_prof; verbose=true)
    print_iso("final", psi_prof)
    check_aliased_invariant(psi_prof; label="final")

    # ── Regression verdict: did aliased regress to dense? ──────────────────
    hon = honest_linkdims(psi_prof)
    _, _, _, _, _, _, ratios = inspect_storage(psi_prof)
    honest_ok = all(h -> h <= maxdim, hon)
    dedup_ok  = !isempty(ratios) && minimum(ratios) > 1.0 + 1e-9
    println("\n=========== REGRESSION VERDICT (aliased vs dense) ===========")
    @printf("  [%s] honest bond dim obeys MAXDIM (no silent rank inflation)\n", honest_ok ? "PASS" : "FAIL")
    @printf("  [%s] alias dedup > 1x at every site (n_templates not collapsed to 1)\n", dedup_ok ? "PASS" : "FAIL")
    if honest_ok && dedup_ok
        println("  → Structure preserved: honest BD ≤ MAXDIM AND alias dedup intact (no one-template collapse).")
        println("    Compare footprint against ../test_sparse_psi/test_dense_kl.jl (PHP dense) and")
        println("    ../test_sparse_psi/test_sparse_kl.jl (BS) at the SAME --N-plaq/--maxdim/--n-sweeps.")
    else
        println("  → POSSIBLE REGRESSION TO DENSE:")
        !honest_ok && println("    - honest bond dim exceeds MAXDIM → truncation not bounding the true rank.")
        !dedup_ok  && println("    - some site has n_templates == n_blocks → one-template collapse (dedup lost).")
        println("    Datastructure regression, NOT mere overhead — diagnose before any fix (hypothesis-tag it).")
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
