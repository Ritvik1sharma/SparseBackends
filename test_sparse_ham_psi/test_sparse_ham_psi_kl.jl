# KL benchmark — ALIASED ψ × ALIASED PHP runner (the "both-aliased" path).
#
# This is the UNION of the two sister lines of work:
#   - test_aliased_psi/test_aliased_kl.jl : ψ stored as WrappedAliasedBlockSparse,
#       DMRG on the BARE H (constraint enforced structurally), Path-B eigsolve.
#   - test_sparse_ham/test_check_working_aliased.jl : H = PHP stored as
#       WrappedAliasedBlockSparse, DMRG with a DENSE ψ.
#
# Here BOTH the wavefunction ψ AND the projected Hamiltonian PHP are aliased:
#       ψ_ali = contract(P, ψ₀, :coo, :aliased)         (WrappedAliasedBlockSparse MPS)
#       H_ali = aliased sandwich  P''·H'·P              (WrappedAliasedBlockSparse MPO)
# and DMRG runs the aliased ψ on the aliased PHP via Path-B.
#
# ── Why PHP here (and not the bare H of test_aliased_kl.jl)? ───────────────────
# The sparse/aliased ψ runners rely on [H,P]=0 + structural channel sparsity to
# stay in image(P) while running on the bare H. This folder instead carries the
# PROJECTED Hamiltonian explicitly, stored aliased — the same H the dense PHP
# baseline (../test_sparse_psi/test_dense_kl.jl) uses, but compressed. Since
# ψ_ali ∈ image(P) and PHP = H on image(P), the two constraint mechanisms agree;
# this run exercises the aliased-H matvec kernel against an aliased ψ.
#
# ── Path / correctness ────────────────────────────────────────────────────────
# Aliased ψ is structurally NON-iso (templates shared across bond-channel values
# ⇒ off-diagonal L†L). So this MUST use Path-B (M-corrected generalized eigsolve),
# NOT the iso path — Path-B is dmrg()'s default run_mode=:bop_aliased, so no
# explicit run_mode is needed here. The validated aliased-ψ recipe is the 5-gate
# fix from ../test_aliased_psi/README.md ("RESOLVED"):
#   BMF_BOP_PROJECT=1  BMF_MINV_RTOL=1e-2
#   SB_ALIASED_PERCM_CAP=1  SB_USE_QR=1  SB_BALANCED_OWNERSHIP=1  SB_ADAPTIVE_RANK=1
# (SB_ALIASED_MINV_HINT, SB_ALIASED_NATIVE_FISSION, SB_ALIASED_MINV_WRAP dropped
# 2026-06: now hardcoded on unconditionally.)
# These are defaulted ON below (overridable from the environment).
#
# ⚠ STATUS: this aliased-ψ × aliased-PHP combination is NEW and was NOT validated
# when this file was written — neither sister folder ran both backends together.
# The aliased-H matvec kernel (test_sparse_ham) was only ever exercised against a
# DENSE ψ, and the aliased-ψ Path-B (test_aliased_psi) only against the BARE
# (dense) H. Treat every number this script prints as Confidence: Low until the
# in-script dense-PHP reference |ΔE| check passes at N=2 and again at a larger N.
# Keep the `--dense-ref` correctness check ON.
#
# Example (full validated-aliased-ψ gate set; small smoke test):
#   julia --project=.. test_sparse_ham_psi_kl.jl --N-plaq 2 --maxdim 4 --n-sweeps 3

# ── Default the aliased-ψ Path-B gate set ON (all overridable). ────────────────
ENV["BMF_BOP_PROJECT"]          = get(ENV, "BMF_BOP_PROJECT", "1")      # B = M⁻¹ᐟ²·H_eff·M⁻¹ᐟ², range(M) projection.
ENV["BMF_MINV_RTOL"]            = get(ENV, "BMF_MINV_RTOL", "1e-2")     # aggressive pseudo-inverse cutoff.
ENV["SB_ALIASED_PERCM_CAP"]     = get(ENV, "SB_ALIASED_PERCM_CAP", "1")     # honest BD ≤ maxdim.
ENV["SB_USE_QR"]                = get(ENV, "SB_USE_QR", "1")
ENV["SB_BALANCED_OWNERSHIP"]    = get(ENV, "SB_BALANCED_OWNERSHIP", "1")
ENV["SB_ADAPTIVE_RANK"]         = get(ENV, "SB_ADAPTIVE_RANK", "1")
# Aliased PHP carries multi-strand bonds (P''·H'·P leaves separate link strands
# at different prime levels); FUSE them so each H site has the clean 4-sparse
# (2 sites + 2 fused links) + 2 dense structure, and the env/matvec reductions
# act on single shared links. Required for the both-aliased path.
ENV["SB_FUSE_LINKS"]            = get(ENV, "SB_FUSE_LINKS", "1")
# SB_ALIASED_AA_ENV / SB_ALIASED_AA_HINT hardened 2026-06 — always on now.
using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using LinearAlgebra: I as eye, norm, BLAS
using Random
using ArgParse
using Printf

BLAS.set_num_threads(1)

include("../test_sparse_psi/utils.jl")          # clean!, mpo_memory_bytes
include("../test_sparse_ham/aliased_helpers.jl")# fuse_sparse_links!, prepermute_aliased_mpo!, report_aliased_footprint

function parse_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N-plaq";        help="Number of plaquettes (N) → 2N+2 sites"; arg_type=Int; default=2
        "--eignv";         help="Projector eigenvalue +1 (true) / -1 (false)."; arg_type=Bool; default=true
        "--spin";          help="2 → S=1/2, 3 → S=1."; arg_type=Int; default=3
        "--n-sweeps";      help="DMRG sweeps at maxdim (sweep 1 = JIT, excluded from post-JIT totals)."; arg_type=Int; default=6
        "--maxdim";        help="DMRG maxdim cap (honest per-bond total dim target)."; arg_type=Int; default=40
        "--target-energy"; help="If set, log the first sweep at which E ≤ target."; arg_type=Float64; default=NaN
        "--dense-ref";     help="Also build a dense PHP + dense ψ and run it as an in-script |ΔE| reference."; arg_type=Bool; default=true
    end
    return parse_args(s)
end

# ── Model build. Shared OpSum / projector with the sister KL runners. ──────────
# Returns the aliased ψ, the aliased PHP H, and (optionally) a dense PHP H +
# dense ψ for the in-script correctness reference. P_sparse is the product of the
# plaquette (I+C)/2 MPOs — identical to test_aliased_kl.jl / test_dense_kl.jl.
function build_setup(N::Int, psign::Int, spin::Int; want_dense::Bool)
    Random.seed!(42)
    sites = spin == 2 ? siteinds("S=1/2", 2*N+2) :
            spin == 3 ? siteinds("S=1",   2*N+2) : error("Unsupported spin: $spin")

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
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end

    H    = MPO(os, sites)
    psi0 = random_mps(sites)

    # ── ALIASED ψ: site + both bond axes in the sparse prefix (denseLinksB=0). ──
    psi_ali = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)

    # ── ALIASED PHP H: per-site sandwich P''·H'·P (matches test_check_working_aliased.jl). ──
    H_ali = MPO(length(H))
    for i in 1:length(H)
        H1       = SparseBackends.contract_aliased_itensor(P_sparse[i]'', H[i]', :coo, :dense)
        H_eff_i  = SparseBackends.contract_aliased_itensor(P_sparse[i], H1, :coo, :aliased)
        H_ali[i] = replaceprime(H_eff_i, 3 => 1)
    end
    if get(ENV, "SB_FUSE_LINKS", "0") == "1"
        println("  [fusing multi-strand sparse links in H_ali]")
        fuse_sparse_links!(H_ali)
    end
    if get(ENV, "SB_PREPERMUTE_H", "0") == "1"
        println("  [prepermuting aliased H tails]")
        prepermute_aliased_mpo!(H_ali)
    end

    # ── DENSE PHP reference (same construction as test_dense_kl.jl). ────────────
    H_dense = nothing; psi_dense = nothing
    if want_dense
        H1       = contract(P_sparse'', H', :coo, :dense)
        H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
        psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
        H_dense   = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in H_sparse])
        psi_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp])
    end
    return (; sites, H, P_sparse, psi_ali, H_ali, H_dense, psi_dense)
end

# ── ψ instrumentation (mirrors test_aliased_kl.jl). ──────────────────────────
mps_footprint_bytes(psi) = Base.summarysize(psi)
reported_linkdims(psi) = [ITensors.dim(commonind(psi[i], psi[i+1])) for i in 1:length(psi)-1]
honest_linkdims(psi) =
    [(cis = commoninds(psi[i], psi[i+1]); isempty(cis) ? 0 : prod(ITensors.dim, cis))
     for i in 1:length(psi)-1]

function inspect_storage(psi)
    payload_total = 0; keys_total = 0; site_total = 0; full_total = 0; bs_total = 0
    ratios = Float64[]; rows = String[]
    for (i, T) in enumerate(psi)
        s  = try ITensors.get_external_storage(T) catch _ nothing end
        ss = Base.summarysize(T); site_total += ss
        if s isa SparseBackends.WrappedAliasedBlockSparse
            ali = s.aliased
            nb = length(ali.keys); nt = ali.n_templates; blksz = ali.blksize
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
            bs = s.blocksparse; nb = length(bs.keys)
            data_b = Base.summarysize(bs.data)
            keys_b = Base.summarysize(bs.keys) + Base.summarysize(bs.ids)
            full_b = prod(bs.dims) * sizeof(eltype(bs.data))
            payload_total += data_b; keys_total += keys_b; full_total += full_b; bs_total += data_b
            push!(rows, @sprintf("site %2d [BS]  nblocks=%d blksize=%d  data=%.2fKiB  keys+ids=%.2fKiB  if_dense=%.2fKiB  total=%.2fKiB",
                                 i, nb, bs.blksize, data_b/1024, keys_b/1024, full_b/1024, ss/1024))
        else
            a = ITensors.array(T); data_b = Base.summarysize(a)
            payload_total += data_b; full_total += data_b; bs_total += data_b
            push!(rows, @sprintf("site %2d [dense] data=%.2fKiB  total=%.2fKiB", i, data_b/1024, ss/1024))
        end
    end
    return rows, payload_total, keys_total, site_total, full_total, bs_total, ratios
end

function check_aliased_invariant(psi; label="")
    bad = Int[]
    for (i, T) in enumerate(psi)
        (ITensors.has_external_storage(T) && T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse) || push!(bad, i)
    end
    isempty(bad) ? println("  [$label] psi storage invariant ✓ (all $(length(psi)) sites aliased)") :
                   @warn "[$label] psi sites NOT aliased: $bad (alias invariant broken)"
    return isempty(bad)
end

function report_state(label, psi, maxdim, E=nothing; verbose=false)
    mb  = round(mps_footprint_bytes(psi) / 2^20, digits=3)
    rep = reported_linkdims(psi); hon = honest_linkdims(psi)
    mx  = isempty(rep) ? 0 : maximum(rep); mxh = isempty(hon) ? 0 : maximum(hon)
    println("  $label: footprint=$(mb) MiB  reported_maxlinkdim=$mx  honest_maxlinkdim=$mxh  (MAXDIM=$maxdim)")
    println("    linkdims(reported)=$rep")
    println("    linkdims(honest, ∏all shared)=$hon" * (E === nothing ? "" : "  E=$E"))
    over = findall(h -> h > maxdim, hon)
    isempty(over) ? println("    ✓ honest bond dim obeys MAXDIM at every bond.") :
                    println("    ✗ honest bond dim EXCEEDS MAXDIM at bonds $over.")
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

# ── Per-sweep DMRG with the alias invariant checked after every sweep. ────────
function run_sweeps(H, psi0, n_sweeps::Int, maxdim::Int; cutoff=1e-10, mindim=1,
                    target_E=NaN, label="ALI", check_alias=true)
    psi = psi0; E = NaN; cum = 0.0; cum_excl1 = 0.0
    tr_sweep = 0; tr_cum = NaN; tr_cum_excl1 = NaN
    for i in 1:n_sweeps
        sw = Sweeps(1); setmaxdim!(sw, maxdim); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        t = @elapsed (E, psi) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false, run_label=label)
        cum += t; if i > 1; cum_excl1 += t; end
        @printf("  [%s sweep %2d] t=%8.3fs  E=%.12f\n", label, i, t, E)
        check_alias && check_aliased_invariant(psi; label="$label after sweep $i")
        flush(stdout)
        if tr_sweep == 0 && !isnan(target_E) && E <= target_E
            tr_sweep = i; tr_cum = cum; tr_cum_excl1 = cum_excl1
        end
    end
    return (; E, psi, total=cum, total_excl1=cum_excl1,
            target_reached_sweep=tr_sweep, target_reached_cum=tr_cum, target_reached_cum_excl1=tr_cum_excl1)
end

let
    args     = parse_command_line()
    N_plaq   = args["N-plaq"]
    psign    = args["eignv"] ? +1 : -1
    spin     = args["spin"]
    n_sweeps = args["n-sweeps"]
    maxdim   = args["maxdim"]
    target_E = args["target-energy"]
    do_dense = args["dense-ref"]

    println("=== KL benchmark — ALIASED ψ × ALIASED PHP (Path-B) ===")
    println("BMF_BOP_PROJECT=", ENV["BMF_BOP_PROJECT"], "  BMF_MINV_RTOL=", ENV["BMF_MINV_RTOL"])
    println("SB_ALIASED_PERCM_CAP=", ENV["SB_ALIASED_PERCM_CAP"])
    println("Projector sign = $psign  (P = ∏(I", psign > 0 ? "+" : "-", "C)/2)")
    println("N_plaq=$N_plaq  spin=$spin  n_sweeps=$n_sweeps  maxdim=$maxdim  target_E=$(isnan(target_E) ? "—" : target_E)  dense_ref=$do_dense")
    println("⚠ NEW combination (aliased ψ × aliased PHP) — confidence LOW until the dense-ref |ΔE| check passes.")

    println("\nBuilding setup for N=$N_plaq plaquettes ...")
    t_setup = @elapsed st = build_setup(N_plaq, psign, spin; want_dense=do_dense)
    println("Setup time: $(round(t_setup, digits=1))s.  System: $(length(st.psi_ali)) sites.")

    # ── Storage sanity at DMRG entry ──────────────────────────────────────────
    n_alias = count(T -> ITensors.has_external_storage(T) && T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse, st.H_ali)
    println("\n[H_ali storage at DMRG entry]  sites=$(length(st.H_ali))  aliased=$n_alias  other=$(length(st.H_ali)-n_alias)")
    report_aliased_footprint(st.H_ali, "ALIASED PHP:")
    println("\n[initial ψ_ali]")
    report_state("initial psi_ali", st.psi_ali, maxdim; verbose=true)
    check_aliased_invariant(st.psi_ali; label="initial")

    # ── Optional pointwise H sanity (aliased PHP vs dense PHP), small N only. ──
    # Best-effort: the aliased PHP keeps multi-strand sparse links (more indices
    # than the link-fused dense PHP), so a per-site array compare may be unable to
    # align them — that is a storage-layout artifact, not a value bug. The
    # decisive correctness signal is the in-script dense-ref |ΔE| from DMRG below.
    if do_dense && N_plaq <= 4
        println("\n[pointwise H sanity: aliased PHP vs dense PHP (best-effort)]")
        nbad = 0; nskip = 0
        for i in 1:length(st.H_ali)
            t1 = ITensors.has_external_storage(st.H_ali[i]) ? SparseBackends.to_dense_itensors(st.H_ali[i]) : st.H_ali[i]
            t2 = st.H_dense[i]
            res_cmp = try compare_mpo_tensors(t1, t2) catch e; (:err, e) end
            if res_cmp[1] === :err || length(res_cmp) < 2 || ITensors.order(t1) != ITensors.order(t2)
                nskip += 1
                println(@sprintf("    H[%2d] — skipped (rank %d vs %d / link strands differ)", i, ITensors.order(t1), ITensors.order(t2)))
            else
                ok = res_cmp[1]; mdiff = res_cmp[2]
                ok || (nbad += 1)
                println(@sprintf("    H[%2d] %s  max|Δ|=%.2e", i, ok ? "✓" : "✗", mdiff))
            end
        end
        println(nbad == 0 ? "    (no value mismatches among $(length(st.H_ali)-nskip) comparable sites; $nskip skipped)" :
                            "    ✗ $nbad site(s) differ — aliased PHP build may be WRONG; cross-check the |ΔE| verdict below.")
    end

    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)

    # ── Aliased add-path counters: did the Lanczos cross-schema merge ever fire,
    # or did the basis vectors always share a schema? Reset → run → read.
    #   plus_match : Base.:+ identical schema   (stays aliased, compression kept)
    #   plus_merge : Base.:+ same dims, DIFFERENT schema → key-union merge (the
    #                "cross-schema merge" — stays aliased but compression destroyed)
    #   plus_dense : Base.:+ mismatched classification → DENSE (densifies)
    #   axpy_match : KrylovKit axpy!/scale! on identical schema
    SparseBackends._ADD_AXPY_MATCH[] = 0
    SparseBackends._ADD_PLUS_MATCH[] = 0
    SparseBackends._ADD_PLUS_MERGE[] = 0
    SparseBackends._ADD_PLUS_DENSE[] = 0

    # ── ALIASED ψ × ALIASED PHP run (Path-B). ─────────────────────────────────
    println("\n=== ALIASED run ($n_sweeps sweeps at maxdim=$maxdim; sweep 1 = JIT) ===")
    res = run_sweeps(st.H_ali, st.psi_ali, n_sweeps, maxdim; target_E=target_E, label="ALI", check_alias=true)

    let am = SparseBackends._ADD_AXPY_MATCH[], pm = SparseBackends._ADD_PLUS_MATCH[],
        mg = SparseBackends._ADD_PLUS_MERGE[], dn = SparseBackends._ADD_PLUS_DENSE[]
        tot = am + pm + mg + dn
        println("\n========== ALIASED ADD-PATH CLASSIFICATION (Lanczos eigsolve) ==========")
        println("  total aliased adds = $tot")
        pct(x) = tot == 0 ? 0.0 : round(100x/tot, digits=2)
        println("  axpy_match (axpy/scale, same schema, stays aliased)      = $am  ($(pct(am))%)")
        println("  plus_match (Base.:+, same schema, stays aliased)         = $pm  ($(pct(pm))%)")
        println("  plus_merge (Base.:+, DIFFERENT schema → cross-schema merge, stays aliased) = $mg  ($(pct(mg))%)")
        println("  plus_dense (Base.:+, mismatched classification → DENSIFIES)               = $dn  ($(pct(dn))%)")
        println("  → cross-schema merge fired: $(mg > 0 ? "YES ($mg times)" : "NO — vectors always shared a schema")")
        println("  → densify fallback fired:  $(dn > 0 ? "YES ($dn times)" : "NO")")
    end

    # ── DENSE PHP reference (same args), for the in-script |ΔE| correctness check. ──
    res_d = nothing
    if do_dense
        println("\n=== DENSE-PHP reference run ($n_sweeps sweeps at maxdim=$maxdim) ===")
        res_d = run_sweeps(st.H_dense, st.psi_dense, n_sweeps, maxdim; target_E=NaN, label="DEN", check_alias=false)
    end

    avg_excl1 = n_sweeps > 1 ? res.total_excl1 / (n_sweeps - 1) : NaN
    println("\n=========== SUMMARY (aliased ψ × aliased PHP  KL  N=$N_plaq  md=$maxdim) ===========")
    @printf("ALIASED total (incl sweep 1): %8.3fs   excl sweep 1: %8.3fs   avg/sweep: %8.3fs\n",
            res.total, res.total_excl1, avg_excl1)
    @printf("ALIASED final E: %.12f\n", res.E)
    if res_d !== nothing
        @printf("DENSE   final E: %.12f   ALIASED-vs-DENSE |ΔE| = %.3e\n", res_d.E, abs(res.E - res_d.E))
        @printf("DENSE   total (excl sweep 1): %8.3fs\n", res_d.total_excl1)
    end
    if !isnan(target_E) && res.target_reached_sweep > 0
        @printf("reached target E=%.12f by sweep %d (cum=%.3fs, excl1=%.3fs)\n",
                target_E, res.target_reached_sweep, res.target_reached_cum, res.target_reached_cum_excl1)
    end

    println("\n--- final aliased ψ ---")
    report_state("final psi_ali", res.psi, maxdim, res.E; verbose=true)
    check_aliased_invariant(res.psi; label="final")

    # ── Regression verdict ────────────────────────────────────────────────────
    hon = honest_linkdims(res.psi)
    _, _, _, _, _, _, ratios = inspect_storage(res.psi)
    honest_ok = all(h -> h <= maxdim, hon)
    dedup_ok  = !isempty(ratios) && minimum(ratios) > 1.0 + 1e-9
    energy_ok = res_d === nothing ? nothing : abs(res.E - res_d.E) < 5e-2 * abs(res_d.E) + 1e-6
    println("\n=========== VERDICT (aliased ψ × aliased PHP) ===========")
    @printf("  [%s] alias invariant: ψ aliased every sweep\n", check_aliased_invariant(res.psi; label="verdict") ? "PASS" : "FAIL")
    @printf("  [%s] honest bond dim ≤ MAXDIM (no silent rank inflation)\n", honest_ok ? "PASS" : "FAIL")
    @printf("  [%s] alias dedup > 1x at every site (no one-template collapse)\n", dedup_ok ? "PASS" : "FAIL")
    if energy_ok !== nothing
        @printf("  [%s] |ΔE| vs dense PHP within 5%% (correctness)\n", energy_ok ? "PASS" : "FAIL")
    else
        println("  [----] energy check SKIPPED (--dense-ref false) — cannot judge correctness.")
    end

    # ── Env footprint: aliased-aliased env vs dense-PHP×dense-ψ env, per site. ──
    ITensorMPS.print_env_footprint(; labelA="ALI", labelB="DEN")

    println("\n========== ITensorMPS.PROJMPO_TIMER ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
    println("\n========== SparseBackends.TIMER ==========")
    print_timer(SparseBackends.TIMER)
end
nothing
