# KL benchmark — ALIASED-ψ runner (single backend).
# Mirror of ../test_sparse_psi/test_sparse_kl.jl, but ψ is carried as
# WrappedAliasedBlockSparse (alias structure frozen across sweeps; only template
# numeric data updates). 
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
# Path-B (M-corrected generalized eigsolve): run_mode=:bop_aliased (the
# dmrg() default). run_mode=:iso (used by the sister sparse KL runner where
# strict-cap SVD keeps ψ iso) is NOT valid for aliased ψ and gives unphysical
# energies — do not use it here.
#
# Example:
#   SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
#     julia --project=.. test_aliased_kl.jl --N-plaq 12 --maxdim 40 --n-sweeps 10
# Eigensolve pathway is selected by the dmrg(...) run_mode kwarg (see dmrg call below),
# not env flags. Aliased ψ uses run_mode=:bop_aliased (Path-B B=M^{-1/2}HM^{-1/2}, no densify).

using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using LinearAlgebra: I as eye, norm, mul!, BLAS
using Random
using ArgParse
using Printf

# KrylovKit threading is a SEPARATE knob from the aliased-kernel threading
# (SB_ALIASED_NTHREADS). KrylovKit's __init__ defaults its count to Threads.nthreads(),
# so launching `julia --threads=N` would silently turn on its threaded orthogonalization
# (and confound a kernel-threading A/B). Hardened 2026-06 to the default serial
# count (was SB_KK_NTHREADS, default-on knob, never varied off 1).
import KrylovKit
KrylovKit.set_num_threads(1)
println("[KrylovKit threads = ", KrylovKit.get_num_threads(),
        "   Julia threads = ", Threads.nthreads(), "]")

include("../test_sparse_psi/utils.jl")
include("../test_aliased_psi/setup.jl")

# Schema for the initial aliased ψ (frozen across sweeps; only template numeric data updates). Used for invariance tracking.
const _INIT_SCHEMA = Ref{Any}(nothing)

# Debug toggles (were SB_SCHEMA_DBG / SB_SCHEMA_TRACK env vars) — flip to true
# manually here to enable; SparseBackends.schema_dbg has its own independent
# _SCHEMA_DBG const in tensor_wrappers_aliased.jl for its internal dumps.
const _SCHEMA_DBG_ON = false
const _SCHEMA_TRACK_ON = false


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
        "--run-mode"
            help = "Eigensolve pathway: bop_aliased (default, no densify) or bop_densify (env-dressed densified seed)."
            arg_type = String
            default = "bop_aliased"
        "--roofline"
            help = "Enable the consolidated roofline/flop-count/env-footprint/perm-capture/GEMM-histogram instrumentation."
            arg_type = Bool
            default = false
        "--minv-from-p"
            help = "minv_from_p kwarg: true (DEFAULT, matches dmrg) → M^{±1/2}=c^{∓…}·Lgram (from-P, skips eigendecomposition); false → eigen path."
            arg_type = Bool
            default = true
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
    # Pre-order each BULK H site tensor ONCE at construction to the matvec's EXACT
    # permB-canonical layout [Site(ket,plev0), left-link, Site(bra,plev1), right-link],
    # so the aliased matvec's permB on the H steps (2,3) is IDENTITY — no runtime
    # reorder. Direction-independent: the matvec chain is always Lenv→H[b]→H[b+1]→Renv,
    # so H[b] always contracts its LEFT link (measured permB uniformly [1,2,4,3] across
    # all bulk bonds AND both sweep directions). Correctness is name-based (position! +
    # matvec contract by index name) → E bit-identical. Bulk only (edges differ).
    for s in 2:(length(H) - 1)
        Is    = collect(inds(H[s]))
        ket   = [I for I in Is if ITensors.plev(I) == 0 && ITensors.hastags(I, "Site")]
        bra   = [I for I in Is if ITensors.plev(I) == 1 && ITensors.hastags(I, "Site")]
        left  = ITensors.commonind(H[s], H[s - 1])
        right = ITensors.commonind(H[s], H[s + 1])
        (length(ket) == 1 && length(bra) == 1 && left !== nothing && right !== nothing) || continue
        tgt = [ket[1], left, bra[1], right]
        Is != tgt && (H[s] = permute(H[s], tgt...))
    end
    psi0     = random_mps(sites)
    # KEY: ψ is ALIASED (vs BS in test_sparse_kl.jl). denseLinksB=0 keeps site +
    # both bond axes in the sparse prefix. DMRG runs on the bare H (above).
    psi_ali  = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)
    return H, psi_ali
end



function run_sweeps(H, psi0, n_sweeps::Int, maxdim::Int; cutoff=1e-10, mindim=1, target_E=NaN, label="ALI", run_mode::Symbol=:bop_aliased, roofline::Bool=false, minv_from_p=nothing)
    psi = psi0
    # _SCHEMA_DBG_ON: print the P-classification (sparse keys vs dense tail) of the
    # freshly-constructed aliased ψ at each site — the reference schema that the
    # DMRG operations (orthogonalize/eigsolve/replacebond/add) should preserve.
    if _SCHEMA_DBG_ON
        for k in 1:length(psi); SparseBackends.schema_dbg("CONSTRUCT site $k", psi[k]); end
    end
    E = NaN
    cum = 0.0; cum_excl1 = 0.0
    target_reached_sweep = 0; target_reached_cum = NaN; target_reached_cum_excl1 = NaN
    for i in 1:n_sweeps
        sw = Sweeps(1); setmaxdim!(sw, maxdim); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        _st = @timed (E, psi, _esw, terr) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false, run_mode=run_mode, roofline=roofline, minv_from_p=minv_from_p)
        t = _st.time; _gct = _st.gctime
        cum += t; if i > 1; cum_excl1 += t; end
        @printf("  [%s sweep %2d] t=%8.3fs  gc=%7.3fs (%.0f%%)  E=%.12f  maxtruncerr=%.3e\n", label, i, t, _gct, 100*_gct/t, E, terr)
        # Reset timers AFTER sweep 1 (JIT/compilation) so the printed breakdown
        # reflects STEADY-STATE only (sweeps 2..n), not JIT-polluted totals.
        if i == 1
            reset_timer!(ITensorMPS.PROJMPO_TIMER)
            reset_timer!(SparseBackends.TIMER)
            roofline && SparseBackends.reset_roofline!(roofline)
        end
        check_aliased_invariant(psi; label="after sweep $i")
        if _SCHEMA_TRACK_ON && _INIT_SCHEMA[] !== nothing
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
    roofline = parsed_args["roofline"]

    println("=== KL benchmark — ALIASED ψ (Path-B) ===")
    println("run_mode=:$(parsed_args["run-mode"])  (Path-B: B = M⁻¹ᐟ²·H_eff·M⁻¹ᐟ²; aliased=no-densify seed, densify=env-dressed densified seed)")
    println("Projector sign = $psign  (P = ∏(I", psign > 0 ? "+" : "-", "C)/2)")
    println("N_plaq=$N_plaq  spin=$spin  n_sweeps=$n_sweeps  maxdim=$maxdim  target_E=$(isnan(target_E) ? "—" : target_E)")
    println("UNCAPPED (mult=maxdim, honest_bd=channel×maxdim — the benchmarked regime) [DEFAULT]")
    println("ψ = ALIASED (P·ψ₀); DMRG on BARE H (constraint enforced structurally by channel sparsity).")
    println("Building setup for N=$N_plaq plaquettes ...")
    t_setup = @elapsed (H, psi_ali) = build_setup(N_plaq, psign, spin)
    println("Setup time: $(round(t_setup, digits=1))s.  System: $(length(psi_ali)) sites.")
    report_state("initial psi_ali", psi_ali, maxdim; verbose=true)
    if _SCHEMA_TRACK_ON
        _INIT_SCHEMA[] = _schema_fingerprint(psi_ali)
        println("  [init] captured alias schema fingerprint (keys/partition/scalars) for invariance tracking")
    end
    check_aliased_invariant(psi_ali; label="initial")
    flush(stdout)

    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)
    SparseBackends.reset_cas_stats!()
    SparseBackends.reset_roofline!(roofline)  # zero accumulators once before the sweep loop below
    println("\n=== RUN ($n_sweeps sweeps at maxdim=$maxdim; sweep 1 = JIT) ===")
    res = run_sweeps(H, psi_ali, n_sweeps, maxdim; target_E=target_E, run_mode=Symbol(parsed_args["run-mode"]), roofline=roofline,
                     minv_from_p=(parsed_args["minv-from-p"] ? true : nothing))
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

    # --roofline true is the ONE switch for the full kernel timing breakdown: the
    # per-phase roofline (GEMM-only vs A-permute / Bconv / accum / prepass /
    # finalize+sortperm / loop bookkeeping), the CAS redundancy counters, the
    # flop counter, env footprint, GEMM-dims histogram, AND the PROJMPO/
    # SparseBackends TimerOutputs trees. Off by default — a normal run prints
    # only the per-sweep energies and the regression verdict.
    if roofline
        print_roofline_ceilings()
        println("\n========== kernel roofline =========="); SparseBackends.show_roofline()
        println("\n========== CAS redundancy stats =========="); SparseBackends.show_cas_stats()
        SparseBackends.report_flops("ALI")
        ITensorMPS.print_env_footprint()
        SparseBackends.show_gemm_dims_hist()
        println("\n========== ITensorMPS.PROJMPO_TIMER ==========")
        print_timer(ITensorMPS.PROJMPO_TIMER)
        println("\n========== SparseBackends.TIMER ==========")
        print_timer(SparseBackends.TIMER)
    end
end
nothing
