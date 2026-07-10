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
#   Path-B is required for correct energies — dmrg()'s default run_mode=:bop_aliased.
#
# Example:
#   julia --project=.. test_aliased_pxp.jl --N 12 --maxdim 40 --n-sweeps 10

using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using LinearAlgebra: I as eye, norm
using Random
using ArgParse
using Printf

# Hardened 2026-06 to the default serial count (was SB_KK_NTHREADS,
# default-on knob, never varied off 1).
import KrylovKit
KrylovKit.set_num_threads(1)
println("[KrylovKit threads = ", KrylovKit.get_num_threads(),
        "   Julia threads = ", Threads.nthreads(), "]")

include("../test_sparse_psi/utils.jl")
include("../test_aliased_psi/setup.jl")

# Schema for the initial aliased ψ (frozen across sweeps; only template numeric data updates). Used for invariance tracking.
const _INIT_SCHEMA = Ref{Any}(nothing)

# Debug toggle (was SB_SCHEMA_TRACK env var) — flip to true manually to enable.
const _SCHEMA_TRACK_ON = false


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
        "--run-mode"
            help = "Path-B eigensolve mode: bop_aliased (default √-frame), rr (Rayleigh-Ritz), minner (M-inner Lanczos), bop_densify."
            arg_type = String
            default = "bop_aliased"
        "--gram-from-h"
            help = "true = covector env-slice gram (assumes traceless H — WRONG for PXP); false = direct ψ†ψ transfer (correct for any H). PXP needs false."
            arg_type = Bool
            default = false
        "--rr-dense-iter"
            help = "run_mode=rr only: run the local eigensolve fully DENSE and re-alias only at φ recovery (snap_dense_to_aliased). Diagnostic."
            arg_type = Bool
            default = false
        "--minv-rtol"
            help = "bop_* only: M^{-1/2} pseudo-inverse cutoff (drop eigenvalues < rtol·maxλ). NaN → default 0.1. Scan to test whether PXP's graded M has any converging cutoff."
            arg_type = Float64
            default = NaN
        "--roofline"
            help = "Enable the consolidated roofline/flop-count/env-footprint/perm-capture/GEMM-histogram instrumentation."
            arg_type = Bool
            default = false
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

# ── Sweep runner ──────────────────────────────────────────────────────────────
function run_sweeps(H, psi0, n_sweeps::Int, maxdim::Int;
                    cutoff=1e-10, mindim=1, target_E=NaN, label="ALI",
                    orthogonal_states=nothing, weight=20.0, roofline::Bool=false,
                    run_mode::Symbol=:bop_aliased, gram_from_h::Bool=false,
                    rr_dense_iter::Bool=false, minv_rtol=nothing)
    psi = psi0
    E = NaN
    cum = 0.0; cum_excl1 = 0.0
    target_reached_sweep = 0; target_reached_cum = NaN; target_reached_cum_excl1 = NaN
    for i in 1:n_sweeps
        sw = Sweeps(1); setmaxdim!(sw, maxdim); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        # minv_from_p=nothing → EIGEN M^{±1/2} (general). PXP's metric is NOT the
        # scaled projector c·Π (it's a directional reduced DM), so the from-P scalar
        # (minv_from_p=true, the dmrg default — valid only for KL) gives garbage
        # energy here. PXP MUST use the eigen path.
        t = if orthogonal_states === nothing
            @elapsed (E, psi, _esw, terr) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false, roofline=roofline,
                minv_from_p=nothing, run_mode=run_mode, gram_from_h=gram_from_h, rr_dense_iter=rr_dense_iter, minv_rtol=minv_rtol)
        else
            @elapsed (E, psi, _esw, terr) = dmrg(H, orthogonal_states, psi, sw;
                outputlevel=0, use_early_exit=false, weight=weight, roofline=roofline,
                minv_from_p=nothing, run_mode=run_mode, gram_from_h=gram_from_h, rr_dense_iter=rr_dense_iter, minv_rtol=minv_rtol)
        end
        cum += t; if i > 1; cum_excl1 += t; end
        @printf("  [%s sweep %2d] t=%8.3fs  E=%.12f  maxtruncerr=%.3e\n", label, i, t, E, terr)
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

# ── Main ──────────────────────────────────────────────────────────────────────
let
    parsed_args = parse_command_line()
    N          = parsed_args["N"]
    n_sweeps   = parsed_args["n-sweeps"]
    maxdim     = parsed_args["maxdim"]
    mindim     = parsed_args["mindim"]
    target_E   = parsed_args["target-energy"]
    no_excited = parsed_args["no-excited"]
    roofline   = parsed_args["roofline"]
    run_mode   = Symbol(parsed_args["run-mode"])
    gram_from_h   = parsed_args["gram-from-h"]
    rr_dense_iter = parsed_args["rr-dense-iter"]
    minv_rtol     = isnan(parsed_args["minv-rtol"]) ? nothing : parsed_args["minv-rtol"]

    println("=== PXP benchmark — ALIASED ψ (Path-B) ===")
    println("run_mode=:$run_mode  gram_from_h=$gram_from_h  rr_dense_iter=$rr_dense_iter  minv_rtol=$(minv_rtol === nothing ? "default(0.1)" : minv_rtol) — non-iso aliased ψ needs Path-B; PXP needs gram_from_h=false (direct ψ†ψ M)")
    println("N=$N  n_sweeps=$n_sweeps  maxdim=$maxdim  mindim=$mindim  target_E=$(isnan(target_E) ? "—" : target_E)")
    println("UNCAPPED (honest_bd=channel×maxdim) [DEFAULT]")
    println("ψ = ALIASED (P·ψ₀ via NotEqlsLoop_R1); DMRG on BARE H (constraint enforced structurally).")

    println("\nBuilding setup for N=$N sites ...")
    t_setup = @elapsed (H, psi_ali) = build_setup(N)
    println("Setup time: $(round(t_setup, digits=1))s.  System: $(length(psi_ali)) sites.")
    report_state("initial psi_ali", psi_ali, maxdim; verbose=true)

    if _SCHEMA_TRACK_ON
        _INIT_SCHEMA[] = _schema_fingerprint(psi_ali)
        println("  [init] captured alias schema fingerprint for invariance tracking")
    end
    check_aliased_invariant(psi_ali; label="initial")
    flush(stdout)

    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)
    SparseBackends.reset_cas_stats!()
    SparseBackends.reset_roofline!(roofline)  # zero accumulators once before the sweep loops below

    println("\n=== GROUND STATE ($n_sweeps sweeps at maxdim=$maxdim, mindim=$mindim; sweep 1 = JIT) ===")
    res_gs = run_sweeps(H, psi_ali, n_sweeps, maxdim; mindim=mindim, target_E=target_E, roofline=roofline,
                        run_mode=run_mode, gram_from_h=gram_from_h, rr_dense_iter=rr_dense_iter, minv_rtol=minv_rtol)
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
                            orthogonal_states=[psi_gs], weight=20.0, roofline=roofline,
                            run_mode=run_mode, gram_from_h=gram_from_h, rr_dense_iter=rr_dense_iter, minv_rtol=minv_rtol)
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

    if roofline
        println("\n========== kernel roofline =========="); SparseBackends.show_roofline()
        println("\n========== CAS redundancy stats =========="); SparseBackends.show_cas_stats()
        SparseBackends.report_flops("ALI")
        ITensorMPS.print_env_footprint()
        SparseBackends.show_gemm_dims_hist()
    end
    println("\n========== ITensorMPS.PROJMPO_TIMER ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
    println("\n========== SparseBackends.TIMER ==========")
    print_timer(SparseBackends.TIMER)
end
nothing
