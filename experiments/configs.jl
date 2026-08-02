# configs.jl — the benchmark grid, in one place.
#
# Every experiment file in this directory is a thin wrapper that names one of
# these config ids and one variant. `run_group.jl` uses the same table to run
# several variants inside a single julia process (so the SparseBackends codegen
# cost is paid once instead of once per variant).
#
# Config ids follow the naming of experiments/kl and experiments/pxp:
#   kl_min1_v<nsites>_bd<maxdim>   — KL, projector sector -1, nsites = 2*nplaq+2
#   pxp_bd<maxdim>                 — PXP, N = 100 sites
#
# Keep this table in sync with the copy under experiments/manual_tests/.

# 15 sweeps (was 25) to keep the sweep tractable. Every result JSON records the
# value actually used, and per-sweep timings are reported separately from the
# total, so raising this later does not invalidate earlier rows.
const NSWEEPS_DEFAULT = 15

const CONFIGS = Dict{String,NamedTuple}(
    # ── KL (Kitaev ladder), spin S=1, projector sector -1 ────────────────────
    # nplaq is ARGS[1] of test_sparse_ham/test_check_working_aliased.jl;
    # the lattice has 2*nplaq+2 sites.
    "kl_min1_v26_bd40" => (
        model   = :kl,
        nplaq   = 12,          # 26 sites
        spin    = 3,           # 3 => S=1, 2 => S=1/2
        psign   = -1.0,        # BENCH_PSIGN: projector eigenvalue sector
        maxdim  = 40,
        nsweeps = NSWEEPS_DEFAULT,
        cutoff  = 1e-12,
    ),
    "kl_min1_v66_bd80" => (
        model   = :kl,
        nplaq   = 32,          # 66 sites
        spin    = 3,
        psign   = -1.0,
        maxdim  = 80,
        nsweeps = NSWEEPS_DEFAULT,
        cutoff  = 1e-12,
    ),
    "kl_min1_v130_bd100" => (
        model   = :kl,
        nplaq   = 64,          # 130 sites
        spin    = 3,
        psign   = -1.0,
        maxdim  = 100,
        nsweeps = NSWEEPS_DEFAULT,
        cutoff  = 1e-12,
    ),

    # ── Padded-H cost experiment ─────────────────────────────────────────────
    # The real Hamiltonians have tiny chi_H (KL 5, PXP 4), so PHP stays small
    # (chi_PHP = chi_P^2 * chi_H = 80 / 16) and the dense path is cheap. These
    # configs raise chi_H to 20 using REAL long-range couplings (see
    # experiments/pad_h_utils.jl), giving chi_PHP ~ 320 for KL and ~80 for PXP so
    # the dense PHP becomes genuinely expensive. Purpose: test whether the aliased
    # speedup then tracks the PHP memory saving.
    #
    # PHYSICS IS DELIBERATELY ALTERED -- these energies are meaningless and not
    # comparable to the unpadded configs. Only per-sweep time matters, so 6 sweeps
    # (2 ramp + 4 steady); convergence is not the goal.
    #
    # Two psi bond dimensions (20, 120) to expose the trend. NOTE PXP uses 16
    # sites, not 8: at 8 sites the exact maximum MPS bond is 3^4 = 81, so a
    # maxdim/mindim of 120 is unreachable and the "bd=120" point would silently
    # not be bd=120.
    "kl_padh20_v18_bd20" => (
        model = :kl, nplaq = 8, spin = 3, psign = -1.0,
        maxdim = 20, nsweeps = 6, cutoff = 1e-12, pad_h_chi = 20,
    ),
    "kl_padh20_v18_bd120" => (
        model = :kl, nplaq = 8, spin = 3, psign = -1.0,
        maxdim = 120, nsweeps = 6, cutoff = 1e-12, pad_h_chi = 20,
    ),
    "pxp_padh20_v16_bd20" => (
        model = :pxp, nsites = 16,
        maxdim = 20, nsweeps = 6, cutoff = 1e-10, weight = 20.0, pad_h_chi = 20,
    ),
    "pxp_padh20_v16_bd120" => (
        model = :pxp, nsites = 16,
        maxdim = 120, nsweeps = 6, cutoff = 1e-10, weight = 20.0, pad_h_chi = 20,
    ),

    # ── Unpadded TWINS of the four padh20 configs ────────────────────────────
    # Identical in EVERY field (N, psi maxdim, nsweeps, cutoff, weight, psign,
    # spin) except pad_h_chi = 0. These exist because no config in the main grid
    # pairs with a padh20 one: the padded runs use small lattices (KL 18 sites,
    # PXP 16) and 6 sweeps, whereas the grid uses KL 26/66/130 and PXP 100 at
    # 10-15 sweeps. Diffing padded-vs-grid therefore confounds chi_H with N and
    # sweep count, so the padded set could only ever be a demonstration.
    # With these twins, (padh20, nopad) differ in chi_H ALONE:
    #   KL   chi_H  5 -> 20   =>  chi_PHP  80 -> 320
    #   PXP  chi_H  4 -> 20   =>  chi_PHP  16 ->  80
    # making the chi_H axis a controlled ablation.
    #
    # Separate config IDs (not an env override of the padh20 ids) on purpose:
    # results are written to <config>__<variant>__seed<N>.json, so reusing an id
    # would overwrite/pool the two arms — the same silent-pooling bug that put
    # chi=12 and chi=16 PXP dense runs in one average.
    "kl_nopad_v18_bd20" => (
        model = :kl, nplaq = 8, spin = 3, psign = -1.0,
        maxdim = 20, nsweeps = 6, cutoff = 1e-12, pad_h_chi = 0,
    ),
    "kl_nopad_v18_bd120" => (
        model = :kl, nplaq = 8, spin = 3, psign = -1.0,
        maxdim = 120, nsweeps = 6, cutoff = 1e-12, pad_h_chi = 0,
    ),
    "pxp_nopad_v16_bd20" => (
        model = :pxp, nsites = 16,
        maxdim = 20, nsweeps = 6, cutoff = 1e-10, weight = 20.0, pad_h_chi = 0,
    ),
    "pxp_nopad_v16_bd120" => (
        model = :pxp, nsites = 16,
        maxdim = 120, nsweeps = 6, cutoff = 1e-10, weight = 20.0, pad_h_chi = 0,
    ),

    # ── Historical-reproduction configs ──────────────────────────────────────
    # These match the recorded 1-BLAS-thread runs of test_check_working_aliased.jl
    # and test_pxp_aliased.jl that reported "KL beats (0.845-0.852), PXP loses
    # (1.52 ground / 1.49 excited)". Only meaningful at BENCH_BLAS_THREADS=1:
    # those numbers are a serial-GEMM comparison, and the aliased path issues
    # thousands of tiny GEMMs that do not thread while dense does a few large
    # ones that do -- so the ratio is expected to move with thread count.
    # Note psign=+1.0 here (the scripts' BENCH_PSIGN default), unlike the -1.0
    # sector used by the main grid.
    "kl_pls1_v26_bd40" => (
        model   = :kl,
        nplaq   = 12,          # 26 sites
        spin    = 3,
        psign   = 1.0,         # script default, matches the historical run
        maxdim  = 40,
        nsweeps = 25,          # historical run was 25 sweeps
        cutoff  = 1e-12,
    ),
    "pxp_v12_bd40" => (
        model   = :pxp,
        nsites  = 12,          # historical PXP comparison size, NOT 100
        maxdim  = 40,
        nsweeps = 6,           # historical run was 6 sweeps
        cutoff  = 1e-10,
        weight  = 20.0,
    ),

    # ── PXP, S=1 sites, N = 100, NotEqlsLoop_R1 constraint (bond dim 2) ──────
    # Ground state + first excited state (orthogonality penalty `weight`).
    "pxp_bd20" => (
        model   = :pxp,
        nsites  = 100,
        maxdim  = 20,
        nsweeps = NSWEEPS_DEFAULT,
        cutoff  = 1e-10,
        weight  = 20.0,
    ),
    # bd40 exists to give the fork-penalty study (sb_dense vs orig_dense, same
    # exact chi=16 operator) a third bond dimension between bd20 and bd60, so the
    # penalty can be plotted against bd rather than inferred from two points.
    "pxp_bd40" => (
        model   = :pxp,
        nsites  = 100,
        maxdim  = 40,
        nsweeps = NSWEEPS_DEFAULT,
        cutoff  = 1e-10,
        weight  = 20.0,
    ),
    "pxp_bd60" => (
        model   = :pxp,
        nsites  = 100,
        maxdim  = 60,
        nsweeps = NSWEEPS_DEFAULT,
        cutoff  = 1e-10,
        weight  = 20.0,
    ),
    "pxp_bd120" => (
        model   = :pxp,
        nsites  = 100,
        maxdim  = 120,
        nsweeps = NSWEEPS_DEFAULT,
        cutoff  = 1e-10,
        weight  = 20.0,
    ),
)

# Order used by run_group.jl: cheapest / most important first, so that if an
# expensive variant dies (OOM, wall clock) the ones already finished have
# written their JSON.
const SB_VARIANT_ORDER = [:sb_aliased, :sb_fused, :sb_dense]

const SEEDS = [0, 1, 2]

"""
    config_or_die(id)

Look up a config. `BENCH_NSWEEPS` / `BENCH_MAXDIM` override the table when set —
for smoke tests only. Both are recorded in the result JSON, so an overridden run
is never silently mistaken for a production one.
"""
function config_or_die(id::AbstractString)
    haskey(CONFIGS, id) ||
        error("unknown config id `$id`; known ids: " * join(sort(collect(keys(CONFIGS))), ", "))
    cfg = CONFIGS[id]
    ns = tryparse(Int, get(ENV, "BENCH_NSWEEPS", ""))
    md = tryparse(Int, get(ENV, "BENCH_MAXDIM", ""))
    ns === nothing || (cfg = merge(cfg, (nsweeps = ns,)))
    md === nothing || (cfg = merge(cfg, (maxdim  = md,)))
    (ns === nothing && md === nothing) ||
        @warn "config $id overridden by env" nsweeps = cfg.nsweeps maxdim = cfg.maxdim
    return cfg
end
