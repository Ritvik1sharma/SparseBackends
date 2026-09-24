# configs.jl — the benchmark grid, in one place.
#
# `run_group.jl` names one of these config ids and a variant list. Keep this file
# in sync with the copy under experiments/manual_tests/ (they must be identical:
# both trees must agree on what a config id means, or the fork comparison is
# comparing two different problems).
#
# Config ids are STABLE IDENTIFIERS, not derived strings. Results are written to
# <config>__<variant>__seed<N>.json, so renaming an id orphans every result
# already on disk. They are therefore spelled out explicitly below even where the
# value could be computed; only the FIELDS are defaulted.
#
#   kl_min1_v<nsites>_bd<maxdim>    KL, projector sector -1, nsites = 2*nplaq+2
#   kl_pls1_...                     KL, sector +1
#   *_padh20_*                      chi_H zero-padded to 20
#   *_nopad_*                       the unpadded twin of a padh20 config
#   pxp_bd<maxdim>                  PXP, N = 100

# 15 sweeps (was 25) to keep the sweep tractable. Every result JSON records the
# value actually used, and per-sweep timings are reported separately from the
# total, so raising this later does not invalidate earlier rows.
const NSWEEPS_DEFAULT = 15

# ─────────────────────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────────────────────
# One per model, carrying every default. A config entry then states only what
# makes it DIFFERENT, so a reader can see at a glance which knob a given
# experiment turns — and a new field gets a default in exactly one place instead
# of being pasted into fifteen literals.
#
# `pad_h_chi = 0` is always present (rather than omitted and read via
# `get(cfg, :pad_h_chi, 0)`) so every config is explicit about whether H is
# inflated.

# Several fields have genuinely fixed admissible values, so they are checked
# HERE, at construction — an out-of-range config then cannot be created at all,
# rather than being caught (or not) somewhere downstream. `psign` is the one that
# matters most: it is a projector EIGENVALUE and only +1 / -1 are meaningful, but
# nothing downstream would reject e.g. 0.5 — `kl_operators` would happily build a
# non-projector and DMRG would converge to a number that looks plausible and is
# meaningless.
function _chk(cond, msg)
    cond || error("config: " * msg)
    return nothing
end

"""KL (Kitaev ladder). `nplaq` plaquettes ⇒ `2*nplaq + 2` sites. `spin=3` ⇒ S=1."""
function kl_cfg(; nplaq::Int, maxdim::Int, psign::Float64 = -1.0, spin::Int = 3,
                  nsweeps::Int = NSWEEPS_DEFAULT, cutoff::Float64 = 1e-12,
                  pad_h_chi::Int = 0)
    _chk(psign == 1.0 || psign == -1.0,
         "psign is a projector eigenvalue sector and must be +1.0 or -1.0, got $psign")
    _chk(spin == 2 || spin == 3, "spin must be 2 (S=1/2) or 3 (S=1), got $spin")
    _chk(nplaq   >= 1, "nplaq must be >= 1, got $nplaq")
    _chk(maxdim  >= 1, "maxdim must be >= 1, got $maxdim")
    _chk(nsweeps >= 1, "nsweeps must be >= 1, got $nsweeps")
    _chk(cutoff  >= 0, "cutoff must be >= 0, got $cutoff")
    # pad_h_chi=0 means "no padding"; anything in 1..chi_H would be a REDUCTION,
    # which pad_hamiltonian cannot express (it only zero-extends).
    _chk(pad_h_chi == 0 || pad_h_chi >= 5,
         "pad_h_chi must be 0 (unpadded) or >= chi_H = 5 for KL, got $pad_h_chi")
    return (model = :kl, nplaq = nplaq, spin = spin, psign = psign, maxdim = maxdim,
            nsweeps = nsweeps, cutoff = cutoff, pad_h_chi = pad_h_chi)
end

"""PXP, S=1 sites, NotEqlsLoop_R1 blockade constraint. `weight` is the
orthogonality penalty for the first-excited state."""
function pxp_cfg(; nsites::Int, maxdim::Int, nsweeps::Int = NSWEEPS_DEFAULT,
                   cutoff::Float64 = 1e-10, weight::Float64 = 20.0,
                   pad_h_chi::Int = 0)
    _chk(nsites  >= 2, "nsites must be >= 2, got $nsites")
    _chk(maxdim  >= 1, "maxdim must be >= 1, got $maxdim")
    _chk(nsweeps >= 1, "nsweeps must be >= 1, got $nsweeps")
    _chk(cutoff  >= 0, "cutoff must be >= 0, got $cutoff")
    _chk(weight   > 0, "weight (excited-state orthogonality penalty) must be > 0, got $weight")
    _chk(pad_h_chi == 0 || pad_h_chi >= 4,
         "pad_h_chi must be 0 (unpadded) or >= chi_H = 4 for PXP, got $pad_h_chi")
    # The exact max MPS bond at the middle of an S=1 chain is 3^(nsites/2); asking
    # for more means the requested bd is silently unreachable and the run is not
    # the bd it claims to be (this is why the padded PXP configs use 16 sites, not 8).
    # Only checked for short chains: 3^half overflows Int64 past half=39, and by
    # half=30 the bound is 2e14, far beyond any bond dimension anyone would run.
    half = nsites ÷ 2
    if half <= 30
        reachable = 3^half
        _chk(maxdim <= reachable,
             "maxdim=$maxdim exceeds the exact max MPS bond 3^$half=$reachable " *
             "at nsites=$nsites; the run would not actually be bd=$maxdim")
    end
    return (model = :pxp, nsites = nsites, maxdim = maxdim, nsweeps = nsweeps,
            cutoff = cutoff, weight = weight, pad_h_chi = pad_h_chi)
end

const CONFIGS = Dict{String,NamedTuple}()

# ── KL main grid, projector sector -1 ────────────────────────────────────────
CONFIGS["kl_min1_v26_bd40"]   = kl_cfg(; nplaq = 12, maxdim = 40)    # 26 sites
CONFIGS["kl_min1_v66_bd80"]   = kl_cfg(; nplaq = 32, maxdim = 80)    # 66 sites
CONFIGS["kl_min1_v130_bd100"] = kl_cfg(; nplaq = 64, maxdim = 100)   # 130 sites

# ── PXP main grid, N = 100. Ground + first excited state ─────────────────────
# bd40 exists to give the fork-penalty study (sb_dense vs orig_dense at the same
# exact chi=16 operator) a third point between bd20 and bd60, so the penalty can
# be plotted against bd rather than inferred from two points.
for bd in (20, 40, 60, 120)
    CONFIGS["pxp_bd$bd"] = pxp_cfg(; nsites = 100, maxdim = bd)
end

# ── chi_H ablation: padded configs and their unpadded twins ──────────────────
# The real Hamiltonians have tiny chi_H (KL 5, PXP 4), so chi_PHP = chi_P^2*chi_H
# stays small (80 / 16) and the dense path is cheap. Zero-padding chi_H to 20
# (see experiments/pad_h_utils.jl) raises chi_PHP to 320 / 80, making the dense
# PHP genuinely expensive. Purpose: test whether the aliased speedup tracks the
# PHP memory saving.
#
# PADDED PHYSICS IS DELIBERATELY ALTERED -- padded energies are meaningless and
# are NOT comparable to any unpadded config. Only time and memory are. Hence 6
# sweeps: convergence is not the goal.
#
# The two arms are emitted from ONE loop so they cannot drift apart. That matters
# because the whole point is that a (padh20, nopad) pair differs in chi_H ALONE;
# no config in the main grid pairs with a padh20 one (the padded runs use KL 18 /
# PXP 16 sites and 6 sweeps, the grid uses KL 26/66/130 and PXP 100 at 10-15), so
# diffing padded-vs-grid would confound chi_H with N and sweep count.
#
# Small lattices on purpose, but NOT tiny: PXP uses 16 sites rather than 8
# because at 8 sites the exact maximum MPS bond is 3^4 = 81, so maxdim/mindim of
# 120 is unreachable and the "bd=120" point would silently not be bd=120.
# The N=100 padded config below is the ONE that needs no twin: it is
# `pxp_bd20` with pad_h_chi=20 and nothing else changed, so `pxp_bd20` itself is
# the control. Every other field is taken from pxp_cfg's defaults, which is
# exactly what pxp_bd20 does -- N=100, maxdim=20, nsweeps=15, cutoff=1e-10 --
# so the pair differs in chi_H ALONE.
#
# This is the comparison the v16/v18 padded configs cannot give. Those changed N
# (100 -> 16) and sweeps (15 -> 6) at the same time as chi_H, and at N=16 the
# damage is worse than it looks: the exact max S=1 bond is min(3^j, 3^(N-j)), so
# the 8 rank-capped edge bonds are an ABSOLUTE count, not a fraction. At bd=120
# only 7 of 15 bonds reach 120 (vs 91 of 99 at N=100), so "bd=120, N=16" is not
# the same operating point as "bd=120, N=100" despite the matching number.
#
# Run it at BENCH_BLAS_THREADS=5 -- the recorded pxp_bd20 rows are BLAS=5, and
# the control only holds at the same thread count.
CONFIGS["pxp_padh20_v100_bd20"] = pxp_cfg(; nsites = 100, maxdim = 20, pad_h_chi = 20)

for bd in (20, 120)
    CONFIGS["kl_padh20_v18_bd$bd"]  = kl_cfg(;  nplaq  = 8,  maxdim = bd, nsweeps = 6, pad_h_chi = 20)
    CONFIGS["kl_nopad_v18_bd$bd"]   = kl_cfg(;  nplaq  = 8,  maxdim = bd, nsweeps = 6, pad_h_chi = 0)
    CONFIGS["pxp_padh20_v16_bd$bd"] = pxp_cfg(; nsites = 16, maxdim = bd, nsweeps = 6, pad_h_chi = 20)
    CONFIGS["pxp_nopad_v16_bd$bd"]  = pxp_cfg(; nsites = 16, maxdim = bd, nsweeps = 6, pad_h_chi = 0)
end

# ── Historical-reproduction configs ──────────────────────────────────────────
# These match the recorded 1-BLAS-thread runs of test_check_working_aliased.jl
# and test_pxp_aliased.jl that reported "KL beats (0.845-0.852), PXP loses
# (1.52 ground / 1.49 excited)". Only meaningful at BENCH_BLAS_THREADS=1: those
# numbers are a serial-GEMM comparison, and the aliased path issues thousands of
# tiny GEMMs that do not thread while dense does a few large ones that do -- the
# ratio moves with thread count (measured: KL v26 aliased scales 1.10x from 1 to
# 5 threads, dense 1.68x).
# Note psign=+1.0 here (the scripts' BENCH_PSIGN default), unlike the -1.0 sector
# used by the main grid.
CONFIGS["kl_pls1_v26_bd40"] = kl_cfg(;  nplaq  = 12, maxdim = 40, psign = 1.0, nsweeps = 25)
CONFIGS["pxp_v12_bd40"]     = pxp_cfg(; nsites = 12, maxdim = 40, nsweeps = 6)

# Order used by run_group.jl: cheapest / most important first, so that if an
# expensive variant dies (OOM, wall clock) the ones already finished have
# written their JSON.
const SB_VARIANT_ORDER = [:sb_aliased, :sb_fused, :sb_dense]

const SEEDS = [0, 1, 2]

"""
    config_or_die(id)

Look up a config by id, or fail loudly listing every known id (a typo'd id would
otherwise surface much later as a confusing `KeyError` inside a batch job).

`BENCH_NSWEEPS` / `BENCH_MAXDIM` override the table when set — for smoke tests
only. Both are recorded in the result JSON and a warning is emitted, so an
overridden run is never silently mistaken for a production one.
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
