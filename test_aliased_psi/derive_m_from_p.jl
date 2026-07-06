# derive_m_from_p.jl — PROVE that the Path-B gram M is, at convergence, a scaled
# projector M = c·Π that is determined by the constraint structure P (alias keys +
# alias-sign mult dressing + environment count), NOT by ψ's numerical data.
#
# Tests, per side (Lgram / Rgram) at the mid bond after convergence:
#   (1) SCALED-PROJECTOR:  eig spectrum is {c (×rank), 0 (×null)} ⇒  M ≈ c·Π,
#       Π = V_kept V_kept'.  Report residual ‖M − c·Π‖/‖M‖ and eigenvalue spread.
#   (2) c = P-COUNT:        c should equal a pure environment multiplicity (power of 2),
#       independent of ψ. Report c and the matching 2^k.
#   (3) M^{-1/2} MATCH:     c^{-1/2}·Π must equal the ACTUAL operator that the sweep
#       uses (SparseBackends.build_half_pair_single). Report ‖Linv_P − Linv_live‖.
#   (4) MULT STRUCTURE:     within Π, per surviving channel the mult part is rank-1 and
#       flat-magnitude (the alias-sign vector). Report mult-rank-1 fraction.
#   (5) CROSS-SEED:         re-converge from a DIFFERENT random seed; c and range(Π)
#       must be identical (principal-subspace distance ≈ 0) ⇒ M is P-determined.
#
# Run (matches the N=12 b=12 converged-bulk case):
#   OPENBLAS_NUM_THREADS=1 \
#     julia --project=.. derive_m_from_p.jl --N-plaq 12 --maxdim 40 --n-sweeps 4
using SparseBackends, ITensors, ITensorMPS, LinearAlgebra, Printf, Random, ArgParse
include("../test_sparse_psi/utils.jl")

# build_setup returns (H, psi, P, env_sites_left_of_midbond, env_sites_right_of_midbond).
# Identical to diag_gram_structure.jl's build_setup but (a) seedable and (b) returns P.
function build_setup(N, psign, spin, seed)
    Random.seed!(seed)
    sites = spin == 2 ? siteinds("S=1/2", 2N + 2) : siteinds("S=1", 2N + 2)
    os = OpSum()
    for j in 1:(N + 1); os += "Sz", 2j - 1, "Sz", 2j; end
    for j in 1:N; os += "Sx", 2j - 1, "Sx", 2j + 2; os += "Sy", 2j, "Sy", 2j + 1; end
    cs = 0.5 * psign; os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2j - 1, "Id", 2j, "Id", 2j + 1, "Id", 2j + 2
        t += cs, "exp(i*pi*Sy)", 2j - 1, "exp(i*pi*Sx)", 2j, "exp(i*pi*Sx)", 2j + 1, "exp(i*pi*Sy)", 2j + 2
        push!(os2, t)
    end
    Cons = [clean!(MPO(os2[j], sites, [2j - 1, 2j, 2j + 1, 2j + 2])) for j in 1:N]
    mulMPO(A, B) = replaceprime(contract(A, prime(B, "Site"), :coo, :coo), 2 => 1)
    P = Cons[1]; for j in 2:length(Cons); P = mulMPO(P, Cons[j]); end
    H = MPO(os, sites); psi0 = random_mps(sites)
    psi = replaceprime(contract(P, copy(psi0), :coo, :aliased; denseLinksB = 0), 1 => 0)
    return H, psi, P
end

# Eigen-analysis of one gram side: returns (c, rank, null, spread, residual, λ, V, kept_mask, dims)
function analyze_gram(G; rtol=1e-1)
    gi = collect(ITensors.inds(G))
    unp = filter(I -> ITensors.plev(I) == 0, gi)
    prm = filter(I -> ITensors.plev(I) == 1, gi)
    d = prod(I -> ITensors.dim(I), unp; init=1)
    Gd = ITensors.has_external_storage(G) ? SparseBackends.to_dense_itensors_unfused(G) : G
    M = reshape(Array(Gd, unp..., prm...), d, d); M = (M + M') / 2
    F = eigen(Hermitian(M)); λ = real.(F.values); V = F.vectors
    mx = maximum(λ); tol = rtol * mx
    kept = λ .> tol
    c = sum(λ[kept]) / max(count(kept), 1)
    spread = count(kept) == 0 ? 0.0 : (maximum(λ[kept]) - minimum(λ[kept])) / abs(c)
    Pi = V[:, kept] * V[:, kept]'
    resid = norm(M - c * Pi) / max(norm(M), eps())
    return (; c, rank=count(kept), null=count(.!kept), spread, resid, λ, V, kept, M,
              unp, prm, dims=[ITensors.dim(I) for I in unp])
end

# NOTE: a "c from P alone" via an output-traced P†P contraction was tested and REMOVED —
# no contraction of P's tensors reproduces c=2^env (it is off-shell: sums the full physical
# space, picking up dim 3, not the on-shell Z2 factor 2). c=2^env is the analytical Z2-count
# (base = involution order, verified below; exponent = #2-site rungs = ⌈env/2⌉).

# Mult-rank-1 test: reshape Π's range to (channel, mult) and, per channel basis dir,
# check the mult-component is (near) rank-1 = one signed direction. Needs the two
# unpaired axes sorted as (channel=smaller-dim, mult=larger-OR-mult-tagged).
function mult_structure(a)
    length(a.unp) == 2 || return (; ok=false, msg="not a 2-axis bond")
    d1, d2 = a.dims[1], a.dims[2]
    # mult axis = the dim-4 alias axis (smaller); channel = the larger sparse-bond axis.
    chan_ax, mult_ax = d1 >= d2 ? (1, 2) : (2, 1)
    dc = a.dims[chan_ax]; dm = a.dims[mult_ax]
    Vk = a.V[:, a.kept]                          # (d) × rank  range basis, d = d1*d2
    # reshape each range vector to (axis1,axis2) then to (chan,mult)
    r1frac = Float64[]
    for k in 1:size(Vk, 2)
        blk = reshape(Vk[:, k], d1, d2)
        blk = chan_ax == 1 ? blk : permutedims(blk, (2, 1))   # → (chan, mult)
        s = svdvals(blk)
        push!(r1frac, s[1]^2 / max(sum(abs2, s), eps()))
    end
    return (; ok=true, dc, dm, mean_rank1=sum(r1frac) / length(r1frac),
              min_rank1=minimum(r1frac))
end

# Gauge-invariant cross-seed comparison. Across runs DMRG fixes a different unitary
# gauge on the bond, so raw ‖Π1−Π2‖ is meaningless (different bases). The P-determined
# invariants are basis-FREE: the eigenvalue spectrum (→ c and rank) and the per-channel
# mult rank-1 signature. Compare sorted kept-spectra.
function spectrum_match(a1, a2)
    s1 = sort(a1.λ[a1.kept]; rev=true); s2 = sort(a2.λ[a2.kept]; rev=true)
    n = min(length(s1), length(s2))
    maxdiff = n == 0 ? 0.0 : maximum(abs.(s1[1:n] .- s2[1:n]))
    return (; rank1=length(s1), rank2=length(s2), maxdiff)
end

function run(N, md, nsw, seed, bond, drive=:loop, psign=+1)
    # psign=+1 matches the test_aliased_kl.jl reference (--eignv default true → P=(Id+YXXY)/2,
    # E≈-17.18). psign=-1 solves the OTHER sector (E≈-16.34) — a different problem.
    H, psi, P = build_setup(N, psign, 3, seed)
    # :loop  = drive sweeps ONE AT A TIME, re-feeding psi (re-inits gram cache each call)
    #          — IDENTICAL to the test_aliased_kl.jl harness `run_sweeps`.
    # :single= one dmrg(Sweeps(nsw)) call — maintains the gram cache INCREMENTALLY across
    #          sweeps. Suspected to under-converge the left side (E=-16.31 vs -17.18).
    local E
    if drive === :single
        sw = Sweeps(nsw); setmaxdim!(sw, md); setmindim!(sw, 1); setcutoff!(sw, 1e-10)
        E, psi = dmrg(H, psi, sw; outputlevel=0, run_mode=:bop_aliased, use_early_exit=false)
    else
        for s in 1:nsw
            sw = Sweeps(1); setmaxdim!(sw, md); setmindim!(sw, 1); setcutoff!(sw, 1e-10)
            E, psi = dmrg(H, psi, sw; outputlevel=0, run_mode=:bop_aliased, use_early_exit=false)
        end
    end
    nb = length(psi); b = bond > 0 ? bond : max(1, div(nb, 2))
    # GAUGE: the gram is gauge-dependent. To read it in the SAME canonical form the
    # eigsolve at bond b uses (sites 1..b-1 left-canonical, b+1..N right-canonical),
    # position the orthogonality center at b BEFORE building the cache. Without this,
    # the final ψ (OC at an end after the last half-sweep) gives one side in the wrong
    # gauge → a non-projector gram (spurious low rank), which is NOT physics.
    ITensorMPS.orthogonalize!(psi, b)
    gc = SparseBackends.init_gram_cache(psi)
    L = SparseBackends.get_left_gram(gc, b); R = SparseBackends.get_right_gram(gc, b)
    # env site counts: Lgram at b = gram of ψ[1..b-1] (b-1 sites); Rgram = ψ[b+2..N]
    n_left = b - 1; n_right = nb - (b + 1)
    return (; E, b, nb, L, R, n_left, n_right)
end

s = ArgParseSettings()
@add_arg_table! s begin
    "--N-plaq"; arg_type=Int; default=12
    "--maxdim"; arg_type=Int; default=40
    "--n-sweeps"; arg_type=Int; default=4
    "--seed"; arg_type=Int; default=42
    "--seed2"; arg_type=Int; default=7    # cross-seed check
    "--bond"; arg_type=Int; default=12    # converged-bulk mid bond (N=12)
    "--psign"; arg_type=Int; default=+1   # +1 = reference sector (E≈-17.18); -1 = other sector
end
args = parse_args(s)
N = args["N-plaq"]; md = args["maxdim"]; nsw = args["n-sweeps"]
seed1 = args["seed"]; seed2 = args["seed2"]; bond = args["bond"]; psign = args["psign"]

println("="^78)
println("DERIVE M FROM P   N_plaq=$N  maxdim=$md  n_sweeps=$nsw  seeds=($seed1,$seed2)")
println("="^78)

# ── STRUCTURAL Z2-COUNT for c (base of c = 2^env, derived from P's structure) ──
# c = (Z2 involution order)^(# 2-site rungs in env). The base is the order of the
# constraint symmetry U_j: each Cons_j = ½(Id + psign·U_j) with U_j a product of
# single-site π-rotations exp(iπS·). For integer spin S=1, exp(iπS)² = exp(2πiS) = Id,
# so each is an involution ⇒ U_j² = Id ⇒ order 2 ⇒ base = 2. Verify structurally:
let sts = siteinds("S=1", 1)
    s = sts[1]; ip = prime(s)
    Oy = Array(op("exp(i*pi*Sy)", s), ip, s); Ox = Array(op("exp(i*pi*Sx)", s), ip, s)
    dy = norm(Oy * Oy - Matrix(LinearAlgebra.I, 3, 3))
    dx = norm(Ox * Ox - Matrix(LinearAlgebra.I, 3, 3))
    order = (dy < 1e-10 && dx < 1e-10) ? 2 : -1
    @printf("STRUCTURAL Z2-COUNT: ||exp(iπSy)²−Id||=%.1e ||exp(iπSx)²−Id||=%.1e ⇒ involution order=%d ⇒ c-base=%d\n",
            dy, dx, order, order)
    println("  ⇒ c(env) = order^(#2-site rungs) = $order^⌈env/2⌉   (structural, from P's Z2 constraint)")
end

# Head-to-head: does single multi-sweep call converge to the same energy as the
# per-sweep loop? (answers "is the incremental gram cache drifting across sweeps?")
println("\n── DRIVE COMPARISON (seed $seed1, $nsw sweeps) ──")
rs = run(N, md, nsw, seed1, bond, :single, psign)
rl = run(N, md, nsw, seed1, bond, :loop, psign)
@printf("  single dmrg(Sweeps(%d)) : E=%.8f\n", nsw, rs.E)
@printf("  loop %d×dmrg(Sweeps(1)) : E=%.8f\n", nsw, rl.E)
@printf("  ΔE = %.3e   %s\n", abs(rs.E - rl.E),
        abs(rs.E - rl.E) < 1e-6 ? "→ SAME (incremental cache OK)" :
                                  "→ DIFFER (single-call under-converges)")

r1 = rl   # use the loop-driven (harness-equivalent) result for the detailed analysis
println("\n[seed $seed1, loop-driven]  E=$(round(r1.E; digits=8))  bond b=$(r1.b)/$(r1.nb)  ",
        "env sites: left=$(r1.n_left) right=$(r1.n_right)")

for (nm, G, nenv) in (("Lgram", r1.L, r1.n_left), ("Rgram", r1.R, r1.n_right))
    a = analyze_gram(G)
    println("\n── $nm  dim=$(prod(a.dims))  axes=$(a.dims) ─────────────────────────")
    @printf("  (1) SCALED PROJECTOR:  rank=%d  null=%d  c=%.6g  eig-spread=%.2e  ‖M−c·Π‖/‖M‖=%.2e\n",
            a.rank, a.null, a.c, a.spread, a.resid)
    # (2) c vs environment count: predicted 2^ceil(env_sites/2) (one factor of 2 per
    # plaquette-pair in the environment — a pure P/constraint count, no ψ data).
    kpred = cld(nenv, 2); k = round(Int, log2(max(a.c, 1.0)))
    @printf("  (2) c-as-P-COUNT:      c=%.6g   observed 2^%d=%d   predicted 2^ceil(%d/2)=2^%d=%d   match=%s\n",
            a.c, k, 2^k, nenv, kpred, 2^kpred, isapprox(a.c, 2.0^kpred; rtol=1e-6))
    # (2b) NON-CIRCULAR residual: use the P-PREDICTED scale c_pred=2^kpred (NOT the
    # fitted mean-eigenvalue) with M's range projector. ~0 ⇒ M = (P-count)·Π exactly,
    # so the scale is derived from P, not fitted to M.
    Pi = a.V[:, a.kept] * a.V[:, a.kept]'
    cpred = 2.0^kpred
    resid_pred = LinearAlgebra.norm(a.M - cpred * Pi) / max(LinearAlgebra.norm(a.M), eps())
    @printf("  (2b) ‖M − c_pred·Π‖/‖M‖ (c_pred from P, not fitted) = %.2e\n", resid_pred)
    # (3) live operator match: c^{-1/2}·Π  vs  build_half_pair_single
    _, Linv_live = SparseBackends.build_half_pair_single(G)
    Lv = reshape(Array(Linv_live, a.unp..., a.prm...), prod(a.dims), prod(a.dims))
    Linv_P = (1 / sqrt(a.c)) * (a.V[:, a.kept] * a.V[:, a.kept]')
    @printf("  (3) M^{-1/2} MATCH:    ‖c^{-1/2}Π − Linv_live‖/‖Linv_live‖ = %.2e\n",
            norm(Linv_P - Lv) / max(norm(Lv), eps()))
    # (4) mult structure
    ms = mult_structure(a)
    if ms.ok
        @printf("  (4) MULT STRUCTURE:    chan=%d mult=%d   per-channel mult rank-1 frac: mean=%.4f min=%.4f\n",
                ms.dc, ms.dm, ms.mean_rank1, ms.min_rank1)
    end
end

# (5) cross-seed invariance
println("\n── (5) CROSS-SEED INVARIANCE (P-determined ⇔ identical across ψ data) ──")
r2 = run(N, md, nsw, seed2, bond, :loop, psign)
println("  [seed $seed2]  E=$(round(r2.E; digits=8))   (vs seed $seed1 E=$(round(r1.E; digits=8)))")
for (nm, G1, G2) in (("Lgram", r1.L, r2.L), ("Rgram", r1.R, r2.R))
    a1 = analyze_gram(G1); a2 = analyze_gram(G2)
    sm = spectrum_match(a1, a2)
    @printf("  %s:  c[%s]=%.6g  c[%s]=%.6g  Δc=%.2e   rank %d/%d   max|Δspectrum|=%.2e  (gauge-invariant)\n",
            nm, string(seed1), a1.c, string(seed2), a2.c, abs(a1.c - a2.c),
            sm.rank1, sm.rank2, sm.maxdiff)
end
println("\nDONE.")
