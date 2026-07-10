# pxp_gram_spectrum.jl — DIAGNOSTIC (no DMRG run_mode change): characterize PXP's
# Path-B metric so we know whether a blowup-free inverse is even possible.
#
# (1) CONVERGED aliased-ψ gram spectrum: build Lgram/Rgram (direct ψ†ψ transfer,
#     gram_from_h=false style) at a mid-chain bond of a converged ψ, and report the
#     eigenvalue spread — condition number, # decades, smooth-decay?, numerical
#     nulls — plus per-connected-block conditioning of Lgram (it's claimed
#     block-diagonal in the FSM channel; if so, are the blocks individually well
#     conditioned → per-block inverse viable?).
#
# (2) PURE-P STRUCTURAL spectrum: fold P†P on the doubled FSM bond from PXP's MPO
#     tensors ALONE (no ψ, no core) and look at the transfer spectrum. Clean gap
#     (exact zeros) ⇒ a structural null exists ⇒ from-P-style blowup-free inverse
#     is buildable. Graded (no gap) ⇒ no clean structural inverse; RR (inverse-free)
#     is the ceiling. CAVEAT: the fully-physical-traced P†P transfer is OFF-SHELL
#     (its magnitude does NOT reproduce the on-shell c=gram-eigenvalue — see
#     derive_m_from_p.jl note); we read only its GAP STRUCTURE (zeros vs graded),
#     which is what the structural-null question needs.

using ITensors, ITensorMPS, SparseBackends, LinearAlgebra, Printf, Random
import KrylovKit; KrylovKit.set_num_threads(1)

ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]
_nz(dims, coords) = (A=zeros(Float64,dims...); for c in coords; A[(c .+ 1)...]=1.0; end; A)

# PXP FSM constraint MPO (NotEqlsLoop_R1) — bond dim 2. Return the raw bulk array too.
const Rf = _nz((3,3,2),[(0,0,0),(1,1,1),(2,2,0)])
const Rb = _nz((3,3,2,2),[(0,0,0,0),(0,0,1,0),(1,1,0,1),(1,1,1,0),(2,2,0,0)])
const Rl = _nz((3,3,2),[(0,0,0),(0,0,1),(1,1,0),(1,1,1),(2,2,0)])
function NotEqlsLoop_R1(sites)
    N=length(sites); bonds=[Index(2,"Link,l=$i") for i in 1:N-1]; W=Vector{ITensor}(undef,N)
    W[1]=ITensor(Rf, sites[1], sites[1]', bonds[1])
    for j in 2:N-1; W[j]=ITensor(Rb, sites[j], sites[j]', bonds[j-1], bonds[j]); end
    W[N]=ITensor(Rl, sites[N], sites[N]', bonds[N-1]); MPO(W)
end
function build_setup_pxp(N)
    Random.seed!(42); sites=siteinds("S=1",N)
    HT=OpSum(); HT+=1,"Xp",1
    for j in 0:N-2; HT+=1,"Px",j+1,"LP",j+2; HT+=1,"RP",j+1,"Xp",j+2; end
    HT+=1,"Px",N; H=MPO(HT,sites)
    P=NotEqlsLoop_R1(sites); psi0=random_mps(sites)
    psi=replaceprime(contract(P,copy(psi0),:coo,:aliased;denseLinksB=0),1=>0)
    return H, psi
end

# reshape a gram ITensor (primed/unprimed link pair) to a Hermitian matrix + eigvals.
function gram_matrix(G)
    gi = collect(ITensors.inds(G))
    unp = filter(I -> ITensors.plev(I)==0, gi); prm = filter(I -> ITensors.plev(I)==1, gi)
    d = prod(I -> ITensors.dim(I), unp; init=1)
    Gd = ITensors.has_external_storage(G) ? SparseBackends.to_dense_itensors_unfused(G) : G
    M = reshape(Array(Gd, unp..., prm...), d, d); (M + M')/2
end

# connected-component blocks of a (near) block-diagonal SPD matrix, via graph of
# |M_ij| > tol·max|M|. Returns list of index groups.
function blocks_of(M; relthresh=1e-9)
    n = size(M,1); thr = relthresh * maximum(abs, M)
    seen = falses(n); comps = Vector{Vector{Int}}()
    for s in 1:n
        seen[s] && continue
        stack=[s]; comp=Int[]
        while !isempty(stack)
            i=pop!(stack); seen[i] && continue; seen[i]=true; push!(comp,i)
            for j in 1:n
                (j!=i && !seen[j] && abs(M[i,j])>thr) && push!(stack,j)
            end
        end
        push!(comps, sort!(comp))
    end
    comps
end

function report_spectrum(name, M)
    λ = sort(real.(eigvals(Hermitian(M))); rev=true)
    mx = λ[1]; posλ = λ[λ .> 0]
    mn_all = λ[end]; mn_pos = isempty(posλ) ? NaN : minimum(posλ)
    tiny = count(l -> l <= 1e-10*mx, λ)                 # numerical nulls
    condn = mn_pos > 0 ? mx/mn_pos : Inf
    decades = mn_pos > 0 ? log10(mx/mn_pos) : Inf
    @printf("\n[%s]  dim=%d  λmax=%.4g  λmin(all)=%.4g  λmin(>0)=%.4g\n", name, length(λ), mx, mn_all, mn_pos)
    @printf("    cond(λmax/λmin>0) = %.3g   decades = %.2f   numerical-nulls(<1e-10·λmax) = %d\n", condn, decades, tiny)
    # ratio of consecutive eigenvalues → smooth decay vs bimodal gap
    ratios = [λ[i]/λ[i+1] for i in 1:length(λ)-1 if λ[i+1] > 1e-14*mx]
    biggap = isempty(ratios) ? 1.0 : maximum(ratios)
    @printf("    eigenvalues (top→bottom): %s\n", join([@sprintf("%.3g",l) for l in λ], "  "))
    @printf("    largest consecutive ratio (gap indicator; ≫1 ⇒ bimodal, ~O(1) ⇒ smooth) = %.3g\n", biggap)
    return λ
end

# ── PART 1: converged aliased-ψ gram spectrum ────────────────────────────────
function part1(N, bd, nsw)
    println("="^78, "\nPART 1  — converged aliased-ψ gram spectrum (N=$N, bd=$bd, $nsw sweeps)\n", "="^78)
    H, psi = build_setup_pxp(N)
    s=Sweeps(nsw); setmaxdim!(s,bd); setmindim!(s,bd); setcutoff!(s,1e-10)
    E,psi2,_,_ = dmrg(H, psi, s; outputlevel=0, use_early_exit=false,
                      run_mode=:bop_aliased, minv_from_p=nothing, gram_from_h=false)
    @printf("converged E=%.8f (ref ground -14.771969)\n", E)
    b = div(N,2)                     # mid-chain bond (two-site block b, b+1)
    ITensorMPS.orthogonalize!(psi2, b)
    dg(from,to) = begin
        T = *(psi2[from], SparseBackends._dag_link_primed(psi2[from]); preserve_bs_output=true)
        i=from; step = from<=to ? 1 : -1
        while i!=to; i+=step
            T = *(T, psi2[i]; preserve_bs_output=true)
            T = *(T, SparseBackends._dag_link_primed(psi2[i]); preserve_bs_output=true)
        end
        T
    end
    Lg = dg(1, b-1); Rg = dg(N, b+2)
    for (nm,G) in (("Lgram (left env, 1..b-1)",Lg), ("Rgram (right env, b+2..N)",Rg))
        M = gram_matrix(G)
        report_spectrum(nm, M)
        comps = blocks_of(M)
        @printf("    block-diagonal structure: %d block(s), sizes %s\n", length(comps), join(length.(comps),","))
        if length(comps) > 1
            for (bi,c) in enumerate(comps)
                sub = (M[c,c] + M[c,c]')/2; λb = sort(real.(eigvals(Hermitian(sub))); rev=true)
                pos = λb[λb .> 1e-14*max(λb[1],eps())]
                mn = isempty(pos) ? NaN : minimum(pos)
                condb = isempty(pos) ? Inf : λb[1]/mn
                @printf("      block %d (size %d): λmax=%.3g  λmin(>0)=%.3g  cond=%.3g  decades=%.2f\n",
                        bi, length(c), λb[1], mn, condb, log10(condb))
            end
        end
    end
end

# ── PART 2: pure-P structural spectrum (no ψ) ────────────────────────────────
# Doubled-bond P†P transfer: Q = Σ_{s,s'} W[j]_{s,s',l,r} · conj(W[j]_{s,s',l',r'}),
# a map (l,l') → (r,r'). Fold from the left boundary and report the accumulated
# transfer's eigenvalue spectrum (gap ⇒ structural null; graded ⇒ none).
function part2(N)
    println("\n", "="^78, "\nPART 2 — pure-P structural spectrum: P†P doubled FSM bond (no ψ), N=$N\n", "="^78)
    D = 2                                  # PXP FSM bond dim
    # single-site bulk transfer on doubled bond, as a D²×D² matrix (l,l')→(r,r')
    # Rb indices: (s, s', l, r) 0-based dims (3,3,2,2)
    Q = zeros(Float64, D*D, D*D)           # rows=(r,r'), cols=(l,l')
    for s in 1:3, sp in 1:3, l in 1:D, r in 1:D, lp in 1:D, rp in 1:D
        Q[(r-1)*D+rp, (l-1)*D+lp] += Rb[s,sp,l,r] * Rb[s,sp,lp,rp]
    end
    # left boundary vector from Rf (s,s',r): v0[(r,r')] = Σ_{s,s'} Rf[s,s',r]·Rf[s,s',r']
    v0 = zeros(Float64, D*D)
    for s in 1:3, sp in 1:3, r in 1:D, rp in 1:D
        v0[(r-1)*D+rp] += Rf[s,sp,r]*Rf[s,sp,rp]
    end
    # Q is NOT symmetric (it's a transfer operator) — report its TRUE (complex)
    # eigenvalues; do NOT symmetrize. A zero eigenvalue ⇒ a bond direction P†P
    # annihilates ⇒ an exact STRUCTURAL null. All-O(1), no zero ⇒ no clean null.
    ev = eigvals(Q); mag = abs.(ev)
    order = sortperm(mag; rev=true)
    mx = maximum(mag)
    nz = count(m -> m < 1e-10*mx, mag)
    @printf("\n[Q — single-site P†P doubled-bond transfer, %d×%d]\n", D*D, D*D)
    @printf("    eigenvalues (by |λ|): %s\n", join([@sprintf("%.4g%+.4gim", real(ev[i]), imag(ev[i])) for i in order], "  "))
    @printf("    |λ|: %s\n", join([@sprintf("%.4g", mag[i]) for i in order], "  "))
    @printf("    exact-zero eigenvalues (|λ|<1e-10·|λ|max) = %d   → %s\n", nz,
            nz>0 ? "structural null EXISTS" : "NO structural null (all O(1))")
    # Accumulated: eigenvalues of Q^k are eig(Q).^k, so the gram's structural
    # conditioning grows as (λ_dom/λ_sub)^k — report that ratio.
    evre = sort(real.(ev); rev=true)
    subdom = length(evre) >= 2 ? evre[2]/evre[1] : NaN
    @printf("    Perron λ_dom=%.4g, λ_sub/λ_dom=%.4g → over k bulk sites the doubled-bond\n", evre[1], subdom)
    @printf("      transfer condition grows ~(λ_dom/|λ_sub|)^k (exponential in system size)\n")
    println("    (boundary v0 on doubled bond = ", round.(v0; digits=4), ")")
end

let
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
    bd = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 40
    nsw = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 12
    part1(N, bd, nsw)
    part2(N)
end
nothing
