# diag_gram_structure.jl — VISUALIZE the gram M (Lgram/Rgram) structure at one bond
# for a small aliased-ψ KL run. Tests three competing pictures of M:
#   (a) low-rank?      → singular-value spectrum + numerical rank
#   (b) block-sparse/structured? → |M| text heatmap (zeros vs nonzeros)
#   (c) how rank compares to φ's n_keys / dedup (the "rank ≈ n_c/dedup" hypothesis)
#
# Run:  DIAG_N_PLAQ=1 DIAG_MAXDIM=10 julia --project=.. diag_gram_structure.jl
ENV["SB_ALIASED_ENABLE"] = "1"
ENV["BMF_ISO_PATH"]      = "0"
ENV["BMF_APPLY_MINV"]    = "1"
using SparseBackends, ITensors, ITensorMPS, LinearAlgebra, Printf, Random
include("../test_sparse_psi/utils.jl")

function build_setup(N, psign, spin)
    Random.seed!(42)
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
    return H, psi
end

function show_matrix(nm, G)
    if ITensors.has_external_storage(G)
        println("\n=== $nm : WRAPPED (not plain dense) — skip"); return
    end
    is = collect(ITensors.inds(G))
    println("\n=== $nm : ndims=$(length(is)) dims=$([ITensors.dim(i) for i in is]) ===")
    for i in is
        println("    leg dim=$(ITensors.dim(i)) plev=$(ITensors.plev(i)) tags=$(ITensors.tags(i))")
    end
    ket = filter(i -> ITensors.plev(i) == 0, is)
    bra = filter(i -> ITensors.plev(i) == 1, is)
    if length(ket) != 2 || length(bra) != 2
        println("    (expected channel+mult ket/bra = 4 legs; got ket=$(length(ket)) bra=$(length(bra)); skip)")
        return
    end
    # within each side: mult = larger dim, channel = smaller dim
    ks = sort(ket; by = i -> -ITensors.dim(i)); bs = sort(bra; by = i -> -ITensors.dim(i))
    mk, ck = ks[1], ks[2]; mb, cb = bs[1], bs[2]
    dm = ITensors.dim(mk); dc = ITensors.dim(ck)
    A4 = Array(G, mk, ck, mb, cb)            # (mult, chan, mult', chan')
    s1 = svdvals(reshape(A4, dm * dc, dm * dc))[1]
    println("  → channel dim n_c=$dc   mult dim=$dm   honest bond=$(dm*dc)")
    svf = svdvals(reshape(A4, dm * dc, dm * dc))
    println("  FULL M  σ/σ₁: ", join([@sprintf("%.3f", s / s1) for s in svf], " "))
    println("  FULL M  numerical rank @1e-6 = ", count(>(1e-6 * s1), svf), " / ", dm * dc)
    println("  CHANNEL-BLOCK ‖M[:,c,:,c']‖/σ₁  (block-diagonal ⇒ off-diagonal ≈ 0):")
    for c in 1:dc
        print("    c=$c: ")
        for cp in 1:dc; @printf("%.3f ", norm(A4[:, c, :, cp]) / s1); end
        println()
    end
    # separability: M[m,c,m',c'] =?= A(c,c') ⊗ B(m,m')  ⇔  rank-1 of (c,c')×(m,m')
    Asep = reshape(permutedims(A4, (2, 4, 1, 3)), dc * dc, dm * dm)
    svs = svdvals(Asep)
    r2 = length(svs) > 1 ? svs[2] / svs[1] : 0.0
    println("  SEPARABILITY (cc')×(mm') σ/σ₁: ", join([@sprintf("%.3f", s / svs[1]) for s in svs], " "))
    println("    → separable channel⊗mult (rank-1)? ", r2 < 1e-6, "   (σ₂/σ₁=$(round(r2; sigdigits=3)))")
end

N  = parse(Int, get(ENV, "DIAG_N_PLAQ", "1"))
md = parse(Int, get(ENV, "DIAG_MAXDIM", "10"))
nsw = parse(Int, get(ENV, "DIAG_NSWEEPS", "2"))
H, psi = build_setup(N, -1, 3)
println("N_plaq=$N  n_sites=$(length(psi))  maxdim=$md")
if nsw > 0
    sw = Sweeps(nsw); setmaxdim!(sw, md); setmindim!(sw, 1); setcutoff!(sw, 1e-10)
    E, psi = dmrg(H, psi, sw; outputlevel = 0, use_early_exit = false)
    println("ran $nsw sweeps, E=$E")
end
nb = length(psi)
b  = max(1, div(nb, 2))
println("\nbond b=$b  (of $nb sites)")
SparseBackends.schema_dbg("phi b=$b", psi[b] * psi[b + 1])

gc = SparseBackends.init_gram_cache(psi)
show_matrix("Lgram b=$b", SparseBackends.get_left_gram(gc, b))
show_matrix("Rgram b=$b", SparseBackends.get_right_gram(gc, b))
nothing
