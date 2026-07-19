# test_fromp_dedup.jl — does the from-P M^{±1/2}=c^{∓...}·G factor keep aliased/BS
# storage + dedup, vs the eigen factor (dense)? Inspect storage type + dedup (nb/nt)
# of: the gram G, the from-P Linv, and the eigen Linv. Also the seed y0=Mhalf·φ.
# Run: OPENBLAS_NUM_THREADS=1 julia --project=.. test_fromp_dedup.jl [--N-plaq 12 --maxdim 40 --n-sweeps 4]
using SparseBackends, ITensors, ITensorMPS, Printf, Random, LinearAlgebra, ArgParse
include("../test_sparse_psi/utils.jl")

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
    return H, psi
end

# report storage class + dedup (nb/nt) of an ITensor
function info(lbl, T)
    if !ITensors.has_external_storage(T)
        @printf("  %-22s DENSE (plain array, no structure/dedup)\n", lbl); return
    end
    s = ITensors.get_external_storage(T)
    tn = string(nameof(typeof(s)))
    nb = "?"; nt = "?"; dedup = "-"
    try
        if s isa SparseBackends.WrappedAliasedBlockSparse
            a = s.aliased; nb = length(a.alias_ids); nt = a.n_templates
            dedup = @sprintf("%.2fx", nb/max(nt,1))
        elseif s isa SparseBackends.WrappedBlockSparse
            bs = s.blocksparse; nb = length(bs.keys); nt = nb; dedup = "1.00x (BS, no aliasing)"
        end
    catch e; end
    @printf("  %-22s %s   nb=%s ntmpl=%s  dedup=%s\n", lbl, tn, string(nb), string(nt), dedup)
end

s = ArgParseSettings()
@add_arg_table! s begin
    "--N-plaq"; arg_type=Int; default=12
    "--maxdim"; arg_type=Int; default=40
    "--n-sweeps"; arg_type=Int; default=4
    "--psign"; arg_type=Int; default=1
    "--bond"; arg_type=Int; default=12
end
a = parse_args(s); N=a["N-plaq"]; md=a["maxdim"]; nsw=a["n-sweeps"]; psign=a["psign"]; bond=a["bond"]

println("="^70)
println("FROM-P DEDUP CHECK  N=$N md=$md nsweeps=$nsw psign=$psign bond=$bond")
println("="^70)
H, psi = build_setup(N, psign, 3, 42)
for i in 1:nsw
    sw = Sweeps(1); setmaxdim!(sw, md); setmindim!(sw, 1); setcutoff!(sw, 1e-10)
    global psi
    _, psi = dmrg(H, psi, sw; outputlevel=0, run_mode=:bop_aliased, use_early_exit=false)
end
gc = SparseBackends.init_gram_cache(psi)
G = SparseBackends.get_left_gram(gc, bond)
# Build the 2-site φ the SAME way dmrg.jl does: `preserve_bs_output=true` routes
# through ITensors' `*` to the aliased-preserving contraction (plain `*` densifies).
phi = *(psi[bond], psi[bond+1]; preserve_bs_output=true)
c = 2.0^cld(bond-1, 2)

println("\n-- inputs --")
info("psi[bond]", psi[bond])
info("psi[bond+1]", psi[bond+1])
info("phi (2-site)", phi)
info("gram G (Lgram)", G)
println("   c = 2^ceil((bond-1)/2) = $c")

println("\n-- from-P factors  M^{±1/2}=c^{∓...}·G --")
Mh_p, Linv_p = SparseBackends.build_half_pair_single_fromP(G, c)
info("from-P Mhalf", Mh_p)
info("from-P Linv", Linv_p)

println("\n-- eigen factors  build_half_pair_single(G) --")
Mh_e, Linv_e = SparseBackends.build_half_pair_single(G)
info("eigen Mhalf", Mh_e)
info("eigen Linv", Linv_e)

println("\n-- seed y0 = Mhalf · phi  (what feeds the Krylov eigsolve) --")
try; info("from-P  y0 (apply)", SparseBackends.apply_minv_preserve_bs(Mh_p, phi, phi; fission=true)); catch e; println("  from-P y0 apply: ", e); end
try; info("eigen   y0 (apply)", SparseBackends.apply_minv_preserve_bs(Mh_e, phi, phi; fission=true)); catch e; println("  eigen y0 apply: ", e); end

# Storage note: a scalar multiply sqrt(c)*phi keeps φ's aliased dedup (Number *
# WrappedAliasedBlockSparse preserves templates/keys/alias_ids), UNLIKE the dense-factor
# apply above which collapses dedup. (The scalar SEED/recovery was tested in DMRG and
# reverted — it converged ~10× further from the dense-PHP ground truth than the full-G
# apply; see dmrg.jl notes. This line just documents the storage behaviour.)
println("\n-- scalar sqrt(c)*phi (dedup-preserving storage demo) --")
info("scalar  y0", sqrt(c) * phi)
println("\nDONE.")
