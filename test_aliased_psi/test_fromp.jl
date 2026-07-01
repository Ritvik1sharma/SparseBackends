# test_fromp.jl — Step 2b: DMRG with from-P M^{±1/2} (dmrg kwarg minv_from_p=true),
# i.e. M^{±1/2}=c^{∓...}·G with c=2^⌈env/2⌉ geometric, NO eigen. Compares to baseline.
# Run: OPENBLAS_NUM_THREADS=1 julia --project=.. test_fromp.jl \
#        --N-plaq 12 --maxdim 40 --n-sweeps 10 [--psign 1] [--from-p true]
using SparseBackends, ITensors, ITensorMPS, Printf, Random, ArgParse
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

s = ArgParseSettings()
@add_arg_table! s begin
    "--N-plaq"; arg_type=Int; default=12
    "--maxdim"; arg_type=Int; default=40
    "--n-sweeps"; arg_type=Int; default=10
    "--psign"; arg_type=Int; default=1
    "--from-p"; arg_type=Bool; default=true   # true → minv_from_p=true; false → baseline eigen
    "--seed"; arg_type=Int; default=42
end
a = parse_args(s)
N=a["N-plaq"]; md=a["maxdim"]; nsw=a["n-sweeps"]; psign=a["psign"]; fromp=a["from-p"]; seed=a["seed"]
mfp = fromp ? true : nothing
println("="^70)
println("FROM-P TEST  N=$N md=$md nsweeps=$nsw psign=$psign  minv_from_p=$mfp")
println("="^70)
H, psi = build_setup(N, psign, 3, seed)
E = NaN
for i in 1:nsw
    sw = Sweeps(1); setmaxdim!(sw, md); setmindim!(sw, 1); setcutoff!(sw, 1e-10)
    global E, psi = dmrg(H, psi, sw; outputlevel=0, run_mode=:bop_aliased,
                         use_early_exit=false, minv_from_p=mfp)
    @printf("  [sweep %2d]  E=%.12f\n", i, E)
end
@printf("\nFINAL E=%.12f  (minv_from_p=%s)\n", E, string(mfp))
