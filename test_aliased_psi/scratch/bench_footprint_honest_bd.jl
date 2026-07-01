# Footprint + honest-bond-dimension benchmark: dense / BlockSparse / Aliased psi.
#
# Same KL-model Hamiltonian + projector as bench_aliased_vs_bs_vs_dense.jl and
# test_sparse_kl.jl (they are all the SAME physical model). Runs all three
# backends head-to-head at a FLAT maxdim for a fixed number of sweeps, then for
# the BS and ALI final wavefunctions reports:
#   - reported link dims (dim of the single shared link index)
#   - HONEST link dims (∏ of all shared-index dims — the true variational bond)
#   - per-site n_blocks / n_templates (alias dedup ratio)
#   - what the same wavefunction would cost as pure dense / pure BS
# and prints an explicit verdict on whether ALI has regressed to dense.
#
# Defaults reproduce the requested run: N_plaq=12, maxdim=40, 10 sweeps (flat).
#   BENCH_N_PLAQ=12  BENCH_MD=40  BENCH_NSWEEPS=10

using SparseBackends, ITensors, ITensorMPS
using Random
using Printf

include("../test_sparse_psi/utils.jl")

const N_PLAQ    = parse(Int, get(ENV, "BENCH_N_PLAQ", "12"))
const MAXDIM    = parse(Int, get(ENV, "BENCH_MD", "40"))
const N_SWEEPS  = parse(Int, get(ENV, "BENCH_NSWEEPS", "10"))

function build_setup(N::Int, psign::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5,          "Id",            2*j-1, "Id",            2*j, "Id",            2*j+1, "Id",            2*j+2
        t += 0.5 * psign,  "exp(i*pi*Sy)",  2*j-1, "exp(i*pi*Sx)",  2*j, "exp(i*pi*Sx)",  2*j+1, "exp(i*pi*Sy)",  2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    psi0     = random_mps(sites)
    return H, P_sparse, psi0
end

mpsize(psi) = Base.summarysize(psi)

# Reported link dim: the dim of the single shared link Index between neighbors.
reported_linkdims(psi) =
    [ITensors.dim(commonind(psi[i], psi[i+1])) for i in 1:length(psi)-1]

# Honest link dim: product of ALL shared-index dims between neighbors. With the
# doubled-link convention the bond appears as (channel × multiplicity); the
# honest bond dim is the product, i.e. the true rank of the bipartition.
function honest_linkdims(psi)
    [(cis = commoninds(psi[i], psi[i+1]); isempty(cis) ? 0 : prod(ITensors.dim, cis))
     for i in 1:length(psi)-1]
end

# Per-site storage breakdown. Returns rows + totals: payload (numeric data),
# what-if-dense, what-if-BS, and the min/mean alias-dedup ratio across sites.
function inspect_storage(psi)
    payload_total = 0
    site_total    = 0
    full_total    = 0   # if stored fully dense
    bs_total      = 0   # if stored as plain BS (n_blocks * blksize)
    ratios        = Float64[]
    rows = String[]
    for (i, T) in enumerate(psi)
        s  = try ITensors.get_external_storage(T) catch _ nothing end
        ss = Base.summarysize(T)
        site_total += ss
        if s isa SparseBackends.WrappedAliasedBlockSparse
            ali    = s.aliased
            nb     = length(ali.keys)
            nt     = ali.n_templates
            blksz  = ali.blksize
            data_b = Base.summarysize(ali.templates)
            full_b = prod(ali.dims) * sizeof(eltype(ali.templates))
            bs_b   = nb * blksz * sizeof(eltype(ali.templates))
            ratio  = nb / max(nt, 1)
            payload_total += data_b; full_total += full_b; bs_total += bs_b
            push!(ratios, ratio)
            push!(rows, @sprintf("site %2d [ALI] nb=%d ntmpl=%d blksize=%d  dedup(nb/nt)=%.2fx  templates=%.2fKiB  if_dense=%.2fKiB",
                                 i, nb, nt, blksz, ratio, data_b/1024, full_b/1024))
        elseif s isa SparseBackends.WrappedBlockSparse
            bs     = s.blocksparse
            nb     = length(bs.keys)
            data_b = Base.summarysize(bs.data)
            full_b = prod(bs.dims) * sizeof(eltype(bs.data))
            payload_total += data_b; full_total += full_b; bs_total += data_b
            push!(rows, @sprintf("site %2d [BS]  nblocks=%d blksize=%d  data=%.2fKiB  if_dense=%.2fKiB",
                                 i, nb, bs.blksize, data_b/1024, full_b/1024))
        else
            a = ITensors.array(T)
            data_b = Base.summarysize(a)
            payload_total += data_b; full_total += data_b; bs_total += data_b
            push!(rows, @sprintf("site %2d [dense] data=%.2fKiB", i, data_b/1024))
        end
    end
    return rows, payload_total, site_total, full_total, bs_total, ratios
end

function run_dmrg_timed(label, H, psi)
    sweep_times = Float64[]
    cum_excl1 = 0.0
    E = NaN
    for k in 1:N_SWEEPS
        sw = Sweeps(1); setmaxdim!(sw, MAXDIM); setmindim!(sw, 1); setcutoff!(sw, 1e-10)
        t = @elapsed (E, psi) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false)
        push!(sweep_times, t)
        if k > 1; cum_excl1 += t; end
        @printf("  [%s sweep %2d / md=%d] %8.3fs  E=%.10f\n", label, k, MAXDIM, t, E)
        flush(stdout)
    end
    n_excl1 = length(sweep_times) - 1
    avg_excl1 = n_excl1 > 0 ? cum_excl1 / n_excl1 : NaN
    return E, psi, sweep_times, cum_excl1, avg_excl1
end

println("=== Footprint + honest-BD benchmark: dense vs BS vs Aliased ===")
println("Model: KL-model S=1 projected ladder (same as bench_aliased_vs_bs_vs_dense.jl / test_sparse_kl.jl)")
println("N_plaq=$N_PLAQ  (=$(2*N_PLAQ+2) sites)   maxdim=$MAXDIM (flat)   sweeps=$N_SWEEPS")
println()

H, P_sparse, psi0 = build_setup(N_PLAQ, +1)
println("Built H, P_sparse, psi0.  System: $(length(psi0)) sites.")
flush(stdout)

println("\n--- Dense DMRG ---")
psi_d = copy(psi0)
E_dense, psi_dense_final, t_dense, cum_d, avg_d = run_dmrg_timed("dense", H, psi_d)
mem_d = mpsize(psi_dense_final)

println("\n--- BS DMRG ---")
psi_bs0 = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
E_bs, psi_bs_final, t_bs, cum_bs, avg_bs = run_dmrg_timed("BS", H, psi_bs0)
mem_bs = mpsize(psi_bs_final)

println("\n--- Aliased DMRG ---")
psi_ali0 = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)
E_ali, psi_ali_final, t_ali, cum_ali, avg_ali = run_dmrg_timed("ALI", H, psi_ali0)
mem_ali = mpsize(psi_ali_final)

println("\n================== SUMMARY ==================")
@printf("Final energies:\n")
@printf("  dense:  E = %.10f\n", E_dense)
@printf("  BS:     E = %.10f\n", E_bs)
@printf("  ALI:    E = %.10f\n", E_ali)

@printf("\nAverages (excl sweep 1 = JIT):\n")
@printf("  dense  avg/sweep = %.3fs\n", avg_d)
@printf("  BS     avg/sweep = %.3fs   speedup_vs_dense = %.2fx\n", avg_bs, avg_d/max(avg_bs,1e-9))
@printf("  ALI    avg/sweep = %.3fs   speedup_vs_dense = %.2fx   speedup_vs_BS = %.2fx\n",
        avg_ali, avg_d/max(avg_ali,1e-9), avg_bs/max(avg_ali,1e-9))

@printf("\nFinal MPS footprint:\n")
@printf("  dense: %8.3f MiB\n", mem_d/2^20)
@printf("  BS:    %8.3f MiB   vs dense = %.2fx %s\n", mem_bs/2^20, mem_d/max(mem_bs,1),
        mem_bs < mem_d ? "(smaller ✓)" : "(NOT smaller ✗)")
@printf("  ALI:   %8.3f MiB   vs dense = %.2fx %s   vs BS = %.2fx\n", mem_ali/2^20,
        mem_d/max(mem_ali,1), mem_ali < mem_d ? "(smaller ✓)" : "(NOT smaller ✗)",
        mem_bs/max(mem_ali,1))

# ── Honest bond dimension + regression verdict ─────────────────────────────
function report_bd(label, psi)
    rep = reported_linkdims(psi)
    hon = honest_linkdims(psi)
    println("\n--- $label bond dimension ---")
    @printf("  reported max linkdim = %d   honest max linkdim = %d   (MAXDIM = %d)\n",
            maximum(rep), maximum(hon), MAXDIM)
    println("  reported linkdims = $rep")
    println("  honest   linkdims = $hon")
    over = findall(h -> h > MAXDIM, hon)
    if isempty(over)
        println("  ✓ honest bond dimension obeys MAXDIM at every bond.")
    else
        println("  ✗ honest bond dimension EXCEEDS MAXDIM at bonds $over — bond-dim invariant violated.")
    end
    return rep, hon
end

rep_bs,  hon_bs  = report_bd("BS",  psi_bs_final)
rep_ali, hon_ali = report_bd("ALI", psi_ali_final)

println("\n--- ALI per-site storage ---")
rows, payload, site_total, full, bs_eq, ratios = inspect_storage(psi_ali_final)
for r in rows; println("    $r"); end
@printf("  ALI numeric payload = %.2f KiB   if_dense = %.2f KiB   if_BS = %.2f KiB\n",
        payload/1024, full/1024, bs_eq/1024)
@printf("  vs-dense payload compression = %.2fx   vs-BS = %.2fx\n",
        full/max(payload,1), bs_eq/max(payload,1))
min_ratio  = isempty(ratios) ? 0.0 : minimum(ratios)
mean_ratio = isempty(ratios) ? 0.0 : sum(ratios)/length(ratios)
@printf("  alias dedup ratio (nb/nt): min = %.2fx   mean = %.2fx\n", min_ratio, mean_ratio)

println("\n================== REGRESSION VERDICT (ALI) ==================")
honest_ok   = all(h -> h <= MAXDIM, hon_ali)
dedup_ok    = min_ratio > 1.0 + 1e-9
footprint_ok = mem_ali < mem_d
@printf("  [%s] honest bond dim obeys MAXDIM (no silent rank inflation)\n", honest_ok   ? "PASS" : "FAIL")
@printf("  [%s] alias dedup > 1x at every site (n_templates not collapsed to 1)\n", dedup_ok ? "PASS" : "FAIL")
@printf("  [%s] ALI footprint < dense\n", footprint_ok ? "PASS" : "FAIL")
if honest_ok && dedup_ok && footprint_ok
    println("  → No regression to dense: structure preserved AND footprint beats dense.")
elseif honest_ok && dedup_ok && !footprint_ok
    println("  → Structure preserved (honest BD ≤ MAXDIM, dedup intact) but footprint ≥ dense:")
    println("    likely IMPLEMENTATION OVERHEAD (metadata/wrappers), not a datastructure regression.")
    println("    Investigate keys/alias_ids/scalars + wrapper bookkeeping, not the SVD truncation.")
else
    println("  → POSSIBLE REGRESSION TO DENSE. At least one structural check failed:")
    !honest_ok && println("    - honest bond dim exceeds MAXDIM → truncation not bounding the true rank.")
    !dedup_ok  && println("    - some site has n_templates == n_blocks → one-template collapse (dedup lost).")
    println("    This is a datastructure regression, NOT mere overhead. Diagnose before any fix.")
end
nothing
