# B-phase DMRG integration test: sparse DMRG on N plaquettes, optionally vs dense.
# Usage: julia test_b_phase.jl <N> [nsweeps] [maxdim] [sparse_only]
#   N           — number of plaquettes (required)
#   nsweeps     — number of sweeps (default 8)
#   maxdim      — max bond dimension (default 40)
#   sparse_only — "true" to skip dense DMRG (default "false")
# Reports: timing, real bond dims, energy gap, fidelity vs dense.
#
# Verbose pre-DMRG diagnostics (absorbed from test_projection_correctness.jl):
#   BPHASE_VERBOSE=1 — runs before DMRG:
#     Section 1: inner products + ProjMPO matvec + iso check (sparse vs dense)
#     Section 2: site-by-site |H_sparse[j] - H_dense[j]| + energy comparisons
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
include("utils.jl")

length(ARGS) < 1 && error("Usage: julia test_b_phase.jl <N> [nsweeps] [maxdim] [sparse_only]")
const N_PLAQ     = parse(Int,  ARGS[1])
const N_SWEEPS   = length(ARGS) >= 2 ? parse(Int,   ARGS[2]) : 8
const MAXDIM     = length(ARGS) >= 3 ? parse(Int,   ARGS[3]) : 40
const SPARSE_ONLY = length(ARGS) >= 4 ? (ARGS[4] == "true") : false

function build_setup(N::Int)
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
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += 0.5, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    H_dense   = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in H_sparse])
    psi_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp])
    return H_sparse, H_dense, psi_sp, psi_dense
end

function real_linkdims(psi)
    [prod(ITensors.dim(I) for I in commoninds(psi[i], psi[i+1]); init=1) for i in 1:length(psi)-1]
end

function shared_inds_summary(psi, i)
    parts = String[]
    for I in collect(commoninds(psi[i], psi[i+1]))
        push!(parts, string(ITensors.dim(I), ITensors.hastags(I, "Link") ? "L" : "M"))
    end
    "[" * join(parts, "x") * "]"
end

let
    println("############################################################")
    println("##  N_plaq = $N_PLAQ   sites = $(2*N_PLAQ+2)   nsweeps = $N_SWEEPS   maxdim = $MAXDIM   sparse_only = $SPARSE_ONLY")
    println("############################################################")

    t_setup = @elapsed (H_sp, H_d, psi_sp, psi_d) = build_setup(N_PLAQ)

    # --- BPHASE_VERBOSE=1: pre-DMRG sanity checks (from test_projection_correctness) ---
    if get(ENV, "BPHASE_VERBOSE", "0") == "1"
        println("\n" * "="^70)
        println("VERBOSE Section 1: Inner product sanity (no DMRG)")
        println("="^70)

        psi_sp_o = ITensorMPS.orthogonalize(psi_sp, 1)
        psi_d_o  = ITensorMPS.orthogonalize(psi_d,  1)

        norm_sp = inner(psi_sp, psi_sp);  norm_d = inner(psi_d, psi_d)
        println("  <psi|psi> sparse=$norm_sp  dense=$norm_d  |diff|=$(abs(norm_sp-norm_d))")

        eh_sp = inner(psi_sp', H_sp, psi_sp);  eh_d = inner(psi_d', H_d, psi_d)
        println("  <psi|H|psi> sparse=$eh_sp  dense=$eh_d  |diff|=$(abs(eh_sp-eh_d))")

        println("  <psi|psi> after orth: sparse=$(inner(psi_sp_o,psi_sp_o))  dense=$(inner(psi_d_o,psi_d_o))")
        println("  E after orth: sparse=$(real(inner(psi_sp_o',H_sp,psi_sp_o)/inner(psi_sp_o,psi_sp_o)))  dense=$(real(inner(psi_d_o',H_d,psi_d_o)/inner(psi_d_o,psi_d_o)))")

        println("\n  ProjMPO matvec <phi|H_eff|phi>/<phi|phi> at bond 1:")
        PH_sp = position!(ProjMPO(H_sp), psi_sp_o, 1)
        PH_d  = position!(ProjMPO(H_d),  psi_d_o,  1)
        phi_sp = psi_sp_o[1] * psi_sp_o[2];  phi_d = psi_d_o[1] * psi_d_o[2]
        Hphi_sp = product(PH_sp, phi_sp);     Hphi_d = product(PH_d, phi_d)
        r_sp = scalar(dag(phi_sp)*Hphi_sp)/scalar(dag(phi_sp)*phi_sp)
        r_d  = scalar(dag(phi_d) *Hphi_d) /scalar(dag(phi_d) *phi_d)
        println("    sparse=$(real(r_sp))  dense=$(real(r_d))  |diff|=$(abs(r_sp-r_d))")

        println("\n  Right-isometry check (densified tensors, all bonds):")
        for (lbl, psi_x) in (("sparse_o", psi_sp_o), ("dense_o", psi_d_o))
            for i in 2:length(psi_x)
                lb = commoninds(psi_x[i-1], psi_x[i])
                T_d = SparseBackends.to_dense_itensors_unfused(psi_x[i])
                Td  = prime(dag(T_d), lb...)
                E   = T_d * Td
                Cl = combiner(lb...; tags="bL");  Cr = combiner(prime.(lb)...; tags="bR")
                Em = Array(E*Cl*Cr, combinedind(Cl), combinedind(Cr))
                D  = ITensors.dim(combinedind(Cl))
                println("    $lbl psi[$i] iso_err=$(round(norm(Em - Matrix{ComplexF64}(I,D,D)); sigdigits=4))  D=$D")
            end
        end

        println("\n" * "="^70)
        println("VERBOSE Section 2: Site-by-site H comparison")
        println("="^70)
        for j in 1:length(H_sp)
            hs = ITensors.has_external_storage(H_sp[j]) ? SparseBackends.to_dense_itensors(H_sp[j]) : H_sp[j]
            hd = H_d[j]
            if issetequal(inds(hs), inds(hd))
                println("  Site $j: |H_sp - H_d| = $(norm(hs - hd))")
            else
                println("  Site $j: index mismatch")
            end
        end
        println("  <psi0|H_sparse|psi0> = $(inner(psi_sp', H_sp, psi_sp))")
        println("  <psi0|H_dense|psi0>  = $(inner(psi_d',  H_d,  psi_d))")
        println("="^70 * "\n")
    end
    println("Setup time: $(round(t_setup, digits=1))s")
    println("psi_sp initial REAL linkdims = ", real_linkdims(psi_sp))
    println("psi_d  initial linkdims      = ", real_linkdims(psi_d))
    println("psi_sp is_sparse_mps? ", SparseBackends.is_sparse_mps(psi_sp))

    sweeps = Sweeps(N_SWEEPS)
    setmaxdim!(sweeps, [min(10*k, MAXDIM) for k in 1:N_SWEEPS]...)
    setmindim!(sweeps, 1)
    setcutoff!(sweeps, 1e-10)

    E_d = nothing
    psi_d_out = nothing
    if !SPARSE_ONLY
        println("\n--- Dense DMRG (N=$N_PLAQ, maxdim=$MAXDIM, sweeps=$N_SWEEPS) ---")
        t_d = @elapsed (E_d, psi_d_out) = dmrg(H_d, psi_d, sweeps; outputlevel=1, use_early_exit=false)
        println("Dense final E = $E_d  (time: $(round(t_d, digits=1))s)")
        println("Dense final REAL linkdims = ", real_linkdims(psi_d_out))
    end

    println("\n--- Sparse DMRG (Path B, N=$N_PLAQ, maxdim=$MAXDIM, sweeps=$N_SWEEPS) ---")
    t_s = @elapsed (E_s, psi_sp_out) = dmrg(H_sp, psi_sp, sweeps; outputlevel=1, use_early_exit=false)
    println("Sparse final E = $E_s  (time: $(round(t_s, digits=1))s)")
    println("Sparse final REAL linkdims  = ", real_linkdims(psi_sp_out))
    println("Sparse final Link-only dims = ", [ITensors.dim(commonind(psi_sp_out[i], psi_sp_out[i+1])) for i in 1:length(psi_sp_out)-1])
    println("Sparse per-bond shared-ind breakdown (L=Link, M=Mult):")
    for i in 1:length(psi_sp_out)-1
        println("   bond $i: $(shared_inds_summary(psi_sp_out, i))")
    end

    if !SPARSE_ONLY && !isnothing(E_d)
        println("\n|E_dense - E_sparse| = ", abs(E_d - E_s))
        println("\n--- Overlap check ---")
        psi_sp_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp_out])
        nrm_sp = sqrt(real(inner(psi_sp_dense, psi_sp_dense)))
        nrm_d  = sqrt(real(inner(psi_d_out,   psi_d_out)))
        fid    = abs(inner(psi_sp_dense, psi_d_out)) / (nrm_sp * nrm_d)
        println("  ||psi_sparse|| = $nrm_sp   ||psi_dense|| = $nrm_d")
        println("  fidelity = $fid   1 - fidelity = $(1 - fid)")
    end
end
nothing
