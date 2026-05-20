# Projection correctness — ladder from inner products to full DMRG.
# Merges: test_innerprod.jl, test_h_compare.jl,
#         test_dmrg_dense_vs_sparse.jl, test_sparse_psi.jl
#
# Usage: julia test_projection_correctness.jl <N_plaq> [nsweeps] [maxdim]
#
# Section 1: Inner product sanity (no DMRG)
#   <psi|psi> and <psi|H|psi> sparse vs dense storage
#   ProjMPO matvec <phi|PH|phi>/<phi|phi>
#   Right-isometry check on densified tensors
# Section 2: Site-by-site H comparison
#   |H_sparse[j] - H_dense[j]| per site
#   Energy via inner products for H_sparse, H_dense, H_original
# Section 3: DMRG convergence
#   Dense DMRG (H_dense, psi_dense) vs Sparse DMRG (H_sparse, psi_sp)
#   |E_dense - E_sparse| final check

using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
include("utils.jl")

length(ARGS) < 1 && error("Usage: julia test_projection_correctness.jl <N_plaq> [nsweeps] [maxdim]")
const _N_PC      = parse(Int, ARGS[1])
const _NS_PC     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const _MAXDIM_PC = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 40

# Build the full setup: H, H_sparse, H_dense, psi0, psi_sp, psi_dense, ConsOpsCombined
function build_setup(N::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)

    os = OpSum()
    for j in 1:N+1
        os += "Sz", 2*j-1, "Sz", 2*j
    end
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
    ConsOps2 = [MPO(os2[j], sites) for j in 1:N]

    function mulMPO(A, B)
        Bp = prime(B, "Site")
        replaceprime(contract(A, Bp, :coo, :coo), 2 => 1)
    end
    P_sparse = ConsOps1[1]
    for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end

    H  = MPO(os, sites)
    H1 = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    H_dense  = replaceprime(contract(P_sparse, H1, :coo, :dense),       3 => 1)

    psi0 = random_mps(sites)
    # Sequential projection for well-behaved initial state
    psi_seq = copy(psi0)
    for j in 1:N
        psi_seq = replaceprime(ConsOps2[j] * psi_seq, 1 => 0)
        normalize!(psi_seq)
    end

    # Sparse MPS from combined projector
    psi_sp    = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    # Dense copy (unfuse external storage so DMRG sees plain Dense ITensors)
    psi_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp])

    return (sites=sites, H=H, H_sparse=H_sparse, H_dense=H_dense,
            psi0=psi_seq, psi_sp=psi_sp, psi_dense=psi_dense,
            P_sparse=P_sparse)
end

let
    s = build_setup(_N_PC)
    H        = s.H
    H_sparse = s.H_sparse
    H_dense  = s.H_dense
    psi0     = s.psi0
    psi_sp   = s.psi_sp
    psi_d    = s.psi_dense

    # =========================================================================
    println("\n" * "="^70)
    println("SECTION 1: Inner product sanity (no DMRG)")
    println("="^70)

    println("\n--- Tensor storage types ---")
    println("psi_sp: ", [ITensors.has_external_storage(T) ? "BS" : "D" for T in psi_sp])
    println("psi_d:  ", [ITensors.has_external_storage(T) ? "BS" : "D" for T in psi_d])
    println("H_sp:   ", [ITensors.has_external_storage(T) ? "BS" : "D" for T in H_sparse])
    println("H_d:    ", [ITensors.has_external_storage(T) ? "BS" : "D" for T in H_dense])

    println("\n--- <psi|psi> ---")
    norm_sp = inner(psi_sp, psi_sp)
    norm_d  = inner(psi_d,  psi_d)
    println("  sparse = ", norm_sp)
    println("  dense  = ", norm_d)
    println("  |diff| = ", abs(norm_sp - norm_d))

    println("\n--- <psi|H|psi> ---")
    eh_sp = inner(psi_sp', H_sparse, psi_sp)
    eh_d  = inner(psi_d',  H_dense,  psi_d)
    println("  sparse = ", eh_sp, "  (Rayleigh = ", real(eh_sp/norm_sp), ")")
    println("  dense  = ", eh_d,  "  (Rayleigh = ", real(eh_d /norm_d),  ")")
    println("  |diff| = ", abs(eh_sp - eh_d))

    println("\n--- After orthogonalize(psi, 1) ---")
    psi_sp_o = ITensorMPS.orthogonalize(psi_sp, 1)
    psi_d_o  = ITensorMPS.orthogonalize(psi_d,  1)
    println("<psi|psi> sparse = ", inner(psi_sp_o, psi_sp_o))
    println("<psi|psi> dense  = ", inner(psi_d_o,  psi_d_o))
    println("E sparse = ", real(inner(psi_sp_o', H_sparse, psi_sp_o) / inner(psi_sp_o, psi_sp_o)))
    println("E dense  = ", real(inner(psi_d_o',  H_dense,  psi_d_o)  / inner(psi_d_o,  psi_d_o)))

    println("\n--- ProjMPO matvec: <phi|PH|phi>/<phi|phi> at bond 1 ---")
    PH_sp = position!(ProjMPO(H_sparse), psi_sp_o, 1)
    PH_d  = position!(ProjMPO(H_dense),  psi_d_o,  1)
    phi_sp = psi_sp_o[1] * psi_sp_o[2]
    phi_d  = psi_d_o[1]  * psi_d_o[2]
    Hphi_sp = product(PH_sp, phi_sp)
    Hphi_d  = product(PH_d,  phi_d)
    num_sp = scalar(dag(phi_sp) * Hphi_sp);  den_sp = scalar(dag(phi_sp) * phi_sp)
    num_d  = scalar(dag(phi_d)  * Hphi_d);   den_d  = scalar(dag(phi_d)  * phi_d)
    println("  sparse: ratio = ", real(num_sp/den_sp))
    println("  dense:  ratio = ", real(num_d /den_d))
    println("  |diff| = ", abs(num_sp/den_sp - num_d/den_d))

    println("\n--- Right-isometry check (on densified tensors) ---")
    for (label, psi_x) in (("sparse_o", psi_sp_o), ("dense_o", psi_d_o))
        for i in 2:length(psi_x)
            left_bond = commoninds(psi_x[i-1], psi_x[i])
            T_d = SparseBackends.to_dense_itensors_unfused(psi_x[i])
            Td  = prime(dag(T_d), left_bond...)
            E   = T_d * Td
            Cl = combiner(left_bond...; tags="bL")
            Cr = combiner(prime.(left_bond)...; tags="bR")
            cl = combinedind(Cl);  cr = combinedind(Cr)
            Em = Array(E * Cl * Cr, cl, cr)
            D  = ITensors.dim(cl)
            err = norm(Em - Matrix{ComplexF64}(I, D, D))
            println("  $label  psi[$i] right-iso err = $err  (D=$D)")
        end
    end

    # =========================================================================
    println("\n" * "="^70)
    println("SECTION 2: Site-by-site H comparison (sparse vs dense)")
    println("="^70)

    println("\n--- |H_sparse[j] - H_dense[j]| per site ---")
    for j in 1:length(H_sparse)
        hs = H_sparse[j]
        hd = H_dense[j]
        hs_d = ITensors.has_external_storage(hs) ? SparseBackends.to_dense_itensors(hs) : hs
        if issetequal(inds(hs_d), inds(hd))
            println("  Site $j: diff = ", norm(hs_d - hd))
        else
            println("  Site $j: index mismatch — H_sparse inds: ", inds(hs_d),
                    "  H_dense inds: ", inds(hd))
        end
    end

    println("\n--- Energy via inner products ---")
    println("  inner(psi0, H_sparse, psi0) = ", inner(psi0', H_sparse, psi0))
    println("  inner(psi0, H_dense,  psi0) = ", inner(psi0', H_dense,  psi0))
    println("  inner(psi0, H,        psi0) = ", inner(psi0', H,        psi0))

    psi_contracted = replaceprime(contract(s.P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    println("  inner(P*psi0, H,        P*psi0) = ", inner(psi_contracted', H,        psi_contracted))
    println("  inner(P*psi0, H_sparse, P*psi0) = ", inner(psi_contracted', H_sparse, psi_contracted))

    # =========================================================================
    println("\n" * "="^70)
    println("SECTION 3: DMRG convergence — sparse vs dense")
    println("="^70)
    println("  nsweeps=$_NS_PC  maxdim=$_MAXDIM_PC")

    psi1 = copy(psi_sp)
    psi2 = copy(psi_d)

    println("\n--- Dense DMRG (H_dense, psi_dense) ---")
    E_d, psi_d_out, sw_d, terr_d = dmrg(H_dense, psi2;
        nsweeps=_NS_PC, maxdim=[_MAXDIM_PC], mindim=[1], cutoff=1e-12,
        target_energy=nothing, use_early_exit=false, outputlevel=1)
    E_d_final = inner(psi_d_out', H, psi_d_out)
    println("[Dense]  E_dmrg=$E_d  sweeps=$sw_d  terr=$terr_d")
    println("[Dense]  E_final (vs H) = $E_d_final")

    println("\n--- Sparse DMRG (H_sparse, psi_sp) ---")
    E_s, psi_s_out, sw_s, terr_s = dmrg(H_sparse, psi1;
        nsweeps=_NS_PC, maxdim=[_MAXDIM_PC], mindim=[1], cutoff=1e-12,
        target_energy=nothing, use_early_exit=false, outputlevel=1)
    E_s_final = inner(psi_s_out', H, psi_s_out)
    println("[Sparse] E_dmrg=$E_s  sweeps=$sw_s  terr=$terr_s")
    println("[Sparse] E_final (vs H) = $E_s_final")

    delta = abs(E_d_final - E_s_final)
    println("\n|E_dense - E_sparse| = $delta")
    if delta < 1e-4
        println("✓ Sparse and dense DMRG agree within tolerance")
    else
        println("✗ Energy mismatch: $delta (expected < 1e-4)")
    end
end
nothing
