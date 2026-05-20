# Most fundamental sanity check: <psi|H|psi> via sparse vs dense.
# No DMRG, no orthogonalize, just direct contractions.
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
include("utils.jl")

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
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
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

length(ARGS) < 1 && error("Usage: julia test_innerprod.jl <N_plaq>")
const _N_IP = parse(Int, ARGS[1])

let
    H_sp, H_d, psi_sp, psi_d = build_setup(_N_IP)

    println("=== Tensor types ===")
    println("psi_sp: ", [ITensors.has_external_storage(T) ? "BS" : "D" for T in psi_sp])
    println("psi_d:  ", [ITensors.has_external_storage(T) ? "BS" : "D" for T in psi_d])
    println("H_sp:   ", [ITensors.has_external_storage(T) ? "BS" : "D" for T in H_sp])
    println("H_d:    ", [ITensors.has_external_storage(T) ? "BS" : "D" for T in H_d])

    # <psi|psi>
    norm_sp = inner(psi_sp, psi_sp)
    norm_d  = inner(psi_d,  psi_d)
    println("\n=== <psi|psi> ===")
    println("  sparse = ", norm_sp)
    println("  dense  = ", norm_d)
    println("  |diff| = ", abs(norm_sp - norm_d))

    # <psi|H|psi>
    eh_sp = inner(psi_sp', H_sp, psi_sp)
    eh_d  = inner(psi_d',  H_d,  psi_d)
    println("\n=== <psi|H|psi> ===")
    println("  sparse = ", eh_sp)
    println("  dense  = ", eh_d)
    println("  |diff| = ", abs(eh_sp - eh_d))

    println("\nE_sparse / <psi|psi>_sparse = ", real(eh_sp/norm_sp))
    println("E_dense  / <psi|psi>_dense  = ", real(eh_d /norm_d))

    # ---- After orthogonalize ----
    psi_sp_o = ITensorMPS.orthogonalize(psi_sp, 1)
    psi_d_o  = ITensorMPS.orthogonalize(psi_d,  1)
    println("\n=== After orthogonalize(psi, 1) ===")
    println("<psi|psi> sparse = ", inner(psi_sp_o, psi_sp_o))
    println("<psi|psi> dense  = ", inner(psi_d_o,  psi_d_o))
    println("<psi|H|psi> sparse = ", inner(psi_sp_o', H_sp, psi_sp_o))
    println("<psi|H|psi> dense  = ", inner(psi_d_o',  H_d,  psi_d_o))
    # Energy (Rayleigh quotient)
    println("E sparse = ", real(inner(psi_sp_o', H_sp, psi_sp_o) / inner(psi_sp_o, psi_sp_o)))
    println("E dense  = ", real(inner(psi_d_o',  H_d,  psi_d_o)  / inner(psi_d_o,  psi_d_o)))

    # ---- ProjMPO check: <phi|PH|phi>/<phi|phi> should equal <psi|H|psi>/<psi|psi> ----
    PH_sp = position!(ProjMPO(H_sp), psi_sp_o, 1)
    PH_d  = position!(ProjMPO(H_d),  psi_d_o,  1)
    phi_sp = psi_sp_o[1] * psi_sp_o[2]
    phi_d  = psi_d_o[1]  * psi_d_o[2]

    Hphi_sp = product(PH_sp, phi_sp)
    Hphi_d  = product(PH_d,  phi_d)

    num_sp = scalar(dag(phi_sp) * Hphi_sp)
    den_sp = scalar(dag(phi_sp) * phi_sp)
    num_d  = scalar(dag(phi_d)  * Hphi_d)
    den_d  = scalar(dag(phi_d)  * phi_d)

    println("\n=== <phi|PH|phi>/<phi|phi> (should equal 0.807...) ===")
    println("  sparse: num=", num_sp, "  den=", den_sp, "  ratio=", real(num_sp/den_sp))
    println("  dense:  num=", num_d,  "  den=", den_d,  "  ratio=", real(num_d/den_d))

    # ---- Right-isometry check on DENSIFIED tensors ----
    # Densify every site, then check the right-iso identity on each i > 1
    println("\n=== Right-isometry check ‖T·dag(T) — δ_{left-bond}‖ (on densified) ===")
    for psi_label in (("sparse_o", psi_sp_o), ("dense_o", psi_d_o))
        label, psi_x = psi_label
        for i in 2:length(psi_x)
            left_bond = commoninds(psi_x[i-1], psi_x[i])
            T_d = SparseBackends.to_dense_itensors_unfused(psi_x[i])
            Td  = dag(T_d)
            Td  = prime(Td, left_bond...)
            E   = T_d * Td
            Cl = combiner(left_bond...; tags="bL")
            Cr = combiner(prime.(left_bond)...; tags="bR")
            cl = combinedind(Cl); cr = combinedind(Cr)
            Em = Array(E * Cl * Cr, cl, cr)
            D  = ITensors.dim(cl)
            err = norm(Em - Matrix{ComplexF64}(I, D, D))
            println("  $label  psi[$i] right-iso err = $err  (bond dim D=$D)")
        end
    end
end
nothing
