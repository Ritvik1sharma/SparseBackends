# Investigation I2 + I3 (from the plan): build the Gram matrix M explicitly
# for the non-canonical sparse psi, verify that <phi|H_eff|phi>/<phi|M|phi>
# matches dense <phi|H_eff|phi>/<phi|phi>, then run a manual generalized
# eigsolve at bond 1 and compare its eigenvalue to dense eigsolve.
#
# If this passes, Path B (generalized-eigsolve DMRG) is sound.
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
using KrylovKit: eigsolve, geneigsolve

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

# Build the right "Gram" environment from sites b+1..N of psi.
# Convention: at every site, dag(psi[i]) has ALL its Link indices primed (both
# the left bond and the right bond), so:
#   - the right bond of dag(psi[i]) (primed) contracts with the primed copy in
#     Mright[i+1],
#   - the left bond of dag(psi[i]) (primed) remains free as the primed copy in
#     Mright[i].
# After the recursion finishes at site b+1, Mright has two indices: the
# (left_bond_of_psi[b+1]) and its primed copy. For canonical psi this equals δ
# on that bond; for non-canonical psi it's the Gram matrix.
function gram_right_env(psi::MPS, b::Int)
    # For 2-site DMRG with phi = psi[b]*psi[b+1], the right environment
    # contracts sites b+2..N. Its free legs are the bond between psi[b+1]
    # and psi[b+2] (primed + unprimed copies).
    N = length(psi)
    Mright = ITensor(1.0)
    for i in N:-1:(b+2)
        T  = psi[i]
        # All Link indices on T → prime them on dag (both left and right bonds).
        link_inds = filter(I -> ITensors.hastags(I, "Link"), collect(inds(T)))
        Td = dag(T)
        for I in link_inds
            Td = prime(Td, I)
        end
        Mright = Mright * T * Td
    end
    return Mright
end

length(ARGS) < 1 && error("Usage: julia test_gram_matrix.jl <N_plaq>")
const _N_GRAM = parse(Int, ARGS[1])

let
    H_sp, H_d, psi_sp, psi_d = build_setup(_N_GRAM)
    N = length(psi_sp)

    # Orthogonalize both to position 1.
    psi_sp_o = ITensorMPS.orthogonalize(psi_sp, 1)
    psi_d_o  = ITensorMPS.orthogonalize(psi_d,  1)

    println("=== Step 0: norms / energies (post-orthogonalize) ===")
    println("  <psi_sp | psi_sp> = ", inner(psi_sp_o, psi_sp_o))
    println("  <psi_d  | psi_d > = ", inner(psi_d_o,  psi_d_o))
    println("  <psi_sp | H | psi_sp> = ", inner(psi_sp_o', H_sp, psi_sp_o))
    println("  <psi_d  | H | psi_d > = ", inner(psi_d_o',  H_d,  psi_d_o))

    # ---- Build M (right gram env) and H_right env for both psi. ----
    b = 1
    M_sp_right = gram_right_env(psi_sp_o, b)
    M_d_right  = gram_right_env(psi_d_o,  b)

    println("\n=== Step 1: Gram right-env at bond $b ===")
    println("  inds(M_sp_right): ", inds(M_sp_right))
    println("  inds(M_d_right):  ", inds(M_d_right))
    # M_d_right should be ≈ identity; M_sp_right may not be.
    println("  M_sp_right is identity? checking norm of (M_sp_right - I)...")
    # Convert to matrix form for visual check
    M_sp_d = SparseBackends.to_dense_itensors_unfused(M_sp_right)
    println("  ||M_sp_right||_F = ", norm(M_sp_d))

    # ---- Build phi and compute the four ratios. ----
    phi_sp = psi_sp_o[1] * psi_sp_o[2]
    phi_d  = psi_d_o[1]  * psi_d_o[2]

    # <phi|phi> (no metric)
    pp_sp = scalar(dag(phi_sp) * phi_sp)
    pp_d  = scalar(dag(phi_d)  * phi_d)

    # <phi|M|phi> = phi acting through right gram env, contracted with dag(phi)
    # For phi at bond 1, the M_right has unprimed left-bond indices (between
    # psi[1] and psi[2]). To form <phi|M|phi>, we need to:
    #   - take dag(phi) primed on the left-bond between psi[1] and psi[2],
    #   - contract through M_sp_right.
    # The free bond of phi on its right side = bond between psi[b+1] and psi[b+2],
    # which is what M_right attaches to. NOT the bond between psi[b] and psi[b+1]
    # — that bond is contracted out inside phi = psi[b]*psi[b+1].
    # Bond between phi's right side (psi[b+1]) and the right environment (psi[b+2]).
    bond_inds = commoninds(psi_sp_o[b+1], psi_sp_o[b+2])
    phi_sp_d = dag(phi_sp)
    for I in bond_inds
        phi_sp_d = prime(phi_sp_d, I)
    end
    pm_sp = scalar(phi_sp_d * M_sp_right * phi_sp)

    bond_inds_d = commoninds(psi_d_o[b+1], psi_d_o[b+2])
    phi_d_dag = dag(phi_d)
    for I in bond_inds_d
        phi_d_dag = prime(phi_d_dag, I)
    end
    pm_d = scalar(phi_d_dag * M_d_right * phi_d)

    println("\n=== Step 2: <phi|phi> vs <phi|M|phi> ===")
    println("  sparse: <phi|phi>=$pp_sp, <phi|M|phi>=$pm_sp")
    println("  dense : <phi|phi>=$pp_d,  <phi|M|phi>=$pm_d")
    println("  Sparse M corrects? expected pm_sp ≈ pp_d (both = <psi|psi>) and pp_sp may differ:")
    println("    pm_sp = $pm_sp")
    println("    pp_d  = $pp_d")
    println("    |pm_sp - pp_d| = ", abs(pm_sp - pp_d))

    # <phi|H_eff|phi> (with PH = ProjMPO action) — easiest is to use existing ProjMPO
    PH_sp = position!(ProjMPO(H_sp), psi_sp_o, 1)
    PH_d  = position!(ProjMPO(H_d),  psi_d_o,  1)

    Hphi_sp = product(PH_sp, phi_sp)
    Hphi_d  = product(PH_d,  phi_d)
    ph_sp = scalar(dag(phi_sp) * Hphi_sp)
    ph_d  = scalar(dag(phi_d)  * Hphi_d)

    println("\n=== Step 3: <phi|H_eff|phi> values ===")
    println("  sparse: <phi|H_eff|phi> = ", ph_sp)
    println("  dense : <phi|H_eff|phi> = ", ph_d)

    println("\n=== Step 4: Energy estimates ===")
    println("  sparse <phi|H_eff|phi>/<phi|phi> (broken, current bug) = ", real(ph_sp / pp_sp))
    println("  sparse <phi|H_eff|phi>/<phi|M|phi> (Path B candidate)  = ", real(ph_sp / pm_sp))
    println("  dense  <phi|H_eff|phi>/<phi|phi>                       = ", real(ph_d  / pp_d))
    println("  ⇒ If Path B candidate matches dense, generalized eigsolve will work.")

    # ===== Step 5: Manual generalized eigsolve at bond 1 (I3) =====
    println("\n=== Step 5: generalized eigsolve at bond b=1 ===")

    # Dense reference: ordinary eigsolve of PH on phi_d.
    vals_d, _ = eigsolve(PH_d, phi_d, 1, :SR; ishermitian=true)
    E_dense = real(vals_d[1])
    println("  dense eigsolve (E)            = ", E_dense)

    # Sparse generalized eigsolve via dense matrices (Investigation: math only).
    # Build a basis of phi-shaped vectors by iterating over phi_sp's index set,
    # build H_eff and M as small dense matrices in that basis, solve generalized
    # eigenproblem with LinearAlgebra.
    phi_inds = collect(inds(phi_sp))
    dims = [ITensors.dim(I) for I in phi_inds]
    total = prod(dims)
    println("  phi_sp dim total = $total, dims = $dims")

    # Probe H_eff and M action: apply to dense ITensor basis vectors built from phi_sp inds.
    # Use a dense ITensor with phi_sp's inds for the probe (so PH_sp(action) is well-typed).
    H_mat = zeros(ComplexF64, total, total)
    M_mat = zeros(ComplexF64, total, total)
    # Build a "template" dense ITensor with same inds as phi_sp.
    template_d = ITensors.itensor(zeros(ComplexF64, dims...), phi_inds...)
    for j in 1:total
        ej = zeros(ComplexF64, dims...)
        ej[j] = 1.0
        x = ITensors.itensor(ej, phi_inds...)
        Hx = product(PH_sp, x)
        Mx = M_sp_right * x
        Mx = replaceprime(Mx, 1 => 0; tags="Link")
        # Densify and read as vectors.
        Hx_d = SparseBackends.to_dense_itensors_unfused(Hx)
        Mx_d = SparseBackends.to_dense_itensors_unfused(Mx)
        # Permute to phi_inds order, then linearize.
        Hx_p = permute(Hx_d, phi_inds...; allow_alias=true)
        Mx_p = permute(Mx_d, phi_inds...; allow_alias=true)
        H_mat[:, j] = vec(Array(Hx_p, phi_inds...))
        M_mat[:, j] = vec(Array(Mx_p, phi_inds...))
    end

    # Hermitize (round-off).
    H_mat = (H_mat + H_mat') / 2
    M_mat = (M_mat + M_mat') / 2

    # Solve generalized eigenproblem H x = E M x. M may be singular (sparse kernel of M
    # corresponds to forbidden block keys); take smallest eigenvalue restricted to M's range.
    F = eigen(H_mat, M_mat)
    finite = filter(isfinite, real.(F.values))
    sort!(finite)
    E_gen = finite[1]
    println("  sparse generalized eig (E)    = ", E_gen)
    println("  |E_gen - E_dense|             = ", abs(E_gen - E_dense))
    println("  M_mat rank = ", rank(M_mat), " / $total")
    println("  H_mat rank = ", rank(H_mat), " / $total")
    M_evs = sort(real.(eigvals(M_mat)))
    println("  M eigenvalues (sorted): ", round.(M_evs, sigdigits=4))
    println("  cond(M) = ", round(M_evs[end] / max(M_evs[1], eps(Float64)); sigdigits=4))

    # ===== Step 6: eigsolve(M⁻¹ H_eff) via small dense Minv on bond legs =====
    println("\n=== Step 6: eigsolve(M⁻¹ H_eff) ===")

    # Build small dense Minv on the bond legs. M_sp_right has inds
    # (bond_unp, bond_prim, mult_unp, mult_prim). Densify into a (bond×mult) ×
    # (bond×mult) matrix, invert, then wrap as an ITensor with same Index layout.
    bond_inds_right  = filter(I -> ITensors.plev(I) == 0, collect(inds(M_sp_right)))
    bond_inds_primed = filter(I -> ITensors.plev(I) == 1, collect(inds(M_sp_right)))
    @assert length(bond_inds_right) == length(bond_inds_primed)
    bond_dim_full = prod(ITensors.dim, bond_inds_right)
    M_dense_it = SparseBackends.to_dense_itensors_unfused(M_sp_right)
    M_mat_small_arr = Array(M_dense_it, bond_inds_right..., bond_inds_primed...)
    M_mat_small = reshape(M_mat_small_arr, bond_dim_full, bond_dim_full)
    M_mat_small = (M_mat_small + M_mat_small') / 2
    Minv_small  = inv(M_mat_small)
    dims_all = [ITensors.dim(I) for I in bond_inds_right]
    append!(dims_all, [ITensors.dim(I) for I in bond_inds_primed])
    Minv_arr = reshape(Minv_small, dims_all...)
    Minv_itensor = ITensors.itensor(Minv_arr, bond_inds_right..., bond_inds_primed...)

    function A_op(x)
        Hx = product(PH_sp, x)                                 # sparse BS
        # Apply Minv on bond legs while keeping BS storage matching x's type.
        MinvHx = SparseBackends.contract_preserve_bs(Minv_itensor, Hx; template=x)
        MinvHx = replaceprime(MinvHx, 1 => 0; tags="Link")
        # Recast to exactly match x's storage type+keys.
        if ITensors.has_external_storage(MinvHx) && ITensors.has_external_storage(x)
            Tw = ITensors.get_external_storage(x)
            Cw = ITensors.get_external_storage(MinvHx)
            if Cw isa SparseBackends.WrappedBlockSparse && Tw isa SparseBackends.WrappedBlockSparse
                MinvHx = ITensors._itensor_from_external_storage(SparseBackends.recast_bs_to_template(Cw, Tw))
            end
        end
        return MinvHx
    end

    Aphi = A_op(phi_sp)
    println("  inds(A_op(phi_sp)): ", inds(Aphi))

    vals_k, vecs_k, info_k = eigsolve(A_op, phi_sp, 1, :SR; ishermitian=false)
    E_k = real(vals_k[1])
    println("  eigsolve(M⁻¹ H_eff) E         = ", E_k)
    println("  |E_k - E_dense|               = ", abs(E_k - E_dense))
    println("  info: ", info_k)

    # --- Section: contraction-kernel cross-check (absorbed from test_eigsolve_kernel.jl) ---
    # Compute ⟨psi|H|psi⟩ in four storage combinations to isolate any BS×BS or BS×dense bug.
    println("\n" * "="^60)
    println("Section: ⟨psi|H|psi⟩ cross-check (sparse/dense H × sparse/dense ψ)")
    println("="^60)
    densify_T(T::ITensor) = ITensors.has_external_storage(T) ? SparseBackends.to_dense_itensors_unfused(T) : T
    densify_M(M) = typeof(M)(length(M); [densify_T(M[i]) for i in 1:length(M)]...)
    psi_sp_ref = orthogonalize(psi_sp, 1)
    H_sp_ref   = H_sparse
    H_dn_ref   = densify_M(H_sparse)
    psi_dn_ref = densify_M(psi_sp_ref)
    E_sp_sp = real(inner(psi_sp_ref', H_sp_ref, psi_sp_ref))
    E_sp_dn = real(inner(psi_sp_ref', H_dn_ref, psi_sp_ref))
    E_dn_sp = real(inner(psi_dn_ref', H_sp_ref, psi_dn_ref))
    E_dn_dn = real(inner(psi_dn_ref', H_dn_ref, psi_dn_ref))
    println("  sparse H × sparse ψ : E = $E_sp_sp")
    println("  dense  H × sparse ψ : E = $E_sp_dn   diff = $(abs(E_sp_sp - E_sp_dn))")
    println("  sparse H × dense  ψ : E = $E_dn_sp   diff = $(abs(E_sp_sp - E_dn_sp))")
    println("  dense  H × dense  ψ : E = $E_dn_dn   diff = $(abs(E_sp_sp - E_dn_dn))")
    println("All four should agree within rounding. Any large discrepancy localises a BS contraction bug.")
end
nothing
