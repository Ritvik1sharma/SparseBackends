# Microbench: at one bond of N=2 system, time each piece of the Path-B
# operator (both Arnoldi/M⁻¹ and Lanczos/M^(±1/2) variants) and the eigsolve
# call. Goal: isolate where the symmetric-transform path is spending time.
using SparseBackends, ITensors, ITensorMPS
using KrylovKit: eigsolve
using Random, LinearAlgebra, Printf
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
    return H_sparse, psi_sp
end

function bench_bond(N::Int, b::Int)
    println("\n\n=========== N=$N, bond=$b ===========")
    H_sp, psi_sp = build_setup(N)
    psi_sp = ITensorMPS.orthogonalize(psi_sp, 1)
    PH = ProjMPO(H_sp)
    position!(PH, psi_sp, b)

    println("--- build gram envs + Minv variants ---")
    t1 = @elapsed Lgram = SparseBackends.gram_left_env(psi_sp,  b)
    t2 = @elapsed Rgram = SparseBackends.gram_right_env(psi_sp, b)
    t3 = @elapsed M_full = Lgram * Rgram
    t4 = @elapsed Minv = SparseBackends.build_minv_itensor(M_full)
    t5 = @elapsed (Mhalf, Linv) = SparseBackends.build_minv_half_pair_itensor(M_full)
    @printf("  Lgram:   %.4fs    Rgram: %.4fs   M_full(L*R): %.4fs\n", t1, t2, t3)
    @printf("  Minv (pinv):              %.4fs\n", t4)
    @printf("  Mhalf/Linv (sym sqrt):    %.4fs\n", t5)

    phi = psi_sp[b] * psi_sp[b+1]
    println("--- one operator application ---")
    A_op_arnoldi = function (x)
        Hx = product(PH, x)
        return SparseBackends.apply_minv_preserve_bs(Minv, Hx, x)
    end
    B_op_lanczos = function (y)
        phi_x = SparseBackends.apply_minv_preserve_bs(Linv, y, y)
        Hphi  = product(PH, phi_x)
        return SparseBackends.apply_minv_preserve_bs(Linv, Hphi, y)
    end
    tA = @elapsed _ = A_op_arnoldi(phi)
    phi_y = SparseBackends.apply_minv_preserve_bs(Mhalf, phi, phi)
    tB = @elapsed _ = B_op_lanczos(phi_y)
    @printf("  A_op (M⁻¹ H):              %.4fs\n", tA)
    @printf("  B_op (Linv H Linv):        %.4fs\n", tB)

    # Dense reference: explicit eigen(H_mat, M_mat) over phi's index basis.
    # ------------------------------------------------------------------
    # Linv correctness checks:
    # (1) Mhalf · Linv should equal I_image (identity on image(M))
    # (2) build B_mat = Linv · H · Linv as a small dense matrix and compare
    #     its smallest eigenvalue to the dense generalized reference.
    # ------------------------------------------------------------------
    println("--- Linv correctness check ---")
    bond_unp = filter(I -> ITensors.plev(I) == 0, collect(inds(M_full)))
    bond_prm = filter(I -> ITensors.plev(I) == 1, collect(inds(M_full)))
    dbond = prod(ITensors.dim, bond_unp; init=1)
    M_full_d  = SparseBackends.to_dense_itensors_unfused(M_full)
    Mhalf_d   = SparseBackends.to_dense_itensors_unfused(Mhalf)
    Linv_d    = SparseBackends.to_dense_itensors_unfused(Linv)
    M_arr     = Array(M_full_d, bond_unp..., bond_prm...)
    Mhalf_arr = Array(Mhalf_d,  bond_unp..., bond_prm...)
    Linv_arr  = Array(Linv_d,   bond_unp..., bond_prm...)
    M_mat_bond     = reshape(M_arr,     dbond, dbond)
    Mhalf_mat_bond = reshape(Mhalf_arr, dbond, dbond)
    Linv_mat_bond  = reshape(Linv_arr,  dbond, dbond)
    err1 = norm(Mhalf_mat_bond * Linv_mat_bond - LinearAlgebra.I * (1.0+0.0im)) / sqrt(dbond)
    err2 = norm(Linv_mat_bond * Mhalf_mat_bond - LinearAlgebra.I * (1.0+0.0im)) / sqrt(dbond)
    err3 = norm(Mhalf_mat_bond^2 - M_mat_bond) / norm(M_mat_bond)
    @printf("  ||Mhalf·Linv - I||/√n        = %.3e   (expect 0 on image(M))\n", err1)
    @printf("  ||Linv·Mhalf - I||/√n        = %.3e   (expect 0 on image(M))\n", err2)
    @printf("  ||Mhalf² - M||/||M||         = %.3e   (expect ~eps)\n", err3)
    # Eigenvalue spectrum of M
    Mevs = sort(real.(LinearAlgebra.eigvals(M_mat_bond)))
    @printf("  M eigenvalues: min=%.3e  max=%.3e  ratio=%.3e\n", Mevs[1], Mevs[end], Mevs[end]/max(Mevs[1], eps()))

    println("--- dense reference (eigen(H, M)) ---")
    phi_inds = collect(inds(phi))
    dims = [ITensors.dim(I) for I in phi_inds]
    total = prod(dims)
    H_mat = zeros(ComplexF64, total, total)
    M_mat = zeros(ComplexF64, total, total)
    M_op_dense = function(x)
        y = Lgram * x * Rgram
        y = replaceprime(y, 1 => 0; tags="Link")
        return y
    end
    for j in 1:total
        ej = zeros(ComplexF64, dims...); ej[j] = 1.0
        x  = ITensors.itensor(ej, phi_inds...)
        Hx = product(PH, x)
        Mx = M_op_dense(x)
        Hx_d = SparseBackends.to_dense_itensors_unfused(Hx)
        Mx_d = SparseBackends.to_dense_itensors_unfused(Mx)
        Hx_p = permute(Hx_d, phi_inds...; allow_alias=true)
        Mx_p = permute(Mx_d, phi_inds...; allow_alias=true)
        H_mat[:, j] = vec(Array(Hx_p, phi_inds...))
        M_mat[:, j] = vec(Array(Mx_p, phi_inds...))
    end
    H_mat = (H_mat + H_mat')/2
    M_mat = (M_mat + M_mat')/2
    F = eigen(H_mat, M_mat)
    vals_finite = sort(filter(isfinite, real.(F.values)))
    @printf("  eigen(H,M) ground state:   E=%.8f   (next: %.6f, %.6f)\n", vals_finite[1], vals_finite[2], vals_finite[3])
    @printf("  M rank=%d / %d,  H rank=%d\n", rank(M_mat), total, rank(H_mat))

    # Build B_mat = Linv · H · Linv as a dense matrix in phi's basis. Its
    # smallest eigenvalue should match eigen(H_mat, M_mat) restricted to image(M).
    # ------------------------------------------------------------------
    # BS-stored Linv pathway (channel-block-diagonal, no recast).
    # ------------------------------------------------------------------
    # Direct measurement: does M have support outside phi's allowed chan-pair set?
    if ITensors.has_external_storage(phi)
        phi_w_local = ITensors.get_external_storage(phi)
        if phi_w_local isa SparseBackends.WrappedBlockSparse
            bs_phi = phi_w_local.blocksparse
            phi_chan_pairs = Set{Tuple{Int,Int}}()
            for k in bs_phi.keys
                push!(phi_chan_pairs, (k[1], k[2]))
            end
            @printf("  phi has %d unique chan-pairs (out of %d possible)\n",
                    length(phi_chan_pairs), bs_phi.dims[1] * bs_phi.dims[2])

            # Recompute ch_u locally from phi_dense_set (same logic as before).
            phi_dense_set_local = Set(collect(SparseBackends.dense_inds(phi_w_local)))
            bu_local2 = filter(I -> ITensors.plev(I) == 0, collect(inds(M_full)))
            ch_u_local = filter(I -> !(I in phi_dense_set_local), bu_local2)
            mu_u_local = filter(I -> (I in phi_dense_set_local),  bu_local2)
            if length(ch_u_local) == 2 && !isempty(ch_u_local)
                cdim_l = ITensors.dim(ch_u_local[1])
                cdim_r = ITensors.dim(ch_u_local[2])
                c_dim_local = cdim_l * cdim_r
                m_dim_local = isempty(mu_u_local) ? 1 : prod(ITensors.dim, mu_u_local)
                bp_local2 = filter(I -> ITensors.plev(I) == 1, collect(inds(M_full)))
                ch_p_local = [first(filter(J -> ITensors.id(J) == ITensors.id(I), bp_local2)) for I in ch_u_local]
                mu_p_local = [first(filter(J -> ITensors.id(J) == ITensors.id(I), bp_local2)) for I in mu_u_local]
                M_d_full2 = SparseBackends.to_dense_itensors_unfused(M_full)
                M_arr_local = Array(M_d_full2, ch_u_local..., ch_p_local..., mu_u_local..., mu_p_local...)
                M_4d_local = reshape(M_arr_local, c_dim_local, c_dim_local, m_dim_local, m_dim_local)
                n_inset  = 0.0
                n_offset = 0.0
                for ic_unp in 1:c_dim_local
                    clu = ((ic_unp - 1) % cdim_l) + 1
                    cru = ((ic_unp - 1) ÷ cdim_l) + 1
                    unp_in_phi = (clu, cru) in phi_chan_pairs
                    for ic_prm in 1:c_dim_local
                        clp = ((ic_prm - 1) % cdim_l) + 1
                        crp = ((ic_prm - 1) ÷ cdim_l) + 1
                        prm_in_phi = (clp, crp) in phi_chan_pairs
                        block_norm2 = sum(abs2, view(M_4d_local, ic_unp, ic_prm, :, :))
                        if unp_in_phi && prm_in_phi
                            n_inset += block_norm2
                        else
                            n_offset += block_norm2
                        end
                    end
                end
                @printf("  M norm² split:  in-phi-chan-set=%.4e  off-phi-chan-set=%.4e  off/total=%.3e\n",
                        n_inset, n_offset, n_offset / max(n_inset + n_offset, eps()))
            end
        end
    end

    # Phi storage key analysis: how many keys does phi actually have?
    if ITensors.has_external_storage(phi)
        phi_w = ITensors.get_external_storage(phi)
        if phi_w isa SparseBackends.WrappedBlockSparse
            phi_bs = phi_w.blocksparse
            n_keys = length(phi_bs.keys)
            # Total possible keys = product of dims of sparse axes
            sparse_dims = phi_bs.dims[1:length(phi_bs.dims)-(length(phi_bs.dims) - length(phi_bs.keys[1]))]
            total_keys = isempty(phi_bs.keys) ? 0 : prod(sparse_dims; init=1)
            @printf("  phi storage: %d stored keys of %d possible (%.1f%% populated)\n",
                    n_keys, total_keys, 100*n_keys/max(total_keys,1))
            @printf("  phi block keys (first 10): %s\n",
                    string([k for k in phi_bs.keys[1:min(end,10)]]))
        end
    end

    println("--- BS-Linv build + correctness vs dense-Linv ---")
    t_bs = @elapsed (Mhalf_bs, Linv_bs) = SparseBackends.build_minv_half_pair_bs(M_full; phi_template=phi)
    @printf("  build_minv_half_pair_bs:   %.4fs\n", t_bs)

    # Verify whether M is actually channel-block-diagonal.
    if ITensors.has_external_storage(phi)
        phi_dense_set = Set(collect(SparseBackends.dense_inds(ITensors.get_external_storage(phi))))
        bu_local = filter(I -> ITensors.plev(I) == 0, collect(inds(M_full)))
        bp_local = filter(I -> ITensors.plev(I) == 1, collect(inds(M_full)))
        ch_u = filter(I -> !(I in phi_dense_set), bu_local)
        mu_u = filter(I -> (I in phi_dense_set),  bu_local)
        ch_p = [first(filter(J -> ITensors.id(J) == ITensors.id(I), bp_local)) for I in ch_u]
        mu_p = [first(filter(J -> ITensors.id(J) == ITensors.id(I), bp_local)) for I in mu_u]
        if !isempty(ch_u)
            c_dim = prod(ITensors.dim, ch_u)
            m_dim = isempty(mu_u) ? 1 : prod(ITensors.dim, mu_u)
            M_d_local = SparseBackends.to_dense_itensors_unfused(M_full)
            M_arr = Array(M_d_local, ch_u..., ch_p..., mu_u..., mu_p...)
            M_4d  = reshape(M_arr, c_dim, c_dim, m_dim, m_dim)
            n_total = norm(M_4d)
            n_diag2 = sum(sum(abs2, view(M_4d, i, i, :, :)) for i in 1:c_dim)
            n_diag  = sqrt(n_diag2)
            n_off   = sqrt(max(n_total^2 - n_diag2, 0.0))
            @printf("  M norms: total=%.4f  diag-channel=%.4f  off-diag-channel=%.4f  ratio_off/total=%.3e\n",
                    n_total, n_diag, n_off, n_off / max(n_total, eps()))
        end
    end
    println("  inds(phi):     ", [(ITensors.id(I), ITensors.dim(I), string(ITensors.tags(I))) for I in inds(phi)])
    println("  inds(Linv_bs): ", [(ITensors.id(I), ITensors.dim(I), string(ITensors.tags(I))) for I in inds(Linv_bs)])
    println("  inds(M_full):  ", [(ITensors.id(I), ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I)) for I in inds(M_full)])
    # Apply both Linvs to a sample BS vector and compare numerically.
    Linv_dense_y  = SparseBackends.apply_minv_preserve_bs(Linv, phi, phi)
    Linv_bs_y_raw = Linv_bs * phi
    Linv_bs_y     = ITensors.replaceprime(Linv_bs_y_raw, 1 => 0; tags="Link")
    # Density-equivalent check
    a_d = SparseBackends.to_dense_itensors_unfused(Linv_dense_y)
    b_d = ITensors.has_external_storage(Linv_bs_y) ?
          SparseBackends.to_dense_itensors_unfused(Linv_bs_y) : Linv_bs_y
    a_arr = Array(a_d, inds(phi)...)
    b_arr = Array(permute(b_d, inds(phi)...; allow_alias=true), inds(phi)...)
    diff_arr = a_arr - b_arr
    @printf("  ||Linv_dense·y - Linv_bs·y|| / ||·|| = %.3e   (expect ~eps)\n",
            norm(diff_arr) / max(norm(a_arr), eps()))
    # Localize discrepancy: top-3 entries by abs(diff)
    flat_a = vec(a_arr); flat_b = vec(b_arr); flat_d = vec(diff_arr)
    sorted_idx = sortperm(abs.(flat_d); rev=true)
    @printf("  Top entries by abs(diff): index | a_dense | b_bs | diff\n")
    for k in 1:min(5, length(sorted_idx))
        i = sorted_idx[k]
        @printf("    [%d] %.6e %+.6ei | %.6e %+.6ei | %.6e %+.6ei\n",
                i, real(flat_a[i]), imag(flat_a[i]),
                real(flat_b[i]), imag(flat_b[i]),
                real(flat_d[i]), imag(flat_d[i]))
    end
    # Also dump Linv as 2D matrix: build a comparable Linv_dense_mat and Linv_bs_mat
    # in (chan_unp ⊗ mult_unp) × (chan_prm ⊗ mult_prm) layout.
    if !isempty(ch_u)
        # Linv_dense already computed in build_minv_itensor — densify it.
        Mhalf_d_full, Linv_d_full = SparseBackends.build_minv_half_pair_itensor(M_full)
        Ld_arr  = Array(SparseBackends.to_dense_itensors_unfused(Linv_d_full),
                       ch_u..., ch_p..., mu_u..., mu_p...)
        Lbs_arr = Array(SparseBackends.to_dense_itensors_unfused(Linv_bs),
                       ch_u..., ch_p..., mu_u..., mu_p...)
        Lbs_minus_Ld = Lbs_arr - Ld_arr
        @printf("  ||Linv_bs - Linv_dense_itensor|| / ||·|| = %.3e\n",
                norm(Lbs_minus_Ld) / max(norm(Ld_arr), eps()))
    end
    # Storage types
    if ITensors.has_external_storage(Linv_bs_y)
      println("  Linv_bs_y storage: ", typeof(ITensors.get_external_storage(Linv_bs_y)))
    end
    println("  phi    storage:    ", typeof(ITensors.get_external_storage(phi)))

    println("--- B_op spectrum check ---")
    B_mat = zeros(ComplexF64, total, total)
    for j in 1:total
        ej = zeros(ComplexF64, dims...); ej[j] = 1.0
        y  = ITensors.itensor(ej, phi_inds...)
        out = B_op_lanczos(y)
        out_d = ITensors.has_external_storage(out) ?
                SparseBackends.to_dense_itensors_unfused(out) : out
        out_p = permute(out_d, phi_inds...; allow_alias=true)
        B_mat[:, j] = vec(Array(out_p, phi_inds...))
    end
    B_mat = (B_mat + B_mat')/2
    Bevs = sort(real.(LinearAlgebra.eigvals(B_mat)))
    @printf("  B_mat smallest 3:  %.6f, %.6f, %.6f\n", Bevs[1], Bevs[2], Bevs[3])
    @printf("  B_mat largest 3:   %.6f, %.6f, %.6f\n", Bevs[end], Bevs[end-1], Bevs[end-2])

    # Time repeated B_op applies to detect any blow-up / accumulation in
    # iteration count or storage handling.
    println("--- repeated B_op applies (10x) ---")
    t10 = @elapsed for _ in 1:10
        _ = B_op_lanczos(phi_y)
    end
    @printf("  10× B_op total: %.4fs   per-call: %.4fs\n", t10, t10/10)
    t10A = @elapsed for _ in 1:10
        _ = A_op_arnoldi(phi)
    end
    @printf("  10× A_op total: %.4fs   per-call: %.4fs\n", t10A, t10A/10)

    println("--- KrylovKit eigsolve ---")
    tArn = @elapsed (valsA, vecsA, infoA) = eigsolve(
        A_op_arnoldi, phi, 1, :SR;
        ishermitian=false, tol=1e-12, krylovdim=30, maxiter=10, verbosity=1
    )
    tLan = @elapsed (valsB, vecsB, infoB) = eigsolve(
        B_op_lanczos, phi_y, 1, :SR;
        ishermitian=true, tol=1e-12, krylovdim=30, maxiter=10, verbosity=1
    )
    @printf("  Arnoldi eigsolve:  %.4fs   E=%s   numops=%d\n", tArn, string(valsA[1]), infoA.numops)
    @printf("  Lanczos eigsolve:  %.4fs   E=%s   numops=%d\n", tLan, string(valsB[1]), infoB.numops)
end

bench_bond(2, 1)
bench_bond(2, 3)
bench_bond(3, 1)
bench_bond(3, 4)
nothing
