# Directly compare dense Linv vs BS Linv (full-storage, no restriction) to
# confirm where they differ — and quantify the discrepancy per bond.
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra, Printf
include("utils.jl")

function build_setup(N::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
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
    psi0 = random_mps(sites)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return psi_sp
end

function compare_linv_at_bond(N_plaq::Int, b::Int)
    println("\n=== N=$N_plaq plaquettes, bond=$b ===")
    psi = build_setup(N_plaq)
    psi = ITensorMPS.orthogonalize(psi, 1)
    cache = SparseBackends.init_gram_cache(psi)
    Lgram = SparseBackends.get_left_gram(cache, b)
    Rgram = SparseBackends.get_right_gram(cache, b)
    phi = psi[b] * psi[b+1]

    # Build both versions of per-side Linv
    Mhalf_L_d, Linv_L_d = SparseBackends.build_half_pair_single(Lgram)
    Mhalf_R_d, Linv_R_d = SparseBackends.build_half_pair_single(Rgram)
    Mhalf_L_bs, Linv_L_bs = SparseBackends.build_minv_half_pair_bs(Lgram; phi_template=phi)
    Mhalf_R_bs, Linv_R_bs = SparseBackends.build_minv_half_pair_bs(Rgram; phi_template=phi)

    # Densify both and compare numerically
    function dense_compare(name, T_dense, T_bs)
        T_dense_d = T_dense
        T_bs_d    = ITensors.has_external_storage(T_bs) ?
                    SparseBackends.to_dense_itensors_unfused(T_bs) : T_bs
        common_inds = collect(inds(T_dense_d))
        if Set(common_inds) != Set(inds(T_bs_d))
            println("  $name: inds mismatch! dense=$(inds(T_dense_d))  bs=$(inds(T_bs_d))")
            return
        end
        a = Array(T_dense_d, common_inds...)
        b = Array(permute(T_bs_d, common_inds...; allow_alias=true), common_inds...)
        diff = norm(a - b)
        rel  = diff / max(norm(a), eps())
        @printf("  %-12s  ||dense - bs|| = %.3e  (rel %.3e)\n", name, diff, rel)
    end

    dense_compare("Linv_L", Linv_L_d, Linv_L_bs)
    dense_compare("Linv_R", Linv_R_d, Linv_R_bs)
    dense_compare("Mhalf_L", Mhalf_L_d, Mhalf_L_bs)
    dense_compare("Mhalf_R", Mhalf_R_d, Mhalf_R_bs)

    # Now apply each to phi and compare outputs.
    println("  --- applied to phi (Linv_L · phi) ---")
    out_dense_L = SparseBackends.apply_minv_preserve_bs(Linv_L_d, phi, phi)
    out_bs_native_L = ITensors.replaceprime(Linv_L_bs * phi, 1 => 0; tags="Link")
    out_bs_via_apply_L = SparseBackends.apply_minv_preserve_bs(Linv_L_bs, phi, phi)
    function compare_outputs(name, a_T, b_T)
        a_d = ITensors.has_external_storage(a_T) ?
              SparseBackends.to_dense_itensors_unfused(a_T) : a_T
        b_d = ITensors.has_external_storage(b_T) ?
              SparseBackends.to_dense_itensors_unfused(b_T) : b_T
        common_inds = collect(inds(a_d))
        if Set(common_inds) != Set(inds(b_d))
            println("  $name: inds mismatch! a=$(inds(a_d))  b=$(inds(b_d))")
            return
        end
        aa = Array(a_d, common_inds...)
        bb = Array(permute(b_d, common_inds...; allow_alias=true), common_inds...)
        diff = norm(aa - bb)
        rel  = diff / max(norm(aa), eps())
        @printf("  %-30s  ||diff|| = %.3e  (rel %.3e)\n", name, diff, rel)
    end
    compare_outputs("dense_apply vs bs_native_*", out_dense_L, out_bs_native_L)
    compare_outputs("dense_apply vs bs_via_apply", out_dense_L, out_bs_via_apply_L)

    # Now the per-side RESTRICTED BS Linv (image-phi-aligned).
    println("  --- per-side restricted BS Linv ---")
    Mhalf_L_r, Linv_L_r = SparseBackends.build_minv_half_pair_bs_side(Lgram, phi, :left)
    Mhalf_R_r, Linv_R_r = SparseBackends.build_minv_half_pair_bs_side(Rgram, phi, :right)
    out_bs_r_native_L = ITensors.replaceprime(Linv_L_r * phi, 1 => 0; tags="Link")
    compare_outputs("dense_apply vs bs_restricted_native", out_dense_L, out_bs_r_native_L)
end

compare_linv_at_bond(2, 1)
compare_linv_at_bond(2, 3)
compare_linv_at_bond(3, 1)
compare_linv_at_bond(3, 4)
nothing
