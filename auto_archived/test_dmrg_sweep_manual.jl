# Investigation I3a: manual full DMRG forward+backward sweep on N=2 plaquettes,
# using non-canonical sparse psi + explicit Gram matrix M + generalized
# eigsolve (via small-matrix LinearAlgebra.eigen, as a placeholder for
# KrylovKit.geneigsolve which is blocked on a BS-output kernel pathway).
#
# Goal: confirm Path B (generalized eigsolve with M) gives the same energy
# as ordinary dense DMRG at EVERY bond of a full forward+backward sweep,
# not just bond 1.
#
# If this passes, Path B is sound for end-to-end DMRG.
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

# Right Gram environment at bond b: contract psi[b+2..N] · dag(psi[b+2..N])
# with all Link inds primed on dag. Free legs are (link_{b+1..b+2}, link_{b+1..b+2}').
function gram_right_env(psi::MPS, b::Int)
    N = length(psi)
    Mright = ITensor(1.0)
    for i in N:-1:(b+2)
        T  = psi[i]
        link_inds = filter(I -> ITensors.hastags(I, "Link"), collect(inds(T)))
        Td = dag(T)
        for I in link_inds; Td = prime(Td, I); end
        Mright = Mright * T * Td
    end
    return Mright
end

# Left Gram environment at bond b: contract psi[1..b-1] · dag(psi[1..b-1])
# with all Link inds primed on dag. Free legs are (link_{b-1..b}, link_{b-1..b}').
function gram_left_env(psi::MPS, b::Int)
    Mleft = ITensor(1.0)
    for i in 1:(b-1)
        T  = psi[i]
        link_inds = filter(I -> ITensors.hastags(I, "Link"), collect(inds(T)))
        Td = dag(T)
        for I in link_inds; Td = prime(Td, I); end
        Mleft = Mleft * T * Td
    end
    return Mleft
end

# Build the full Gram operator at bond b for 2-site phi = psi[b]*psi[b+1].
# Acts as: M_op(phi) = Lgram * phi * Rgram with appropriate priming so output
# inds match phi inds.
function gram_op(Lgram::ITensor, Rgram::ITensor, phi::ITensor)
    y = Lgram * phi * Rgram
    y = replaceprime(y, 1 => 0; tags="Link")
    return y
end

# Solve generalized eigenproblem H_eff phi = E M phi by densifying both
# operators in phi's index basis. Returns (E, phi_new) where phi_new is a
# dense ITensor with the SAME inds as phi (un-primed).
function solve_geneig_dense(PH, Lgram::ITensor, Rgram::ITensor, phi::ITensor)
    phi_inds = collect(inds(phi))
    dims = [ITensors.dim(I) for I in phi_inds]
    total = prod(dims)

    H_mat = zeros(ComplexF64, total, total)
    M_mat = zeros(ComplexF64, total, total)
    for j in 1:total
        ej = zeros(ComplexF64, dims...)
        ej[j] = 1.0
        x = ITensors.itensor(ej, phi_inds...)
        Hx = product(PH, x)
        Mx = gram_op(Lgram, Rgram, x)
        Hx_d = SparseBackends.to_dense_itensors_unfused(Hx)
        Mx_d = SparseBackends.to_dense_itensors_unfused(Mx)
        Hx_p = permute(Hx_d, phi_inds...; allow_alias=true)
        Mx_p = permute(Mx_d, phi_inds...; allow_alias=true)
        H_mat[:, j] = vec(Array(Hx_p, phi_inds...))
        M_mat[:, j] = vec(Array(Mx_p, phi_inds...))
    end
    H_mat = (H_mat + H_mat') / 2
    M_mat = (M_mat + M_mat') / 2

    F = eigen(H_mat, M_mat)
    vals = real.(F.values)
    perm = sortperm(vals)
    # take smallest finite eigenvalue
    E = NaN
    vec_out = zeros(ComplexF64, total)
    for k in perm
        if isfinite(vals[k])
            E = vals[k]
            vec_out = F.vectors[:, k]
            break
        end
    end
    # normalize w.r.t. M-inner product
    nrm2 = real(vec_out' * (M_mat * vec_out))
    vec_out ./= sqrt(abs(nrm2))
    phi_new_arr = reshape(vec_out, dims...)
    phi_new = ITensors.itensor(phi_new_arr, phi_inds...)
    return E, phi_new
end

let
    H_sp, H_d, psi_sp_init, psi_d_init = build_setup(2)
    N = length(psi_sp_init)
    println("=== Manual DMRG sweep prototype (N=2 plaquettes, $N sites) ===\n")

    # --- Run a reference dense DMRG sweep using ITensorMPS for comparison. ---
    psi_d_o = ITensorMPS.orthogonalize(psi_d_init, 1)
    PH_d_ref = ProjMPO(H_d)

    psi_sp_o = ITensorMPS.orthogonalize(psi_sp_init, 1)
    PH_sp    = ProjMPO(H_sp)

    function step_dense(psi::MPS, PH::ProjMPO, b::Int, sweep_dir::Symbol)
        position!(PH, psi, b)
        phi = psi[b] * psi[b+1]
        vals, vecs = eigsolve(PH, phi, 1, :SR; ishermitian=true)
        E = real(vals[1])
        phi_new = vecs[1]
        # SVD back into sites — use replacebond! for dense psi.
        spec = ITensorMPS.replacebond!(
            psi, b, phi_new;
            maxdim=200, cutoff=1e-12,
            ortho = (sweep_dir == :right ? "left" : "right"),
            normalize=true,
        )
        return E
    end

    # Build M (full Gram operator at bond b) as a single ITensor with two-sided
    # bond inds (primed + unprimed). Lgram has the LEFT-side bond inds free
    # (between psi[b-1] and psi[b]); Rgram has the RIGHT-side bond inds free
    # (between psi[b+1] and psi[b+2]). They commute (act on disjoint bonds),
    # so M = Lgram ⊗ Rgram as a tensor product.
    function build_M_full(Lgram::ITensor, Rgram::ITensor)
        return Lgram * Rgram
    end

    # Build the small dense Minv ITensor on phi's bond legs by densifying the
    # full M tensor, treating (unprimed-bond) × (primed-bond) as a matrix, and
    # inverting. Returns Minv as an ITensor with the same Index layout as M.
    function build_Minv_itensor(M_full::ITensor)
        bond_unp = filter(I -> ITensors.plev(I) == 0, collect(inds(M_full)))
        bond_prm = filter(I -> ITensors.plev(I) == 1, collect(inds(M_full)))
        @assert length(bond_unp) == length(bond_prm)
        d = prod(ITensors.dim, bond_unp; init=1)
        M_d_it = if ITensors.has_external_storage(M_full)
            SparseBackends.to_dense_itensors_unfused(M_full)
        else
            M_full
        end
        M_mat_arr = Array(M_d_it, bond_unp..., bond_prm...)
        M_mat = reshape(M_mat_arr, d, d)
        M_mat = (M_mat + M_mat') / 2
        # M may be rank-deficient when psi[i] is not strictly iso (common after
        # channel-aware SVD with truncation). Use pseudoinverse so directions
        # in M's null-space are projected out instead of amplified.
        Minv_mat = pinv(M_mat; rtol=1e-10)
        dims_all = [ITensors.dim(I) for I in bond_unp]
        append!(dims_all, [ITensors.dim(I) for I in bond_prm])
        Minv_arr = reshape(Minv_mat, dims_all...)
        return ITensors.itensor(Minv_arr, bond_unp..., bond_prm...)
    end

    # Krylov variant: use eigsolve(M⁻¹ H_eff) with BS-preserving contractions.
    function step_sparse_krylov(psi::MPS, PH::ProjMPO, b::Int, sweep_dir::Symbol)
        position!(PH, psi, b)
        Lgram = gram_left_env(psi, b)
        Rgram = gram_right_env(psi, b)
        M_full = build_M_full(Lgram, Rgram)
        Minv   = build_Minv_itensor(M_full)
        phi = psi[b] * psi[b+1]

        function A_op(x)
            Hx = product(PH, x)
            MinvHx = SparseBackends.contract_preserve_bs(Minv, Hx; template=x)
            MinvHx = replaceprime(MinvHx, 1 => 0; tags="Link")
            if ITensors.has_external_storage(MinvHx) && ITensors.has_external_storage(x)
                Tw = ITensors.get_external_storage(x)
                Cw = ITensors.get_external_storage(MinvHx)
                if Cw isa SparseBackends.WrappedBlockSparse && Tw isa SparseBackends.WrappedBlockSparse
                    MinvHx = ITensors._itensor_from_external_storage(SparseBackends.recast_bs_to_template(Cw, Tw))
                end
            end
            return MinvHx
        end

        vals, vecs, info = eigsolve(A_op, phi, 1, :SR; ishermitian=false)
        E = real(vals[1])
        phi_new = vecs[1]
        spec = ITensorMPS.replacebond!(
            psi, b, phi_new;
            mindim=1, maxdim=200, cutoff=1e-12,
            ortho = (sweep_dir == :right ? "left" : "right"),
            normalize=true,
        )
        return E
    end

    function step_sparse(psi::MPS, PH::ProjMPO, b::Int, sweep_dir::Symbol)
        position!(PH, psi, b)
        # Gram envs from CURRENT (non-canonical) sparse psi.
        Lgram = gram_left_env(psi, b)
        Rgram = gram_right_env(psi, b)
        phi = psi[b] * psi[b+1]
        E, phi_new = solve_geneig_dense(PH, Lgram, Rgram, phi)
        # SVD back using the channel-aware sparse pathway.
        spec = ITensorMPS.replacebond!(
            psi, b, phi_new;
            maxdim=200, cutoff=1e-12,
            ortho = (sweep_dir == :right ? "left" : "right"),
            normalize=true,
        )
        return E
    end

    function do_sweep!(psi_d, PH_d, psi_s, PH_s, sweep_idx)
        println("--- Sweep $sweep_idx forward (b = 1..$(N-1)) ---")
        for b in 1:N-1
            Ed = step_dense(psi_d, PH_d, b, :right)
            Es = step_sparse(psi_s, PH_s, b, :right)
            ΔE = abs(Ed - Es)
            ok = ΔE < 1e-8 ? "OK" : "MISMATCH"
            println("  bond $b: E_dense = $(round(Ed, digits=10)), E_sparse = $(round(Es, digits=10)), |ΔE| = $(round(ΔE, sigdigits=3))  [$ok]")
        end
        println("--- Sweep $sweep_idx backward (b = $(N-1)..1) ---")
        for b in N-1:-1:1
            Ed = step_dense(psi_d, PH_d, b, :left)
            Es = step_sparse(psi_s, PH_s, b, :left)
            ΔE = abs(Ed - Es)
            ok = ΔE < 1e-8 ? "OK" : "MISMATCH"
            println("  bond $b: E_dense = $(round(Ed, digits=10)), E_sparse = $(round(Es, digits=10)), |ΔE| = $(round(ΔE, sigdigits=3))  [$ok]")
        end
    end

    do_sweep!(psi_d_o, PH_d_ref, psi_sp_o, PH_sp, 1)
    do_sweep!(psi_d_o, PH_d_ref, psi_sp_o, PH_sp, 2)

    # --- Krylov-based variant on a FRESH pair of MPS to verify eigsolve(M⁻¹ H_eff) ---
    println("\n\n=== Krylov pass: eigsolve(M⁻¹ H_eff) ===")
    psi_d_k  = ITensorMPS.orthogonalize(psi_d_init,  1)
    psi_sp_k = ITensorMPS.orthogonalize(psi_sp_init, 1)
    PH_d_k   = ProjMPO(H_d)
    PH_sp_k  = ProjMPO(H_sp)

    function do_sweep_krylov!(psi_d, PH_d, psi_s, PH_s, sweep_idx)
        println("--- Krylov sweep $sweep_idx forward (b = 1..$(N-1)) ---")
        for b in 1:N-1
            Ed = step_dense(psi_d, PH_d, b, :right)
            Es = step_sparse_krylov(psi_s, PH_s, b, :right)
            ΔE = abs(Ed - Es)
            ok = ΔE < 1e-8 ? "OK" : "MISMATCH"
            println("  bond $b: E_dense = $(round(Ed, digits=10)), E_sparse = $(round(Es, digits=10)), |ΔE| = $(round(ΔE, sigdigits=3))  [$ok]")
        end
        println("--- Krylov sweep $sweep_idx backward (b = $(N-1)..1) ---")
        for b in N-1:-1:1
            Ed = step_dense(psi_d, PH_d, b, :left)
            Es = step_sparse_krylov(psi_s, PH_s, b, :left)
            ΔE = abs(Ed - Es)
            ok = ΔE < 1e-8 ? "OK" : "MISMATCH"
            println("  bond $b: E_dense = $(round(Ed, digits=10)), E_sparse = $(round(Es, digits=10)), |ΔE| = $(round(ΔE, sigdigits=3))  [$ok]")
        end
    end
    do_sweep_krylov!(psi_d_k, PH_d_k, psi_sp_k, PH_sp_k, 1)
    do_sweep_krylov!(psi_d_k, PH_d_k, psi_sp_k, PH_sp_k, 2)

    println("\n=== Final Krylov energies ===")
    println("  dense  <psi|H|psi> / <psi|psi> = ", real(inner(psi_d_k',  H_d,  psi_d_k)  / inner(psi_d_k,  psi_d_k)))
    println("  sparse <psi|H|psi> / <psi|psi> = ", real(inner(psi_sp_k', H_sp, psi_sp_k) / inner(psi_sp_k, psi_sp_k)))

    println("\n=== Final energies ===")
    println("  dense  <psi|H|psi> / <psi|psi> = ", real(inner(psi_d_o',  H_d,  psi_d_o)  / inner(psi_d_o,  psi_d_o)))
    println("  sparse <psi|H|psi> / <psi|psi> = ", real(inner(psi_sp_o', H_sp, psi_sp_o) / inner(psi_sp_o, psi_sp_o)))
end
nothing
