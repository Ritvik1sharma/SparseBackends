# Manual per-bond DMRG: sparse vs dense comparison at every bond.
# Usage: julia test_dmrg_manual.jl <N>
# Two passes:
#   Pass 1 — simple eigsolve per bond (forward sweep), compare sparse vs dense energy
#   Pass 2 — full forward+backward sweep with generalized eigsolve (Gram matrix)
#             plus a Krylov variant (eigsolve(M⁻¹H_eff))
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
using KrylovKit: eigsolve, geneigsolve

include("utils.jl")

length(ARGS) < 1 && error("Usage: julia test_dmrg_manual.jl <N_plaq>")
const N_PLAQ = parse(Int, ARGS[1])

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

bs_info(T) = ITensors.has_external_storage(T) ?
    "BS($(length(ITensors.get_external_storage(T).blocksparse.keys)) blk)" : "D"
describe(label, psi) = println("  [$label] ", [bs_info(T) for T in psi])

# -----------------------------------------------------------------------
# Pass 1: simple per-bond eigsolve comparison (forward only)
# -----------------------------------------------------------------------
function pass1_simple_eigsolve(H_sp, H_d, psi_sp_init, psi_d_init)
    println("\n" * "="^60)
    println("Pass 1: per-bond eigsolve (forward sweep, sparse vs dense)")
    println("="^60)
    L = length(psi_sp_init)

    psi_sp = ITensorMPS.orthogonalize(psi_sp_init, 1)
    psi_d  = ITensorMPS.orthogonalize(psi_d_init,  1)
    describe("psi_sp after ortho", psi_sp)
    describe("psi_d  after ortho", psi_d)

    PH_sp = position!(ProjMPO(H_sp), psi_sp, 1)
    PH_d  = position!(ProjMPO(H_d),  psi_d,  1)

    for b in 1:(L-1)
        b > 1 && (position!(PH_sp, psi_sp, b); position!(PH_d, psi_d, b))
        phi_sp = psi_sp[b] * psi_sp[b+1]
        phi_d  = psi_d[b]  * psi_d[b+1]
        E_sp = real(eigsolve(PH_sp, phi_sp, 1, :SR; ishermitian=true, tol=1e-14, krylovdim=3, maxiter=1, verbosity=0)[1][1])
        E_d  = real(eigsolve(PH_d,  phi_d,  1, :SR; ishermitian=true, tol=1e-14, krylovdim=3, maxiter=1, verbosity=0)[1][1])
        ΔE = abs(E_sp - E_d)
        println("  bond $b: E_sp=$(round(E_sp,digits=10))  E_d=$(round(E_d,digits=10))  |ΔE|=$(round(ΔE,sigdigits=3))  $(ΔE < 1e-8 ? "OK" : "MISMATCH")")
        phi_sp_new = eigsolve(PH_sp, phi_sp, 1, :SR; ishermitian=true)[2][1]
        phi_d_new  = eigsolve(PH_d,  phi_d,  1, :SR; ishermitian=true)[2][1]
        ITensorMPS.replacebond!(psi_sp, b, phi_sp_new; ortho="left", maxdim=200, mindim=1, cutoff=1e-12, normalize=true)
        ITensorMPS.replacebond!(psi_d,  b, phi_d_new;  ortho="left", maxdim=200, mindim=1, cutoff=1e-12, normalize=true)
    end
end

# -----------------------------------------------------------------------
# Pass 2: full sweep with generalized eigsolve (Gram matrix) + Krylov
# -----------------------------------------------------------------------
function gram_left_env(psi::MPS, b::Int)
    Ml = ITensor(1.0)
    for i in 1:(b-1)
        T  = psi[i]; Td = dag(T)
        for I in filter(I -> ITensors.hastags(I, "Link"), collect(inds(T))); Td = prime(Td, I); end
        Ml = Ml * T * Td
    end
    return Ml
end

function gram_right_env(psi::MPS, b::Int)
    L = length(psi); Mr = ITensor(1.0)
    for i in L:-1:(b+2)
        T  = psi[i]; Td = dag(T)
        for I in filter(I -> ITensors.hastags(I, "Link"), collect(inds(T))); Td = prime(Td, I); end
        Mr = Mr * T * Td
    end
    return Mr
end

function build_Minv_itensor(Lgram::ITensor, Rgram::ITensor)
    M_full  = Lgram * Rgram
    bond_u  = filter(I -> ITensors.plev(I) == 0, collect(inds(M_full)))
    bond_p  = filter(I -> ITensors.plev(I) == 1, collect(inds(M_full)))
    d       = prod(ITensors.dim, bond_u; init=1)
    M_d_it  = ITensors.has_external_storage(M_full) ? SparseBackends.to_dense_itensors_unfused(M_full) : M_full
    M_mat   = reshape(Array(M_d_it, bond_u..., bond_p...), d, d)
    M_mat   = (M_mat + M_mat') / 2
    Minv_mat = pinv(M_mat; rtol=1e-10)
    dims_all = vcat([ITensors.dim(I) for I in bond_u], [ITensors.dim(I) for I in bond_p])
    return ITensors.itensor(reshape(Minv_mat, dims_all...), bond_u..., bond_p...)
end

function solve_geneig_dense(PH, Lgram, Rgram, phi)
    phi_inds = collect(inds(phi))
    dims     = [ITensors.dim(I) for I in phi_inds]
    total    = prod(dims)
    H_mat    = zeros(ComplexF64, total, total)
    M_mat    = zeros(ComplexF64, total, total)
    M_full   = Lgram * Rgram
    for j in 1:total
        ej = zeros(ComplexF64, dims...); ej[j] = 1.0
        x  = ITensors.itensor(ej, phi_inds...)
        Hx = product(PH, x)
        Mx = replaceprime(M_full * x * dag(prime(x, filter(I->ITensors.plev(I)==0, collect(inds(M_full))))), 1 => 0; tags="Link")
        Hx_d = SparseBackends.to_dense_itensors_unfused(Hx)
        Mx_d = try SparseBackends.to_dense_itensors_unfused(Mx) catch; Mx end
        H_mat[:, j] = vec(Array(permute(Hx_d, phi_inds...; allow_alias=true), phi_inds...))
        M_mat[:, j] = vec(Array(permute(Mx_d, phi_inds...; allow_alias=true), phi_inds...))
    end
    H_mat = (H_mat + H_mat') / 2; M_mat = (M_mat + M_mat') / 2
    F    = eigen(H_mat, M_mat); vals = real.(F.values); perm = sortperm(vals)
    E    = NaN; v = zeros(ComplexF64, total)
    for k in perm; if isfinite(vals[k]); E = vals[k]; v = F.vectors[:, k]; break; end; end
    nrm2 = real(v' * (M_mat * v))
    v ./= sqrt(abs(nrm2))
    return E, ITensors.itensor(reshape(v, dims...), phi_inds...)
end

function step_dense(psi::MPS, PH::ProjMPO, b::Int, dir::Symbol)
    position!(PH, psi, b)
    phi = psi[b] * psi[b+1]
    vals, vecs = eigsolve(PH, phi, 1, :SR; ishermitian=true)
    ITensorMPS.replacebond!(psi, b, vecs[1]; maxdim=200, cutoff=1e-12,
        ortho=(dir==:right ? "left" : "right"), normalize=true)
    return real(vals[1])
end

function step_sparse_gram(psi::MPS, PH::ProjMPO, b::Int, dir::Symbol)
    position!(PH, psi, b)
    phi = psi[b] * psi[b+1]
    E, phi_new = solve_geneig_dense(PH, gram_left_env(psi, b), gram_right_env(psi, b), phi)
    ITensorMPS.replacebond!(psi, b, phi_new; maxdim=200, cutoff=1e-12,
        ortho=(dir==:right ? "left" : "right"), normalize=true)
    return E
end

function step_sparse_krylov(psi::MPS, PH::ProjMPO, b::Int, dir::Symbol)
    position!(PH, psi, b)
    Minv = build_Minv_itensor(gram_left_env(psi, b), gram_right_env(psi, b))
    phi  = psi[b] * psi[b+1]
    A_op(x) = begin
        Hx = product(PH, x)
        MinvHx = SparseBackends.contract_preserve_bs(Minv, Hx; template=x)
        MinvHx = replaceprime(MinvHx, 1 => 0; tags="Link")
        if ITensors.has_external_storage(MinvHx) && ITensors.has_external_storage(x)
            Tw = ITensors.get_external_storage(x); Cw = ITensors.get_external_storage(MinvHx)
            if Cw isa SparseBackends.WrappedBlockSparse && Tw isa SparseBackends.WrappedBlockSparse
                MinvHx = ITensors._itensor_from_external_storage(SparseBackends.recast_bs_to_template(Cw, Tw))
            end
        end
        MinvHx
    end
    vals, vecs, _ = eigsolve(A_op, phi, 1, :SR; ishermitian=false)
    ITensorMPS.replacebond!(psi, b, vecs[1]; mindim=1, maxdim=200, cutoff=1e-12,
        ortho=(dir==:right ? "left" : "right"), normalize=true)
    return real(vals[1])
end

function do_sweep!(psi_d, PH_d, psi_s, PH_s, sweep_idx, step_sparse_fn, label)
    L = length(psi_d)
    println("  --- $label sweep $sweep_idx forward ---")
    for b in 1:L-1
        Ed = step_dense(psi_d, PH_d, b, :right)
        Es = step_sparse_fn(psi_s, PH_s, b, :right)
        ΔE = abs(Ed - Es)
        println("    bond $b: E_d=$(round(Ed,digits=10))  E_s=$(round(Es,digits=10))  |ΔE|=$(round(ΔE,sigdigits=3))  $(ΔE<1e-8 ? "OK" : "MISMATCH")")
    end
    println("  --- $label sweep $sweep_idx backward ---")
    for b in L-1:-1:1
        Ed = step_dense(psi_d, PH_d, b, :left)
        Es = step_sparse_fn(psi_s, PH_s, b, :left)
        ΔE = abs(Ed - Es)
        println("    bond $b: E_d=$(round(Ed,digits=10))  E_s=$(round(Es,digits=10))  |ΔE|=$(round(ΔE,sigdigits=3))  $(ΔE<1e-8 ? "OK" : "MISMATCH")")
    end
end

function pass2_gram_and_krylov(H_sp, H_d, psi_sp_init, psi_d_init)
    println("\n" * "="^60)
    println("Pass 2a: full sweep with generalized eigsolve (Gram matrix)")
    println("="^60)
    psi_d_o  = ITensorMPS.orthogonalize(psi_d_init,  1)
    psi_sp_o = ITensorMPS.orthogonalize(psi_sp_init, 1)
    PH_d  = ProjMPO(H_d);  PH_sp = ProjMPO(H_sp)
    do_sweep!(psi_d_o, PH_d, psi_sp_o, PH_sp, 1, step_sparse_gram, "Gram")
    do_sweep!(psi_d_o, PH_d, psi_sp_o, PH_sp, 2, step_sparse_gram, "Gram")
    println("  Final dense  ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ = ", real(inner(psi_d_o',  H_d,  psi_d_o)  / inner(psi_d_o,  psi_d_o)))
    println("  Final sparse ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ = ", real(inner(psi_sp_o', H_sp, psi_sp_o) / inner(psi_sp_o, psi_sp_o)))

    println("\n" * "="^60)
    println("Pass 2b: full sweep with Krylov eigsolve(M⁻¹H_eff)")
    println("="^60)
    psi_d_k  = ITensorMPS.orthogonalize(psi_d_init,  1)
    psi_sp_k = ITensorMPS.orthogonalize(psi_sp_init, 1)
    PH_d_k   = ProjMPO(H_d); PH_sp_k = ProjMPO(H_sp)
    do_sweep!(psi_d_k, PH_d_k, psi_sp_k, PH_sp_k, 1, step_sparse_krylov, "Krylov")
    do_sweep!(psi_d_k, PH_d_k, psi_sp_k, PH_sp_k, 2, step_sparse_krylov, "Krylov")
    println("  Final dense  ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ = ", real(inner(psi_d_k',  H_d,  psi_d_k)  / inner(psi_d_k,  psi_d_k)))
    println("  Final sparse ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ = ", real(inner(psi_sp_k', H_sp, psi_sp_k) / inner(psi_sp_k, psi_sp_k)))
end

let
    println("=== Manual DMRG (N=$N_PLAQ plaquettes, $(2*N_PLAQ+2) sites) ===")
    H_sp, H_d, psi_sp, psi_d = build_setup(N_PLAQ)
    pass1_simple_eigsolve(H_sp, H_d, psi_sp, psi_d)
    pass2_gram_and_krylov(H_sp, H_d, psi_sp, psi_d)
end
nothing
