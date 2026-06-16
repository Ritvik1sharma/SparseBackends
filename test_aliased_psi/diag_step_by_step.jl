# Step-by-step diagnostic: compare aliased vs dense pipeline at each stage
# of one DMRG step. Reports norm differences and iso check on factorize.
ENV["BMF_ISO_PATH"] = "1"
ENV["SB_ALIASED_ENABLE"] = "1"

using SparseBackends, ITensors, ITensorMPS
using Random
using Printf
using LinearAlgebra: norm, I as eye

include("../test_sparse_psi/utils.jl")

const N_PLAQ = 2

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
        t += 0.5,         "Id",            2*j-1, "Id",            2*j, "Id",            2*j+1, "Id",            2*j+2
        t += 0.5 * psign, "exp(i*pi*Sy)",  2*j-1, "exp(i*pi*Sx)",  2*j, "exp(i*pi*Sx)",  2*j+1, "exp(i*pi*Sy)",  2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    psi0     = random_mps(sites)
    return H, P_sparse, psi0
end

# Convert an aliased ITensor to a dense ITensor with the same inds/values.
function densify(T::ITensor)
    if !ITensors.has_external_storage(T)
        return T
    end
    s = ITensors.get_external_storage(T)
    if s isa SparseBackends.WrappedAliasedBlockSparse
        return ITensors.itensor(SparseBackends.to_dense(s.aliased), s.inds...)
    elseif s isa SparseBackends.WrappedBlockSparse
        return ITensors.itensor(SparseBackends.to_dense(s.blocksparse), s.inds...)
    end
    return T
end

# Cast two ITensors to the same dense layout by matching inds (by id/tags).
function aligned_diff_norm(t_ali::ITensor, t_dense::ITensor)
    a = densify(t_ali)
    b = densify(t_dense)
    common = collect(inds(a))
    # Permute b to match a's index order.
    b_perm = permute(b, common...; allow_alias=true)
    return norm(array(a) .- array(b_perm))
end

function relative_diff(a::ITensor, b::ITensor)
    diff = aligned_diff_norm(a, b)
    return diff / max(norm(a), 1e-30)
end

H, P_sparse, psi0 = build_setup(N_PLAQ, +1)
n_sites = length(psi0)
println("=== N_plaq=$N_PLAQ, n_sites=$n_sites ===\n")

# Build psi_dense and psi_ali from the same psi0 (so values match initially).
psi_d   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense),   1 => 0)
psi_ali = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)

println("--- Step 1: initial psi value check (psi_dense vs psi_aliased) ---")
for k in 1:n_sites
    d = relative_diff(psi_ali[k], psi_d[k])
    @printf("  psi[%d]: relative_diff = %.3e   norm(psi)=%.3e\n", k, d, norm(densify(psi_ali[k])))
end

# Run one matvec step on each.
println("\n--- Step 2: H * psi at bond b=1 (eigenvector test vector) ---")
b = 1
phi_d   = psi_d[b]   * psi_d[b+1]
# Aliased phi: route through wrapped_contract_aliased with preserve_bs_output=true
# (same as dmrg.jl does for the eigsolve test vector).
let
    Aw = ITensors.get_external_storage(psi_ali[b])
    Bw = ITensors.get_external_storage(psi_ali[b+1])
    Cw = SparseBackends.wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true)
    global phi_ali = Cw isa ITensor ? Cw : ITensors._itensor_from_external_storage(Cw)
end
@printf("phi_ali storage = %s\n", ITensors.has_external_storage(phi_ali) ? typeof(phi_ali.tensor.data) : "dense")
@printf("phi_d   norm = %.4e\n", norm(densify(phi_d)))
@printf("phi_ali norm = %.4e\n", norm(densify(phi_ali)))
@printf("phi value diff (ali vs dense) = %.3e\n", aligned_diff_norm(phi_ali, phi_d))

# Apply H via ProjMPO.
PH_d   = ProjMPO(H); position!(PH_d, psi_d,   b)
PH_ali = ProjMPO(H); position!(PH_ali, psi_ali, b)
Hphi_d   = ITensorMPS.product(PH_d,   phi_d)
Hphi_ali = ITensorMPS.product(PH_ali, phi_ali)
@printf("Hphi_d   norm = %.4e\n", norm(densify(Hphi_d)))
@printf("Hphi_ali norm = %.4e\n", norm(densify(Hphi_ali)))
@printf("Hphi value diff (ali vs dense) = %.3e\n", aligned_diff_norm(Hphi_ali, Hphi_d))

# Inner product check.
inner_d   = scalar(dag(phi_d) * Hphi_d)
inner_ali = scalar(dag(densify(phi_ali)) * densify(Hphi_ali))
@printf("inner(phi, Hphi) dense=%.6f  ali=%.6f  diff=%.3e\n",
        real(inner_d), real(inner_ali), abs(inner_d - inner_ali))

# Factorize phi → L, R for the aliased path, check L*R = phi and iso.
println("\n--- Step 3: factorize phi → L, R (aliased path) ---")
L_ali, R_ali, spec = SparseBackends.itensor_aliased_factorize(
    phi_ali, psi_ali[b], psi_ali[b+1];
    ortho="left", maxdim=4, mindim=1, cutoff=1e-12)
LR_ali = L_ali * R_ali
@printf("L_ali  storage = %s\n", typeof(L_ali.tensor.data))
@printf("R_ali  storage = %s\n", typeof(R_ali.tensor.data))
@printf("L*R reconstruction error = norm(L*R - phi) = %.3e\n", aligned_diff_norm(LR_ali, phi_ali))
@printf("L*R norm = %.4e   phi norm = %.4e\n", norm(densify(LR_ali)), norm(densify(phi_ali)))

# Iso check: for ortho="left", L should be left-iso. L's bond inds (shared
# with R) form the "out" axes; iso means contracting L with dag(prime(L, bond_inds))
# gives identity on bond_inds × bond_inds'.
println("\n--- Step 4: iso check on L (left-ortho) ---")
bond_inds = collect(commoninds(L_ali, R_ali))
@printf("  bond inds: %s\n", [(dim(I), tags(I)) for I in bond_inds])
L_d = densify(L_ali)
LdagL = L_d * dag(prime(L_d, bond_inds...))
bond_pr = [prime(I) for I in bond_inds]
total_bond_dim = prod(dim, bond_inds)
LdagL_arr = array(LdagL, bond_inds..., bond_pr...)
LdagL_mat = reshape(LdagL_arr, total_bond_dim, total_bond_dim)
I_target  = Matrix{ComplexF64}(eye, total_bond_dim, total_bond_dim)
iso_err   = norm(LdagL_mat - I_target) / sqrt(total_bond_dim)
@printf("  iso error: norm(L'L - I) / sqrt(dim) = %.3e   (total bond dim = %d)\n",
        iso_err, total_bond_dim)
@printf("  L'L matrix:\n")
for i in 1:total_bond_dim
    for j in 1:total_bond_dim
        @printf("    [%d,%d] = %.4f%+.4fim\n", i, j, real(LdagL_mat[i,j]), imag(LdagL_mat[i,j]))
    end
end

# Also check norm of L*R vs phi (full energy preservation).
println("\n--- Step 5: phi reconstruction + norm preservation ---")
@printf("  norm(phi)     = %.6f\n", norm(densify(phi_ali)))
@printf("  norm(L*R)     = %.6f\n", norm(densify(LR_ali)))
@printf("  norm(L)       = %.6f\n", norm(densify(L_ali)))
@printf("  norm(R)       = %.6f   (= norm(phi) for left-ortho)\n", norm(densify(R_ali)))
nothing
