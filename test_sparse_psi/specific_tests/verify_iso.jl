# Verify strict-iso BS SVD actually produces an isometric L tensor.
# For each bond in psi_sp:
#   1. Compute phi = psi[b] * psi[b+1]
#   2. SVD with relax_iso_cap=false (strict) → L, R
#   3. Check L^dag * L = I on the new bond
#   4. Report ‖L^dag L − I‖ at each bond
# Pinpoints exactly which bonds (if any) fail strict isometry and prints details.
#
# Usage:  julia --project=.. verify_iso.jl [N]

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra
using Random
include("../utils.jl")

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
    H = MPO(os, sites)

    # Verify [P, H] = 0 by computing ‖P·H·ψ − H·P·ψ‖ on a random dense ψ.
    # Convert P_sparse to dense MPO first to avoid BS-vs-dense apply issues.
    P_dense = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in P_sparse])
    psi_test = random_mps(sites; linkdims=2)
    PH_psi = apply(P_dense, apply(H, psi_test; cutoff=1e-15, maxdim=400); cutoff=1e-15, maxdim=400)
    HP_psi = apply(H, apply(P_dense, psi_test; cutoff=1e-15, maxdim=400); cutoff=1e-15, maxdim=400)
    nph = norm(PH_psi); nhp = norm(HP_psi)
    diff_psi = +(PH_psi, -1.0 * HP_psi; cutoff=1e-15, maxdim=400)
    println("[H,P] sanity: ‖P·H·ψ‖=$(round(nph;sigdigits=4))  ‖H·P·ψ‖=$(round(nhp;sigdigits=4))  ‖[H,P]·ψ‖=$(round(norm(diff_psi);sigdigits=4))  rel=$(round(norm(diff_psi)/max(nph,1e-30);sigdigits=4))")

    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return psi_sp
end

# For an L tensor that's supposed to be isometric on the new bond,
# compute L^dag * L as a (D × D) dense matrix and compare to identity.
# Convert L to dense first to avoid BS storage edge cases in the contraction.
function check_iso(L::ITensor, new_bond_inds, label::String)
    # Use UNFUSED densification so bond_sparse and bond_mult stay separate.
    L_dense = ITensors.has_external_storage(L) ?
              SparseBackends.to_dense_itensors_unfused(L) : L
    bond_inds = collect(new_bond_inds)
    D = isempty(bond_inds) ? 1 : prod(dim, bond_inds)
    all_inds = collect(inds(L_dense))
    other = filter(I -> !any(b -> ITensors.id(I) == ITensors.id(b), bond_inds), all_inds)
    if length(other) + length(bond_inds) != length(all_inds)
        println("  [$label] bond_inds not found in L_dense inds; skipping")
        return
    end
    L_arr = Array(L_dense, other..., bond_inds...)
    nrows = div(length(L_arr), D)
    L_mat = reshape(L_arr, nrows, D)
    LdL = L_mat' * L_mat
    Id  = Matrix{eltype(LdL)}(I, D, D)
    diff_norm = norm(LdL - Id)
    max_off = maximum(abs(LdL[i,j]) for i in 1:D, j in 1:D if i != j; init=0.0)
    println("  [$label] bond_dim=$D  rows=$nrows  ‖L'L - I‖ = $(round(diff_norm, sigdigits=4))  max|off|=$(round(max_off, sigdigits=4))")
end

# Check reconstruction error: ‖L·R − phi‖ / ‖phi‖.
function check_recon(L::ITensor, R::ITensor, phi::ITensor, label::String)
    phi_recon = L * R
    phi_d  = ITensors.has_external_storage(phi)       ? SparseBackends.to_dense_itensors_unfused(phi)       : phi
    pr_d   = ITensors.has_external_storage(phi_recon) ? SparseBackends.to_dense_itensors_unfused(phi_recon) : phi_recon
    common = collect(inds(phi_d))
    a = Array(phi_d, common...)
    # Try to match recon's index ordering to phi
    b_inds = collect(inds(pr_d))
    perm = [findfirst(I -> ITensors.id(I) == ITensors.id(ci), b_inds) for ci in common]
    if any(isnothing, perm)
        println("  [$label] recon inds don't match phi; skipping recon check")
        return
    end
    b = Array(pr_d, [b_inds[p] for p in perm]...)
    nrm = norm(a)
    err = norm(a .- b)
    println("  [$label] ‖phi‖=$(round(nrm, sigdigits=4))  ‖L·R−phi‖=$(round(err, sigdigits=4))  rel=$(round(err/max(nrm,1e-30), sigdigits=4))")
end

let
    N_plaq = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
    println("=== Building setup for N=$N_plaq plaquettes ===")
    psi = build_setup(N_plaq)
    println("System: $(length(psi)) sites; orthogonalizing to bond 1...")
    psi = orthogonalize!(psi, 1)

    println("\n=== Strict-iso SVD at each bond (relax_iso_cap=false) ===")
    for b in 1:length(psi)-1
        phi = psi[b] * psi[b+1]
        if !ITensors.has_external_storage(phi)
            continue
        end
        try
            factor_fn = if get(ENV, "SB_USE_OWNED_SVD", "0") == "1"
                SparseBackends.itensor_blocksparse_svd_owned_channel_aware
            elseif get(ENV, "SB_USE_QR", "0") == "1"
                SparseBackends.itensor_blocksparse_qr_channel_aware
            else
                SparseBackends.itensor_blocksparse_svd_channel_aware
            end
            L, R, spec = factor_fn(
                phi, psi[b], psi[b+1];
                ortho="left", maxdim=10000, mindim=1, cutoff=0.0,
            )
            new_bonds = collect(commoninds(L, R))
            println("\n--- bond $b ---  new bond dims = $([dim(b) for b in new_bonds])  (n_inds=$(length(new_bonds)))")
            check_iso(L, new_bonds, "L'L")
            check_recon(L, R, phi, "L·R recon")
            psi[b]   = L
            psi[b+1] = R
        catch e
            println("\n--- bond $b ---  SVD FAILED: ", sprint(showerror, e))
        end
    end
end
nothing
