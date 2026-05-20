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
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return psi_sp
end

# For an L tensor that's supposed to be isometric on the new bond,
# compute L^dag * L as a (D × D) dense matrix and compare to identity.
# Convert L to dense first to avoid BS storage edge cases in the contraction.
function check_iso(L::ITensor, new_bond::Index, label::String)
    L_dense = ITensors.has_external_storage(L) ?
              SparseBackends.to_dense_itensors(L) : L
    D = dim(new_bond)
    # Use the dense tensor's actual inds (may differ from L's inds after unfusing)
    all_inds = collect(inds(L_dense))
    # Find which is the bond axis (by id, robust to plev/tag changes)
    bond_pos = findfirst(I -> ITensors.id(I) == ITensors.id(new_bond), all_inds)
    if bond_pos === nothing
        println("  [$label] bond_dim=$D  (could not locate new_bond in dense inds; skipping)")
        return
    end
    other = [I for (i, I) in enumerate(all_inds) if i != bond_pos]
    # Permute the array so new_bond is the LAST axis, then reshape to (rows, D)
    L_arr = Array(L_dense, other..., all_inds[bond_pos])
    nrows = div(length(L_arr), D)
    L_mat = reshape(L_arr, nrows, D)
    LdL = L_mat' * L_mat   # D × D
    Id  = Matrix{eltype(LdL)}(I, D, D)
    diff = LdL - Id
    diff_norm = norm(diff)
    max_off = 0.0
    for i in 1:D, j in 1:D
        i == j && continue
        v = abs(LdL[i, j])
        max_off = max(max_off, v)
    end
    diag_vals = [real(LdL[i, i]) for i in 1:D]
    println("  [$label] bond_dim=$D  rows=$nrows  ‖L'L - I‖ = $(round(diff_norm, sigdigits=4))")
    println("         diag = ", [round(d, digits=6) for d in diag_vals])
    println("         max |off-diag| = $(round(max_off, sigdigits=4))")
    is_iso = diff_norm < 1e-8
    println("         ISO=", is_iso)
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
            L, R, spec = SparseBackends.itensor_blocksparse_svd_channel_aware(
                phi, psi[b], psi[b+1];
                ortho="left", maxdim=40, mindim=1, cutoff=0.0,
                relax_iso_cap=false,
            )
            new_bond = first(I for I in commoninds(L, R))
            println("\n--- bond $b ---  new bond dim = $(dim(new_bond))")
            check_iso(L, new_bond, "L'L (should be I)")
            psi[b]   = L
            psi[b+1] = R
        catch e
            println("\n--- bond $b ---  SVD FAILED: ", sprint(showerror, e))
        end
    end
end
nothing
