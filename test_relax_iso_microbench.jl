# Microbenchmark: probe channel-aware SVD at every bond, both with and without
# relax_iso_cap.  Checks:
#   (1) Exact factorization:  ||phi - L*R||  ≈ 0
#   (2) Block-key preservation: L's keys ⊆ M_b's keys, R's keys ⊆ M_b1's keys
#   (3) Mult capacity: n_new_d (relax) ≥ n_new_d (strict) — relax lets it grow
#   (4) Iso behavior: strict path may still be non-iso at factor-overlap bonds
#   (5) Dense-SVD reference: ||phi - Ld*Rd|| and Rd iso err for cross-check
# Usage: julia test_relax_iso_microbench.jl <N_plaq>
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
include("utils.jl")

function build_psi(N::Int)
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
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    psi0 = random_mps(sites)
    return replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
end

densify(T) = ITensors.has_external_storage(T) ? SparseBackends.to_dense_itensors_unfused(T) : T

# Get the block-key set (set of NTuple sparse-key tuples) of a sparse tensor.
function bs_keys(T::ITensor)
    w = ITensors.get_external_storage(T)
    return Set(collect(w.blocksparse.keys))
end

function check_right_iso(T::ITensor, left_bond, label)
    Td = densify(T); Tdg = prime(dag(Td), left_bond...)
    E = Td * Tdg
    Cl = combiner(left_bond...; tags="bL")
    Cr = combiner(prime.(left_bond)...; tags="bR")
    cl = combinedind(Cl); cr = combinedind(Cr)
    Em = Array(E * Cl * Cr, cl, cr)
    D = ITensors.dim(cl)
    err = norm(Em - Matrix{ComplexF64}(I, D, D))
    println("    [$label] right-iso err = $(round(err; sigdigits=4))  bond_full_dim=$D")
    return err
end

function probe_bond(psi, b)
    println("\n========= bond $b  (psi[$b] × psi[$(b+1)]) =========")
    phi = psi[b] * psi[b+1]
    Mb_keys  = bs_keys(psi[b])
    Mb1_keys = bs_keys(psi[b+1])
    println("OLD M_b keys=$(length(Mb_keys))  M_b1 keys=$(length(Mb1_keys))")

    for relax in (false, true)
        tag = relax ? "RELAX" : "STRICT"
        L, R, _ = SparseBackends.itensor_blocksparse_svd_channel_aware(
            phi, psi[b], psi[b+1];
            ortho="right", maxdim=typemax(Int), mindim=1, cutoff=0.0,
            relax_iso_cap = relax)
        # (1) factorization
        err = norm(densify(phi) - densify(L * R))
        # (2) block-key subset check
        Lk = bs_keys(L); Rk = bs_keys(R)
        Lk_sub = issubset(Lk, Mb_keys)
        Rk_sub = issubset(Rk, Mb1_keys)
        # bond dims
        nb = collect(commoninds(L, R))
        sp_dim = prod(ITensors.dim(I) for I in nb if ITensors.hastags(I, "Link"); init=1)
        mu_dim = prod(ITensors.dim(I) for I in nb if !ITensors.hastags(I, "Link"); init=1)
        println("  $tag  ||phi - L*R||=$(round(err; sigdigits=4))  L_keys=$(length(Lk)) (⊆? $Lk_sub)  R_keys=$(length(Rk)) (⊆? $Rk_sub)  new_sp=$sp_dim new_d=$mu_dim full=$(sp_dim*mu_dim)")
        # (3) iso check (only meaningful for R since ortho="right")
        check_right_iso(R, nb, tag)
    end

    # (5) Dense-SVD reference at this same bond, for cross-check on factorization
    # error and iso. Uses ITensors.svd on the densified phi.
    phi_d = densify(phi)
    indsMb = [I for I in inds(phi) if I in inds(psi[b]) && !(I in inds(psi[b+1]))]
    Ud, Sd, Vd, _, _, _ = ITensors.svd(phi_d, indsMb;
        lefttags=TagSet("Link,l=$b"), righttags=TagSet("Link,l=$b"))
    Ld = Ud * Sd; Rd = Vd
    err_d = norm(densify(phi) - densify(Ld * Rd))
    nb_d = collect(commoninds(Ld, Rd))
    println("  DENSE_REF  ||phi - Ld*Rd||=$(round(err_d; sigdigits=4))  bond_dim=$(prod(ITensors.dim(I) for I in nb_d; init=1))")
    check_right_iso(Rd, nb_d, "DENSE_REF")
end

length(ARGS) < 1 && error("Usage: julia test_relax_iso_microbench.jl <N_plaq>")
const _N_RELAX = parse(Int, ARGS[1])

let
    psi = build_psi(_N_RELAX)
    println("psi length=$(length(psi)) (N=$_N_RELAX, $(2*_N_RELAX+2) sites)")
    # Probe a bulk factor-overlap bond (b=4 in 8-site chain has overlap of C₂ across both free links).
    for b in 2:length(psi)-1
        probe_bond(psi, b)
    end
end
nothing
