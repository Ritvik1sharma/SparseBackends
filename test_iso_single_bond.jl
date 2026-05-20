# Probe the channel-aware SVD at a single bond: is R iso?
# Tests bond 3 (bulk-bulk) and bond 5 (bulk-boundary) explicitly to find why
# psi[4] / psi[6] iso fails after orthogonalize.
using SparseBackends, ITensors, ITensorMPS
using Random, LinearAlgebra
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

bs_info(T) = ITensors.has_external_storage(T) ?
    "BS($(length(ITensors.get_external_storage(T).blocksparse.keys)) blk, $(length(ITensors.get_external_storage(T).blocksparse.data))/$(prod(ITensors.get_external_storage(T).blocksparse.dims)))" :
    "Dense"

densify(T) = ITensors.has_external_storage(T) ? SparseBackends.to_dense_itensors_unfused(T) : T

# Check that a tensor `T` is right-iso: T · dag(T) over its right-side legs == δ on left_bond.
function check_right_iso(T::ITensor, left_bond::Vector{<:Index}, label::String)
    T_d = densify(T)
    Td  = dag(T_d)
    Td  = prime(Td, left_bond...)
    E   = T_d * Td
    Cl = combiner(left_bond...; tags="bL")
    Cr = combiner(prime.(left_bond)...; tags="bR")
    cl = combinedind(Cl); cr = combinedind(Cr)
    Em = Array(E * Cl * Cr, cl, cr)
    D  = ITensors.dim(cl)
    err = norm(Em - Matrix{ComplexF64}(I, D, D))
    println("  $label  right-iso err = $err  (bond dim D=$D)")
    if err > 1e-10
        println("    sample of (M*M† - I):")
        show(stdout, "text/plain", Em - Matrix{ComplexF64}(I, D, D))
        println()
    end
    return err
end

length(ARGS) < 1 && error("Usage: julia test_iso_single_bond.jl <N_plaq>")
const _N_ISO = parse(Int, ARGS[1])

let
    psi = build_setup(_N_ISO)
    println("Initial psi:")
    for i in 1:length(psi); println("  psi[$i]: ", bs_info(psi[i])); end

    println("\n=== Single SVD at bond 3 (bulk-bulk) with ortho=\"right\" ===")
    phi = psi[3] * psi[4]
    println("phi: ", bs_info(phi))
    L, R, _ = SparseBackends.itensor_blocksparse_svd_channel_aware(
        phi, psi[3], psi[4];
        ortho="right", maxdim=typemax(Int), mindim=1, cutoff=0.0)
    println("L: ", bs_info(L), "   inds=", inds(L))
    println("R: ", bs_info(R), "   inds=", inds(R))

    # ||phi - L*R||
    println("||phi - L*R||  = ", norm(densify(phi) - densify(L * R)))

    # Check R iso (since ortho="right"): contract over R's non-bond legs
    new_bond = commoninds(L, R)
    println("new bond indices: ", new_bond)
    check_right_iso(R, collect(new_bond), "R (=new M[b+1])")

    # Compare to dense SVD
    println("\n=== Dense reference SVD at bond 3 ===")
    phi_d = densify(phi)
    indsMb = [I for I in inds(phi) if I in inds(psi[3]) && !(I in inds(psi[4]))]
    Ud, Sd, Vd, _, _, _ = ITensors.svd(phi_d, indsMb;
        lefttags=TagSet("Link,l=3"), righttags=TagSet("Link,l=3"))
    Ld = Ud * Sd
    Rd = Vd
    println("Ld: ", bs_info(Ld), "   inds=", inds(Ld))
    println("Rd: ", bs_info(Rd), "   inds=", inds(Rd))
    println("||phi - Ld*Rd|| = ", norm(densify(phi) - densify(Ld * Rd)))
    new_bond_d = commoninds(Ld, Rd)
    check_right_iso(Rd, collect(new_bond_d), "Rd (dense ref)")
end
nothing
