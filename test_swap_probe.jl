# Probe ortho="right" swap: check L*R ≈ phi, isometry of isometric factor.
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

describe_T(T, name) = begin
    if ITensors.has_external_storage(T)
        w = ITensors.get_external_storage(T)
        println("  $name: BS($(length(w.blocksparse.keys)) blk, $(length(w.blocksparse.data))/$(prod(w.blocksparse.dims)))")
    else
        println("  $name: D")
    end
end

# Convert sparse ITensor to dense ITensor (keeps Index identity for ITensor arithmetic)
densify(T::ITensor) = ITensors.has_external_storage(T) ? SparseBackends.to_dense_itensors_unfused(T) : T

length(ARGS) < 1 && error("Usage: julia test_swap_probe.jl <N_plaq>")
const _N_SWAP = parse(Int, ARGS[1])

let
    psi = build_setup(_N_SWAP)
    b = 2
    phi = psi[b] * psi[b+1]
    println("phi: ", inds(phi))
    describe_T(phi, "phi")
    indsMb = collect(inds(psi[b]))   # left side of phi (replacebond! convention)

    # ===== Run all three paths =====
    for (label, ortho_kw, use_swap, binr) in (
        ("ortho=left,  bin_by_right=true  (current forward path)", "left",  false, true),
        ("ortho=left,  bin_by_right=false (plain left-binned)",    "left",  false, false),
        ("ortho=right, bin_by_right=true  (current back path)",    "right", false, true),
        ("ortho=right, bin_by_right=false (plain left-binned)",    "right", false, false),
        ("ortho=right, via swap (bin_by_right=true under hood)",   "right", true,  true),
    )
        println("\n----- $label -----")
        if use_swap
            phi_inds_all = collect(ITensors.get_external_storage(phi).inds)
            complement   = filter(i -> !(i in indsMb), phi_inds_all)
            L_sw, R_sw, spec = SparseBackends.itensor_blocksparse_svd(
                phi, complement;
                ortho = "left",
                tags  = TagSet("Link,l=$b"),
                maxdim=typemax(Int), mindim=1, cutoff=0.0,
                bin_by_right = binr)
            L, R = R_sw, L_sw     # swap-back to caller (L,R) convention
        else
            L, R, spec = SparseBackends.itensor_blocksparse_svd(
                phi, indsMb;
                ortho = ortho_kw,
                tags  = TagSet("Link,l=$b"),
                maxdim=typemax(Int), mindim=1, cutoff=0.0,
                bin_by_right = binr)
        end
        describe_T(L, "L"); describe_T(R, "R")
        println("  L inds: ", inds(L))
        println("  R inds: ", inds(R))
        nbs = commoninds(L, R)
        println("  new bond legs (sparse + dense multiplicity): ", nbs)
        println("  total bond dim = ", prod(ITensors.dim(i) for i in nbs))

        # ----- L*R reconstruction check (densify both then subtract) -----
        recon = L * R
        phi_d   = densify(phi)
        recon_d = densify(recon)
        diff = phi_d - recon_d
        err_recon = norm(diff)
        println("  ||phi - L*R|| (dense) = ", err_recon)

        # ----- isometry check on the side that should be isometric -----
        # ortho="left" → L isometric; ortho="right" → R isometric (after swap-back)
        iso_T, iso_name = ortho_kw == "left" ? (L, "L") : (R, "R")
        iso_d  = densify(iso_T)
        # Prime ALL shared bond legs (sparse + multiplicity) for the dag copy
        iso_dag = dag(iso_d)
        for nb in nbs
            iso_dag = prime(iso_dag, nb)
        end
        eye_like = iso_d * iso_dag    # should be ≈ identity on each (nb, nb')
        # Combine sparse+multiplicity legs on each side into a single combined leg, then check vs I
        Cl = ITensors.combiner(nbs...; tags="bL")
        cl = ITensors.combinedind(Cl)
        Cr = ITensors.combiner(prime.(nbs)...; tags="bR")
        cr = ITensors.combinedind(Cr)
        eye_mat_T = eye_like * Cl * Cr
        Dtot = ITensors.dim(cl)
        eye_arr = Array(eye_mat_T, cl, cr)
        err_iso = norm(eye_arr - Matrix{ComplexF64}(I, Dtot, Dtot))
        println("  ||$(iso_name)·$(iso_name)† - I|| over non-bond legs = ", err_iso, "  (total bond dim D=$Dtot)")
    end
end
nothing
