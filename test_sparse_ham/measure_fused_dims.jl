# Measure the fused-2&3 relevant dims (n_m, mid mult dr, per-site channel/mult,
# template counts) directly from H_new_aliased on a bulk bond — no ProjMPO/sweep
# needed, since these depend only on the two MPO tensors H1,H2.
using SparseBackends, Random
using ITensors, ITensorMPS
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
include("aliased_helpers.jl")
include("../test_sparse_psi/utils.jl")

function sandwich_mpo(P::MPO, H::MPO; output_hint::Symbol = :default)
    H1    = contract(P'', H', :coo, :dense; Cbackend=output_hint)
    H_eff = contract(P, H1, :coo, output_hint; Cbackend=output_hint)
    return replaceprime(H_eff, 3 => 1)
end
mulMPO(A::MPO, B::MPO) = replaceprime(contract(A, prime(B, "Site"), :coo, :coo), 2 => 1)
function multiplyVecMPOtoMPO(vec::Vector{MPO})
    result = vec[1]
    for j in 2:length(vec); result = mulMPO(result, vec[j]); end
    return result
end

let
    spin = 3
    spin_sector = parse(Float64, get(ENV, "BENCH_PSIGN", "1.0"))
    N = parse(Int, get(ENV, "N_PLAQ", "12"))
    states = 2*N + 2
    sites = siteinds("S=1", states)

    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    os2 = OpSum[]
    for j in 1:N
        temp = OpSum()
        temp += 0.5,              "Id",2*j-1,"Id",2*j,"Id",2*j+1,"Id",2*j+2
        temp += spin_sector*0.5,  "exp(i*pi*Sy)",2*j-1,"exp(i*pi*Sx)",2*j,"exp(i*pi*Sx)",2*j+1,"exp(i*pi*Sy)",2*j+2
        push!(os2, temp)
    end
    ConsOps1 = MPO[]
    for j in 1:N
        operator = MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])
        clean!(operator; tol=1e-12)
        push!(ConsOps1, operator)
    end
    ConsOpsCombined = multiplyVecMPOtoMPO(ConsOps1)
    H = MPO(os, sites)

    H_new_aliased = sandwich_mpo(ConsOpsCombined, copy(H); output_hint = :aliased)
    fuse_sparse_links!(H_new_aliased)
    prepermute_aliased_mpo!(H_new_aliased)

    L = length(H_new_aliased)
    println("\n=== H_new_aliased: $L tensors (N=$N, states=$states) ===")

    # Per-site sparse-channel (Link) dims and dense-mult dims + template counts.
    println("\nsite | sparse-Link dims        | dense-mult dims | n_tmpl  blksize")
    for i in 1:L
        ITensors.has_external_storage(H_new_aliased[i]) || (println("  $i  | (no aliased storage)"); continue)
        Hw = ITensors.get_external_storage(H_new_aliased[i])
        hinds = inds(Hw)
        length(hinds) < 6 && (println("  $i  | boundary (", length(hinds), " inds)"); continue)
        sparse_inds = hinds[1:4]; dense_inds = hinds[5:6]
        A = Hw.aliased
        linkdims = [dim(I) for I in sparse_inds if hastags(I, "Link")]
        ddims = [dim(I) for I in dense_inds]
        println("  $i  | Link=", linkdims, "  Site=",
                [dim(I) for I in sparse_inds if hastags(I,"Site")],
                "  | mult=", ddims, "  | ", A.n_templates, "  ", A.blksize)
    end

    # Classify a handful of bulk bonds (si, si+1) the way build_matvec_context does.
    println("\n=== bulk-bond fused dims (mid channel n_m, mid mult dr) ===")
    for si in [3, 4, 5, 6, 7, 8]
        (si+1 > L) && continue
        H1 = H_new_aliased[si]; H2 = H_new_aliased[si+1]
        H1w = ITensors.get_external_storage(H1); H2w = ITensors.get_external_storage(H2)
        mids = commoninds(H1, H2)
        h1_sparse_ids = Set(ITensors.id(i) for i in inds(H1w)[1:4])
        mid_channel = only(filter(i ->   ITensors.id(i) in h1_sparse_ids, mids))
        mid_mult    = only(filter(i -> !(ITensors.id(i) in h1_sparse_ids), mids))
        n_m = dim(mid_channel); dr = dim(mid_mult)
        # outer (non-shared) sparse Link on each side
        h1_links = [i for i in inds(H1w)[1:4] if hastags(i,"Link")]
        h2_links = [i for i in inds(H2w)[1:4] if hastags(i,"Link")]
        n_l1 = dim(only(filter(i -> ITensors.id(i) != ITensors.id(mid_channel), h1_links)))
        n_r2 = dim(only(filter(i -> ITensors.id(i) != ITensors.id(mid_channel), h2_links)))
        A1 = H1w.aliased; A2 = H2w.aliased
        full_mid = n_m * dr
        println("bond ($si,$(si+1)): n_m(mid chan)=$n_m  mid-mult dr=$dr  full mid link=$full_mid  ",
                "| n_l1=$n_l1 n_r2=$n_r2 | H1 nblk=$(length(A1.keys)) ntmpl=$(A1.n_templates) blksz=$(A1.blksize)",
                "  H2 nblk=$(length(A2.keys)) ntmpl=$(A2.n_templates) blksz=$(A2.blksize)")
    end
end
