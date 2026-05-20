using SparseBackends, Random
using ITensors, ITensorMPS
using LinearAlgebra
include("utils.jl")

function sandwich_mpo(P::MPO, H::MPO)
    H1 = contract(P'', H', :coo, :dense)
    replaceprime(contract(P, H1, :coo, :blocksparse), 3 => 1)
end
function mulMPO(A::MPO, B::MPO)
    Bp = prime(B, "Site")
    replaceprime(contract(A, Bp, :coo, :coo), 2 => 1)
end
function multiplyVecMPOtoMPO(vec::Vector{MPO})
    r = vec[1]; for j in 2:length(vec); r = mulMPO(r, vec[j]); end; r
end

ok(msg, cond) = println(cond ? "  ✓ $msg" : "  ✗ $msg")

length(ARGS) < 1 && error("Usage: julia test_sparse_components.jl <N_plaq>")

let
    Random.seed!(42)
    N = parse(Int, ARGS[1])
    spin_sector = 1.0
    states = 2*N+2
    sites = siteinds("S=1", states)

    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end

    os2 = OpSum[]
    for j in 1:N
        c = 0.5
        t = OpSum()
        t += c, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += spin_sector*c, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                            "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end

    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    ConsOps2 = [MPO(os2[j], sites) for j in 1:N]
    P_sparse = multiplyVecMPOtoMPO(ConsOps1)

    H        = MPO(os, sites)
    H_sparse = sandwich_mpo(P_sparse, copy(H))

    psi0 = copy(random_mps(sites))
    for j in 1:N
        psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0)
        normalize!(psi0)
    end

    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    println("psi_sp link dims: ", [dim(linkind(psi_sp, i)) for i in 1:length(psi_sp)-1])
    println("psi_sp tensor types: ",
            [ITensors.has_external_storage(psi_sp[i]) ? "BS" : "dense" for i in 1:length(psi_sp)])

    # ────────────────────────────────────────────────────────────────────────
    # [1] Initial state sanity
    println("\n[1] initial-state sanity")
    n_sp  = inner(psi_sp, psi_sp)
    E_sp  = inner(psi_sp', H, psi_sp)
    E_sp2 = inner(psi_sp', H_sparse, psi_sp)
    println("    norm² = $n_sp")
    println("    ⟨psi_sp|H|psi_sp⟩       = $E_sp")
    println("    ⟨psi_sp|H_sparse|psi_sp⟩ = $E_sp2")
    ok("⟨H⟩ = ⟨H_sparse⟩ on initial state",
       isapprox(real(E_sp), real(E_sp2); atol=1e-8))
    ok("isortho is false (expected)", !isortho(psi_sp))

    # ────────────────────────────────────────────────────────────────────────
    # [2] Does orthogonalize!(psi_sp, k) preserve the state?
    println("\n[2] orthogonalize! preserves state? (gauge fix only, no truncation)")
    site1_before = inds(psi_sp[1])
    println("    site inds(psi_sp[1]) BEFORE = $site1_before")
    for k in (1, 3, length(psi_sp))
        psi_try = copy(psi_sp)
        n_b = inner(psi_try, psi_try)
        E_b_v = try inner(psi_try', H, psi_try) catch e "THREW: $(sprint(showerror,e))" end
        threw = false
        try
            orthogonalize!(psi_try, k)
        catch e
            println("    orthogonalize!(_, $k) THREW: ", sprint(showerror, e)); threw = true
        end
        threw && continue
        site1_after = inds(psi_try[1])
        types_after = [ITensors.has_external_storage(psi_try[i]) ? "BS" : "d" for i in 1:length(psi_try)]
        dims_after  = [dim(linkind(psi_try, i)) for i in 1:length(psi_try)-1]
        n_a = try inner(psi_try, psi_try) catch e "THREW" end
        E_a_v = try inner(psi_try', H, psi_try) catch e "THREW: $(sprint(showerror,e))" end
        println("    k=$k:")
        println("        norm before=$(real(n_b))  norm after=$n_a")
        println("        ⟨H⟩  before=$E_b_v")
        println("        ⟨H⟩  after =$E_a_v")
        println("        isortho=$(isortho(psi_try))  oc=$(isortho(psi_try) ? orthocenter(psi_try) : -1)")
        println("        link dims after: $dims_after")
        println("        types     after: $types_after")
        println("        inds(psi_try[1]) after = $site1_after")
    end

    # ────────────────────────────────────────────────────────────────────────
    # [3] Does pre-orthogonalizing fix DMRG?
    println("\n[3] DMRG with manually orthogonalized psi_sp")
    psi_sp_ortho = copy(psi_sp)
    try
        orthogonalize!(psi_sp_ortho, 1)
        normalize!(psi_sp_ortho)
        E_d, _, sw, terr = dmrg(H_sparse, psi_sp_ortho;
            nsweeps=3, maxdim=[20], mindim=[20], cutoff=1e-12,
            target_energy=nothing, use_early_exit=false, outputlevel=1)
        println("    [orthogonalized init] sparse DMRG E=$E_d  terr=$terr  (dense ref ≈ -3.729)")
    catch e
        println("    threw: ", sprint(showerror, e))
    end

    # ────────────────────────────────────────────────────────────────────────
    # [4a] DIRECT itensor_blocksparse_svd on a sparse phi (the path DMRG hits)
    println("\n[4a] itensor_blocksparse_svd round-trip on phi = psi_sp[1] * psi_sp[2]")
    try
        phi = psi_sp[1] * psi_sp[2]
        Linds = uniqueinds(psi_sp[1], psi_sp[2])
        L, R, spec = SparseBackends.itensor_blocksparse_svd(phi, Linds;
            ortho="left", maxdim=400, mindim=1, cutoff=1e-14,
            tags=ITensors.TagSet("Link,l=1"))
        recon = L * R
        a = SparseBackends.to_dense(phi); b = SparseBackends.to_dense(recon)
        err = norm(vec(a) .- vec(b)) / max(norm(vec(a)), 1e-300)
        println("    ‖phi − L*R‖/‖phi‖ = $err   spec=$spec")
        ok("itensor_blocksparse_svd reconstructs phi", err < 1e-8)
        println("    inds(L) = $(inds(L))")
        println("    inds(R) = $(inds(R))")
    catch e
        println("    threw: ", sprint(showerror, e))
    end

    # ────────────────────────────────────────────────────────────────────────
    # [4] Sparse SVD round-trip on phi from the first DMRG bond
    println("\n[4] svd(phi_sparse, Linds; ...) round-trip")
    psi_for_svd = copy(psi_sp)
    try
        # Without orthogonalization first (mirrors what the broken DMRG sees on bond 1)
        phi = psi_for_svd[1] * psi_for_svd[2]
        Linds = uniqueinds(psi_for_svd[1], psi_for_svd[2])
        U, S, V, spec = svd(phi, Linds; cutoff=1e-14, maxdim=400)
        recon = U * S * V
        # Compare via dense arrays
        a = SparseBackends.to_dense(phi)
        b = SparseBackends.to_dense(recon)
        err = norm(vec(a) .- vec(b)) / max(norm(vec(a)), 1e-300)
        println("    ‖phi - U S V‖ / ‖phi‖ = $err   (truncerr stored: ", spec, ")")
        ok("SVD reconstructs phi", err < 1e-6)
    catch e
        println("    svd threw: ", sprint(showerror, e))
    end

    # ────────────────────────────────────────────────────────────────────────
    # [5] product(PH, phi) sanity at bond 1
    println("\n[5] product(PH_sparse, phi_sparse) Rayleigh-quotient check")
    try
        psi_w = copy(psi_sp); orthogonalize!(psi_w, 1)
        PH = ProjMPO(H_sparse); PH.nsite = 2
        position!(PH, psi_w, 1)
        phi = psi_w[1] * psi_w[2]
        Hphi = product(PH, phi)
        rq = scalar(dag(phi) * Hphi) / scalar(dag(phi) * phi)
        println("    ⟨phi|PH|phi⟩ / ⟨phi|phi⟩ = $rq   (should equal ⟨psi|H_sparse|psi⟩ = $(real(E_sp2)))")
        ok("RQ matches ⟨H_sparse⟩", isapprox(real(rq), real(E_sp2); rtol=1e-6))
    catch e
        println("    threw: ", sprint(showerror, e))
    end
end
nothing
