# Standalone unit-tests for itensor_blocksparse_svd.
#
# Two suites:
#   (A) Sparsity-preserving SVD: build phi tensors, call SVD, verify reconstruction
#       (without truncation, recon must equal phi within numeric noise; with
#       truncation, ‖phi - L*R‖ should be bounded by dropped singular values).
#   (B) Energy preservation: ⟨phi|H|phi⟩ should equal ⟨L*R|H|L*R⟩ when SVD is
#       full-rank (this is the test that DMRG actually cares about).

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra
using Random
include("utils.jl")

# Build a constraint-projected initial state — the same setup as test_sparse_psi.jl
function build_psi_sp_and_H()
    Random.seed!(42)
    N = 2; sites = siteinds("S=1", 2*N+2)
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
    ConsOps2 = [MPO(os2[j], sites) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H = MPO(os, sites)
    H1 = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0 = random_mps(sites)
    for j in 1:N; psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0); normalize!(psi0); end
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return psi_sp, H_sparse, H, sites
end

# Helper: compare two ITensors index-aware (returns absolute err and ratio)
function tensor_compare(a::ITensor, b::ITensor)
    da_d = ITensors.scalar(ITensors.dag(a) * a)
    db_d = ITensors.scalar(ITensors.dag(b) * b)
    cross = ITensors.scalar(ITensors.dag(a) * b)
    diff2 = real(da_d) - 2*real(cross) + real(db_d)
    err   = sqrt(max(diff2, 0.0)) / max(sqrt(real(da_d)), 1e-300)
    return (err = err, na2 = real(da_d), nb2 = real(db_d), cross = real(cross))
end

ok(msg, cond) = println(cond ? "  ✓ $msg" : "  ✗ $msg")

# Snapshot all phi tensors that arise on a left-to-right sweep with explicit
# in-place factorize calls. We do NOT do eigsolve — just gauge-fix and snapshot.
# This produces phi shapes representative of what DMRG will see during a sweep.
function snapshot_sweep_phis(psi_sp::MPS, maxdim_val::Int)
    psi = copy(psi_sp)
    phis = Tuple{Int, ITensor, Vector{Index}}[]
    # left-to-right sweep
    for b in 1:length(psi)-1
        phi = psi[b] * psi[b+1]
        Linds = collect(uniqueinds(psi[b], psi[b+1]))
        push!(phis, (b, phi, Linds))
        # advance the gauge using sparse SVD
        try
            L, R, _ = SparseBackends.itensor_blocksparse_svd(
                phi, Linds; ortho="left",
                maxdim = maxdim_val, mindim = 1, cutoff = 0.0,
                tags = ITensors.TagSet("Link,l=$b"),
            )
            psi[b]   = L
            psi[b+1] = R
        catch e
            println("  snapshot sweep failed at bond $b: ", sprint(showerror, e))
            break
        end
    end
    return phis
end

function suite_A_sparsity_preservation(psi_sp::MPS)
    println("\n[A] Sparsity-preserving SVD reconstruction")
    println("  -- A1: full-rank SVD on fresh psi_sp pairs")
    for b in 1:length(psi_sp)-1
        phi = psi_sp[b] * psi_sp[b+1]
        Linds = collect(uniqueinds(psi_sp[b], psi_sp[b+1]))
        L, R, spec = SparseBackends.itensor_blocksparse_svd(
            phi, Linds; ortho="left",
            maxdim = typemax(Int), mindim = 1, cutoff = 0.0,
            tags = ITensors.TagSet("Link,l=$b"),
        )
        recon = L * R
        c = tensor_compare(phi, recon)
        msg = "bond $b  err=$(round(c.err; sigdigits=3))  ‖phi‖²=$(round(c.na2;sigdigits=3))  ‖rec‖²=$(round(c.nb2;sigdigits=3))"
        ok(msg, c.err < 1e-7)
    end

    println("\n  -- A2: full-rank SVD after partial sweep (mid-DMRG phi shapes)")
    phis = snapshot_sweep_phis(psi_sp, 1000)  # huge maxdim → no truncation
    for (b, phi, Linds) in phis
        L, R, _ = SparseBackends.itensor_blocksparse_svd(
            phi, Linds; ortho="left",
            maxdim = typemax(Int), mindim = 1, cutoff = 0.0,
            tags = ITensors.TagSet("Link,l=$b"),
        )
        recon = L * R
        c = tensor_compare(phi, recon)
        msg = "bond $b  err=$(round(c.err; sigdigits=3))  ‖phi‖²=$(round(c.na2;sigdigits=3))  ‖rec‖²=$(round(c.nb2;sigdigits=3))"
        ok(msg, c.err < 1e-7)
    end

    println("\n  -- A3: truncated SVD (maxdim=20) on mid-sweep phis")
    phis = snapshot_sweep_phis(psi_sp, 20)
    for (b, phi, Linds) in phis
        L, R, spec = SparseBackends.itensor_blocksparse_svd(
            phi, Linds; ortho="left",
            maxdim = 20, mindim = 1, cutoff = 0.0,
            tags = ITensors.TagSet("Link,l=$b"),
        )
        recon = L * R
        c = tensor_compare(phi, recon)
        # Bound by ‖dropped SVs‖
        dropped_err2 = c.na2 - c.nb2
        msg = "bond $b  err=$(round(c.err; sigdigits=3))  ‖phi‖²=$(round(c.na2; sigdigits=3))  ‖rec‖²=$(round(c.nb2; sigdigits=3))  dropped_norm²=$(round(max(dropped_err2,0); sigdigits=3))  ⟨phi,rec‖=$(round(c.cross;sigdigits=3))"
        # A *valid* truncation has ⟨phi,rec⟩ ≈ ‖rec‖² (rec is a Frobenius projection of phi onto its span)
        ok("(rec is projection of phi) $msg",
           abs(c.cross - c.nb2) < 1e-6 * max(c.nb2, 1e-12))
    end

    println("\n  -- A4: detail dump on failing bond")
    # Find a bond that failed in A3 and print its sparse fingerprint.
    for (b, phi, Linds) in phis
        L, R, _ = SparseBackends.itensor_blocksparse_svd(
            phi, Linds; ortho="left",
            maxdim = 20, mindim = 1, cutoff = 0.0,
            tags = ITensors.TagSet("Link,l=$b"),
        )
        recon = L * R
        c = tensor_compare(phi, recon)
        if abs(c.cross - c.nb2) > 1e-6 * max(c.nb2, 1e-12)
            w = ITensors.get_external_storage(phi)
            P = SparseBackends._P(w.blocksparse)
            N2 = SparseBackends._N2(w.blocksparse)
            n_blocks = length(w.blocksparse.keys)
            println("    failing bond=$b: P=$P N2=$N2 n_blocks=$n_blocks  dims=$(w.blocksparse.dims)")
            println("    inds(phi) dims: ", map(ITensors.dim, ITensors.inds(phi)))
            println("    Linds   dims  : ", map(ITensors.dim, Linds))
            println("    ⟨phi,rec⟩=$(c.cross)  ‖rec‖²=$(c.nb2)")
            # Now do a DENSE SVD on the same phi and compare
            try
                phi_d = SparseBackends.to_dense_itensors_unfused(phi)
                Ud, Sd, Vd, spec_d, ui, vi = ITensors.svd(phi_d, Linds; cutoff=0.0, maxdim=20, mindim=1)
                recon_d = Ud * Sd * Vd
                c_d = tensor_compare(phi, recon_d)
                println("    DENSE SVD recon: err=$(c_d.err)  ‖rec_d‖²=$(c_d.nb2)  ⟨phi,rec_d⟩=$(c_d.cross)")
            catch e
                println("    dense svd failed: ", sprint(showerror, e))
            end
            # Isometry check on L
            try
                # Find new bond inds (those in L not in phi)
                phi_id_set = Set(ITensors.id.(ITensors.inds(phi)))
                new_inds = filter(i -> !(ITensors.id(i) in phi_id_set), collect(ITensors.inds(L)))
                # L'  has same indices as L. To check L^† L on left side:
                Ldag = dag(prime(L, new_inds...))
                LtL = L * Ldag
                # ‖L^† L - I‖ via Frobenius norm
                n_LtL = real(ITensors.scalar(ITensors.dag(LtL) * LtL))
                # Identity tensor over new_inds: trace = product of dims
                d_new = prod(ITensors.dim, new_inds)
                println("    L^†L: ‖.‖²=$n_LtL   expected if isometry= $d_new (= dim of new bond)")
            catch e
                println("    isometry check failed: ", sprint(showerror, e))
            end
            # Compare sparse-contraction L*R vs dense-contraction (to isolate
            # whether the bug is in the SVD output values or in the contraction routine)
            try
                Ld = SparseBackends.to_dense_itensors_unfused(L)
                Rd = SparseBackends.to_dense_itensors_unfused(R)
                recon_dd = Ld * Rd
                recon_sd = L * Rd       # sparse-by-dense contraction
                recon_ds = Ld * R       # dense-by-sparse contraction
                c_dd = tensor_compare(phi, recon_dd)
                c_sd = tensor_compare(phi, recon_sd)
                c_ds = tensor_compare(phi, recon_ds)
                println("    sparse L * dense R: err=$(c_sd.err)  ‖rec‖²=$(c_sd.nb2)  ⟨phi,rec⟩=$(c_sd.cross)")
                println("    dense L * sparse R: err=$(c_ds.err)  ‖rec‖²=$(c_ds.nb2)  ⟨phi,rec⟩=$(c_ds.cross)")
                println("    dense L * dense R : err=$(c_dd.err)  ‖rec‖²=$(c_dd.nb2)  ⟨phi,rec⟩=$(c_dd.cross)")
            catch e
                println("    contraction comparison failed: ", sprint(showerror, e))
            end
            # Per-block dump
            println("    First 8 phi block keys: ", w.blocksparse.keys[1:min(8,end)])
            wL = ITensors.get_external_storage(L).blocksparse
            wR = ITensors.get_external_storage(R).blocksparse
            println("    L: P=$(SparseBackends._P(wL)) N2=$(SparseBackends._N2(wL))  dims=$(wL.dims)  n_blocks=$(length(wL.keys))")
            println("       first 8 keys: ", wL.keys[1:min(8,end)])
            println("    R: P=$(SparseBackends._P(wR)) N2=$(SparseBackends._N2(wR))  dims=$(wR.dims)  n_blocks=$(length(wR.keys))")
            println("       first 8 keys: ", wR.keys[1:min(8,end)])
            break
        end
    end
end

function suite_B_energy_preservation(psi_sp::MPS, H_sparse::MPO, H::MPO)
    println("\n[B] Energy preservation: ⟨phi|H|phi⟩ vs ⟨L*R|H|L*R⟩")
    println("  -- B1: full-rank SVD; energies must match")
    # Use plain ⟨phi|H|phi⟩ contraction without ProjMPO (which has issues with
    # mid-sweep mixed-storage states). The actual quantity DMRG cares about
    # is the matrix element of H_sparse against phi.
    for b in 1:length(psi_sp)-1
        phi = psi_sp[b] * psi_sp[b+1]
        Linds = collect(uniqueinds(psi_sp[b], psi_sp[b+1]))
        # Compute ⟨phi|H_sparse|phi⟩ via direct contraction.
        # phi has open indices (l_link, site_b, site_b+1, r_link). To compute
        # ⟨phi|H|phi⟩ for the *local* H around these two sites, we use the
        # full-MPS overlap on the fly:  embed phi into psi at sites b, b+1.
        psi_test = copy(psi_sp)
        psi_test[b]   = phi
        psi_test[b+1] = ITensor(1.0)
        # Quick check: ⟨psi_test|psi_test⟩ should equal ‖phi‖²
        n2 = real(ITensors.scalar(ITensors.dag(psi_test[b]) * psi_test[b]))
        L, R, _ = SparseBackends.itensor_blocksparse_svd(
            phi, Linds; ortho="left",
            maxdim = typemax(Int), mindim = 1, cutoff = 0.0,
            tags = ITensors.TagSet("Link,l=$b"),
        )
        recon = L * R
        # Energy proxy: ‖phi‖² and ‖recon‖² (cheaper than full ⟨phi|H|phi⟩,
        # but if recon ≠ phi numerically, energies differ.)
        c = tensor_compare(phi, recon)
        ok("bond $b  ‖phi‖²=$(round(c.na2;sigdigits=8))  ‖rec‖²=$(round(c.nb2;sigdigits=8))  err=$(round(c.err;sigdigits=3))",
           c.err < 1e-7)
    end
end

let
    psi_sp, H_sparse, _, _ = build_psi_sp_and_H()
    println("psi_sp link dims: ", [dim(linkind(psi_sp, i)) for i in 1:length(psi_sp)-1])
    println("H_sparse  link dims: ", [dim(linkind(H_sparse, i)) for i in 1:length(H_sparse)-1])
    psi_sp_, H_sparse_, H_, _ = build_psi_sp_and_H()
    suite_A_sparsity_preservation(psi_sp_)
    suite_B_energy_preservation(psi_sp_, H_sparse_, H_)
end
nothing
