# test_stable_factorize.jl
#
# End-to-end tests for stable_factorize covering:
#   (A) Exact-S grouping — degenerate SV groups identified the same way
#       across runs, even when phi differs by O(1e-13).
#   (B) Gauge-invariant canonicalization — basis within each degenerate
#       block is fixed by a span-only rule, not by the input gauge.
#
# WORKFLOW:
#   1. Run as-is. Tests with degenerate blocks should FAIL because the
#      current canonicalize_degenerate_block (QR + svd-cleanup) is gauge-dependent.
#   2. Inside stable_factorize, swap the single line:
#          Q, U = canonicalize_degenerate_block(B; atol=atol)
#      for:
#          Q, U = canonicalize_degenerate_block_fixed(B; atol=atol)
#   3. Re-run. Everything passes at machine precision.

using Test
using ITensors
using LinearAlgebra
using Random

function stable_factorize(
    phi::ITensor,
    indsMb;           # already sorted by canonicalize_phi_inds
    ortho               = "left",
    mindim              = nothing,
    maxdim              = nothing,
    cutoff              = nothing,
    svd_alg             = nothing,
    use_absolute_cutoff = nothing,
    use_relative_cutoff = nothing,
    min_blockdim        = nothing,
    tags                = ITensors.ts"Link,l",
    atol                = 1e-12,
    rtol                = 1e-10,
    null_atol           = 1e-10,
    null_rtol           = 1e-6,
    debug               = false,
)
    # ── 1. Direct SVD ─────────────────────────────────────────────────────────
    result = ITensors.svd(
        phi, indsMb;
        mindim,
        maxdim,
        cutoff,
        alg                 = something(svd_alg, "divide_and_conquer"),
        use_absolute_cutoff,
        use_relative_cutoff,
        min_blockdim,
        lefttags            = tags,
        righttags           = tags,
    )
    isnothing(result) && error("stable_factorize: SVD returned nothing")
    U_it, S_it, V_it, spec, u_idx, v_idx = result
    D = dim(u_idx)
    # ── 2. Exact singular values from S diagonal ───────────────────────────────
    svs = [S_it[u_idx => j, v_idx => j] for j in 1:D]
    # ── 3. Fuse non-bond indices → plain matrices ──────────────────────────────
    # U: (physical_left..., u_idx)  → n × D matrix
    # V: (v_idx, physical_right...) → D × m matrix (V† in the decomposition)
    U_phys = filter(i -> i ≠ u_idx && i ≠ dag(u_idx), collect(inds(U_it)))
    V_phys = filter(i -> i ≠ v_idx && i ≠ dag(v_idx), collect(inds(V_it)))
    CU = ITensors.combiner(U_phys...; tags="cU")
    CV = ITensors.combiner(V_phys...; tags="cV")
    cU = ITensors.combinedind(CU)
    cV = ITensors.combinedind(CV)
    Umat  = Array(U_it * CU, cU, u_idx)        # n × D
    Vmat  = Array(V_it * CV, v_idx, cV)         # D × m  (V† rows)
    # ── 4. Canonical channel ordering ─────────────────────────────────────────
    sv_scale  = max(maximum(svs), atol)
    group_tol = max(atol, rtol * sv_scale)
    null_tol  = max(null_atol, null_rtol * sv_scale)
    # Snap degenerate group representatives.
    sv_rep = copy(svs)
    for k in 1:D, j in k+1:D
        abs(svs[j] - sv_rep[k]) <= group_tol && (sv_rep[j] = sv_rep[k])
    end
    _TOPK = 8
    col_fp      = [(sum(abs(x)^3 for x in @view Umat[:, j]))^(1/3) for j in 1:D]
    row_fp      = [(sum(abs(x)^3 for x in @view Vmat[j, :]))^(1/3) for j in 1:D]
    col_profile = [ntuple(k -> k <= size(Umat,1) ? -sort(abs.(@view Umat[:,j]), rev=true)[k] : 0.0, _TOPK) for j in 1:D]
    row_profile = [ntuple(k -> k <= size(Vmat,2) ? -sort(abs.(@view Vmat[j,:]), rev=true)[k] : 0.0, _TOPK) for j in 1:D]
    perm = sort(collect(1:D); by = j -> (
        -sv_rep[j], -col_fp[j], -row_fp[j], col_profile[j], row_profile[j],
    ))
    Umat, Vmat, svs = Umat[:, perm], Vmat[perm, :], svs[perm]
    # ── 4.5. Degenerate-block canonicalization ────────────────────────────────
    # Within each group of equal singular values, LAPACK can produce an arbitrary
    # unitary rotation of the degenerate subspace. The fingerprint-based sort
    # above cannot resolve this — both runs get different rotated bases whose
    # fingerprints differ, so the sort order within the block still differs.
    # Replace the basis of each degenerate block with the deterministic QR-based
    # basis from canonicalize_degenerate_block, then rotate Vmat rows consistently.
    let k = 1
        while k <= D
            g_end = k
            while g_end < D && abs(svs[g_end + 1] - svs[k]) <= group_tol
                g_end += 1
            end
            if g_end > k && svs[k] >= null_tol   # degenerate, non-null block
                B = copy(Umat[:, k:g_end])
                Q, U = canonicalize_degenerate_block_fixed(B; atol=atol)
                Umat[:, k:g_end]  = Q
                Vmat[k:g_end, :] = U * copy(Vmat[k:g_end, :])
                # Q = canonicalize_degenerate_block(B; atol=atol)
                # W = Q' * B                        # unitary rotation within subspace
                # Umat[:, k:g_end] = Q
                # Vmat[k:g_end, :] = W * Vmat[k:g_end, :]
            end
            k = g_end + 1
        end
    end
    # ── 5. Canonical phase per channel (weighted-sum) ──────────────────────────
    for j in 1:D
        if svs[j] < null_tol
            Umat[:, j] .= zero(eltype(Umat))
            Vmat[j, :] .= zero(eltype(Vmat))
            continue
        end
        col     = @view Umat[:, j]
        col_abs = abs.(col)
        col_max = isempty(col_abs) ? zero(real(eltype(col))) : maximum(col_abs)
        if col_max > 10 * atol
            z = zero(eltype(col))
            @inbounds for i in eachindex(col)
                ai = col_abs[i]; ai > atol && (z += ai * col[i])
            end
            if abs(z) > atol
                ph = conj(z) / abs(z)
                Umat[:, j] .*= ph
                Vmat[j, :] .*= conj(ph)
            end
        end
    end
    # ── 6. Reconstruct ITensors ────────────────────────────────────────────────
    # Create a fresh bond index with the canonical ordering.
    u_new = ITensors.Index(D, ITensors.tags(u_idx))
    if ortho == "left"
        # L = U_canonical (isometry), R = diag(S) * V†_canonical (singular tensor)
        L_mat = ITensor(Umat, cU, u_new)
        L_it  = L_mat * dag(CU)                        # unfuse physical inds
        SV    = Diagonal(svs) * Vmat                   # D × m
        R_mat = ITensor(SV, u_new, cV)
        R_it  = R_mat * dag(CV)                        # unfuse physical inds
    else   # ortho == "right"
        # L = U * diag(S) (singular tensor), R = V†_canonical (isometry)
        US    = Umat * Diagonal(svs)                   # n × D
        L_mat = ITensor(US, cU, u_new)
        L_it  = L_mat * dag(CU)
        R_mat = ITensor(Vmat, u_new, cV)
        R_it  = R_mat * dag(CV)
    end
    if debug
        println("stable_factorize: D=$D  svs=$(round.(svs, sigdigits=4))")
    end
    return L_it, R_it, spec
end


# ============================================================
# The replacement canonicalizer.
#
# Returns Q, U with B = Q*U, where Q depends ONLY on span(B),
# not on the input basis. This is what makes the test pass.
#
# Mechanism: M = B' * diag(w) * B is Hermitian and transforms as
# M -> W'·M·W under B -> B·W. eigen(M) gives a basis V tied to
# the span, not the input gauge. Q = B*V is invariant up to
# per-column phase, fixed by the weighted-sum convention.
# ============================================================
function canonicalize_degenerate_block_fixed(
    B::AbstractMatrix; atol = 1e-12, max_attempts = 3,
)
    n, m = size(B)
    m == 0 && return copy(B), Matrix{eltype(B)}(I, 0, 0)
    if m == 1
        col = copy(B[:, 1])
        ph  = _wsum_phase(col; atol = atol)
        col .*= ph
        return reshape(col, n, 1), reshape([conj(ph)], 1, 1)
    end

    T = real(eltype(B))
    for attempt in 1:max_attempts
        w = _make_weight(T, n, attempt)
        M = B' * (w .* B); M = (M + M') / 2

        F  = eigen(Hermitian(M))
        ev = F.values
        ev_scale = max(maximum(abs, ev), one(T))
        gap = m > 1 ? minimum(ev[j] - ev[j-1] for j in 2:m) : ev_scale

        if gap > 1e-10 * ev_scale || attempt == max_attempts
            V = F.vectors
            Q = B * V
            phases = Vector{eltype(B)}(undef, m)
            for j in 1:m
                phases[j] = _wsum_phase(@view(Q[:, j]); atol = atol)
                Q[:, j] .*= phases[j]
            end
            U = Diagonal(conj.(phases)) * V'
            return Matrix(Q), Matrix(U)
        end
    end
end

function _wsum_phase(col; atol = 1e-12)
    z = zero(eltype(col))
    @inbounds for x in col
        ax = abs(x)
        ax > atol && (z += ax * x)
    end
    return abs(z) > atol ? conj(z) / abs(z) : one(eltype(col))
end

function _make_weight(::Type{T}, n, attempt) where {T}
    if attempt == 1
        return T[T(i) + T(i)^2 / T(n + 1) for i in 1:n]
    elseif attempt == 2
        return T[sin(T(i) * T(0.7) + T(0.3)) + T(i) / T(n + 1) for i in 1:n]
    else
        return T[T(i)^T(1.3) + cos(T(i) * T(0.3)) for i in 1:n]
    end
end

# ============================================================
# Helpers
# ============================================================

# Compare L,R produced by two stable_factorize calls. Bond indices
# have different IDs but the same dim+tags, so we extract Arrays
# with explicit index orderings.
function compare_LR(L1, R1, L2, R2, li, ri)
    l1 = commonind(L1, R1)
    l2 = commonind(L2, R2)
    @assert dim(l1) == dim(l2) "bond dimensions differ: $(dim(l1)) vs $(dim(l2))"
    L1m = Array(L1, li, l1); L2m = Array(L2, li, l2)
    R1m = Array(R1, l1, ri); R2m = Array(R2, l2, ri)
    return norm(L1m - L2m), norm(R1m - R2m)
end

function build_phi(svs, L_dim, R_dim; seed = 0)
    seed != 0 && Random.seed!(seed)
    Umat = Matrix(qr(randn(L_dim, L_dim)).Q)
    Vmat = Matrix(qr(randn(R_dim, R_dim)).Q)
    return Umat * Diagonal(svs) * Vmat', Umat, Vmat
end

# ============================================================
# Test (A): perturbation stability with degenerate blocks.
#
# The dominant failure mode in DMRG. eigsolve returns phi vectors
# that differ by O(1e-13) between sparse and dense paths. The
# factorized L and R must drift by at most that much, not by O(1).
# ============================================================
@testset "stable_factorize: perturbation stability (degenerate blocks)" begin
    Random.seed!(42)
    L_dim, R_dim = 6, 6
    svs_target = [3.0, 3.0, 1.5, 1.5, 0.7, 0.2]

    phi_mat, _, _ = build_phi(svs_target, L_dim, R_dim)

    li = Index(L_dim, "left")
    ri = Index(R_dim, "right")
    phi1 = ITensor(phi_mat, li, ri)

    eps = 1e-13
    delta = randn(L_dim, R_dim); delta ./= norm(delta)
    phi2 = ITensor(phi_mat .+ eps .* delta, li, ri)

    L1, R1, _ = stable_factorize(phi1, [li]; ortho = "left")
    L2, R2, _ = stable_factorize(phi2, [li]; ortho = "left")

    # Reconstruction must always hold, regardless of the canonicalizer.
    @test norm(Array(L1 * R1, li, ri) - phi_mat) < 1e-10
    @test norm(Array(L2 * R2, li, ri) - (phi_mat .+ eps .* delta)) < 1e-10

    diffL, diffR = compare_LR(L1, R1, L2, R2, li, ri)
    @info "(A) perturbation: ‖L1-L2‖ = $diffL  ‖R1-R2‖ = $diffR"

    # The actual stability requirement.
    @test diffL < 1e-9
    @test diffR < 1e-9
end

# ============================================================
# Test (B): exact gauge invariance.
#
# phi1 == phi2 (modulo O(1e-15)) but the inputs are presented in
# different bases inside the degenerate blocks. L and R must agree
# to machine precision — the canonicalizer should erase the gauge.
# ============================================================
@testset "stable_factorize: exact gauge invariance (degenerate blocks)" begin
    Random.seed!(123)
    L_dim, R_dim = 6, 6
    svs = [3.0, 3.0, 1.5, 0.7, 0.7, 0.7]

    phi_mat1, Umat, Vmat = build_phi(svs, L_dim, R_dim)

    rot = Matrix{Float64}(I, L_dim, L_dim)
    rot[1:2, 1:2] = Matrix(qr(randn(2, 2)).Q)
    rot[4:6, 4:6] = Matrix(qr(randn(3, 3)).Q)
    phi_mat2 = (Umat * rot) * Diagonal(svs) * (Vmat * rot)'

    @test norm(phi_mat1 - phi_mat2) < 1e-12

    li = Index(L_dim, "left"); ri = Index(R_dim, "right")
    phi1 = ITensor(phi_mat1, li, ri)
    phi2 = ITensor(phi_mat2, li, ri)

    L1, R1, _ = stable_factorize(phi1, [li]; ortho = "left")
    L2, R2, _ = stable_factorize(phi2, [li]; ortho = "left")

    diffL, diffR = compare_LR(L1, R1, L2, R2, li, ri)
    @info "(B) gauge:        ‖L1-L2‖ = $diffL  ‖R1-R2‖ = $diffR"

    @test diffL < 1e-10
    @test diffR < 1e-10
end

# ============================================================
# Test (C): near-degenerate SVs.
#
# Validates the *exact-S grouping* part of stable_factorize.
# SVs differ by < group_tol, so they must be classified into the
# same group across both runs. If grouping uses row-norm estimates
# instead of exact S, summation-order noise can split the group
# differently and produce divergent canonicalization.
# ============================================================
@testset "stable_factorize: near-degenerate SVs (exact-S grouping)" begin
    Random.seed!(7)
    L_dim, R_dim = 6, 6
    # Two pairs that are "near degenerate" — within group_tol but not equal.
    svs_target = [3.0, 3.0 + 1e-13, 1.5, 1.5 - 1e-13, 0.7, 0.2]

    phi_mat, _, _ = build_phi(svs_target, L_dim, R_dim)

    li = Index(L_dim, "left"); ri = Index(R_dim, "right")
    phi1 = ITensor(phi_mat, li, ri)

    eps = 1e-13
    delta = randn(L_dim, R_dim); delta ./= norm(delta)
    phi2 = ITensor(phi_mat .+ eps .* delta, li, ri)

    L1, R1, _ = stable_factorize(phi1, [li]; ortho = "left")
    L2, R2, _ = stable_factorize(phi2, [li]; ortho = "left")

    diffL, diffR = compare_LR(L1, R1, L2, R2, li, ri)
    @info "(C) near-degen:   ‖L1-L2‖ = $diffL  ‖R1-R2‖ = $diffR"

    @test diffL < 1e-9
    @test diffR < 1e-9
end

# ============================================================
# Test (D): non-degenerate baseline.
#
# All SVs well-separated. This should pass with both the OLD and
# NEW canonicalizers, and acts as a sanity check that nothing in
# the test setup itself is broken.
# ============================================================
@testset "stable_factorize: non-degenerate baseline" begin
    Random.seed!(99)
    L_dim, R_dim = 6, 6
    svs_target = [3.0, 2.4, 1.7, 1.1, 0.6, 0.2]

    phi_mat, _, _ = build_phi(svs_target, L_dim, R_dim)

    li = Index(L_dim, "left"); ri = Index(R_dim, "right")
    phi1 = ITensor(phi_mat, li, ri)

    eps = 1e-13
    delta = randn(L_dim, R_dim); delta ./= norm(delta)
    phi2 = ITensor(phi_mat .+ eps .* delta, li, ri)

    L1, R1, _ = stable_factorize(phi1, [li]; ortho = "left")
    L2, R2, _ = stable_factorize(phi2, [li]; ortho = "left")

    diffL, diffR = compare_LR(L1, R1, L2, R2, li, ri)
    @info "(D) non-degen:    ‖L1-L2‖ = $diffL  ‖R1-R2‖ = $diffR"

    @test diffL < 1e-10
    @test diffR < 1e-10
end

# ============================================================
# Test (E): ortho="right" path.
#
# Same perturbation test as (A) but with the singular tensor on
# the left. Ensures both branches of the reconstruction in
# stable_factorize are covered.
# ============================================================
@testset "stable_factorize: ortho=right perturbation stability" begin
    Random.seed!(42)
    L_dim, R_dim = 6, 6
    svs_target = [3.0, 3.0, 1.5, 1.5, 0.7, 0.2]

    phi_mat, _, _ = build_phi(svs_target, L_dim, R_dim)

    li = Index(L_dim, "left"); ri = Index(R_dim, "right")
    phi1 = ITensor(phi_mat, li, ri)

    eps = 1e-13
    delta = randn(L_dim, R_dim); delta ./= norm(delta)
    phi2 = ITensor(phi_mat .+ eps .* delta, li, ri)

    L1, R1, _ = stable_factorize(phi1, [li]; ortho = "right")
    L2, R2, _ = stable_factorize(phi2, [li]; ortho = "right")

    diffL, diffR = compare_LR(L1, R1, L2, R2, li, ri)
    @info "(E) ortho=right:  ‖L1-L2‖ = $diffL  ‖R1-R2‖ = $diffR"

    @test diffL < 1e-9
    @test diffR < 1e-9
end