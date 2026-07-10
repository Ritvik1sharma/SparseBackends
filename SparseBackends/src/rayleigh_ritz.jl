# rayleigh_ritz.jl — Rayleigh-Ritz local eigensolve for aliased Path-B DMRG.
#
# Split out of path_b_helpers.jl (2026-07) to keep the RR machinery in one place,
# separate from the M^{±1/2} / apply_minv helpers. `include`d into the
# SparseBackends module after path_b_helpers.jl, so all module-level names
# (ITensors, LinearAlgebra, SparseBackends.TIMER, to_dense_itensors_unfused,
# WrappedAliasedBlockSparse, …) are in scope. Public entry:
# `rayleigh_ritz_local_eigsolve`; helpers `solve_small_geneig`, `_rr_select_index`.
# The dmrg-side sweep wiring lives in ITensorMPS.jl/src/rayleigh_ritz_sweep.jl.
#
# ------------------------------------------------------------------
# Generalized Rayleigh-Ritz local eigensolve (BMF_RAYLEIGH_RITZ path).
#
# Solves the local generalized problem  H_eff·φ = E·M·φ  WITHOUT ever applying
# M^{±1/2} to a vector (the operation that discards aliasing in the B_op/A_op
# paths). It projects onto a small aliased Krylov subspace built only from H·v,
# forms tiny k×k matrices H_small / M_small via SCALAR inner products (M applied
# with the RAW Lgram/Rgram — no square root, no BMF_MINV_RTOL pseudoinverse), and
# solves the k×k generalized eig densely with a per-block null projection. The
# Ritz vector φ_new = Σ cᵢ vᵢ is an aliased linear combo of φ-schema vectors, so
# it stays aliased (combos of same-(P,N2)-schema aliased tensors never densify).
# ------------------------------------------------------------------

# Pick the eigenvalue index matching KrylovKit's `which` selector. DMRG ground
# state uses :SR (smallest real) → most-negative algebraic eigenvalue.
function _rr_select_index(vals, which::Symbol)
    rv = real.(vals)
    if which in (:LR, :LA, :largest, :LM)
        return argmax(rv)
    else                      # :SR, :SA, :smallest, default
        return argmin(rv)
    end
end

# Solve H_small c = λ M_small c, k×k, with M_small symmetric PSD but possibly
# rank-deficient (M is structurally rank-deficient for aliased ψ). Project out
# M_small's near-null directions (per-block analog of the per-side pseudoinverse),
# whiten, solve the reduced standard symmetric eig, and recover c in the original
# basis (already M-normalized: cᵀ·M_small·c = 1).
function solve_small_geneig(Hs::AbstractMatrix, Ms::AbstractMatrix, which::Symbol; rtol::Real=1e-8)
    k = size(Hs, 1)
    Hsym = LinearAlgebra.Hermitian((Hs + Hs') / 2)
    Msym = LinearAlgebra.Hermitian((Ms + Ms') / 2)
    Fm = LinearAlgebra.eigen(Msym)            # ascending eigenvalues
    mu = Fm.values
    U  = Fm.vectors
    mumax = isempty(mu) ? 0.0 : maximum(mu)
    if mumax <= 0                              # degenerate M_small → plain eig of Hs
        Fh = LinearAlgebra.eigen(Hsym)
        sel = _rr_select_index(Fh.values, which)
        return (real(Fh.values[sel]), Fh.vectors[:, sel])
    end
    keep = findall(>(rtol * mumax), mu)
    UK = U[:, keep]
    invsqrt = LinearAlgebra.Diagonal(1 ./ sqrt.(mu[keep]))
    B = invsqrt * (UK' * (Matrix(Hsym) * UK)) * invsqrt
    B = LinearAlgebra.Hermitian((B + B') / 2)
    Fb = LinearAlgebra.eigen(B)
    sel = _rr_select_index(Fb.values, which)
    lam = real(Fb.values[sel])
    c = UK * (invsqrt * Fb.vectors[:, sel])
    return (lam, c)
end

# Driver. `Hop` is the H_eff apply closure (built in dmrg.jl as
# v -> recast_to_phi(product(PH, v)) — must NOT be built here: SparseBackends
# does not depend on ITensorMPS). `phi` is the current local tensor (the schema
# template + starting vector). Lgram/Rgram are the raw bond grams. Returns
# (vals, vecs) matching the B_op/A_op contract: vals[1] real, vecs[1] aliased.
function rayleigh_ritz_local_eigsolve(Hop::Function, phi::ITensors.ITensor,
        Lgram::ITensors.ITensor, Rgram::ITensors.ITensor;
        which::Symbol = :SR, tol::Real = 1e-12,
        krylovdim::Int = 8, maxiter::Int = 100,
        rtol::Real = 1e-8, b::Int = 0, ha::Int = 0, sw::Int = 0)
 @timeit SparseBackends.TIMER "rayleigh_ritz" begin
    # M·v via RAW gram (no M^{1/2}). The result is consumed only by a scalar
    # `inner`, so any transient densification here does not enter the basis.
    Mop = v -> apply_minv_preserve_bs(Lgram, apply_minv_preserve_bs(Rgram, v, phi), phi)
    _ip(a, c) = real(ITensors.inner(a, c))
    _nrm(a) = sqrt(max(_ip(a, a), 0.0))
    orth_tol = 1e-12
    # RR-iteration debug print, disabled; flip to `true` (and restore the check
    # below) to re-enable. Note: this whole Rayleigh-Ritz eigensolve is parked
    # (never called in production, ~2.5x slower than the default B_op path).
    dbg = false
    kdim = max(krylovdim, 2)

    n0 = _nrm(phi)
    n0 == 0 && return ([0.0], [phi])
    x = (1.0 / n0) * phi
    V = ITensors.ITensor[x]
    lam = 0.0
    lam_prev = Inf

    for outer_it in 1:maxiter
        # ── grow block to kdim with DGKS double re-orthogonalization (standard
        #    inner product — keeps φ-schema combos aliased; M handled in the
        #    k×k solve) ──
        while length(V) < kdim
            w = Hop(V[end])
            for _pass in 1:2, u in V
                w = w - _ip(u, w) * u
            end
            nw = _nrm(w)
            nw < orth_tol && break              # breakdown → block complete
            push!(V, (1.0 / nw) * w)
        end
        k = length(V)

        # ── small matrices: H_small / M_small via scalar inner products ──
        HV = [Hop(V[j]) for j in 1:k]
        MV = [Mop(V[j]) for j in 1:k]
        Hs = Array{Float64}(undef, k, k)
        Ms = Array{Float64}(undef, k, k)
        for i in 1:k, j in 1:k
            Hs[i, j] = _ip(V[i], HV[j])
            Ms[i, j] = _ip(V[i], MV[j])
        end

        lam, c = solve_small_geneig(Hs, Ms, which; rtol=rtol)

        # ── Ritz vector + its H/M images via the SAME coefficients ──
        x  = c[1] * V[1]
        Hx = c[1] * HV[1]
        Mx = c[1] * MV[1]
        for j in 2:k
            x  = x  + c[j] * V[j]
            Hx = Hx + c[j] * HV[j]
            Mx = Mx + c[j] * MV[j]
        end

        # ── generalized residual r = H x - λ M x ──
        r = Hx - lam * Mx
        rnorm = _nrm(r)
        # if dbg
        #     println("[RR b=$b ha=$ha sw=$sw] it=$outer_it k=$k lam=$lam rnorm=$rnorm")
        #     flush(stdout)
        # end
        if rnorm < tol || abs(lam - lam_prev) < tol
            return ([lam], [x])
        end
        lam_prev = lam

        # ── thick restart: new basis = {Ritz vector, residual direction} ──
        rr = r - _ip(x, r) * x
        nrr = _nrm(rr)
        V = nrr < orth_tol ? ITensors.ITensor[x] :
                             ITensors.ITensor[x, (1.0 / nrr) * rr]
    end
    return ([isfinite(lam_prev) ? lam_prev : lam], [x])
 end
end
