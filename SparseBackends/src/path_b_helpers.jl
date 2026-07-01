# Path-B helpers for sparse DMRG: build the Gram environments L/R from a
# non-canonical sparse MPS, assemble the small bond-leg Minv, and detect
# sparse-MPS storage. Kept here so ITensorMPS depends only on a public API,
# not on internals.

import ITensors
import LinearAlgebra
import LinearAlgebra: pinv
using TimerOutputs: @timeit

# ------------------------------------------------------------------
# Incremental Gram-environment cache.
#
# For Path-B DMRG we need:
#   Lgram[b] = ∑ over sites 1..b-1 of ψ[1..b-1] · dag(ψ[1..b-1])'
#   Rgram[b] = ∑ over sites b+2..N of ψ[b+2..N] · dag(ψ[b+2..N])'
# Naively rebuilt from scratch each bond, this is O(N) tensor contractions
# per bond × O(N) bonds per sweep = O(N²) per sweep — the dominant cost at
# N=4 maxdim=40 (75.7% of sweep time).
#
# The cache maintains L[i] = gram of ψ[1..i-1] and R[i] = gram of ψ[i..N]
# for all i = 1..N+1. A sweep step b→b+1 in the forward direction touches
# only L[b+1] (one contraction). Backward sweep b→b-1 touches only R[b+1].
# Direction switches do not require rebuilding (the cache stays valid for
# the side that was being updated, the other side stayed valid throughout).
# Net cost: O(N) contractions per sweep (down from O(N²)).
mutable struct GramCache
    L::Vector{ITensors.ITensor}
    R::Vector{ITensors.ITensor}
end

# Helper: dag(T) with all Link indices primed (the convention used in the
# gram contractions).
function _dag_link_primed(T::ITensors.ITensor)
    Td = ITensors.dag(T)
    for I in ITensors.inds(T)
        if ITensors.hastags(I, "Link")
            Td = ITensors.prime(Td, I)
        end
    end
    return Td
end

# Build both L and R caches from scratch. Called once at the start of DMRG.
# Cost: 2N contractions (one full pass left-to-right + one right-to-left).
function init_gram_cache(psi)
    N = length(psi)
    L = Vector{ITensors.ITensor}(undef, N + 1)
    R = Vector{ITensors.ITensor}(undef, N + 1)
    L[1]     = ITensors.ITensor(1.0)
    R[N + 1] = ITensors.ITensor(1.0)
    @inbounds for i in 1:N
        L[i + 1] = L[i] * psi[i] * _dag_link_primed(psi[i])
    end
    @inbounds for i in N:-1:1
        R[i] = R[i + 1] * psi[i] * _dag_link_primed(psi[i])
    end
    return GramCache(L, R)
end

# Update L[i+1] after psi[i] has changed. Called after replacebond! at b
# in the forward direction (psi[b] is now the newly-fixed left tensor).
function update_left!(cache::GramCache, psi, i::Int)
    @inbounds cache.L[i + 1] = cache.L[i] * psi[i] * _dag_link_primed(psi[i])
    return cache
end

# Update R[i] after psi[i] has changed. Called after replacebond! at b
# in the backward direction (psi[b+1] is now the newly-fixed right tensor;
# pass i=b+1).
function update_right!(cache::GramCache, psi, i::Int)
    @inbounds cache.R[i] = cache.R[i + 1] * psi[i] * _dag_link_primed(psi[i])
    return cache
end

# Retrieve gram envs for bond b. Lgram at b = gram of ψ[1..b-1] = cache.L[b].
# Rgram at b = gram of ψ[b+2..N] = cache.R[b+2].
get_left_gram(cache::GramCache, b::Int)  = @inbounds cache.L[b]
get_right_gram(cache::GramCache, b::Int) = @inbounds cache.R[b + 2]

# Is this MPS using sparse storage? Used to gate Path-B code paths in DMRG.
function is_sparse_mps(psi)::Bool
  length(psi) == 0 && return false
  T = psi[1]
  ITensors.has_external_storage(T) || return false
  s = ITensors.get_external_storage(T)
  return s isa WrappedBlockSparse || s isa WrappedAliasedBlockSparse
end

# Single-side gram → Mhalf, Linv as BS ITensors restricted to phi's allowed
# chan-values on the appropriate side. Returns nothing if phi is not BS or
# if the channel structure can't be auto-detected.
#
# Idea: per-side Linv (Lgram or Rgram) lives on ONE bond. Phi has keys
# parameterized by (chan_l, chan_r, sites..., mult). The set of allowed
# chan_l values on the LEFT side = unique c_l values in phi's keys. For
# Lgram-derived Linv, we restrict to (c_unp, c_prm) pairs where both ∈
# this allowed set. Same logic for Rgram on the right side.
#
# When Linv is stored with the restricted key set, `Linv_BS * phi_BS` produces
# output keys that match phi's structure → no recast needed → big perf win.
function build_minv_half_pair_bs_side(G::ITensors.ITensor,
                                      phi_template::ITensors.ITensor,
                                      side::Symbol;  # :left or :right
                                      rtol::Real=1e-10)
  @assert side in (:left, :right) "side must be :left or :right"
  G_inds = collect(ITensors.inds(G))
  if isempty(G_inds)
    return ITensors.ITensor(1.0), ITensors.ITensor(1.0)
  end
  if !ITensors.has_external_storage(phi_template) ||
     !(ITensors.get_external_storage(phi_template) isa WrappedBlockSparse)
    # phi must be BS to extract allowed chan values
    return build_half_pair_single(G; rtol)
  end
  pw = ITensors.get_external_storage(phi_template)
  phi_bs = pw.blocksparse
  phi_inds_all = pw.inds

  # Detect chan vs mult inds on G's bond (using phi's dense_inds classification).
  unp = filter(I -> ITensors.plev(I) == 0, G_inds)
  prm = filter(I -> ITensors.plev(I) == 1, G_inds)
  @assert length(unp) == length(prm)
  phi_dense_set = Set(collect(dense_inds(pw)))
  chan_unp_list = filter(I -> !(I in phi_dense_set), unp)
  mult_unp_list = filter(I -> (I in phi_dense_set),  unp)
  if isempty(chan_unp_list)
    # No channel axes — single-axis case. Fall back to dense (no restriction needed).
    return build_half_pair_single(G; rtol)
  end

  function _primed_of(I::ITensors.Index, all_prm::Vector)
    for J in all_prm
      ITensors.id(I) == ITensors.id(J) && return J
    end
    error("No primed counterpart found for $I")
  end
  chan_prm_list = [_primed_of(I, prm) for I in chan_unp_list]
  mult_prm_list = [_primed_of(I, prm) for I in mult_unp_list]
  Pc = length(chan_unp_list)

  # Find phi's stored chan values for THIS side.
  # phi.inds order: [sparse_link..., sparse_nonlink..., dense_tail].
  # phi's keys are NTuple{P, Int} where P = # sparse axes. The first
  # entries correspond to sparse_link Indices (= channel axes), in phi.inds order.
  # We need to locate which entries of phi's keys correspond to G's chan axes.
  phi_key_idx_of = Dict{ITensors.Index, Int}()
  let key_idx = 0
    for (i, I) in enumerate(phi_inds_all)
      if !(I in phi_dense_set)
        key_idx += 1
        phi_key_idx_of[I] = key_idx
      end
    end
  end
  # For each G chan-axis, find its position in phi's keys.
  G_chan_pos_in_phi = Int[]
  for I in chan_unp_list
    haskey(phi_key_idx_of, I) || begin
      # Channel Index of G not present in phi → can't restrict, fall back to full BS.
      return build_minv_half_pair_bs(G; rtol, phi_template)
    end
    push!(G_chan_pos_in_phi, phi_key_idx_of[I])
  end

  # Compute EQUIVALENCE CLASSES on this-side chan values:
  #   c_u ~ c_p iff they share the SAME set of opposite-side chan-values in phi's keys.
  # For Lgram (side=:left): c_l_u ~ c_l_p iff they pair with the same set of c_r values.
  # For Rgram (side=:right): c_r_u ~ c_r_p iff they pair with the same set of c_l values.
  # Linv ends up block-diagonal in these equivalence classes → restricted key set
  # is intra-class only. Output of Linv_BS · phi_BS keeps phi's exact key set.
  #
  # Position of "this side" chan tuple in phi's keys:
  this_side_positions = G_chan_pos_in_phi
  # Position of "other side" chan(s) in phi's keys: all sparse_link positions of
  # phi that are NOT this side's.
  other_side_positions = Int[]
  for (idx, I) in enumerate(phi_inds_all)
    if !(I in phi_dense_set) && ITensors.hastags(I, "Link") && !(I in chan_unp_list)
      push!(other_side_positions, phi_key_idx_of[I])
    end
  end

  # For each this-side chan tuple t, collect the SET of other-side chan tuples
  # it pairs with in phi's keys.
  pair_partners = Dict{NTuple{Pc,Int}, Set{NTuple{length(other_side_positions),Int}}}()
  for k in phi_bs.keys
    t = ntuple(i -> k[this_side_positions[i]], Pc)
    o = ntuple(i -> k[other_side_positions[i]], length(other_side_positions))
    !haskey(pair_partners, t) && (pair_partners[t] = Set{NTuple{length(other_side_positions),Int}}())
    push!(pair_partners[t], o)
  end
  # Group this-side tuples by identical partner sets → equivalence classes.
  classes_dict = Dict{Set{NTuple{length(other_side_positions),Int}}, Vector{NTuple{Pc,Int}}}()
  for (t, partners) in pair_partners
    !haskey(classes_dict, partners) && (classes_dict[partners] = NTuple{Pc,Int}[])
    push!(classes_dict[partners], t)
  end
  classes = collect(values(classes_dict))
  for c in classes; sort!(c); end
  sort!(classes; by = c -> c[1])

  # Densify G and extract per-class sub-matrices for eigen.
  G_d = ITensors.has_external_storage(G) ? to_dense_itensors_unfused(G) : G
  G_arr = Array(G_d, chan_unp_list..., chan_prm_list..., mult_unp_list..., mult_prm_list...)
  c_total = prod(I -> ITensors.dim(I), chan_unp_list; init=1)
  m_total = prod(I -> ITensors.dim(I), mult_unp_list; init=1)
  G_4d = reshape(G_arr, c_total, c_total, m_total, m_total)

  TC = eltype(G_arr)
  chan_dims_unp = ntuple(i -> ITensors.dim(chan_unp_list[i]), Pc)
  function _chan_tup_to_lin(t::NTuple{P,Int}, dims::NTuple{P,Int}) where {P}
    lin = 0; stride = 1
    @inbounds for i in 1:P
      lin += (t[i] - 1) * stride
      stride *= dims[i]
    end
    return lin + 1
  end

  # Build per-class Mhalf, Linv via eigen on each class's sub-matrix.
  # Then collect ALL keys (intra-class only) and pack into BS storage.
  keys_vec   = Vector{NTuple{2*Pc,Int}}()
  Linv_data  = TC[]
  Mhalf_data = TC[]
  block_payload_size = m_total^2

  for class in classes
    nC = length(class)
    class_lin = [_chan_tup_to_lin(t, chan_dims_unp) for t in class]
    d_C = nC * m_total
    G_C = zeros(TC, d_C, d_C)
    @inbounds for i in 1:nC, j in 1:nC
      ci = class_lin[i]; cj = class_lin[j]
      G_C[(i-1)*m_total+1 : i*m_total, (j-1)*m_total+1 : j*m_total] .=
        view(G_4d, ci, cj, :, :)
    end
    G_C = (G_C + G_C') / 2
    F = LinearAlgebra.eigen(LinearAlgebra.Hermitian(G_C))
    λ = F.values; V = F.vectors
    maxλ = isempty(λ) ? zero(real(TC)) : maximum(real, λ)
    tol  = real(rtol * maxλ)
    sqrt_λ     = [real(l) > tol ? sqrt(real(l))     : zero(real(TC)) for l in λ]
    inv_sqrt_λ = [real(l) > tol ? 1 / sqrt(real(l)) : zero(real(TC)) for l in λ]
    Mhalf_C = V * LinearAlgebra.Diagonal(sqrt_λ)     * V'
    Linv_C  = V * LinearAlgebra.Diagonal(inv_sqrt_λ) * V'
    Mhalf_C = (Mhalf_C + Mhalf_C') / 2
    Linv_C  = (Linv_C  + Linv_C')  / 2
    # Scatter back: keys are (c_u_tup..., c_p_tup...) for c_u, c_p ∈ class.
    @inbounds for i in 1:nC, j in 1:nC
      c1_tup = class[i]; c2_tup = class[j]
      rblk = (i-1)*m_total+1 : i*m_total
      cblk = (j-1)*m_total+1 : j*m_total
      Linv_block_mat  = Linv_C[rblk, cblk]
      Mhalf_block_mat = Mhalf_C[rblk, cblk]
      push!(keys_vec, (c1_tup..., c2_tup...))
      for k in 1:block_payload_size
        push!(Linv_data,  Linv_block_mat[k])
        push!(Mhalf_data, Mhalf_block_mat[k])
      end
    end
  end

  ids_vec = collect(1:length(keys_vec))
  Linv_inds  = (chan_unp_list..., chan_prm_list..., mult_unp_list..., mult_prm_list...)
  N_total    = length(Linv_inds)
  N2_dense   = N_total - 2 * Pc
  dims_full  = ntuple(i -> ITensors.dim(Linv_inds[i]), N_total)

  Linv_bs = NewBlockSparseSorted{TC, N_total, N2_dense, 2*Pc, Int}(
    dims_full, block_payload_size, keys_vec, copy(ids_vec), Linv_data
  )
  Mhalf_bs = NewBlockSparseSorted{TC, N_total, N2_dense, 2*Pc, Int}(
    dims_full, block_payload_size, copy(keys_vec), copy(ids_vec), Mhalf_data
  )
  Linv_it  = ITensors._itensor_from_external_storage(
    WrappedBlockSparse{TC, N_total, N2_dense, 2*Pc}(Linv_bs,  Linv_inds)
  )
  Mhalf_it = ITensors._itensor_from_external_storage(
    WrappedBlockSparse{TC, N_total, N2_dense, 2*Pc}(Mhalf_bs, Linv_inds)
  )
  return Mhalf_it, Linv_it
end

# Single-side gram → Mhalf, Linv as small dense ITensors.
# Given a 2-side gram tensor G with paired plev=0/plev=1 Link inds (e.g.
# Lgram from psi[1..b-1] or Rgram from psi[b+2..N]), eigen-decompose it as
# a small dense matrix on (chan × mult) and produce G^(1/2), G^(-1/2) as
# dense ITensors with the same Index layout.
#
# Used by `build_minv_half_pair_factored` below: M_full = Lgram ⊗ Rgram is an
# outer product (no shared inds), so M^(±1/2) = Lgram^(±1/2) ⊗ Rgram^(±1/2).
# Per-side eigen is on a (bond_dim_total)² matrix (~160² for N=4 mid bond)
# instead of the combined (bond_dim²)² matrix (~25600²) — ~2,000,000× fewer
# eigen ops AND avoids the multi-GB densification of the combined M.
function build_half_pair_single(G::ITensors.ITensor; rtol::Real=1e-10)
  # Env override for the pseudo-inverse cutoff. The aliased gram M is
  # structurally rank-deficient (dedup + channel structure → near-linearly-
  # dependent directions), so the default rtol=1e-10 retains near-null
  # eigenvalues whose 1/√λ blows up M⁻¹ and destabilises the Path-B eigsolve at
  # large bond dim (cond(M) ~ 1e8–1e10 observed at md=40). A larger cutoff drops
  # those null directions (textbook null-space-projected M^{-1/2}).
  # Hardened default 1e-1 (2026-06): the aliased gram M is structurally
  # rank-deficient. The dense-DMRG default (1e-10) keeps near-null eigenvalues
  # whose 1/√λ blows up M⁻¹ and destabilises the Path-B eigsolve. An rtol of r
  # is equivalent to capping the condition number of the KEPT block of M at 1/r
  # (we drop every eigenvalue below r·maxλ, so cond(M_kept) ≤ 1/r and the worst
  # amplification is 1/√λ_min ≤ √(1/r)/√maxλ). The N=12 md=40 KL scan shows the
  # stable, converged band is rtol ∈ [~1e-3, ~7e-1] (cond_kept 1.4–1000); below
  # ~1e-3 the eigsolve OSCILLATES and diverges (E rises from -17.16 to -5..-7),
  # and rtol≥9e-1 begins over-truncating genuine DOF on marginal bonds. Within
  # the band the energy is flat to the 5th digit, but rtol=1e-1 (cond_kept ≤ 10)
  # is the empirical optimum: lowest E (-17.16104 vs -17.16046 at 1e-2), lowest
  # final truncerr, and fastest convergence (plateau by sweep 3 vs sweep 5). The
  # mechanism is conditioning, not variational-subspace: a tighter cond cap
  # cleans up the marginal/redundant directions on the GRADED bonds (cond up to
  # ~98 at md=40 — the spectrum is graded, not cleanly bimodal), so each local
  # eigsolve is better conditioned and lands in a slightly better minimum.
  # Harmless for canonical M=I (no eigenvalue is dropped). Overridable via
  # BMF_MINV_RTOL. NOTE: revisit at larger md/N — if genuine DOF ever extend
  # below 0.1·maxλ, the cond≤10 cap would over-truncate and rtol must be relaxed
  # (staying above the ~1e-3 stability cliff).
  rtol = parse(Float64, get(ENV, "BMF_MINV_RTOL", "1e-1"))
  G_inds = collect(ITensors.inds(G))
  if isempty(G_inds)
    # Scalar gram: pass-through. Both Mhalf and Linv are scalar 1.
    return ITensors.ITensor(1.0), ITensors.ITensor(1.0)
  end
  unp = filter(I -> ITensors.plev(I) == 0, G_inds)
  prm = filter(I -> ITensors.plev(I) == 1, G_inds)
  @assert length(unp) == length(prm) "G inds must pair primed/unprimed"
  d = prod(I -> ITensors.dim(I), unp; init=1)
  G_d = ITensors.has_external_storage(G) ? to_dense_itensors_unfused(G) : G
  G_arr = Array(G_d, unp..., prm...)
  G_mat = reshape(G_arr, d, d)
  G_mat = (G_mat + G_mat') / 2
  F = LinearAlgebra.eigen(LinearAlgebra.Hermitian(G_mat))
  λ = F.values; V = F.vectors
  TC = eltype(G_mat)
  maxλ = isempty(λ) ? zero(real(TC)) : maximum(real, λ)
  tol  = real(rtol * maxλ)
  sqrt_λ     = [real(l) > tol ? sqrt(real(l))     : zero(real(TC)) for l in λ]
  inv_sqrt_λ = [real(l) > tol ? 1 / sqrt(real(l)) : zero(real(TC)) for l in λ]
  # STEP 2a — flat-c override (BMF_MINV_FLATC=1, default off): force the kept spectrum
  # flat to c = mean(kept λ), i.e. use M^{±1/2} = c^{±1/2}·Π (the ideal scaled-projector
  # form) instead of the actual graded spectrum. Uses the REAL range (V) but discards the
  # graded tail. At convergence the spectrum is ALREADY flat (=2^⌈env/2⌉) so this is a
  # no-op / byte-identical; during the ramp it flattens the tail. Tests whether the ideal
  # c·Π form still converges. Byte-identical for canonical M=I (single kept eigenvalue).
  if get(ENV, "BMF_MINV_FLATC", "0") == "1"
    _kept = [real(l) for l in λ if real(l) > tol]
    _cflat = isempty(_kept) ? one(real(TC)) : sum(_kept) / length(_kept)
    sqrt_λ     = [real(l) > tol ? sqrt(_cflat)     : zero(real(TC)) for l in λ]
    inv_sqrt_λ = [real(l) > tol ? 1 / sqrt(_cflat) : zero(real(TC)) for l in λ]
  end
  # M-conditioning diagnostic (BMF_MINV_DIAG=1): the M⁻¹ correction is unstable
  # when an eigenvalue sits just ABOVE the rtol threshold → huge 1/√λ. Log the
  # spectrum so we can see if Linv blows up at large bond dim / many sweeps.
  if get(ENV, "BMF_MINV_DIAG", "0") == "1"
    kept    = [real(l) for l in λ if real(l) > tol]
    n_drop  = length(λ) - length(kept)
    minkept = isempty(kept) ? 0.0 : minimum(kept)
    cond    = minkept > 0 ? maxλ / minkept : Inf
    maxinv  = minkept > 0 ? 1 / sqrt(minkept) : Inf
    println("[BMF_MINV_DIAG] d=$d  rtol=$(round(rtol,sigdigits=3))  maxλ=$(round(maxλ,sigdigits=4))  minkeptλ=$(round(minkept,sigdigits=4))  cond=$(round(cond,sigdigits=4))  n_dropped=$n_drop/$(length(λ))  max(1/√λ)=$(round(maxinv,sigdigits=4))")
  end
  Ghalf_mat = V * LinearAlgebra.Diagonal(sqrt_λ)     * V'
  Linv_mat  = V * LinearAlgebra.Diagonal(inv_sqrt_λ) * V'
  Ghalf_mat = (Ghalf_mat + Ghalf_mat') / 2
  Linv_mat  = (Linv_mat  + Linv_mat')  / 2
  dims_all = Int[]
  for I in unp; push!(dims_all, ITensors.dim(I)); end
  for I in prm; push!(dims_all, ITensors.dim(I)); end
  Ghalf_arr = reshape(Ghalf_mat, dims_all...)
  Linv_arr  = reshape(Linv_mat,  dims_all...)
  Ghalf_it  = ITensors.itensor(Ghalf_arr, unp..., prm...)
  Linv_it   = ITensors.itensor(Linv_arr,  unp..., prm...)
  return Ghalf_it, Linv_it
end

# Factored Mhalf, Linv via per-side eigen of Lgram and Rgram separately.
# Returns 4 ITensors: (Mhalf_L, Linv_L, Mhalf_R, Linv_R), each acting on
# ONE bond side only (left or right). The combined operator M^(±1/2) is
# Mhalf_L ⊗ Mhalf_R (or Linv_L ⊗ Linv_R) — applied by contracting each
# half independently against phi's matching bond.
#
# When phi_template is provided AND has WrappedBlockSparse storage, the
# returned Mhalf/Linv are also BS-stored with the same axis classification
# as phi → the subsequent BS×BS contraction in apply_minv_preserve_bs is
# recast-free (no densify+permute+gather per call). Otherwise (no phi or
# dense phi), returns dense ITensors.
function build_minv_half_pair_factored(Lgram::ITensors.ITensor,
                                       Rgram::ITensors.ITensor;
                                       rtol::Real=1e-10,
                                       phi_template::Union{Nothing,ITensors.ITensor}=nothing,
                                       use_bs_restricted::Bool=false,
                                       both_aliased::Bool=false)
  # When use_bs_restricted=true AND phi_template is BS, use equivalence-class
  # restricted BS storage → no recast in apply_minv_preserve_bs.
 @timeit SparseBackends.TIMER "build_minv_half_pair_factored" begin
  if use_bs_restricted && phi_template !== nothing &&
     ITensors.has_external_storage(phi_template) &&
     ITensors.get_external_storage(phi_template) isa WrappedBlockSparse
    Mhalf_L, Linv_L = build_minv_half_pair_bs_side(Lgram, phi_template, :left;  rtol)
    Mhalf_R, Linv_R = build_minv_half_pair_bs_side(Rgram, phi_template, :right; rtol)
  else
    @timeit SparseBackends.TIMER "bmf.eigen" begin
      Mhalf_L, Linv_L = build_half_pair_single(Lgram; rtol)
      Mhalf_R, Linv_R = build_half_pair_single(Rgram; rtol)
    end
    # BS-wrap dense Linv/Mhalf using phi_template's classification
    # (chan=sparse, mult=dense). Lets the BS×BS / Dense×BS kernel produce
    # output with axis classification matching phi → recast falls to the
    # Layer 1 fast path (key-filter only, no densify).
    # Gate via env (default ON when phi_template is BS).
    if get(ENV, "BMF_BSWRAP", "0") == "1" && phi_template !== nothing &&
       ITensors.has_external_storage(phi_template) &&
       ITensors.get_external_storage(phi_template) isa WrappedBlockSparse
      @timeit SparseBackends.TIMER "bmf.bswrap" begin
        Mhalf_L = wrap_dense_as_bs_via_template(Mhalf_L, phi_template)
        Linv_L  = wrap_dense_as_bs_via_template(Linv_L,  phi_template)
        Mhalf_R = wrap_dense_as_bs_via_template(Mhalf_R, phi_template)
        Linv_R  = wrap_dense_as_bs_via_template(Linv_R,  phi_template)
      end
    # ALIASED φ (case 2/4): relayout the dense factors as aliased carrying φ's
    # {channel→prefix, mult→dense} split, so the M^{−1/2} apply produces a
    # canonical output natively (channel in prefix) — no output hint, no forced
    # fission. This removes the case-4 md=16 crossover at its source (the factor
    # was fully dense ⇒ its channel landed in denseA ⇒ the no-hint fallback
    # parked the output channel in the dense tail). Pure relayout, values
    # bit-identical. Gated SB_ALIASED_MINV_WRAP (default ON for aliased φ);
    # cases 1/3 have dense/BS φ and never enter this branch.
    elseif both_aliased && get(ENV, "SB_ALIASED_MINV_WRAP", "1") == "1" && phi_template !== nothing &&
       ITensors.has_external_storage(phi_template) &&
       ITensors.get_external_storage(phi_template) isa WrappedAliasedBlockSparse
      @timeit SparseBackends.TIMER "bmf.aliaswrap" begin
        Mhalf_L = wrap_dense_as_aliased_via_template(Mhalf_L, phi_template)
        Linv_L  = wrap_dense_as_aliased_via_template(Linv_L,  phi_template)
        Mhalf_R = wrap_dense_as_aliased_via_template(Mhalf_R, phi_template)
        Linv_R  = wrap_dense_as_aliased_via_template(Linv_R,  phi_template)
      end
    end
  end
  return Mhalf_L, Linv_L, Mhalf_R, Linv_R
 end
end

# Wrap a dense ITensor T as BS storage using phi_template's axis classification.
# Each axis of T is classified as:
#   - dense if it shares Index id with a dense axis in phi_template (ignoring plev)
#   - sparse otherwise (always a Link-tagged axis after our context)
# This injects the missing chan-vs-mult distinction into Linv/Mhalf so the
# downstream contract kernel produces output classified the same way as phi.
function wrap_dense_as_bs_via_template(T::ITensors.ITensor,
                                       phi_template::ITensors.ITensor)
  # Edge case: scalar / empty Linv/Mhalf (boundary bond).
  if length(ITensors.inds(T)) == 0
    return T
  end
  debug = get(ENV, "BSWRAP_DEBUG", "0") == "1"
  # Identify phi's dense Index ids.
  pw = ITensors.get_external_storage(phi_template)
  phi_dense_set = dense_inds(pw)
  phi_dense_ids = Set(ITensors.id(I) for I in phi_dense_set)
  if debug
    println("\n[wrap_dense_as_bs] T inds:")
    for (i, I) in enumerate(ITensors.inds(T))
      println("  T[$i]: id=", ITensors.id(I), " dim=", ITensors.dim(I),
              " tags=", ITensors.tags(I), " plev=", ITensors.plev(I))
    end
    println("  phi_template inds:")
    for (i, I) in enumerate(ITensors.inds(phi_template))
      println("  phi[$i]: id=", ITensors.id(I), " dim=", ITensors.dim(I),
              " tags=", ITensors.tags(I), " plev=", ITensors.plev(I),
              "  in_phi_dense_set=", I in phi_dense_set)
    end
    println("  phi P=", length(pw.inds) - length(phi_dense_set),
            " N2(dense)=", length(phi_dense_set),
            " dense_ids=", phi_dense_ids)
  end

  inds_T = collect(ITensors.inds(T))
  sparse_inds = ITensors.Index[]
  dense_list  = ITensors.Index[]
  for I in inds_T
    if ITensors.id(I) in phi_dense_ids
      push!(dense_list, I)
    else
      push!(sparse_inds, I)
    end
  end
  if debug
    println("  classified sparse: ", [(ITensors.id(I), ITensors.dim(I), ITensors.plev(I)) for I in sparse_inds])
    println("  classified dense : ", [(ITensors.id(I), ITensors.dim(I), ITensors.plev(I)) for I in dense_list])
  end
  # If no axes ended up dense, falling back to all-sparse would change semantics
  # — but BS storage requires at least 0 sparse + N2 dense is fine. We allow any
  # split. If everything is dense, return T as-is (no BS needed).
  if isempty(sparse_inds)
    debug && println("  → all-dense: returning T unchanged")
    return T
  end
  inds_reordered = (sparse_inds..., dense_list...)
  TC = eltype(T)
  arr = Array(T, inds_reordered...)
  bs = SparseBackends.blocksparse_from_dense(arr, Val(length(dense_list)))
  out = ITensors._itensor_from_external_storage(
      WrappedBlockSparse(bs, Tuple(inds_reordered))
  )
  if debug
    ow = ITensors.get_external_storage(out)
    println("  → wrapped: P=", length(ow.inds) - length(dense_inds(ow)),
            "  N2=", length(dense_inds(ow)), "  blksize=", ow.blocksparse.blksize,
            "  nkeys=", length(ow.blocksparse.keys))
  end
  return out
end

# Aliased analogue of wrap_dense_as_bs_via_template: relayout a dense factor
# (Mhalf/Linv) as a WrappedAliasedBlockSparse carrying φ's {channel→prefix,
# mult→dense} classification, keyed by Index id (φ's dense_inds). This is a PURE
# RELAYOUT — every nonzero becomes its own trivial template (one template per
# block, scalar=1, no dedup), so values are bit-identical to the dense factor;
# only the prefix/dense split changes. It does NOT assume the metric is
# block-diagonal in the channel: cross-channel coupling stays inside the dense
# tail of each prefix block. With the channel axis classified into the prefix,
# a subsequent aliased×aliased contract produces output with the channel in the
# prefix WITHOUT needing an output hint — so the deferred-fission (fission=false)
# pre-H apply stays canonical and no crossover occurs (the case-4 md=16 fix).
function wrap_dense_as_aliased_via_template(T::ITensors.ITensor,
                                            phi_template::ITensors.ITensor)
  # Edge case: scalar / empty factor (boundary bond).
  if length(ITensors.inds(T)) == 0
    return T
  end
  pw = ITensors.get_external_storage(phi_template)
  phi_dense_ids = Set(ITensors.id(I) for I in dense_inds(pw))
  inds_T = collect(ITensors.inds(T))
  sparse_inds = ITensors.Index[]
  dense_list  = ITensors.Index[]
  for I in inds_T
    if ITensors.id(I) in phi_dense_ids
      push!(dense_list, I)
    else
      push!(sparse_inds, I)
    end
  end
  # No sparse axes ⇒ nothing to enumerate as a prefix; leave dense.
  if isempty(sparse_inds)
    return T
  end
  inds_reordered = (sparse_inds..., dense_list...)
  N   = length(inds_reordered)
  N2  = length(dense_list)
  arr = Array(T, inds_reordered...)
  TC  = eltype(arr)
  dims = ntuple(i -> ITensors.dim(inds_reordered[i]), Val(N))
  # Build trivially-aliased storage (one template per nonzero block), mirroring
  # the WrappedAliasedBlockSparse(T, denseLinks) constructor body.
  ali = AliasedBlockSparse{TC,N,N2}(dims)
  bs  = blocksparse_from_dense(arr, Val(N2))
  for (i, key) in enumerate(bs.keys)
    ali.n_templates += 1
    blk_off = (bs.ids[i] - 1) * bs.blksize
    append!(ali.templates, @view bs.data[blk_off+1 : blk_off+bs.blksize])
    push!(ali.keys,      key)
    push!(ali.alias_ids, _alias_id(eltype(ali.alias_ids), ali.n_templates))
    push!(ali.scalars,   one(TC))
  end
  return ITensors._itensor_from_external_storage(
      WrappedAliasedBlockSparse(ali, Tuple(inds_reordered)))
end

# Right Gram environment at bond b: contract psi[b+2..N] · dag(psi[b+2..N])
# with ALL Link inds primed on dag. Free legs are the bond between psi[b+1]
# and psi[b+2] (primed and unprimed copies). For strictly right-iso psi this
# equals δ on that bond; for non-canonical psi it's a non-trivial Gram matrix.
function gram_right_env(psi, b::Int)
  N = length(psi)
  M = ITensors.ITensor(1.0)
  for i in N:-1:(b + 2)
    T = psi[i]
    Td = ITensors.dag(T)
    for I in ITensors.inds(T)
      if ITensors.hastags(I, "Link")
        Td = ITensors.prime(Td, I)
      end
    end
    M = M * T * Td
  end
  return M
end

# Left Gram environment at bond b: mirror of gram_right_env over sites 1..b-1.
function gram_left_env(psi, b::Int)
  M = ITensors.ITensor(1.0)
  for i in 1:(b - 1)
    T = psi[i]
    Td = ITensors.dag(T)
    for I in ITensors.inds(T)
      if ITensors.hastags(I, "Link")
        Td = ITensors.prime(Td, I)
      end
    end
    M = M * T * Td
  end
  return M
end

# Build M^(1/2) and M^(-1/2) as block-sparse ITensors that exploit M's
# channel-block-diagonal structure. M comes from psi_right · dag(psi_right)
# where each psi[i] is channel-conserving (channel index is a "good quantum
# number" for P = ∏(I+C)). Therefore M[(c1,m1),(c2,m2)] = 0 whenever c1 ≠ c2,
# i.e. M is block-diagonal in the channel index.
#
# We exploit this by storing Linv and Mhalf as WrappedBlockSparse with sparse
# axes = channel-bond Indices, dense axes = mult-bond Indices, one block per
# diagonal channel key (c,c). Each block is (M_c)^(±1/2) computed by an
# eigendecomp of the mult_dim × mult_dim sub-matrix.
#
# Crucially, the resulting BS storage matches phi's bond-leg parameterization
# exactly: when we contract `Linv_bs * y_bs`, the BS×BS kernel produces
# output with the same (N2, P) as y, no recast needed. That eliminates the
# floating-point asymmetry that breaks Lanczos.
function build_minv_half_pair_bs(M_full::ITensors.ITensor;
                                 rtol::Real=1e-10,
                                 channel_inds::Union{Nothing,Tuple{Vararg{ITensors.Index}}}=nothing,
                                 mult_inds::Union{Nothing,Tuple{Vararg{ITensors.Index}}}=nothing,
                                 phi_template::Union{Nothing,ITensors.ITensor}=nothing)
  bond_unp = filter(I -> ITensors.plev(I) == 0, collect(ITensors.inds(M_full)))
  bond_prm = filter(I -> ITensors.plev(I) == 1, collect(ITensors.inds(M_full)))
  @assert length(bond_unp) == length(bond_prm) "M_full inds must pair primed/unprimed"
  if isempty(bond_unp)
    return ITensors.ITensor(1.0), ITensors.ITensor(1.0)
  end

  # Auto-detect channel (sparse) vs mult (dense) inds if not given.
  # Source of truth: phi_template's BS storage classification (since BS-Linv
  # must match phi for the BS×BS contraction to be valid). The "channel"
  # index is the one phi classifies as SPARSE; the "mult" index is the one
  # phi classifies as DENSE.
  if channel_inds === nothing
    classified = false
    if phi_template !== nothing &&
       ITensors.has_external_storage(phi_template) &&
       ITensors.get_external_storage(phi_template) isa WrappedBlockSparse
      pw = ITensors.get_external_storage(phi_template)
      phi_dense_set = Set(collect(dense_inds(pw)))
      channel_unp_list = ITensors.Index[]
      mult_unp_list    = ITensors.Index[]
      for I in bond_unp
        if I in phi_dense_set
          push!(mult_unp_list, I)
        else
          push!(channel_unp_list, I)
        end
      end
      channel_inds = tuple(channel_unp_list...)
      mult_inds    = tuple(mult_unp_list...)
      classified = true
    end
    if !classified
      if length(bond_unp) == 1
        channel_inds = (bond_unp[1],)
        mult_inds    = ()
      else
        # Last-resort heuristic: largest-dim Link is channel (since in our
        # setup channel = #channels which is typically the larger sparse
        # axis at overlap bonds; mult = multiplicity within channel often
        # collapsed to 1 at boundaries). NOT guaranteed correct.
        sorted = sort(collect(bond_unp); by=I -> -ITensors.dim(I))  # descending
        channel_inds = (sorted[1],)
        mult_inds    = tuple(sorted[2:end]...)
      end
    end
  end
  # Map channel/mult to their primed counterparts.
  function _primed_of(I::ITensors.Index, all_prm::Vector)
    for J in all_prm
      ITensors.id(I) == ITensors.id(J) && return J
    end
    error("No primed counterpart found for $I in $all_prm")
  end
  channel_prm = tuple((_primed_of(I, bond_prm) for I in channel_inds)...)
  mult_prm    = tuple((_primed_of(I, bond_prm) for I in mult_inds)...)

  # Densify M_full once to extract per-channel sub-matrices.
  M_d = ITensors.has_external_storage(M_full) ?
        to_dense_itensors_unfused(M_full) : M_full
  # Permute to (channel_unp..., channel_prm..., mult_unp..., mult_prm...) order.
  perm_inds = (channel_inds..., channel_prm..., mult_inds..., mult_prm...)
  M_arr_raw = Array(M_d, perm_inds...)
  c_total = prod(I -> ITensors.dim(I), channel_inds; init=1)
  m_total = prod(I -> ITensors.dim(I), mult_inds; init=1)
  # Shape: (c_total, c_total, m_total, m_total) viewed as nested.
  M_4d = reshape(M_arr_raw, c_total, c_total, m_total, m_total)

  TC = eltype(M_4d)
  Pc = length(channel_inds)  # # sparse axes in Linv_bs (channel only, unprimed side)
  # Linv has same inds as M_full: (channel_unp..., channel_prm..., mult..., mult')
  # We'll wrap with sparse axes = (channel_unp..., channel_prm...) (count 2*Pc)
  # and dense axes = (mult..., mult')  (count 2*length(mult_inds)).
  Linv_inds  = (channel_inds..., channel_prm..., mult_inds..., mult_prm...)
  Mhalf_inds = Linv_inds
  N_total    = length(Linv_inds)
  N2_dense   = N_total - 2 * Pc

  # FULL-matrix eigendecomposition (M is generically NOT channel-block-diagonal
  # because P's "channel" Index encodes an overcomplete basis decomposition,
  # not a symmetry quantum number). Eigen-decompose the full d×d matrix once,
  # then scatter the resulting full Linv/Mhalf into BS storage with ALL
  # channel-pair keys (c1, c2) present. The "sparsity" is the BS-storage
  # alignment with phi (no recast needed for B×B contractions), not memory
  # savings.
  channel_dims_unp = ntuple(i -> ITensors.dim(channel_inds[i]), Pc)
  channel_dims_prm = ntuple(i -> ITensors.dim(channel_prm[i]),  Pc)
  ci_unp = CartesianIndices(channel_dims_unp)
  ci_prm = CartesianIndices(channel_dims_prm)
  c_total_prm = c_total  # same dim on primed side

  d_full = c_total * m_total
  M_full_mat = reshape(permutedims(M_4d, (1, 3, 2, 4)), d_full, d_full)
  M_full_mat = (M_full_mat + M_full_mat') / 2
  F = LinearAlgebra.eigen(LinearAlgebra.Hermitian(M_full_mat))
  λ = F.values; V = F.vectors
  maxλ = isempty(λ) ? zero(real(TC)) : maximum(real, λ)
  tol = real(rtol * maxλ)
  sqrt_λ     = [real(l) > tol ? sqrt(real(l))   : zero(real(TC)) for l in λ]
  inv_sqrt_λ = [real(l) > tol ? 1 / sqrt(real(l)) : zero(real(TC)) for l in λ]
  Mhalf_full = V * LinearAlgebra.Diagonal(sqrt_λ)     * V'
  Linv_full  = V * LinearAlgebra.Diagonal(inv_sqrt_λ) * V'
  Mhalf_full = (Mhalf_full + Mhalf_full') / 2
  Linv_full  = (Linv_full  + Linv_full')  / 2
  # Reshape back to (c_unp, m_unp, c_prm, m_prm) then permute to (c_unp, c_prm, m_unp, m_prm)
  # to match the storage layout (sparse axes 1..2*Pc, dense axes after).
  Mhalf_4d = permutedims(reshape(Mhalf_full, c_total, m_total, c_total_prm, m_total),
                        (1, 3, 2, 4))
  Linv_4d  = permutedims(reshape(Linv_full,  c_total, m_total, c_total_prm, m_total),
                        (1, 3, 2, 4))

  block_payload_size = m_total^2
  keys_vec   = Vector{NTuple{2*Pc,Int}}()
  Linv_data  = Vector{TC}()
  Mhalf_data = Vector{TC}()

  # Scatter every (c_unp, c_prm) key into BS storage.
  for c1_lin in 1:c_total
    c1_tup = Tuple(ci_unp[c1_lin])
    for c2_lin in 1:c_total_prm
      c2_tup = Tuple(ci_prm[c2_lin])
      # Block has shape (m_total, m_total), stored in column-major order
      # consistent with Cartesian (m_unp, m_prm).
      Linv_block  = view(Linv_4d,  c1_lin, c2_lin, :, :)
      Mhalf_block = view(Mhalf_4d, c1_lin, c2_lin, :, :)
      push!(keys_vec, (c1_tup..., c2_tup...))
      append!(Linv_data,  vec(Matrix{TC}(Linv_block)))
      append!(Mhalf_data, vec(Matrix{TC}(Mhalf_block)))
    end
  end
  ids_vec = collect(1:length(keys_vec))

  dims_full = ntuple(i -> ITensors.dim(Linv_inds[i]), N_total)
  Linv_bs = NewBlockSparseSorted{TC, N_total, N2_dense, 2*Pc, Int}(
    dims_full, block_payload_size, keys_vec, copy(ids_vec), Linv_data
  )
  Mhalf_bs = NewBlockSparseSorted{TC, N_total, N2_dense, 2*Pc, Int}(
    dims_full, block_payload_size, copy(keys_vec), copy(ids_vec), Mhalf_data
  )
  Linv_it  = ITensors._itensor_from_external_storage(
    WrappedBlockSparse{TC, N_total, N2_dense, 2*Pc}(Linv_bs,  Linv_inds)
  )
  Mhalf_it = ITensors._itensor_from_external_storage(
    WrappedBlockSparse{TC, N_total, N2_dense, 2*Pc}(Mhalf_bs, Mhalf_inds)
  )
  return Mhalf_it, Linv_it
end

# Phi-key-restricted BS-Linv builder.
#
# Same idea as build_minv_half_pair_bs, but instead of storing Linv data at
# ALL chan-pair quadruples (c1_unp, c2_unp, c1_prm, c2_prm), we store only
# the SUBSET where (c1_unp, c2_unp) AND (c1_prm, c2_prm) appear in phi's
# stored chan-pair set. The eigendecomposition is performed on a smaller
# `(|S|·m_total) × (|S|·m_total)` matrix (where |S| = # allowed chan-pairs
# in phi) instead of the full `c_total · m_total × c_total · m_total`.
#
# Why this matters:
#   - The contraction Linv_BS · phi_BS naturally produces output keys that
#     are a subset of phi's keys (no recast needed → no FP asymmetry → Lanczos
#     works).
#   - The eigendecomp is smaller → faster.
#   - It matches the implicit projection that the dense + recast path was
#     already doing, so DMRG semantics are unchanged.
function build_minv_half_pair_bs_restricted(M_full::ITensors.ITensor,
                                            phi_template::ITensors.ITensor;
                                            rtol::Real=1e-10)
  @assert ITensors.has_external_storage(phi_template) "phi_template must be BS"
  pw = ITensors.get_external_storage(phi_template)
  @assert pw isa WrappedBlockSparse "phi_template must have WrappedBlockSparse storage"
  phi_bs = pw.blocksparse

  bond_unp = filter(I -> ITensors.plev(I) == 0, collect(ITensors.inds(M_full)))
  bond_prm = filter(I -> ITensors.plev(I) == 1, collect(ITensors.inds(M_full)))
  @assert length(bond_unp) == length(bond_prm) "M_full inds must pair primed/unprimed"
  if isempty(bond_unp)
    return ITensors.ITensor(1.0), ITensors.ITensor(1.0)
  end

  # Auto-detect channel vs mult inds from phi's BS classification.
  phi_dense_set = Set(collect(dense_inds(pw)))
  channel_unp_list = ITensors.Index[]
  mult_unp_list    = ITensors.Index[]
  for I in bond_unp
    (I in phi_dense_set) ? push!(mult_unp_list, I) : push!(channel_unp_list, I)
  end
  channel_inds = tuple(channel_unp_list...)
  mult_inds    = tuple(mult_unp_list...)
  Pc = length(channel_inds)

  function _primed_of(I::ITensors.Index, all_prm::Vector)
    for J in all_prm
      ITensors.id(I) == ITensors.id(J) && return J
    end
    error("No primed counterpart found for $I")
  end
  channel_prm = tuple((_primed_of(I, bond_prm) for I in channel_inds)...)
  mult_prm    = tuple((_primed_of(I, bond_prm) for I in mult_inds)...)

  # Find the positions of channel axes within phi's stored keys. phi's keys
  # are NTuple{phi_Pc, Int} where phi_Pc = # sparse axes of phi. The first
  # Pc entries of each key correspond to channel-axis values (since phi's
  # sparse axes are ordered: channel-links first, then non-link sparse, etc.
  # — we rely on this ordering from `output_inds`).
  phi_chan_pair_set = Set{NTuple{Pc,Int}}()
  for k in phi_bs.keys
    push!(phi_chan_pair_set, ntuple(i -> k[i], Pc))
  end
  S = sort(collect(phi_chan_pair_set))  # consistent ordering
  nS = length(S)

  # Densify M_full and permute to (chan_unp..., chan_prm..., mult_unp..., mult_prm...).
  M_d = ITensors.has_external_storage(M_full) ?
        to_dense_itensors_unfused(M_full) : M_full
  perm_inds = (channel_inds..., channel_prm..., mult_inds..., mult_prm...)
  M_arr_raw = Array(M_d, perm_inds...)
  c_total = prod(I -> ITensors.dim(I), channel_inds; init=1)
  m_total = prod(I -> ITensors.dim(I), mult_inds; init=1)
  TC = eltype(M_arr_raw)

  # Build M_4d of shape (c_total, c_total, m_total, m_total) viewed as nested.
  M_4d = reshape(M_arr_raw, c_total, c_total, m_total, m_total)

  # Build M_restricted of size (nS·m_total) × (nS·m_total) by extracting
  # blocks at allowed chan-pair combinations.
  channel_dims_unp = ntuple(i -> ITensors.dim(channel_inds[i]), Pc)
  # Map chan-pair tuple → linear index in c_total via column-major.
  function _chan_tup_to_lin(t::NTuple{P,Int}, dims::NTuple{P,Int}) where {P}
    lin = 0; stride = 1
    @inbounds for i in 1:P
      lin += (t[i] - 1) * stride
      stride *= dims[i]
    end
    return lin + 1
  end
  S_lin = [_chan_tup_to_lin(s, channel_dims_unp) for s in S]

  d_rest = nS * m_total
  M_restricted = zeros(TC, d_rest, d_rest)
  @inbounds for i in 1:nS, j in 1:nS
    ci = S_lin[i]; cj = S_lin[j]
    block = view(M_4d, ci, cj, :, :)  # m_total × m_total
    M_restricted[(i-1)*m_total+1 : i*m_total, (j-1)*m_total+1 : j*m_total] .= block
  end
  M_restricted = (M_restricted + M_restricted') / 2

  F = LinearAlgebra.eigen(LinearAlgebra.Hermitian(M_restricted))
  λ = F.values; V = F.vectors
  maxλ = isempty(λ) ? zero(real(TC)) : maximum(real, λ)
  tol  = real(rtol * maxλ)
  sqrt_λ     = [real(l) > tol ? sqrt(real(l))     : zero(real(TC)) for l in λ]
  inv_sqrt_λ = [real(l) > tol ? 1 / sqrt(real(l)) : zero(real(TC)) for l in λ]
  Mhalf_rest = V * LinearAlgebra.Diagonal(sqrt_λ)     * V'
  Linv_rest  = V * LinearAlgebra.Diagonal(inv_sqrt_λ) * V'
  Mhalf_rest = (Mhalf_rest + Mhalf_rest') / 2
  Linv_rest  = (Linv_rest  + Linv_rest')  / 2

  # Unstack into BS storage: keys = (c1_tup..., c2_tup...), one per (S × S).
  block_payload_size = m_total^2
  keys_vec   = Vector{NTuple{2*Pc,Int}}(undef, nS * nS)
  Linv_data  = Vector{TC}(undef, nS * nS * block_payload_size)
  Mhalf_data = Vector{TC}(undef, nS * nS * block_payload_size)
  idx = 0
  @inbounds for i in 1:nS, j in 1:nS
    idx += 1
    c1_tup = S[i]; c2_tup = S[j]
    keys_vec[idx] = (c1_tup..., c2_tup...)
    # Extract (m_total, m_total) block from Linv_rest / Mhalf_rest at
    # rows i, cols j.
    rblk = (i-1)*m_total+1 : i*m_total
    cblk = (j-1)*m_total+1 : j*m_total
    Linv_block_mat  = Linv_rest[rblk, cblk]
    Mhalf_block_mat = Mhalf_rest[rblk, cblk]
    base = (idx-1)*block_payload_size
    for k in 1:block_payload_size
      Linv_data[base+k]  = Linv_block_mat[k]
      Mhalf_data[base+k] = Mhalf_block_mat[k]
    end
  end
  ids_vec = collect(1:length(keys_vec))

  Linv_inds  = (channel_inds..., channel_prm..., mult_inds..., mult_prm...)
  Mhalf_inds = Linv_inds
  N_total    = length(Linv_inds)
  N2_dense   = N_total - 2 * Pc
  dims_full  = ntuple(i -> ITensors.dim(Linv_inds[i]), N_total)

  Linv_bs = NewBlockSparseSorted{TC, N_total, N2_dense, 2*Pc, Int}(
    dims_full, block_payload_size, keys_vec, copy(ids_vec), Linv_data
  )
  Mhalf_bs = NewBlockSparseSorted{TC, N_total, N2_dense, 2*Pc, Int}(
    dims_full, block_payload_size, copy(keys_vec), copy(ids_vec), Mhalf_data
  )
  Linv_it  = ITensors._itensor_from_external_storage(
    WrappedBlockSparse{TC, N_total, N2_dense, 2*Pc}(Linv_bs,  Linv_inds)
  )
  Mhalf_it = ITensors._itensor_from_external_storage(
    WrappedBlockSparse{TC, N_total, N2_dense, 2*Pc}(Mhalf_bs, Mhalf_inds)
  )
  return Mhalf_it, Linv_it
end

# Build both the symmetric sqrt M^(1/2) and sqrt-inverse M^(-1/2) on phi's
# bond legs from a full Gram tensor M_full (with paired primed/unprimed bond
# inds). Returns (Mhalf, Linv) as ITensors with the same Index layout.
# On M's null space (eigenvalues < rtol · max_eig), both are zero — this
# implicitly projects the eigenproblem onto image(M), which is what we want
# to avoid spurious zero eigenvalues from M's kernel.
function build_minv_half_pair_itensor(M_full::ITensors.ITensor; rtol::Real=1e-10)
  bond_unp = filter(I -> ITensors.plev(I) == 0, collect(ITensors.inds(M_full)))
  bond_prm = filter(I -> ITensors.plev(I) == 1, collect(ITensors.inds(M_full)))
  @assert length(bond_unp) == length(bond_prm) "M_full inds must pair primed/unprimed"
  if isempty(bond_unp)
    return ITensors.ITensor(1.0), ITensors.ITensor(1.0)
  end
  d = prod(I -> ITensors.dim(I), bond_unp; init=1)
  M_d = ITensors.has_external_storage(M_full) ?
        to_dense_itensors_unfused(M_full) : M_full
  M_arr = Array(M_d, bond_unp..., bond_prm...)
  M_mat = reshape(M_arr, d, d)
  M_mat = (M_mat + M_mat') / 2
  F = LinearAlgebra.eigen(LinearAlgebra.Hermitian(M_mat))
  λ = F.values
  V = F.vectors
  maxλ = isempty(λ) ? zero(real(eltype(M_mat))) : maximum(real, λ)
  tol = real(rtol * maxλ)
  sqrt_λ     = [real(l) > tol ? sqrt(real(l))   : zero(real(eltype(M_mat))) for l in λ]
  inv_sqrt_λ = [real(l) > tol ? 1 / sqrt(real(l)) : zero(real(eltype(M_mat))) for l in λ]
  Mhalf_mat = V * LinearAlgebra.Diagonal(sqrt_λ)     * V'
  Linv_mat  = V * LinearAlgebra.Diagonal(inv_sqrt_λ) * V'
  Mhalf_mat = (Mhalf_mat + Mhalf_mat') / 2
  Linv_mat  = (Linv_mat  + Linv_mat')  / 2
  dims_all = Vector{Int}(undef, length(bond_unp) + length(bond_prm))
  @inbounds for i in eachindex(bond_unp); dims_all[i] = ITensors.dim(bond_unp[i]); end
  @inbounds for i in eachindex(bond_prm); dims_all[length(bond_unp) + i] = ITensors.dim(bond_prm[i]); end
  Mhalf_arr = reshape(Mhalf_mat, dims_all...)
  Linv_arr  = reshape(Linv_mat,  dims_all...)
  Mhalf_it = ITensors.itensor(Mhalf_arr, bond_unp..., bond_prm...)
  Linv_it  = ITensors.itensor(Linv_arr,  bond_unp..., bond_prm...)
  return Mhalf_it, Linv_it
end

# Build the *symmetric* sqrt-inverse Linv = M^(-1/2) on phi's bond legs from
# a full Gram tensor M_full (with paired primed/unprimed bond inds).
#
# This is used to recast the generalized eigenproblem `H phi = E M phi` into
# the Hermitian standard problem `B y = E y` where:
#     B  = Linv · H · Linv†                 (Hermitian — Lanczos works)
#     y  = M^(1/2) phi   ⇔   phi = Linv y   (inverse change of variables)
#
# Using the *symmetric* square root (not Cholesky) keeps B Hermitian under
# the standard inner product, lets KrylovKit call `eigsolve(..., ishermitian=true)`
# (Lanczos), and converges in O(10) iterations vs Arnoldi's O(30+).
#
# M may be rank-deficient; we threshold its eigenvalues at `rtol * max_eig`
# and set Linv = 0 on those directions (pseudoinverse-style).
function build_minv_half_itensor(M_full::ITensors.ITensor; rtol::Real=1e-10)
  bond_unp = filter(I -> ITensors.plev(I) == 0, collect(ITensors.inds(M_full)))
  bond_prm = filter(I -> ITensors.plev(I) == 1, collect(ITensors.inds(M_full)))
  @assert length(bond_unp) == length(bond_prm) "M_full inds must pair primed/unprimed"
  if isempty(bond_unp)
    return ITensors.ITensor(1.0)
  end
  d = prod(I -> ITensors.dim(I), bond_unp; init=1)
  M_d = ITensors.has_external_storage(M_full) ?
        to_dense_itensors_unfused(M_full) : M_full
  M_arr = Array(M_d, bond_unp..., bond_prm...)
  M_mat = reshape(M_arr, d, d)
  M_mat = (M_mat + M_mat') / 2  # Hermitize.
  # Symmetric square root inverse via eigendecomposition.
  F = LinearAlgebra.eigen(LinearAlgebra.Hermitian(M_mat))
  λ = F.values
  V = F.vectors
  maxλ = isempty(λ) ? zero(eltype(M_mat)) : maximum(real, λ)
  tol = real(rtol * maxλ)
  inv_sqrt_λ = [real(l) > tol ? 1 / sqrt(real(l)) : zero(real(eltype(M_mat))) for l in λ]
  Linv_mat = V * LinearAlgebra.Diagonal(inv_sqrt_λ) * V'
  Linv_mat = (Linv_mat + Linv_mat') / 2  # Re-Hermitize.
  dims_all = Vector{Int}(undef, length(bond_unp) + length(bond_prm))
  @inbounds for i in eachindex(bond_unp); dims_all[i] = ITensors.dim(bond_unp[i]); end
  @inbounds for i in eachindex(bond_prm); dims_all[length(bond_unp) + i] = ITensors.dim(bond_prm[i]); end
  Linv_arr = reshape(Linv_mat, dims_all...)
  return ITensors.itensor(Linv_arr, bond_unp..., bond_prm...)
end

# Legacy: full Minv. Kept for compatibility with Path-B prototypes.
function build_minv_itensor(M_full::ITensors.ITensor; rtol::Real=1e-6)
  bond_unp = filter(I -> ITensors.plev(I) == 0, collect(ITensors.inds(M_full)))
  bond_prm = filter(I -> ITensors.plev(I) == 1, collect(ITensors.inds(M_full)))
  @assert length(bond_unp) == length(bond_prm) "M_full inds must pair primed/unprimed"
  # Edge case: no bond legs (e.g. boundary with empty env). Minv = scalar 1.
  if isempty(bond_unp)
    return ITensors.ITensor(1.0)
  end
  d = prod(I -> ITensors.dim(I), bond_unp; init=1)
  # Densify M_full and reshape to a d×d matrix in (unp..., prm...) order.
  M_d = ITensors.has_external_storage(M_full) ?
        to_dense_itensors_unfused(M_full) : M_full
  M_arr = Array(M_d, bond_unp..., bond_prm...)
  M_mat = reshape(M_arr, d, d)
  M_mat = (M_mat + M_mat') / 2  # Hermitize for stability.
  Minv_mat = pinv(M_mat; rtol=rtol)
  dims_all = Vector{Int}(undef, length(bond_unp) + length(bond_prm))
  @inbounds for i in eachindex(bond_unp); dims_all[i] = ITensors.dim(bond_unp[i]); end
  @inbounds for i in eachindex(bond_prm); dims_all[length(bond_unp) + i] = ITensors.dim(bond_prm[i]); end
  Minv_arr = reshape(Minv_mat, dims_all...)
  return ITensors.itensor(Minv_arr, bond_unp..., bond_prm...)
end

# Apply M⁻¹ to a sparse vector x while preserving x's WrappedBlockSparse
# storage type+keys. Used to wrap H_eff(x) as A_op(x) = M⁻¹ H_eff(x) for
# KrylovKit.eigsolve in Path-B DMRG.
# Recast-only: align z's BS layout (keys+order+sparse-dense split) to template's,
# without doing any contraction. Used to do a SINGLE final image-of-P projection
# at the end of B_op so the operator is exactly Hermitian → Lanczos becomes valid.
function recast_to_template(z::ITensors.ITensor, template::ITensors.ITensor)
 @timeit SparseBackends.TIMER "recast_to_template" begin
  if ITensors.has_external_storage(z) && ITensors.has_external_storage(template)
    Tw = ITensors.get_external_storage(template)
    Cw = ITensors.get_external_storage(z)
    if Cw isa WrappedBlockSparse && Tw isa WrappedBlockSparse
      return ITensors._itensor_from_external_storage(recast_bs_to_template(Cw, Tw))
    elseif Cw isa WrappedAliasedBlockSparse && Tw isa WrappedAliasedBlockSparse
      return ITensors._itensor_from_external_storage(recast_aliased_to_template(Cw, Tw))
    end
  end
  return z
 end
end

# Contract Minv·y WITHOUT recast (no image-of-P projection). Used inside the
# new Hermitian-Lanczos B_op where the projection happens only once at the end.
function apply_minv_no_recast(Minv::ITensors.ITensor, y::ITensors.ITensor)
 @timeit SparseBackends.TIMER "apply_minv_no_recast" begin
  if length(ITensors.inds(Minv)) == 0
    return y
  end
  z = contract_preserve_bs(Minv, y; template=nothing)
  z = ITensors.replaceprime(z, 1 => 0; tags="Link")
  return z
 end
end

const _AMP_DBG_BUDGET = Ref(4)
# `fission` (Lever 2 / SB_ALIASED_MINV_DEFER): when false, contract WITHOUT the φ-template
# hint (output stays in its natural, big-block classification — no per-channel s'-fission)
# and skip the schema recast. Used for the INTERMEDIATE apply in apply_half (its result is
# consumed by the next apply, which re-establishes φ's schema), so the apply_half OUTPUT is
# unchanged while the costly fission on the first apply is avoided (mirrors the matvec
# hint-lastonly win for the M⁻¹ path). When true (default): φ-schema fission + recast as before.
function apply_minv_preserve_bs(Minv::ITensors.ITensor, y::ITensors.ITensor, template::ITensors.ITensor;
                                fission::Bool=true)
 @timeit SparseBackends.TIMER "apply_minv_preserve_bs" begin
  if length(ITensors.inds(Minv)) == 0
    return y
  end
  do_dbg = get(ENV, "BSWRAP_DEBUG", "0") == "1" && _AMP_DBG_BUDGET[] > 0
  if do_dbg
    println("\n[apply_minv DBG] Minv inds: ", ITensors.inds(Minv))
    println("  y inds: ", ITensors.inds(y))
    if ITensors.has_external_storage(y)
      yw = ITensors.get_external_storage(y)
      if yw isa WrappedBlockSparse
        println("  y storage: BS P=", length(yw.inds) - length(dense_inds(yw)),
                " N2=", length(dense_inds(yw)), " blksize=", yw.blocksparse.blksize)
      end
    end
    println("  template inds: ", ITensors.inds(template))
    if ITensors.has_external_storage(template)
      tw = ITensors.get_external_storage(template)
      if tw isa WrappedBlockSparse
        println("  template storage: BS P=", length(tw.inds) - length(dense_inds(tw)),
                " N2=", length(dense_inds(tw)), " blksize=", tw.blocksparse.blksize)
      end
    end
  end
  # SB_MINV_DIAG=1: trace the prefix/dense split of the M⁻¹-apply output at each
  # stage for ALIASED tensors, to localize where φ's canonical split is lost
  # (channel moved into the dense tail). Prints P / prefix / dense per stage.
  _minv_diag = get(ENV, "SB_MINV_DIAG", "0") == "1" && _AMP_DBG_BUDGET[] > 0
  _mdump = function(lbl, T)
      if ITensors.has_external_storage(T) && T.tensor.data isa WrappedAliasedBlockSparse
          w = T.tensor.data; P = SparseBackends._abs_head_len(w); N = ndims(w.aliased)
          _tg(I) = (ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I))
          println("   [MINV_DIAG ", lbl, "] P=$P  prefix=", [_tg(w.inds[i]) for i in 1:P],
                  "  dense=", [_tg(w.inds[i]) for i in P+1:N])
      else
          println("   [MINV_DIAG ", lbl, "] storage=", ITensors.has_external_storage(T) ? string(typeof(T.tensor.data)) : "dense")
      end
  end
  if _minv_diag
      _AMP_DBG_BUDGET[] -= 1
      println("[MINV_DIAG apply] Minv inds=", [(ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I)) for I in ITensors.inds(Minv)])
      _mdump("y(in)", y); _mdump("template", template)
  end
  schema_dbg("apply_minv INPUT y", y)
  z = @timeit SparseBackends.TIMER "amp.cpb" contract_preserve_bs(Minv, y; template = (fission ? template : nothing))
  _minv_diag && _mdump("z after contract_preserve_bs", z)
  schema_dbg("apply_minv z = Minv·y", z)
  z = @timeit SparseBackends.TIMER "amp.replaceprime" ITensors.replaceprime(z, 1 => 0; tags="Link")
  _minv_diag && _mdump("z after replaceprime", z)
  if do_dbg
    println("  AFTER contract+replaceprime, z inds: ", ITensors.inds(z))
    if ITensors.has_external_storage(z)
      zw = ITensors.get_external_storage(z)
      if zw isa WrappedBlockSparse
        println("  z storage: BS P=", length(zw.inds) - length(dense_inds(zw)),
                " N2=", length(dense_inds(zw)), " blksize=", zw.blocksparse.blksize)
      else
        println("  z storage type: ", typeof(zw))
      end
    else
      println("  z has NO external storage (plain ITensor)")
    end
  end
  if fission && ITensors.has_external_storage(z) && ITensors.has_external_storage(template)
    Tw = ITensors.get_external_storage(template)
    Cw = ITensors.get_external_storage(z)
    if Cw isa WrappedBlockSparse && Tw isa WrappedBlockSparse
      if do_dbg
        set_eq = Set(Cw.inds) == Set(Tw.inds)
        println("  recast check: Set(Cw.inds)==Set(Tw.inds)? ", set_eq)
        if !set_eq
          println("    Cw.inds: ", Cw.inds)
          println("    Tw.inds: ", Tw.inds)
          println("    in Cw not Tw: ", setdiff(Set(Cw.inds), Set(Tw.inds)))
          println("    in Tw not Cw: ", setdiff(Set(Tw.inds), Set(Cw.inds)))
        end
      end
      z = @timeit SparseBackends.TIMER "amp.recast2" ITensors._itensor_from_external_storage(recast_bs_to_template(Cw, Tw))
      if do_dbg
        zw = ITensors.get_external_storage(z)
        if zw isa WrappedBlockSparse
          println("  AFTER recast, z storage: BS P=", length(zw.inds) - length(dense_inds(zw)),
                  " N2=", length(dense_inds(zw)), " blksize=", zw.blocksparse.blksize)
        end
      end
    elseif Cw isa WrappedAliasedBlockSparse && Tw isa WrappedAliasedBlockSparse
      # Aliased recast: align Hv's inds order to phi-template's inds order
      # so subsequent Path-B apply_minv calls see consistent classification.
      z = @timeit SparseBackends.TIMER "amp.recast2_aliased" ITensors._itensor_from_external_storage(recast_aliased_to_template(Cw, Tw))
      _minv_diag && _mdump("z after recast_aliased_to_template", z)
    end
  end
  _minv_diag && _mdump("z RETURNED", z)
  return z
 end
end

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
    dbg = get(ENV, "SB_RR_DBG", "0") == "1"
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
        if dbg
            println("[RR b=$b ha=$ha sw=$sw] it=$outer_it k=$k lam=$lam rnorm=$rnorm")
            flush(stdout)
        end
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
