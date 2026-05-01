# Move axis `axisr` of a dense array with Nd dims to the last position
@inline function _move_axis_to_last_perm(Nd::Int, axisr::Int)
  v = Vector{Int}(undef, Nd)
  k = 1
  @inbounds for d in 1:Nd
    d == axisr && continue
    v[k] = d
    k += 1
  end
  v[Nd] = axisr
  return v
end

"""
    contract_prefix_outer_bd!(
        C, labelsC,
        A, labelsA,
        B, labelsB,
        mapA, mapB, rlab
    )
BlockSparse A × Dense B.
Assumes:
  * Exactly one shared label rlab.
  * rlab is in A prefix (axisAr <= PA) and reduced (rlab ∉ labelsC).
  * B is dense and contains rlab as one of its dims.
  * C dense labels are either [A_dense..., B_wo_r...] or [B_wo_r..., A_dense...].
  * C prefix labels come only from A prefix excluding rlab (any order).
  * A has been (or will be) permuted so rlab is last prefix axis (axisAr == PA).

Effect:
  For each A block with key (..., r=rv), do:
    C_block += outer(A_block_dense, B_slice(rv))
"""
function contract_prefix_outer_bd!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::AbstractVector,
    A::NewBlockSparseSorted{TA,NA,N2A,PA},
    labelsA::AbstractVector,
    B::StridedArray{TB,NB},
    labelsB::AbstractVector,
    mapA::Dict,
    mapB::Dict,
    rlab
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}
  
  time1 = @elapsed begin
    @assert !(rlab in labelsC) "rlab must be reduced (not in labelsC)"
    axisAr = mapA[rlab]
    @assert axisAr <= PA "rlab must be in A prefix/head"
    axisBr = mapB[rlab]
    @assert 1 <= axisBr <= NB "rlab must be a dim of dense B"
  end
  # println("\tValidated assumptions and located rlab axes in $time1 seconds")

  time2 = @elapsed begin
    # ---- permute A so rlab is last prefix axis (slowest-changing within prefix)
    A, labelsA, mapA, _ = _permute_r_to_last_prefix(A, labelsA, mapA, rlab)
    @assert mapA[rlab] == PA
  end
  # println("\t\tPermuted A to put rlab last in prefix in $time2 seconds")

  time3 = @elapsed begin
    # ---- Determine B dims and make B_r_last with contiguous r-columns
    dimsB = size(B)
    R = dimsB[axisBr]
    permB = (axisBr == NB) ? nothing : _move_axis_to_last_perm(NB, axisBr)
    B_r_last = (permB === nothing) ? B : permutedims(B, permB)  # copy if permuted
  end
  # println("\t\tPermuted B to put rlab last in dims in $time3 seconds")
  
  time4 = @elapsed begin
    # Now B_r_last has dims (..., R) with r as last dim.
    # Reshape so each r slice is a contiguous column.
    chunkB = Int(length(B_r_last) ÷ R)
    Bmat = reshape(B_r_last, chunkB, R)  # column r is contiguous

    # ---- Validate C dense tail order: [A_dense..., B_wo_r...] or swapped
    Adense = labelsA[PA+1:NA]  # all dense labels of A
    Bwo = [lab for lab in labelsB if lab != rlab]  # dense labels of B excluding r

    Cdense = collect(labelsC[PC+1:NC])
    dense_order =
      Cdense == vcat(Adense, Bwo) ? :AB :
      Cdense == vcat(Bwo, Adense) ? :BA :
      error("C dense tail must be [A_dense..., B_wo_r...] or [B_wo_r..., A_dense...]. ",
            "Found ", Cdense, " expected ", vcat(Adense, Bwo), " or ", vcat(Bwo, Adense))

    # ---- C block size checks
    chunkA = A.blksize                    # dense payload length per A-block
    @assert C.blksize == chunkA * chunkB "C.blksize must equal A.blksize * (prod(B dims without r))"

    # ---- overwrite semantics
    empty!(C.keys); empty!(C.ids); empty!(C.data)

    # ---- prefix sourcing for ckey (C prefix labels must come from A prefix excluding r)
    # Build src[j] = axis in A.key for labelsC[j]
    src = Vector{Int}(undef, PC)
    @inbounds for j in 1:PC
      lab = labelsC[j]
      @assert haskey(mapA, lab) "C prefix label $lab not in A"
      ax = mapA[lab]
      @assert ax <= PA && lab != rlab "C prefix must be from A prefix excluding rlab"
      src[j] = ax
    end
  end
  # println("\t\tValidated C dense order and set up prefix sourcing in $time4 seconds")


  time5 = @elapsed begin
    α1 = one(TC)

    if get(ENV, "SB_TRACE", "0") == "1"
      println("[SB_TRACE]   prefix_outer_bd  PA=$PA  PC=$PC  chunkA=$chunkA  chunkB=$chunkB  R=$R  nblocks(A)=$(length(A.keys))")
    end

    nA = length(A.keys)

    # --- Fix C: build ckeys array once, sort once, then sweep contiguous runs.
    # This replaces the Dict + group_order approach (eliminates Dict alloc,
    # hash overhead, and per-group Int-Vector allocs).
    ckeys = Vector{NTuple{PC,Int}}(undef, nA)
    @inbounds for iA in 1:nA
      akey = A.keys[iA]
      ckeys[iA] = ntuple(j -> akey[src[j]], Val(PC))
    end
    perm = sortperm(ckeys)   # O(n log n), n << 100 for typical DMRG

    if get(ENV, "SB_TRACE", "0") == "1"
      ngs = (nA == 0) ? 0 : sum(i == 1 || ckeys[perm[i]] != ckeys[perm[i-1]] for i in 1:nA)
      println("[SB_TRACE]   prefix_outer_bd: $ngs ckey-groups")
    end

    # --- Fix C: pre-allocate staging buffers once (max nb per group <= R).
    # Views slice into these per-group; Kahan comp buffers are reset per group.
    Amat       = Matrix{TC}(undef, chunkA, R)
    Bmat_local = Matrix{TC}(undef, chunkB, R)
    comp_AB    = Matrix{TC}(undef, chunkA, chunkB)
    comp_BA    = Matrix{TC}(undef, chunkB, chunkA)

    i = 1
    @inbounds while i <= nA
      ckey = ckeys[perm[i]]
      cid  = _ensure_block!(C, ckey)
      Cvec = _block_view(C, cid)

      # Find the end of this run
      j = i + 1
      while j <= nA && ckeys[perm[j]] == ckey
        j += 1
      end
      nb = j - i

      # Gather A blocks and B slices for this group into pre-allocated staging
      for k in 1:nb
        iA_k = perm[i + k - 1]
        rv   = A.keys[iA_k][PA]
        @assert 1 <= rv <= R "A block key r=$rv out of bounds for dense B (R=$R)"
        Avec = _block_view(A, A.ids[iA_k])
        @inbounds for ii in 1:chunkA
          Amat[ii, k] = convert(TC, Avec[ii])
        end
        @inbounds for jj in 1:chunkB
          Bmat_local[jj, k] = Bmat[jj, rv]
        end
      end

      # Kahan-compensated outer product accumulation over the nb blocks.
      if dense_order === :AB
        Cmat = reshape(Cvec, chunkA, chunkB)
        fill!(comp_AB, zero(TC))
        for k in 1:nb
          @inbounds for jj in 1:chunkB
            bval = Bmat_local[jj, k]
            @inbounds for ii in 1:chunkA
              y = α1 * Amat[ii,k] * bval - comp_AB[ii,jj]
              t = Cmat[ii,jj] + y
              comp_AB[ii,jj] = (t - Cmat[ii,jj]) - y
              Cmat[ii,jj] = t
            end
          end
        end
      else
        Cmat = reshape(Cvec, chunkB, chunkA)
        fill!(comp_BA, zero(TC))
        for k in 1:nb
          @inbounds for jj in 1:chunkA
            aval = Amat[jj, k]
            @inbounds for ii in 1:chunkB
              y = α1 * Bmat_local[ii,k] * aval - comp_BA[ii,jj]
              t = Cmat[ii,jj] + y
              comp_BA[ii,jj] = (t - Cmat[ii,jj]) - y
              Cmat[ii,jj] = t
            end
          end
        end
      end

      i = j
    end
  end
  # println("\t\tCompleted contraction in $time5 seconds")
  # println("Total time is $(time1 + time2 + time3 + time4 + time5) seconds")
  return C
end


function contract!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::AbstractVector,
    A::NewBlockSparseSorted{TA,NA,N2A,PA},
    labelsA::AbstractVector,
    B::StridedArray{TB,NB},
    labelsB::AbstractVector,
    mapA::Dict,
    mapB::Dict,
    rlab
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}

  # sanity
  @assert !(rlab in labelsC) "rlab must be reduced (not in labelsC)"
  @assert haskey(mapA, rlab) "rlab not found in mapA"
  @assert haskey(mapB, rlab) "rlab not found in mapB"

  axisAr = mapA[rlab]
  if get(ENV, "SB_TRACE", "0") == "1"
    println("[SB_TRACE] contract_bs_dense.contract!  BS(blocks=", length(A.keys),
            ", PA=", PA, ", N2A=", N2A, ") × Dense(dims=", size(B), ")",
            "  axisAr=", axisAr,
            (axisAr <= PA ? "  → prefix_outer_bd" : "  → dense_bd"))
  end
  if axisAr <= PA
    time = @elapsed begin
      # Permute A so rlab is last prefix axis (slowest-changing within prefix)
      A, labelsA, mapA, _ = _permute_r_to_last_prefix(A, labelsA, mapA, rlab)
    end 
    # println("\t\tPermuted A to put rlab last in prefix in $time seconds")
    time = @elapsed begin
      result = contract_prefix_outer_bd!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    end
    # println("\tContracted with prefix outer product in $time seconds")
    return result
  else
    return contract_dense_bd!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
  end
end


function contract_dense_bd!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::AbstractVector,
    A::NewBlockSparseSorted{TA,NA,N2A,PA},
    labelsA::AbstractVector,
    B::AbstractArray{TB,NB},
    labelsB::AbstractVector,
    mapA::Dict,
    mapB::Dict,
    rlab;
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}

  @assert !(rlab in labelsC) "rlab must be reduced (not in labelsC)"

  # --- locate reduction axis (must be in dense tail of A; in B it's just a dense axis)
  axisAr = mapA[rlab]; @assert axisAr > PA "rlab must be in A dense tail"
  axisBr = mapB[rlab]; @assert 1 <= axisBr <= NB "rlab must be an axis of B"

  # --- permute A so that rlab is last axis (in-place)
  if axisAr != NA
    permA_global = _find_permutation_for_tensor(A, mapA, rlab)
    A = permutedims(A, permA_global)
    # permutedims!(A, permA_global)
    labelsA = labelsA[permA_global]
    # axisAr becomes NA by construction
  end
  @assert labelsA[end] == rlab

  # --- permute B so that rlab is last axis (materialize once; avoids per-slice shuffling)
  Bp = B
  if axisBr != NB
    permB = vcat(collect(1:axisBr-1), collect(axisBr+1:NB), axisBr)
    Bp = permutedims(B, permB)
    labelsB = labelsB[permB]
  end
  @assert labelsB[end] == rlab

  # ---- dims/chunks
  dimsA_dense = ntuple(i -> A.dims[PA+i], Val(N2A))
  dimsB_dense = size(Bp)  # NB-tuple
  R = dimsA_dense[end]
  @assert dimsB_dense[end] == R

  chunkA = Int(A.blksize / R)
  chunkB = Int(prod(dimsB_dense[1:NB-1]; init=1))

  @assert N2C == (N2A - 1) + (NB - 1)
  @assert C.blksize == chunkA * chunkB

  # ---- validate C dense tail order
  Awo = labelsA[PA+1:NA-1]      # A dense labels without rlab
  Bwo = labelsB[1:NB-1]         # B dense labels without rlab
  dense_order =
    collect(labelsC[PC+1:NC]) == vcat(Awo, Bwo) ? :AB :
    collect(labelsC[PC+1:NC]) == vcat(Bwo, Awo) ? :BA :
    error("C dense tail must be [A_dense..., B_dense...] or [B_dense..., A_dense...] ",
          "found ", collect(labelsC[PC+1:NC]), " expected ", vcat(Awo, Bwo), " or ", vcat(Bwo, Awo))

  # ---- overwrite semantics
  empty!(C.keys); empty!(C.ids); empty!(C.data)

  # ---- prefix sourcing for ckey: for BD, prefix must come from A sparse head
  srcA = zeros(Int, PC)
  @inbounds for j in 1:PC
    lab = labelsC[j]
    ax = mapA[lab]
    @assert ax <= PA "C sparse prefix label $lab must come from A sparse head"
    srcA[j] = ax
  end

  # ---- dense B access as contiguous slices for each r
  Bvec_full = vec(Bp)               # length chunkB*R
  # optional: if can_blas, reshape views for GER
  can_blas = (TC == TA == TB) && (TC <: LinearAlgebra.BlasFloat)
  α1 = one(TC)

  # ---- main
  @inbounds for ai in eachindex(A.keys)
    akey = A.keys[ai]
    aid  = A.ids[ai]
    Avec_full = _block_view(A, aid)   # length chunkA*R

    # one ckey per A-block (dense B has no sparse prefix)
    ckey = ntuple(j -> akey[srcA[j]], Val(PC))
    created = false
    Cvec = Vector{TC}()  # valid iff created
    Cmat = nothing

    if dense_order === :AB
      # C interpreted as (chunkA × chunkB) column-major
      for r in 1:R
        aoff = (r - 1) * chunkA
        boff = (r - 1) * chunkB
        _slice_any_nonzero(Avec_full, aoff, chunkA) || continue
        _slice_any_nonzero(Bvec_full, boff, chunkB) || continue

        if !created
          cid  = _ensure_block!(C, ckey)
          Cvec = _block_view(C, cid)
          created = true
          if can_blas
            Cmat = reshape(Cvec, chunkA, chunkB)
          end
        end

        As = @view Avec_full[aoff+1 : aoff+chunkA]
        Bs = @view Bvec_full[boff+1 : boff+chunkB]

        if can_blas
          _rank1_add_blas!(Cmat, α1, As, Bs)
        else
          _rank1_add_generic!(Cvec, α1, As, Bs)
        end
      end
    else
      # :BA : C interpreted as (chunkB × chunkA) column-major
      for r in 1:R
        aoff = (r - 1) * chunkA
        boff = (r - 1) * chunkB
        _slice_any_nonzero(Avec_full, aoff, chunkA) || continue
        _slice_any_nonzero(Bvec_full, boff, chunkB) || continue

        if !created
          cid  = _ensure_block!(C, ckey)
          Cvec = _block_view(C, cid)
          created = true
          if can_blas
            Cmat = reshape(Cvec, chunkB, chunkA)
          end
        end

        As = @view Avec_full[aoff+1 : aoff+chunkA]
        Bs = @view Bvec_full[boff+1 : boff+chunkB]

        if can_blas
          _rank1_add_blas!(Cmat, α1, Bs, As)  # swapped
        else
          _rank1_add_generic!(Cvec, α1, Bs, As)
        end
      end
    end
  end

  return C
end


"""
BlockSparse * Dense contraction with multiple shared (reduced) labels.

- `A` is `NewBlockSparseSorted{TA,NA,N2A,PA}` (prefix=head sparse, tail=dense)
- `B` is a dense tensor (any `AbstractArray{TB,NB}`)
- `shared_labels` are reduced and may live in A's prefix head or dense tail
- `shared_labels` may appear anywhere in B (all dims are dense)
- `C` is `NewBlockSparseSorted{TC,NC,N2C,PC}`

Assumptions (kept intentionally minimal / efficient):
1) All shared labels are reduced (do NOT appear in labelsC).
2) C prefix labels come from A prefix labels only (no B-in-prefix).
3) `permutedims!(A, permA)` is available (as in your existing code).
4) Existing utilities are available: `_cdense_grouping_and_orders`,
   `_advance_run`, `_block_view`, `_ensure_block!`.
"""
function contract_shared!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::AbstractVector,
    A::NewBlockSparseSorted{TA,NA,N2A,PA},
    labelsA::AbstractVector,
    B::AbstractArray{TB,NB},
    labelsB::AbstractVector,
    mapA::Dict,
    mapB::Dict,
    shared_labels::Vector;
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}

  # -----------------------------
  # 0) Shared labels are ALWAYS reduced
  # -----------------------------
  @inbounds for lab in shared_labels
    @assert !(lab in labelsC) "shared label $lab must be reduced (must not appear in labelsC)"
  end

  # -----------------------------
  # 1) Classify shared labels by A (prefix vs dense)
  #     - shared_prefix: reduced labels in A prefix head
  #     - shared_dense : reduced labels in A dense tail
  # -----------------------------
  shared_prefix = eltype(shared_labels)[]
  shared_dense  = eltype(shared_labels)[]
  @inbounds for lab in shared_labels
    @assert haskey(mapA, lab) "shared label $lab must exist in A"
    @assert haskey(mapB, lab) "shared label $lab must exist in B"
    a_pos = mapA[lab]
    if a_pos <= PA
      push!(shared_prefix, lab)
    else
      push!(shared_dense, lab)
    end
  end

  # -----------------------------
  # 2) Build keep/red sets from CURRENT labels (pre-permute)
  #    - red_dense order is A dense order filtered by shared_dense
  # -----------------------------
  Adense0 = labelsA[PA+1:NA]
  Bdense0 = labelsB                  # all dense for B
  redset  = Set(shared_dense)

  red_dense = [lab for lab in Adense0 if lab in redset]              # reduction order (from A tail)
  keepA0    = [lab for lab in Adense0 if !(lab in redset)]           # A kept dense labels
  keepB0    = [lab for lab in Bdense0 if !(lab in Set(shared_labels))] # B kept labels (everything not reduced)

  # C dense ordering decides whether we do AthenB or BthenA (no interleaving)
  Cdense = labelsC[PC+1:NC]
  mode, desired_keepA, desired_keepB = _cdense_grouping_and_orders(Cdense, keepA0, keepB0)
  mode == :interleaved && error("Cdense interleaves A/B kept dims; would require permuting C (unsupported; sort/fallback)")

  # sanity: desired orders must include all kept dims exactly once
  @assert length(desired_keepA) == length(keepA0) && Set(desired_keepA) == Set(keepA0)
  @assert length(desired_keepB) == length(keepB0) && Set(desired_keepB) == Set(keepB0)

  # -----------------------------
  # 3) ONE perm for A (join-friendly prefix, GEMM-friendly dense)
  #    and ONE perm for B (shared_prefix, red_dense, keepB in desired C order)
  # -----------------------------
  permA = _find_perm_for_A_join_and_dense_order(A, labelsA, mapA, shared_prefix, desired_keepA, red_dense)

  # For B (dense): we want Bp dims = [shared_prefix..., red_dense..., desired_keepB...]
  # using B's own label positions.
  sp_axes_B  = Int[mapB[lab] for lab in shared_prefix]
  rd_axes_B  = Int[mapB[lab] for lab in red_dense]
  kb_axes_B  = Int[mapB[lab] for lab in desired_keepB]
  permB      = vcat(sp_axes_B, rd_axes_B, kb_axes_B)
  @assert length(permB) == NB "B perm length mismatch; labelsB must match B ndims"

  # apply perms
  if permA != collect(1:NA)
    A = permutedims(A, permA)
    labelsA = labelsA[permA]
    mapA = Dict(l => i for (i,l) in enumerate(labelsA))
  end
  # For B: if already in desired order, avoid copy; else permute once (contiguous) for fast GEMM/slicing
  Bp = if permB == collect(1:NB)
    B
  else
    permutedims(B, permB)
  end
  labelsB = labelsB[permB]
  mapB = Dict(l => i for (i,l) in enumerate(labelsB))

  keepA = desired_keepA
  keepB = desired_keepB
  n_keepA = length(keepA)
  n_keepB = length(keepB)
  n_red   = length(red_dense)
  n_sp    = length(shared_prefix)

  # -----------------------------
  # 4) Validate/assemble C prefix mapping (C prefix must come from A prefix)
  # -----------------------------
  # C prefix labels must be from A prefix and NOT from reduced shared_prefix
  c_src_axes = Vector{Int}(undef, PC)
  @inbounds for j in 1:PC
    lab = labelsC[j]
    @assert haskey(mapA, lab) "C prefix label $lab must exist in A"
    apos = mapA[lab]
    @assert apos <= PA "C prefix label $lab must come from A prefix"
    @assert !(lab in Set(shared_prefix)) "C prefix label $lab cannot be a reduced shared_prefix label"
    c_src_axes[j] = apos
  end

  # join positions in A prefix (after permA, shared_prefix were moved to the end of prefix)
  join_posA = Int[mapA[lab] for lab in shared_prefix]

  # -----------------------------
  # 5) Dense contraction shapes
  #   A dense is [keepA..., red_dense...]
  #   Bp dense is [shared_prefix..., red_dense..., keepB...]
  #   For each fixed shared_prefix assignment, Bsub is [red_dense..., keepB...]
  # -----------------------------
  dimsA_dense = A.dims[PA+1:NA]
  dimsB       = size(Bp)

  M = (n_keepA == 0) ? 1 : prod(dimsA_dense[1:n_keepA])
  K = (n_red   == 0) ? 1 : prod(dimsA_dense[n_keepA+1:end])
  N = (n_keepB == 0) ? 1 : prod(dimsB[(n_sp + n_red + 1):end])

  if n_red > 0
    @assert K == prod(dimsB[(n_sp + 1):(n_sp + n_red)]) "Reduced dense extents mismatch between A and B"
  end
  @assert C.blksize == M * N "C.blksize must equal M*N (got $(C.blksize), expected $(M*N))"

  # -----------------------------
  # 6) Main loop:
  #   - Iterate runs of A that share the same shared_prefix values (join_posA)
  #   - For each run, slice Bp at those shared_prefix indices once
  #   - For each A block in the run: GEMM with same Bslice
  # -----------------------------
  empty!(C.keys); empty!(C.ids); empty!(C.data)

  iA = firstindex(A.keys)
  nA = lastindex(A.keys)

  @inbounds while iA <= nA
    # group A blocks by shared_prefix tuple (so we slice B once per group)
    iA2 = _advance_run(A.keys, iA, nA, join_posA)

    # Build Bsub for this shared_prefix assignment:
    # shared_prefix dims are first in Bp, so index by akey[join_posA[*]] (same order as shared_prefix)
    akey0 = A.keys[iA]
    if n_sp == 0
      # no shared_prefix: whole Bp is the slice
      Bsub = Bp
      Bmat = reshape(Bsub, K, N)  # Bsub dims == [red_dense..., keepB...]
      for ii in iA:(iA2-1)
        akey = A.keys[ii]
        Avec = _block_view(A, A.ids[ii])
        Amat = reshape(Avec, M, K)

        ckey = ntuple(Val(PC)) do j
          akey[c_src_axes[j]]
        end
        cid  = _ensure_block!(C, ckey)
        Cvec = _block_view(C, cid)

        if mode == :AthenB
          Cmat = reshape(Cvec, M, N)
          mul!(Cmat, Amat, Bmat, one(TC), one(TC))
        else
          Cmat = reshape(Cvec, N, M)
          mul!(Cmat, transpose(Bmat), transpose(Amat), one(TC), one(TC))
        end
      end
    else
      # slice Bp at the shared_prefix indices
      sp_vals = ntuple(Val(n_sp)) do t
        # join_posA corresponds to shared_prefix order, which matches sp_axes_B order used in permB
        akey0[join_posA[t]]
      end

      # indices: (sp_vals..., :, :, ...) over remaining dims
      idx = (sp_vals..., ntuple(_ -> Colon(), NB - n_sp)...)
      @views Bsub = Bp[idx...]
      # Bsub dims == [red_dense..., keepB...]
      Bmat = reshape(Bsub, K, N)

      for ii in iA:(iA2-1)
        akey = A.keys[ii]
        Avec = _block_view(A, A.ids[ii])
        Amat = reshape(Avec, M, K)

        ckey = ntuple(Val(PC)) do j
          akey[c_src_axes[j]]
        end
        cid  = _ensure_block!(C, ckey)
        Cvec = _block_view(C, cid)

        if mode == :AthenB
          Cmat = reshape(Cvec, M, N)
          mul!(Cmat, Amat, Bmat, one(TC), one(TC))
        else
          Cmat = reshape(Cvec, N, M)
          mul!(Cmat, transpose(Bmat), transpose(Amat), one(TC), one(TC))
        end
      end
    end

    iA = iA2
  end

  return C
end