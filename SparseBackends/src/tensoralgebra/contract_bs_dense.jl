using LinearAlgebra: LinearAlgebra

# Reusable scratch buffer for `contract_bs_dense_to_dense!`'s canonical
# accumulation array. One per element type; resized monotonically — this
# eliminates the per-call malloc + page-fault that defeated our previous
# attempt to use a canonical buffer.
const _BDD_CTGT_BUFS = IdDict{Type, Vector}()
function _bdd_ctgt_buffer(::Type{T}, n::Int) where {T}
    buf = get!(() -> T[], _BDD_CTGT_BUFS, T)::Vector{T}
    length(buf) < n && resize!(buf, n)
    return buf
end

# Reusable scratch buffer for `permute_B`. Same pattern as `_BDD_CTGT_BUFS`:
# avoids per-call malloc + GC pressure when the dense state Hv is reordered
# before each BlockSparse×Dense kernel call.
const _BDD_PERMB_BUFS = IdDict{Type, Vector}()
function _bdd_permB_buffer(::Type{T}, n::Int) where {T}
    buf = get!(() -> T[], _BDD_PERMB_BUFS, T)::Vector{T}
    length(buf) < n && resize!(buf, n)
    return buf
end

# --- GEMM input dumper (env-var gated by SB_GEMM_DUMP=<path>) ---
# Serializes per-call kernel inputs (M,K,N, per-A-block A_mat + sp_vals + cpfx_idx,
# the permuted Bp, and canonical C dims) so that fused-K and alternative-ordering
# kernels can be benchmarked offline on real DMRG shapes without touching the
# production kernel. Cap with SB_GEMM_DUMP_MAXCALLS=<int> (default 2000).
using Serialization: serialize
const _GEMM_DUMP_IO    = Ref{Union{Nothing,IO}}(nothing)
const _GEMM_DUMP_PATH  = Ref{Union{Nothing,String}}(nothing)
const _GEMM_DUMP_COUNT = Ref{Int}(0)
const _GEMM_DUMP_MAX   = Ref{Int}(2000)
const _GEMM_DUMP_INIT  = Ref{Bool}(false)

# --- Permute profile logger (env SB_PERMUTE_PROFILE=<path>) ---
# Appends per-call records for the sparse BS×Dense kernel to the same file
# that ITensorMPS appends matvec.denseH_denseV records to. Call site is taken
# from the `in_position` argument threaded down from dmrg.jl's position! call.
const _PERMUTE_PROFILE_IO_SB = Ref{Union{Nothing,IO}}(nothing)
const _PERMUTE_PROFILE_INIT_SB = Ref{Bool}(false)
function _permute_profile_io_sb()
    if !_PERMUTE_PROFILE_INIT_SB[]
        _PERMUTE_PROFILE_INIT_SB[] = true
        # Was SB_PERMUTE_PROFILE=<path>; now gated on _roofline_on(), opening
        # the fixed _RF_PERMUTE_PATH set by reset_roofline!'s own argument.
        if _roofline_on()
            _PERMUTE_PROFILE_IO_SB[] = open(_RF_PERMUTE_PATH[], "a")
            atexit() do
                io = _PERMUTE_PROFILE_IO_SB[]
                if io !== nothing
                    close(io)
                    _PERMUTE_PROFILE_IO_SB[] = nothing
                end
            end
        end
    end
    return _PERMUTE_PROFILE_IO_SB[]
end

function _gemm_dump_io()
    if !_GEMM_DUMP_INIT[]
        _GEMM_DUMP_INIT[] = true
        path = get(ENV, "SB_GEMM_DUMP", "")
        if !isempty(path)
            _GEMM_DUMP_PATH[]  = path
            _GEMM_DUMP_IO[]    = open(path, "w")
            _GEMM_DUMP_MAX[]   = parse(Int, get(ENV, "SB_GEMM_DUMP_MAXCALLS", "2000"))
            atexit() do
                io = _GEMM_DUMP_IO[]
                if io !== nothing
                    close(io)
                    _GEMM_DUMP_IO[] = nothing
                    println("[SB_GEMM_DUMP] wrote $(_GEMM_DUMP_COUNT[]) records to $(_GEMM_DUMP_PATH[])")
                end
            end
        end
    end
    return _GEMM_DUMP_IO[]
end

# Diagnostic counters (env-var gated by SB_CPFX_STATS=1). Inside each shared_prefix
# run we tally how many A blocks share a c_prefix tuple — measures the potential
# payoff of fusing scatters across same-c_prefix blocks.
const CPFX_STATS = Dict{Symbol, Int}(
    :n_runs          => 0,   # total shared_prefix runs across all kernel calls
    :n_blocks        => 0,   # total A blocks visited (sum of nb)
    :n_unique_cpfx   => 0,   # total unique cpfx tuples (sum of unique counts per run)
    :sum_max_group   => 0,   # sum over runs of max cpfx group size
    :n_runs_with_dup => 0,   # runs where at least one cpfx group has size >= 2
    :n_blocks_in_dup => 0,   # blocks that are part of a duplicate cpfx group
    :runs_nb_1       => 0,   # how many runs had nb == 1
    :runs_nb_le_4    => 0,   # nb in [2,4]
    :runs_nb_le_16   => 0,   # nb in [5,16]
    :runs_nb_gt_16   => 0,   # nb > 16
)
function reset_cpfx_stats!()
    for k in keys(CPFX_STATS); CPFX_STATS[k] = 0; end
    return CPFX_STATS
end
function print_cpfx_stats(io::IO=stdout)
    s = CPFX_STATS
    println(io, "\n--- c_prefix duplication stats (SB_CPFX_STATS) ---")
    println(io, "  total shared_prefix runs       : ", s[:n_runs])
    println(io, "  total A blocks scattered       : ", s[:n_blocks])
    println(io, "  total unique cpfx tuples       : ", s[:n_unique_cpfx])
    if s[:n_runs] > 0
        println(io, "  avg run size (nb)              : ", round(s[:n_blocks]/s[:n_runs]; digits=2))
        println(io, "  avg cpfx groups per run        : ", round(s[:n_unique_cpfx]/s[:n_runs]; digits=2))
        println(io, "  avg max cpfx group size / run  : ", round(s[:sum_max_group]/s[:n_runs]; digits=2))
        println(io, "  runs with any cpfx duplicate   : ", s[:n_runs_with_dup],
                    "  (", round(100*s[:n_runs_with_dup]/s[:n_runs]; digits=1), "%)")
    end
    if s[:n_blocks] > 0
        println(io, "  blocks in duplicate cpfx group : ", s[:n_blocks_in_dup],
                    "  (", round(100*s[:n_blocks_in_dup]/s[:n_blocks]; digits=1), "%)")
        println(io, "  potential scatter reduction    : ",
                    round(100*(s[:n_blocks] - s[:n_unique_cpfx])/s[:n_blocks]; digits=1), "%")
    end
    println(io, "  run-size hist: nb=1 ", s[:runs_nb_1],
                "  nb=[2,4] ", s[:runs_nb_le_4],
                "  nb=[5,16] ", s[:runs_nb_le_16],
                "  nb>16 ", s[:runs_nb_gt_16])
end

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
    @timeit TIMER "kbd.permute_A" begin
      A, labelsA, mapA, _ = _permute_r_to_last_prefix(A, labelsA, mapA, rlab)
    end
    @assert mapA[rlab] == PA
  end
  # println("\t\tPermuted A to put rlab last in prefix in $time2 seconds")

  time3 = @elapsed begin
    # ---- Determine B dims and make B_r_last with contiguous r-columns
    @timeit TIMER "kbd.permute_B" begin
      dimsB = size(B)
      R = dimsB[axisBr]
      permB = (axisBr == NB) ? nothing : _move_axis_to_last_perm(NB, axisBr)
      B_r_last = (permB === nothing) ? B : permutedims(B, permB)  # copy if permuted
    end
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

    if false  # SB_TRACE — flip to true here for debug output
      println("[SB_TRACE]   prefix_outer_bd  PA=$PA  PC=$PC  chunkA=$chunkA  chunkB=$chunkB  R=$R  nblocks(A)=$(length(A.keys))")
    end

    nA = length(A.keys)

    # --- Fix C: build ckeys array once, sort once, then sweep contiguous runs.
    # This replaces the Dict + group_order approach (eliminates Dict alloc,
    # hash overhead, and per-group Int-Vector allocs).
    @timeit TIMER "kbd.ckeys+sortperm" begin
      ckeys = Vector{NTuple{PC,Int}}(undef, nA)
      @inbounds for iA in 1:nA
        akey = A.keys[iA]
        ckeys[iA] = ntuple(j -> akey[src[j]], Val(PC))
      end
      perm = sortperm(ckeys)   # O(n log n), n << 100 for typical DMRG
    end

    if false  # SB_TRACE — flip to true here for debug output
      ngs = (nA == 0) ? 0 : sum(i == 1 || ckeys[perm[i]] != ckeys[perm[i-1]] for i in 1:nA)
      println("[SB_TRACE]   prefix_outer_bd: $ngs ckey-groups")
    end

    # --- Opt (1): bulk pre-reserve C.data CAPACITY (not length) so per-block
    # `_alloc_block!` `resize!` calls don't trigger Vector growth/malloc/memcpy.
    # Count unique ckeys (run-starts) and `sizehint!` once. The actual length
    # still grows one block at a time inside _ensure_block!, but the underlying
    # buffer is already big enough.
    @timeit TIMER "kbd.reserve_C" begin
      nblocks_est = 0
      @inbounds for k in 1:nA
        if k == 1 || ckeys[perm[k]] != ckeys[perm[k-1]]
          nblocks_est += 1
        end
      end
      total_data_size = nblocks_est * C.blksize
      if total_data_size > 0
        sizehint!(C.data, length(C.data) + total_data_size)
        sizehint!(C.keys, length(C.keys) + nblocks_est)
        sizehint!(C.ids,  length(C.ids)  + nblocks_est)
      end
    end

    # --- Fix C: pre-allocate staging buffers once (max nb per group <= R).
    Amat       = Matrix{TC}(undef, chunkA, R)
    Bmat_local = Matrix{TC}(undef, chunkB, R)

    i = 1
    @inbounds while i <= nA
      ckey = ckeys[perm[i]]
      @timeit TIMER "kbd.ensure_block" begin
        cid  = _ensure_block!(C, ckey)
        Cvec = _block_view(C, cid)
      end

      # Find the end of this run
      j = i + 1
      while j <= nA && ckeys[perm[j]] == ckey
        j += 1
      end
      nb = j - i

      # Gather A blocks and B slices for this group into pre-allocated staging
      @timeit TIMER "kbd.gather" begin
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
      end

      # BLAS rank-nb update over the gathered slabs.
      @timeit TIMER "kbd.gemm" begin
        Av = view(Amat,       1:chunkA, 1:nb)
        Bv = view(Bmat_local, 1:chunkB, 1:nb)
        if dense_order === :AB
          Cmat = reshape(Cvec, chunkA, chunkB)
          LinearAlgebra.mul!(Cmat, Av, transpose(Bv), one(TC), one(TC))
        else
          Cmat = reshape(Cvec, chunkB, chunkA)
          LinearAlgebra.mul!(Cmat, Bv, transpose(Av), one(TC), one(TC))
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
  if false  # SB_TRACE — flip to true here for debug output
    println("[SB_TRACE] contract_bs_dense.contract!  BS(blocks=", length(A.keys),
            ", PA=", PA, ", N2A=", N2A, ") × Dense(dims=", size(B), ")",
            "  axisAr=", axisAr,
            (axisAr <= PA ? "  → prefix_outer_bd" : "  → dense_bd"))
  end
  if axisAr <= PA
    @timeit TIMER "kbd.dispatch.prefix_outer_bd" begin
      A, labelsA, mapA, _ = _permute_r_to_last_prefix(A, labelsA, mapA, rlab)
      return contract_prefix_outer_bd!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    end
  else
    @timeit TIMER "kbd.dispatch.dense_bd" begin
      return contract_dense_bd!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, rlab)
    end
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
#= OLD contract_shared! — batched gather/GEMM/scatter version.
   Replaced by a single-loop version below that mirrors the
   contract_bs_dense_to_dense! strategy: per A block, slice B once and
   GEMM directly into the destination C block with β=1. The batched
   variant paid for an Astack gather and a Cstack scatter on every run;
   in practice nb is small enough that the per-run buffer resizes,
   copyto!s, and additive scatters dominated the win from one large GEMM.

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


  @timeit TIMER "kdb.dispatch.prep_labels!" begin
    # -----------------------------
    # 0) Shared labels are ALWAYS reduced
    # -----------------------------
    @inbounds for lab in shared_labels
      @assert !(lab in labelsC) "shared label $lab must be reduced (must not appear in labelsC)"
    end

    # -----------------------------
    # 1) Classify shared labels by A (prefix vs dense)
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
    # -----------------------------
    Adense0 = labelsA[PA+1:NA]
    Bdense0 = labelsB
    redset  = Set(shared_dense)
    red_dense = [lab for lab in Adense0 if lab in redset]
    keepA0    = [lab for lab in Adense0 if !(lab in redset)]
    keepB0    = [lab for lab in Bdense0 if !(lab in Set(shared_labels))]

    Cdense = labelsC[PC+1:NC]
    mode, desired_keepA, desired_keepB = _cdense_grouping_and_orders(Cdense, keepA0, keepB0)
    mode == :interleaved && error("Cdense interleaves A/B kept dims; would require permuting C (unsupported; sort/fallback)")

    @assert length(desired_keepA) == length(keepA0) && Set(desired_keepA) == Set(keepA0)
    @assert length(desired_keepB) == length(keepB0) && Set(desired_keepB) == Set(keepB0)
  end
  

  @timeit TIMER "kdb.dispatch.permute_and_validate!" begin
    # -----------------------------
    # 3) ONE perm for A and ONE perm for B
    # -----------------------------
    permA = _find_perm_for_A_join_and_dense_order(A, labelsA, mapA, shared_prefix, desired_keepA, red_dense)

    sp_axes_B  = Int[mapB[lab] for lab in shared_prefix]
    rd_axes_B  = Int[mapB[lab] for lab in red_dense]
    kb_axes_B  = Int[mapB[lab] for lab in desired_keepB]
    permB      = vcat(sp_axes_B, rd_axes_B, kb_axes_B)
    @assert length(permB) == NB "B perm length mismatch; labelsB must match B ndims"

    if permA != collect(1:NA)
      A = permutedims(A, permA)
      labelsA = labelsA[permA]
      mapA = Dict(l => i for (i,l) in enumerate(labelsA))
    end
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
  end

  @timeit TIMER "kdb.dispatch.validate_and_prep!" begin

    # -----------------------------
    # 4) Validate/assemble C prefix mapping
    # -----------------------------
    c_src_axes = Vector{Int}(undef, PC)
    @inbounds for j in 1:PC
      lab = labelsC[j]
      @assert haskey(mapA, lab) "C prefix label $lab must exist in A"
      apos = mapA[lab]
      @assert apos <= PA "C prefix label $lab must come from A prefix"
      @assert !(lab in Set(shared_prefix)) "C prefix label $lab cannot be a reduced shared_prefix label"
      c_src_axes[j] = apos
    end

    join_posA = Int[mapA[lab] for lab in shared_prefix]
  end

  @timeit TIMER "kdb.dispatch.contract!" begin
    # -----------------------------
    # 5) Dense contraction shapes
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
  end

  @timeit TIMER "kdb.dispatch.main_loop!" begin
    # -----------------------------
    # 6) Main loop:
    #   - Iterate runs of A that share the same shared_prefix values (join_posA)
    #   - For each run, slice Bp at those shared_prefix indices once
    #   - For each A block in the run: batched GEMM with same Bslice
    # -----------------------------
    empty!(C.keys); empty!(C.ids); empty!(C.data)

    iA = firstindex(A.keys)
    nA = lastindex(A.keys)

    # ---------- Phase 0: batched allocation of all C blocks ----------
    # _ensure_block! per call does binary-search + insert!(C.keys/ids) which is
    # O(n) per call → O(n²) total over the kernel. Instead, deduplicate ckeys
    # upfront (Dict-based) assigning cids in *iteration order* — this keeps
    # cids sequential within a single run, so the later scatter into C.data is
    # cache-friendly. C.keys is then re-sorted (cids may be out of order) so
    # the BS data structure satisfies its sorted-keys invariant.
    @timeit TIMER "kdb.dispatch.alloc_C" begin
      cid_for = Vector{Int}(undef, nA)
      if nA > 0
        ck_to_cid = Dict{NTuple{PC,Int}, Int}()
        sizehint!(ck_to_cid, nA)
        n_unique = 0
        @inbounds for ii in 1:nA
          akey = A.keys[ii]
          ckey = ntuple(j -> akey[c_src_axes[j]], Val(PC))
          cid  = get(ck_to_cid, ckey, 0)
          if cid == 0
            n_unique += 1
            ck_to_cid[ckey] = n_unique
            cid = n_unique
          end
          cid_for[ii] = cid
        end
        # Build sorted (ckey, cid) pairs for C.keys / C.ids.
        sorted_pairs = collect(ck_to_cid)
        sort!(sorted_pairs; by = first)
        resize!(C.keys, n_unique)
        resize!(C.ids,  n_unique)
        @inbounds for k in 1:n_unique
          C.keys[k] = sorted_pairs[k].first
          C.ids[k]  = sorted_pairs[k].second
        end
        # One resize + one zero-fill, replacing n_unique individual
        # _alloc_block! calls and their per-call zero-vector temporaries.
        total_data = n_unique * C.blksize
        resize!(C.data, total_data)
        fill!(C.data, zero(TC))
      end
    end

    # Batched-GEMM staging buffers (pre-allocated, grown per run as needed).
    # Astack holds nb stacked A blocks of shape (M, K) → (M*nb, K).
    # Cstack holds the batched result before scattering back to per-block Cmats.
    Astack_buf = Vector{TC}(undef, 0)
    Cstack_buf = Vector{TC}(undef, 0)
    cids_buf   = Vector{Int}(undef, 0)

    @inbounds while iA <= nA
      # group A blocks by shared_prefix tuple (so we slice B once per group)
      iA2 = _advance_run(A.keys, iA, nA, join_posA)
      nb  = iA2 - iA

      # Build Bsub for this shared_prefix assignment:
      akey0 = A.keys[iA]
      if n_sp == 0
        Bsub = Bp
        Bmat = reshape(Bsub, K, N)
      else
        sp_vals = ntuple(Val(n_sp)) do t
          akey0[join_posA[t]]
        end
        idx = (sp_vals..., ntuple(_ -> Colon(), NB - n_sp)...)
        @views Bsub = Bp[idx...]
        Bmat = reshape(Bsub, K, N)
      end

      # ---------- Phase 1: per-run gather (tight loop; cids already known) ----------
      # Stack all nb A blocks of this run into Astack (shape M*nb × K). The
      # destination cids were precomputed in phase 0 (`cid_for`) so no
      # _ensure_block! call is needed here.
      @timeit TIMER "kdb.run.gather" begin
        resize!(Astack_buf, M * nb * K)
        Astack = reshape(view(Astack_buf, 1:M*nb*K), M * nb, K)
        resize!(cids_buf, nb)
        for k in 1:nb
          ii   = iA + k - 1
          Avec = _block_view(A, A.ids[ii])
          # Copy this A block into rows (k-1)*M+1 : k*M of Astack.
          Adst = view(Astack, (k-1)*M+1 : k*M, :)
          copyto!(Adst, reshape(Avec, M, K))
          cids_buf[k] = cid_for[ii]
        end
      end

      # ---------- Phase 2: one BLAS GEMM for the whole run ----------
      # Cstack_run = Astack * Bmat   (shape M*nb × N)
      # then scatter each (k-1)*M+1 : k*M block of Cstack_run additively into
      # the corresponding C block. ONE big BLAS call replaces nb tiny ones.
      @timeit TIMER "kdb.gemm" begin
        if mode == :AthenB
          resize!(Cstack_buf, M * nb * N)
          Cstack = reshape(view(Cstack_buf, 1:M*nb*N), M * nb, N)
          mul!(Cstack, Astack, Bmat)               # β=0: fresh write
          for k in 1:nb
            Cmat = reshape(_block_view(C, cids_buf[k]), M, N)
            Cmat .+= view(Cstack, (k-1)*M+1 : k*M, :)
          end
        else
          # :BA — output transposed (N × M per block)
          resize!(Cstack_buf, N * M * nb)
          Cstack = reshape(view(Cstack_buf, 1:N*M*nb), N, M * nb)
          mul!(Cstack, transpose(Bmat), transpose(Astack))   # β=0
          for k in 1:nb
            Cmat = reshape(_block_view(C, cids_buf[k]), N, M)
            Cmat .+= view(Cstack, :, (k-1)*M+1 : k*M)
          end
        end
      end

      iA = iA2
    end
  end
  return C
end
=#


function contract_shared!(
    C::NewBlockSparseSorted{TC,NC,N2C,PC},
    labelsC::AbstractVector,
    A::NewBlockSparseSorted{TA,NA,N2A,PA},
    labelsA::AbstractVector,
    B::AbstractArray{TB,NB},
    labelsB::AbstractVector,
    mapA::Dict,
    mapB::Dict,
    shared_labels::Vector,
) where {TC,NC,N2C,PC,TA,NA,N2A,PA,TB,NB}

  @timeit TIMER "kdb.dispatch.prep_labels!" begin
    @inbounds for lab in shared_labels
      @assert !(lab in labelsC) "shared label $lab must be reduced (must not appear in labelsC)"
    end

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

    Adense0 = labelsA[PA+1:NA]
    Bdense0 = labelsB
    redset  = Set(shared_dense)
    red_dense = [lab for lab in Adense0 if lab in redset]
    keepA0    = [lab for lab in Adense0 if !(lab in redset)]
    keepB0    = [lab for lab in Bdense0 if !(lab in Set(shared_labels))]

    Cdense = labelsC[PC+1:NC]
    mode, desired_keepA, desired_keepB = _cdense_grouping_and_orders(Cdense, keepA0, keepB0)
    mode == :interleaved && _layout_error("block-sparse × dense",
        "C's dense block interleaves A's and B's indices; list them as two groups")
    # Moving a dense index into C's key (fission) is not implemented here.
    (length(desired_keepA) == length(keepA0) && Set(desired_keepA) == Set(keepA0) &&
     length(desired_keepB) == length(keepB0) && Set(desired_keepB) == Set(keepB0)) ||
        _layout_error("block-sparse × dense", "C's dense block must hold every " *
            "uncontracted dense index of A and B; moving a dense index into the key is not supported")
  end

  @timeit TIMER "kdb.dispatch.permute_and_validate!" begin
    permA = _find_perm_for_A_join_and_dense_order(A, labelsA, mapA, shared_prefix, desired_keepA, red_dense)

    sp_axes_B  = Int[mapB[lab] for lab in shared_prefix]
    rd_axes_B  = Int[mapB[lab] for lab in red_dense]
    kb_axes_B  = Int[mapB[lab] for lab in desired_keepB]
    permB      = vcat(sp_axes_B, rd_axes_B, kb_axes_B)
    @assert length(permB) == NB "B perm length mismatch; labelsB must match B ndims"

    if permA != collect(1:NA)
      A = permutedims(A, permA)
      labelsA = labelsA[permA]
      mapA = Dict(l => i for (i,l) in enumerate(labelsA))
    end
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
  end

  @timeit TIMER "kdb.dispatch.validate_and_prep!" begin
    c_src_axes = Vector{Int}(undef, PC)
    @inbounds for j in 1:PC
      lab = labelsC[j]
      @assert haskey(mapA, lab) "C prefix label $lab must exist in A"
      apos = mapA[lab]
      @assert apos <= PA "C prefix label $lab must come from A prefix"
      @assert !(lab in Set(shared_prefix)) "C prefix label $lab cannot be a reduced shared_prefix label"
      c_src_axes[j] = apos
    end

    join_posA = Int[mapA[lab] for lab in shared_prefix]
  end

  @timeit TIMER "kdb.dispatch.contract!" begin
    dimsA_dense = A.dims[PA+1:NA]
    dimsB       = size(Bp)

    M = (n_keepA == 0) ? 1 : prod(dimsA_dense[1:n_keepA])
    K = (n_red   == 0) ? 1 : prod(dimsA_dense[n_keepA+1:end])
    N = (n_keepB == 0) ? 1 : prod(dimsB[(n_sp + n_red + 1):end])

    if n_red > 0
      @assert K == prod(dimsB[(n_sp + 1):(n_sp + n_red)]) "Reduced dense extents mismatch between A and B"
    end
    @assert C.blksize == M * N "C.blksize must equal M*N (got $(C.blksize), expected $(M*N))"
  end

  @timeit TIMER "kdb.dispatch.main_loop!" begin
    empty!(C.keys); empty!(C.ids); empty!(C.data)

    iA = firstindex(A.keys)
    nA = lastindex(A.keys)

    # ---------- Phase 0: batched allocation of all C blocks ----------
    # Same dedup/alloc as before — C.data is zero-filled so the main loop
    # below can use β=1 from the start.
    @timeit TIMER "kdb.dispatch.alloc_C" begin
      cid_for = Vector{Int}(undef, nA)
      if nA > 0
        ck_to_cid = Dict{NTuple{PC,Int}, Int}()
        sizehint!(ck_to_cid, nA)
        n_unique = 0
        @inbounds for ii in 1:nA
          akey = A.keys[ii]
          ckey = ntuple(j -> akey[c_src_axes[j]], Val(PC))
          cid  = get(ck_to_cid, ckey, 0)
          if cid == 0
            n_unique += 1
            ck_to_cid[ckey] = n_unique
            cid = n_unique
          end
          cid_for[ii] = cid
        end
        sorted_pairs = collect(ck_to_cid)
        sort!(sorted_pairs; by = first)
        resize!(C.keys, n_unique)
        resize!(C.ids,  n_unique)
        @inbounds for k in 1:n_unique
          C.keys[k] = sorted_pairs[k].first
          C.ids[k]  = sorted_pairs[k].second
        end
        total_data = n_unique * C.blksize
        resize!(C.data, total_data)
        fill!(C.data, zero(TC))
      end
    end

    # ---------- Main loop: one GEMM per A block, directly into C ----------
    # Mirrors contract_bs_dense_to_dense!: group by shared_prefix only to
    # slice Bp once per run (cheap; just a view), then for each A block
    # in the run do a single mul! with β=1 straight into its destination
    # C block. No Astack/Cstack, no gather/scatter, no per-run buffer
    # resizes — relies on BLAS for the per-block GEMM.
    @inbounds while iA <= nA
      iA2 = _advance_run(A.keys, iA, nA, join_posA)

      akey0 = A.keys[iA]
      if n_sp == 0
        Bmat = reshape(Bp, K, N)
      else
        sp_vals = ntuple(Val(n_sp)) do t
          akey0[join_posA[t]]
        end
        idx = (sp_vals..., ntuple(_ -> Colon(), NB - n_sp)...)
        @views Bsub = Bp[idx...]
        Bmat = reshape(Bsub, K, N)
      end

      @timeit TIMER "kdb.gemm" begin
        if mode == :AthenB
          for ii in iA:(iA2 - 1)
            A_mat = reshape(_block_view(A, A.ids[ii]), M, K)
            C_mat = reshape(_block_view(C, cid_for[ii]), M, N)
            mul!(C_mat, A_mat, Bmat, one(TC), one(TC))
          end
        else
          # :BA — output transposed (N × M per block)
          for ii in iA:(iA2 - 1)
            A_mat = reshape(_block_view(A, A.ids[ii]), M, K)
            C_mat = reshape(_block_view(C, cid_for[ii]), N, M)
            mul!(C_mat, transpose(Bmat), transpose(A_mat), one(TC), one(TC))
          end
        end
      end

      iA = iA2
    end
  end
  return C
end



# Dim histogram (cumulative), filled by GEMM_DIMS_HIST instrumentation; currently no
# kernel records into it (the BS×Dense fission path that did was removed).
const _GEMM_DIMS_HIST = Dict{NTuple{4,Int}, Int}()

"""
    show_gemm_dims_hist()

Print top-20 most frequent (M=D_remA, K, N=N_full, n_in_run) buckets accumulated
when GEMM_DIMS_HIST=1. Each call's mul!-count is `n_in_run`, so total mul!s
for bucket = bucket_count × n_in_run.
"""
function show_gemm_dims_hist()
  if isempty(_GEMM_DIMS_HIST)
    println("[GEMM_DIMS_HIST] (empty — run dmrg(...; roofline=true) to populate)")
    return
  end
  pairs = collect(_GEMM_DIMS_HIST)
  sort!(pairs; by = p -> -p[2])  # by frequency desc
  total_calls = sum(p[2] for p in pairs)
  total_mul   = sum(p[2] * p[1][4] for p in pairs)
  total_flops = sum(p[2] * p[1][4] * p[1][1] * p[1][2] * p[1][3] * 2 for p in pairs)
  println("\n[GEMM_DIMS_HIST] unique_buckets=$(length(pairs))  hint_calls=$total_calls  total_mul!=$total_mul  total_flops=$total_flops")
  println("  rank   (M=D_remA, K, N=N_full, n_in_run)   bucket_count    mul!_count    flops_in_bucket")
  cum_mul = 0; cum_flops = 0
  for (i, (key, cnt)) in enumerate(pairs[1:min(20, end)])
    m, k, n, nr = key
    bucket_muls = cnt * nr
    bucket_flops = bucket_muls * m * k * n * 2
    cum_mul += bucket_muls
    cum_flops += bucket_flops
    pct_mul = round(100 * cum_mul / total_mul, digits=1)
    pct_flop = round(100 * cum_flops / total_flops, digits=1)
    println("  $i  ($m, $k, $n, $nr)  cnt=$cnt  mul!s=$bucket_muls  flops=$bucket_flops  [cum mul!%=$pct_mul, cum flop%=$pct_flop]")
  end
end


function contract_bs_dense_to_dense!(
    C::AbstractArray{TC},
    labelsC::AbstractVector{Label},
    A::NewBlockSparseSorted{TA,NA,NA2,PA},
    labelsA::AbstractVector{Label},
    B::AbstractArray{TB,NB},
    labelsB::AbstractVector{Label};
    in_position::Bool=false,
) where {TC,TA,NA,NA2,PA,TB,NB}
    NC = ndims(C)
    if isempty(A.keys)
        @timeit TIMER "bdd.zero_C" fill!(C, zero(TC))
        return C
    end

    @timeit TIMER "bdd.classify" begin
        mapA = Dict(l => i for (i,l) in enumerate(labelsA))
        mapB = Dict(l => i for (i,l) in enumerate(labelsB))
        mapC = Dict(l => i for (i,l) in enumerate(labelsC))

        shared_prefix = [l for l in labelsA[1:PA]     if  haskey(mapB, l) && !haskey(mapC, l)]
        c_prefix      = [l for l in labelsA[1:PA]     if  haskey(mapC, l)]
        keepA         = [l for l in labelsA[PA+1:end] if  haskey(mapC, l)]
        red_dense     = [l for l in labelsA[PA+1:end] if !haskey(mapC, l)]
        keepB         = [l for l in labelsB           if  haskey(mapC, l)]

        n_sp    = length(shared_prefix)
        n_rd    = length(red_dense)
        n_keepA = length(keepA)
        n_keepB = length(keepB)
        n_cpfx  = length(c_prefix)
    end

    _bdd_profile_active = _roofline_on()
    _bdd_t_pA = 0.0
    _bdd_t_pB = 0.0
    @timeit TIMER "bdd.permute_A" begin
        _t0 = _bdd_profile_active ? time_ns() : UInt64(0)
        dense_perm = vcat([mapA[l] - PA for l in keepA], [mapA[l] - PA for l in red_dense])
        if dense_perm != collect(1:NA-PA)
            full_perm = vcat(collect(1:PA), dense_perm .+ PA)
            A       = permutedims(A, full_perm)
            labelsA = labelsA[full_perm]
            mapA    = Dict(l => i for (i,l) in enumerate(labelsA))
        end
        _bdd_profile_active && (_bdd_t_pA = (time_ns() - _t0) / 1e9)
    end

    # Layout: [rd, keepB, sp] — sp trailing so Bsub slice is contiguous
    @timeit TIMER "bdd.permute_B" begin
        _t0 = _bdd_profile_active ? time_ns() : UInt64(0)
        permB = Vector{Int}(undef, NB)
        let i = 1
            @inbounds for l in red_dense;     permB[i] = mapB[l]; i += 1; end
            @inbounds for l in keepB;         permB[i] = mapB[l]; i += 1; end
            @inbounds for l in shared_prefix; permB[i] = mapB[l]; i += 1; end
        end
        if permB == collect(1:NB)
            Bp = B
        else
            dimsBp = ntuple(i -> size(B, permB[i]), Val(NB))
            nB     = length(B)
            permB_buf = _bdd_permB_buffer(TB, nB)
            Bp = reshape(view(permB_buf, 1:nB), dimsBp)
            Base.permutedims!(Bp, B, permB)
        end
        _bdd_profile_active && (_bdd_t_pB = (time_ns() - _t0) / 1e9)
    end

    @timeit TIMER "bdd.setup" begin
        dimsA_d = A.dims[PA+1:end]

        # M from A's permuted dense dims (keepA leads after permute_A)
        M = n_keepA == 0 ? 1 : prod(dimsA_d[1:n_keepA])
        # K and N from original label sizes — independent of any layout choice
        K = n_rd    == 0 ? 1 : prod(size(B, mapB[l]) for l in red_dense)
        N = n_keepB == 0 ? 1 : prod(size(B, mapB[l]) for l in keepB)

        join_posA         = [mapA[l] for l in shared_prefix]
        c_prefix_pos_in_A = [mapA[l] for l in c_prefix]

        # Canonical C layout: keepA, keepB first then c_prefix last
        # so each C_slice is a contiguous (M*N) block
        canon_labels = vcat(keepA, keepB, c_prefix)
        perm_C       = [mapC[l] for l in canon_labels]
        if perm_C == collect(1:NC)
            Ctgt = C
            canon_owns_buffer = false
            # C was freshly allocated via zeros(...) in wc.alloc_dense
            # (tensor_wrappers.jl), so it is already zero. Skip the redundant
            # fill! (saved ~18% of kernel time in offline replay). Gate-able
            # via SB_FORCE_ZERO_C=1 to verify regressions.
            if get(ENV, "SB_FORCE_ZERO_C", "0") == "1"
                @timeit TIMER "bdd.zero_C" fill!(C, zero(TC))
            end
        else
            canon_dims = ntuple(i -> size(C, perm_C[i]), Val(NC))
            n_canon    = prod(canon_dims)
            flat_buf   = _bdd_ctgt_buffer(TC, n_canon)
            Ctgt = reshape(view(flat_buf, 1:n_canon), canon_dims)
            # Pooled buffer carries stale data from prior calls — must zero.
            @timeit TIMER "bdd.zero_C" fill!(Ctgt, zero(TC))
            canon_owns_buffer = true
        end
    end

    nA = length(A.keys)

    @timeit TIMER "bdd.main_loop" begin

        # Optional: track how many sp values contribute to each cpfx slot.
        # This tells us how much reduction work each C slot is doing.
        if get(ENV, "SB_CPFX_STATS", "0") == "1"
            cpfx_hits = Dict{NTuple{n_cpfx,Int}, Int}()
            sizehint!(cpfx_hits, nA)
        end

        _bdd_t_gemm = 0.0
        for ii in 1:nA
            akey     = A.keys[ii]
            sp_vals  = ntuple(t -> akey[join_posA[t]],         n_sp)
            cpfx_idx = ntuple(j -> akey[c_prefix_pos_in_A[j]], n_cpfx)

            @timeit TIMER "bdd.slice_B" begin
                # sp is trailing in Bp → fixing it leaves a contiguous (K×N) block
                Bsub  = (n_sp == 0) ? Bp :
                        @view Bp[ntuple(_ -> Colon(), NB - n_sp)..., sp_vals...]
                B_mat = reshape(Bsub, K, N)
                A_mat = reshape(_block_view(A, A.ids[ii]), M, K)
            end

            @timeit TIMER "bdd.gemm" begin
                # C_slice is contiguous because c_prefix is trailing in Ctgt
                C_slice = @view Ctgt[ntuple(_ -> Colon(), n_keepA + n_keepB)..., cpfx_idx...]
                # β=1 accumulates across sp values naturally — this IS the sparse reduction
                _t0 = _bdd_profile_active ? time_ns() : UInt64(0)
                mul!(reshape(C_slice, M, N), A_mat, B_mat, one(TC), one(TC))
                _bdd_profile_active && (_bdd_t_gemm += (time_ns() - _t0) / 1e9)
            end

            if get(ENV, "SB_CPFX_STATS", "0") == "1"
                cpfx_hits[cpfx_idx] = get(cpfx_hits, cpfx_idx, 0) + 1
            end
        end

        if get(ENV, "SB_CPFX_STATS", "0") == "1"
            n_unique = length(cpfx_hits)
            max_hits = maximum(values(cpfx_hits); init=0)
            n_multi  = count(v -> v > 1, values(cpfx_hits))
            println("--- c_prefix hit stats ---")
            println("  total A blocks          : ", nA)
            println("  unique cpfx slots in C  : ", n_unique)
            println("  max sp hits per slot    : ", max_hits)
            println("  slots with >1 sp hit    : ", n_multi,
                    "  (", round(100*n_multi/max(n_unique,1), digits=1), "%)")
            println("  avg sp hits per slot    : ",
                    round(nA / max(n_unique,1), digits=2))
        end

    end  # bdd.main_loop

    # --- GEMM input dump (SB_GEMM_DUMP=<path>) ---
    let io = _gemm_dump_io()
        if io !== nothing && _GEMM_DUMP_COUNT[] < _GEMM_DUMP_MAX[]
            @timeit TIMER "bdd.gemm_dump" begin
                blocks = Vector{NamedTuple{(:A_mat,:sp_vals,:cpfx_idx),
                                          Tuple{Matrix{TA},NTuple{n_sp,Int},NTuple{n_cpfx,Int}}}}(undef, nA)
                for ii in 1:nA
                    akey     = A.keys[ii]
                    sp_vals  = ntuple(t -> akey[join_posA[t]],         n_sp)
                    cpfx_idx = ntuple(j -> akey[c_prefix_pos_in_A[j]], n_cpfx)
                    A_mat    = copy(reshape(_block_view(A, A.ids[ii]), M, K))
                    blocks[ii] = (A_mat = A_mat, sp_vals = sp_vals, cpfx_idx = cpfx_idx)
                end
                rec = (
                    call_id    = _GEMM_DUMP_COUNT[] + 1,
                    M          = M,
                    K          = K,
                    N          = N,
                    n_sp       = n_sp,
                    n_cpfx     = n_cpfx,
                    n_keepA    = n_keepA,
                    n_keepB    = n_keepB,
                    NB         = NB,
                    Bp         = Array(Bp),
                    B_orig     = Array(B),
                    permB      = copy(permB),
                    canon_dims = size(Ctgt),
                    blocks     = blocks,
                    TA         = TA,
                    TB         = TB,
                    TC         = TC,
                )
                serialize(io, rec)
                _GEMM_DUMP_COUNT[] += 1
            end
        end
    end

    # --- Permute profile record ---
    if _bdd_profile_active
        _pp_io = _permute_profile_io_sb()
        if _pp_io !== nothing
            _pp_base = in_position ? "position" : "matvec"
            _pp_run  = CURRENT_RUN_LABEL[]
            _pp_bond = get(ENV, "SB_BOND", "-1")
            _pp_step = something(CURRENT_STEP[], -1)
            _pp_site = string(_pp_base, "|", _pp_run, "|", _pp_bond, "|", _pp_step)
            _pp_total = _bdd_t_pA + _bdd_t_pB + _bdd_t_gemm
            _pp_shared_pos_A = vcat([findfirst(==(l), labelsA) for l in shared_prefix],
                                     [findfirst(==(l), labelsA) for l in red_dense])
            _pp_shared_pos_B_orig = vcat([findfirst(==(l), labelsB) for l in shared_prefix],
                                          [findfirst(==(l), labelsB) for l in red_dense])
            println(_pp_io, _pp_site, "\tsparseH\t",
                    NA, "\t", NB, "\t", NC, "\t",
                    n_sp + n_rd, "\t",
                    _bdd_t_pA, "\t", _bdd_t_pB, "\t", _bdd_t_gemm, "\t", _pp_total, "\t",
                    join(_pp_shared_pos_A, ";"), "\t",
                    join(_pp_shared_pos_B_orig, ";"), "\t",
                    "keepA=", n_keepA, ";keepB=", n_keepB, ";cpfx=", n_cpfx, "\t",
                    "M=", M, ";K=", K, ";N=", N, "\t",
                    "ctgt_size=", join(string.(size(Ctgt)), ","))
        end
    end

    if canon_owns_buffer
        @timeit TIMER "bdd.permute_back" begin
            Base.permutedims!(C, Ctgt, Tuple(invperm(perm_C)))
        end
    end

    return C
end



# function contract_bs_dense_to_dense!(
#     C::AbstractArray{TC},
#     labelsC::AbstractVector{Label},
#     A::NewBlockSparseSorted{TA,NA,NA2,PA},
#     labelsA::AbstractVector{Label},
#     B::AbstractArray{TB,NB},
#     labelsB::AbstractVector{Label},
# ) where {TC,TA,NA,NA2,PA,TB,NB}
#     NC    = ndims(C)
#     dimsC = ntuple(i -> size(C, i), Val(NC))
#     C_bs  = NewBlockSparseSorted{TC, NC, NC}(dimsC)
#     contract!(C_bs, labelsC, A, labelsA, B, labelsB)
#     if !isempty(C_bs.keys)
#         blk = _block_view(C_bs, C_bs.ids[1])
#         NC == 0 ? (C[] = blk[1]) : copyto!(C, reshape(blk, dimsC))
#     end
#     return C
# end



# function contract_bs_dense_to_dense!(
#     C::AbstractArray{TC},
#     labelsC::AbstractVector{Label},
#     A::NewBlockSparseSorted{TA,NA,NA2,PA},
#     labelsA::AbstractVector{Label},
#     B::AbstractArray{TB,NB},
#     labelsB::AbstractVector{Label},
# ) where {TC,TA,NA,NA2,PA,TB,NB}
#     NC = ndims(C)
#     isempty(A.keys) && (fill!(C, zero(TC)); return C)

#     # ── label maps & classify (unchanged) ─────────────────────────────────────
#     mapA = Dict(l => i for (i,l) in enumerate(labelsA))
#     mapB = Dict(l => i for (i,l) in enumerate(labelsB))
#     mapC = Dict(l => i for (i,l) in enumerate(labelsC))

#     shared_prefix = [l for l in labelsA[1:PA]     if  haskey(mapB,l) && !haskey(mapC,l)]
#     c_prefix      = [l for l in labelsA[1:PA]     if  haskey(mapC,l)]
#     keepA         = [l for l in labelsA[PA+1:end] if  haskey(mapC,l)]
#     red_dense     = [l for l in labelsA[PA+1:end] if !haskey(mapC,l)]
#     keepB         = [l for l in labelsB           if  haskey(mapC,l)]

#     n_sp    = length(shared_prefix)
#     n_rd    = length(red_dense)
#     n_keepA = length(keepA)
#     n_keepB = length(keepB)
#     n_cpfx  = length(c_prefix)

#     # ── permute A dense dims (unchanged) ─────────────────────────────────────
#     dense_perm = vcat([mapA[l]-PA for l in keepA], [mapA[l]-PA for l in red_dense])
#     if dense_perm != collect(1:NA-PA)
#         fp = vcat(collect(1:PA), dense_perm .+ PA)
#         A = permutedims(A, fp); labelsA = labelsA[fp]
#         mapA = Dict(l => i for (i,l) in enumerate(labelsA))
#     end

#     # ── dimensions ────────────────────────────────────────────────────────────
#     dimsA_d = A.dims[PA+1:end]
#     M = n_keepA == 0 ? 1 : prod(dimsA_d[1:n_keepA])
#     K = n_rd    == 0 ? 1 : prod(dimsA_d[n_keepA+1:end])
#     join_posA         = [mapA[l] for l in shared_prefix]
#     c_prefix_pos_in_A = [mapA[l] for l in c_prefix]

#     # ── permute_B: avoid full copy when per-run Bmat is cheaper ───────────────
#     #
#     # FIX: instead of one permutedims(B, permB) costing 550 µs + 823 KB every
#     # call, precompute k_offsets / n_offsets once and fill a small (K×N) scratch
#     # buffer per run.  Total data touched = n_runs × K × N ≤ length(B), but
#     # in column-major chunks that stay hot in cache right before the GEMM.
#     #
#     sp_pos_B = [mapB[l] for l in shared_prefix]
#     rd_pos_B = [mapB[l] for l in red_dense]
#     kB_pos_B = [mapB[l] for l in keepB]
#     permB    = vcat(sp_pos_B, rd_pos_B, kB_pos_B)
#     need_perm = (permB != collect(1:NB))

#     # N from B dims directly (needed before we know whether Bp exists)
#     N = n_keepB == 0 ? 1 : prod(size(B, kB_pos_B[j]) for j in 1:n_keepB)

#     local Bp                         # only assigned in the !per_run path
#     local sp_strides_B, k_offsets, n_offsets, B_flat, bmat_scratch

#     nA = length(A.keys)
#     use_per_run_bmat = need_perm

#     if !need_perm
#         Bp = B                       # identity permutation: free
#         use_per_run_bmat = false
#     else
#         # Precompute once: flat offset in B for each K and N multi-index.
#         sp_strides_B = Int[stride(B, sp_pos_B[t]) for t in 1:n_sp]
#         rd_strides_B = Int[stride(B, rd_pos_B[i]) for i in 1:n_rd]
#         kB_strides_B = Int[stride(B, kB_pos_B[j]) for j in 1:n_keepB]
#         rd_dims      = [size(B, rd_pos_B[i]) for i in 1:n_rd]
#         kB_dims      = [size(B, kB_pos_B[j]) for j in 1:n_keepB]

#         k_offsets = Vector{Int}(undef, K)
#         rd_idx = ones(Int, n_rd)
#         for k in 1:K
#             off = 0
#             @inbounds for i in 1:n_rd; off += (rd_idx[i]-1) * rd_strides_B[i]; end
#             k_offsets[k] = off
#             @inbounds for i in 1:n_rd
#                 rd_idx[i] += 1; rd_idx[i] <= rd_dims[i] && break; rd_idx[i] = 1
#             end
#         end

#         n_offsets = Vector{Int}(undef, N)
#         kB_idx = ones(Int, n_keepB)
#         for n in 1:N
#             off = 0
#             @inbounds for j in 1:n_keepB; off += (kB_idx[j]-1) * kB_strides_B[j]; end
#             n_offsets[n] = off
#             @inbounds for j in 1:n_keepB
#                 kB_idx[j] += 1; kB_idx[j] <= kB_dims[j] && break; kB_idx[j] = 1
#             end
#         end

#         B_flat       = reshape(B, :)           # O(1) flat view of B
#         bmat_scratch = Vector{TC}(undef, K*N)  # one K×N buffer, reused each run
#     end

#     MN = M * N

#     # ── Ctgt: canonical layout (keepA, keepB, c_prefix LAST) ─────────────────
#     canon_labels = vcat(keepA, keepB, c_prefix)
#     perm_C       = [mapC[l] for l in canon_labels]

#     if perm_C == collect(1:NC)
#         Ctgt = C; canon_owns_buffer = false; fill!(C, zero(TC))
#     else
#         canon_dims = ntuple(i -> size(C, perm_C[i]), Val(NC))
#         n_canon    = prod(canon_dims)
#         flat_buf   = _bdd_ctgt_buffer(TC, n_canon)
#         Ctgt       = reshape(view(flat_buf, 1:n_canon), canon_dims)
#         fill!(Ctgt, zero(TC)); canon_owns_buffer = true
#     end

#     # FIX (scatter): flat view + precomputed int strides → fully type-stable.
#     # view(Ctgt_vec, base:base+MN-1) is a 1-D SubArray{TC} with stride 1;
#     # reshape to (M,N) gives a StridedMatrix BLAS can write into directly.
#     # No dynamic ntuple-of-Colons, no dynamic dispatch on .+=.
#     Ctgt_vec = vec(Ctgt)
#     cpfx_strides_ctgt = n_cpfx == 0 ? Int[] :
#         Int[stride(Ctgt, n_keepA + n_keepB + j) for j in 1:n_cpfx]

#     # ── Sort-based grouping: O(nA log nA), O(nA) total alloc, no Dict ─────────
#     #
#     # FIX: replaces Dict{NTuple,Vector{Int}} which creates one heap object per
#     # unique sp_val group.  sort! + a boundary scan is faster and GC-free.
#     sp_order = collect(1:nA)
#     n_sp > 0 && sort!(sp_order,
#         by = i -> ntuple(t -> A.keys[i][join_posA[t]], n_sp))

#     run_starts = Int[1]
#     @inbounds for idx in 2:nA
#         ip, ic = sp_order[idx-1], sp_order[idx]
#         for t in 1:n_sp
#             if A.keys[ic][join_posA[t]] != A.keys[ip][join_posA[t]]
#                 push!(run_starts, idx); break
#             end
#         end
#     end
#     push!(run_starts, nA + 1)
#     n_runs = length(run_starts) - 1

#     # ── Main loop ─────────────────────────────────────────────────────────────
#     @inbounds for run_idx in 1:n_runs
#         r_start = run_starts[run_idx]
#         r_end   = run_starts[run_idx+1] - 1

#         ii_first = sp_order[r_start]
#         sp_vals  = ntuple(t -> A.keys[ii_first][join_posA[t]], n_sp)

#         # Build Bmat for this run ─────────────────────────────────────────────
#         Bmat = if use_per_run_bmat
#             # Scatter-gather from B using precomputed offsets; no large allocation.
#             sp_base = 0
#             for t in 1:n_sp; sp_base += (sp_vals[t]-1) * sp_strides_B[t]; end
#             Bm = reshape(view(bmat_scratch, 1:K*N), K, N)
#             for n in 1:N, k in 1:K
#                 Bm[k,n] = B_flat[sp_base + k_offsets[k] + n_offsets[n] + 1]
#             end
#             Bm
#         else
#             Bsub = n_sp == 0 ? Bp :
#                    @view Bp[sp_vals..., ntuple(_->Colon(), NB-n_sp)...]
#             reshape(Bsub, K, N)
#         end

#         # One mul! per block, directly into its Ctgt slab ─────────────────────
#         #
#         # FIX: eliminates Astack, Cstack, gather_A, and the type-unstable scatter.
#         # Because c_prefix is LAST in Ctgt (column-major), each block's MN
#         # elements are contiguous → reshape(view(Ctgt_vec,…),M,N) is a valid
#         # StridedMatrix, and mul! with β=1 accumulates in-place via BLAS.
#         for idx in r_start:r_end
#             ii   = sp_order[idx]
#             akey = A.keys[ii]

#             base = 1
#             for j in 1:n_cpfx
#                 base += (akey[c_prefix_pos_in_A[j]] - 1) * cpfx_strides_ctgt[j]
#             end

#             Amat   = reshape(_block_view(A, A.ids[ii]), M, K)
#             Cslice = reshape(view(Ctgt_vec, base:base+MN-1), M, N)
#             mul!(Cslice, Amat, Bmat, true, true)   # β=1: accumulate, not overwrite
#         end
#     end

#     canon_owns_buffer && Base.permutedims!(C, Ctgt, Tuple(invperm(perm_C)))
#     return C
# end

# function contract_bs_dense_to_dense!(
#     C::AbstractArray{TC},
#     labelsC::AbstractVector{Label},
#     A::NewBlockSparseSorted{TA,NA,NA2,PA},
#     labelsA::AbstractVector{Label},
#     B::AbstractArray{TB,NB},
#     labelsB::AbstractVector{Label},
# ) where {TC,TA,NA,NA2,PA,TB,NB}
#    NC = ndims(C)
#    if isempty(A.keys)
#        @timeit TIMER "bdd.zero_C" fill!(C, zero(TC))
#        return C
#    end

#    @timeit TIMER "bdd.classify" begin
#     # ---- label maps ----
#     mapA = Dict(l => i for (i,l) in enumerate(labelsA))
#     mapB = Dict(l => i for (i,l) in enumerate(labelsB))
#     mapC = Dict(l => i for (i,l) in enumerate(labelsC))

#     # ---- classify labels ----
#     shared_prefix = [l for l in labelsA[1:PA]     if  haskey(mapB, l) && !haskey(mapC, l)]
#     c_prefix      = [l for l in labelsA[1:PA]     if  haskey(mapC, l)]
#     keepA         = [l for l in labelsA[PA+1:end] if  haskey(mapC, l)]
#     red_dense     = [l for l in labelsA[PA+1:end] if !haskey(mapC, l)]
#     keepB         = [l for l in labelsB           if  haskey(mapC, l)]

#     n_sp    = length(shared_prefix)
#     n_rd    = length(red_dense)
#     n_keepA = length(keepA)
#     n_keepB = length(keepB)
#     n_cpfx  = length(c_prefix)

#     # if get(ENV, "INDEXSTATS", "0") == "1"
#     #     @show shared_prefix c_prefix keepA red_dense keepB
#     # end

#    end

#    @timeit TIMER "bdd.permute_A" begin
#     dense_perm = vcat([mapA[l] - PA for l in keepA], [mapA[l] - PA for l in red_dense])
#     if dense_perm != collect(1:NA-PA)
#         full_perm = vcat(collect(1:PA), dense_perm .+ PA)
#         A       = permutedims(A, full_perm)
#         labelsA = labelsA[full_perm]
#         mapA    = Dict(l => i for (i,l) in enumerate(labelsA))
#     end
#    end

#    @timeit TIMER "bdd.permute_B" begin
#     permB = Vector{Int}(undef, NB)
#     let i = 1
#         @inbounds for l in shared_prefix; permB[i] = mapB[l]; i += 1; end
#         @inbounds for l in red_dense;     permB[i] = mapB[l]; i += 1; end
#         @inbounds for l in keepB;         permB[i] = mapB[l]; i += 1; end
#       end
#     if permB == collect(1:NB)
#         Bp = B
#     else
#         dimsBp = ntuple(i -> size(B, permB[i]), Val(NB))
#         nB = length(B)
#         permB_buf = _bdd_permB_buffer(TB, nB)
#         Bp = reshape(view(permB_buf, 1:nB), dimsBp)
#         Base.permutedims!(Bp, B, permB)
#     end
#    end



#    @timeit TIMER "bdd.setup" begin
#     dimsA_d = A.dims[PA+1:end]
#     dimsB_p = size(Bp)
#     M = n_keepA == 0 ? 1 : prod(dimsA_d[1:n_keepA])
#     K = n_rd    == 0 ? 1 : prod(dimsA_d[n_keepA+1:end])
#     # N = n_keepB == 0 ? 1 : prod(dimsB_p[n_rd+1:n_rd+n_keepB])
#     N = n_keepB == 0 ? 1 : prod(dimsB_p[n_sp+n_rd+1:end])

#     join_posA         = [mapA[l] for l in shared_prefix]
#     c_prefix_pos_in_A = [mapA[l] for l in c_prefix]

#     # Canonical layout: c_prefix LAST so `Ctgt[:, :, ..., cpfx_idx...]` is a
#     # CONTIGUOUS (M*N) slab → scatter writes adjacent memory (cache-friendly,
#     # ~4× faster than strided into a PermutedDimsArray view).
#     canon_labels = vcat(keepA, keepB, c_prefix)
#     perm_C       = [mapC[l] for l in canon_labels]
#     if perm_C == collect(1:NC)
#         # labelsC already canonical — write directly into C.
#         Ctgt = C
#         canon_owns_buffer = false
#         @timeit TIMER "bdd.zero_C" fill!(C, zero(TC))
#     else
#         # Re-use a thread-local scratch buffer to avoid per-call malloc.
#         canon_dims = ntuple(i -> size(C, perm_C[i]), Val(NC))
#         n_canon    = prod(canon_dims)
#         flat_buf   = _bdd_ctgt_buffer(TC, n_canon)
#         Ctgt = reshape(view(flat_buf, 1:n_canon), canon_dims)
#         @timeit TIMER "bdd.zero_C" fill!(Ctgt, zero(TC))
#         canon_owns_buffer = true
#     end
#    end

#    # ---------- Group A.keys by shared_prefix via Dict (O(nA), no sort) ----------
#    # Dict insertion is cheaper than `sortperm + comparator` at this size; we
#    # don't need stable ordering for correctness, only same-sp-value grouping.
#    nA = length(A.keys)
#    @timeit TIMER "bdd.group_keys" begin
#     sp_order = Vector{Int}(undef, nA)
#     if n_sp == 0
#         @inbounds for i in 1:nA; sp_order[i] = i; end
#     else
#         groups = Dict{NTuple{n_sp,Int}, Vector{Int}}()
#         sizehint!(groups, nA)
#         @inbounds for ii in 1:nA
#             sp = ntuple(t -> A.keys[ii][join_posA[t]], n_sp)
#             push!(get!(() -> Int[], groups, sp), ii)
#         end
#         i = 1
#         for (_, idxs) in groups
#             @inbounds for ii in idxs
#                 sp_order[i] = ii
#                 i += 1
#             end
#         end
#     end
#    end

#    # Re-usable staging buffers for batched GEMM. Bsub stays constant inside a
#    # run, so we stack all A blocks of the run into Astack and do ONE big BLAS
#    # call (M*nb × K) × (K × N) → Cstack (M*nb × N), then scatter back.
#    Astack_buf = Vector{TC}(undef, 0)
#    Cstack_buf = Vector{TC}(undef, 0)

#    @timeit TIMER "bdd.main_loop" begin
#     iA = 1
#     while iA <= nA
#         # Identify the run [iA, iA2-1] of consecutive sp_order entries that
#         # share the same shared_prefix value.
#         ii_first = sp_order[iA]
#         akey0    = A.keys[ii_first]
#         sp_vals  = (n_sp == 0) ? () : ntuple(t -> akey0[join_posA[t]], n_sp)
#         iA2 = iA + 1
#         if n_sp > 0
#             @inbounds while iA2 <= nA
#                 next_key = A.keys[sp_order[iA2]]
#                 same = true
#                 for t in 1:n_sp
#                     if next_key[join_posA[t]] != sp_vals[t]
#                         same = false; break
#                     end
#                 end
#                 same || break
#                 iA2 += 1
#             end
#         else
#             iA2 = nA + 1   # one big run
#         end
#         nb = iA2 - iA

#         @timeit TIMER "bdd.slice_B" begin
#           # Bsub = (n_sp == 0) ? Bp :
#           #    @view Bp[ntuple(_ -> Colon(), NB - n_sp)..., sp_vals...]
#           Bsub = (n_sp == 0) ? Bp :
#                   @view Bp[sp_vals..., ntuple(_ -> Colon(), NB - n_sp)...]
#           Bmat = reshape(Bsub, K, N)
#         end

#         @timeit TIMER "bdd.gather_A" begin
#             resize!(Astack_buf, M * nb * K)
#             Astack = reshape(view(Astack_buf, 1:M*nb*K), M * nb, K)
#             for k in 1:nb
#                 ii   = sp_order[iA + k - 1]
#                 Avec = _block_view(A, A.ids[ii])
#                 copyto!(view(Astack, (k-1)*M+1 : k*M, :), reshape(Avec, M, K))
#             end
#         end

#         @timeit TIMER "bdd.gemm" begin
#             resize!(Cstack_buf, M * nb * N)
#             Cstack = reshape(view(Cstack_buf, 1:M*nb*N), M * nb, N)
#             mul!(Cstack, Astack, Bmat)   # β=0 fresh write
#         end

#         @timeit TIMER "bdd.scatter" begin
#             # Optional diagnostic: tally how often A blocks within this run
#             # share a cpfx_idx tuple. Cheap (Dict per run) and disabled when
#             # SB_CPFX_STATS is unset.
#             if get(ENV, "SB_CPFX_STATS", "0") == "1"
#                 cpfx_count = Dict{NTuple{n_cpfx,Int}, Int}()
#                 sizehint!(cpfx_count, nb)
#                 for k in 1:nb
#                     ii = sp_order[iA + k - 1]
#                     akey = A.keys[ii]
#                     cp = ntuple(j -> akey[c_prefix_pos_in_A[j]], n_cpfx)
#                     cpfx_count[cp] = get(cpfx_count, cp, 0) + 1
#                 end
#                 n_unique = length(cpfx_count)
#                 max_g = 0
#                 n_in_dup = 0
#                 for (_, c) in cpfx_count
#                     max_g = max(max_g, c)
#                     c >= 2 && (n_in_dup += c)
#                 end
#                 CPFX_STATS[:n_runs]          += 1
#                 CPFX_STATS[:n_blocks]        += nb
#                 CPFX_STATS[:n_unique_cpfx]   += n_unique
#                 CPFX_STATS[:sum_max_group]   += max_g
#                 max_g >= 2 && (CPFX_STATS[:n_runs_with_dup] += 1)
#                 CPFX_STATS[:n_blocks_in_dup] += n_in_dup
#                 if nb == 1
#                     CPFX_STATS[:runs_nb_1] += 1
#                 elseif nb <= 4
#                     CPFX_STATS[:runs_nb_le_4] += 1
#                 elseif nb <= 16
#                     CPFX_STATS[:runs_nb_le_16] += 1
#                 else
#                     CPFX_STATS[:runs_nb_gt_16] += 1
#                 end
#             end

#             for k in 1:nb
#                 ii       = sp_order[iA + k - 1]
#                 akey     = A.keys[ii]
#                 cpfx_idx = ntuple(j -> akey[c_prefix_pos_in_A[j]], n_cpfx)
#                 # Ctgt has keepA/keepB FIRST and c_prefix LAST → this slice is
#                 # ONE contiguous (M*N) memory block, so `.+=` runs at memory-
#                 # bandwidth speed instead of strided-access speed.
#                 Cslice   = @view Ctgt[ntuple(_ -> Colon(), n_keepA + n_keepB)..., cpfx_idx...]
#                 Cchunk   = view(Cstack, (k-1)*M+1 : k*M, :)
#                 Cslice  .+= reshape(Cchunk, size(Cslice))
#             end
#         end

#         iA = iA2
#     end
#    end  # bdd.main_loop

#    if canon_owns_buffer
#        @timeit TIMER "bdd.permute_back" begin
#            Base.permutedims!(C, Ctgt, Tuple(invperm(perm_C)))
#        end
#    end

#    return C
# end

