# blocksparse/factorize.jl
#
# Block-sparse (non-aliased) single-site factorize. The bond is one sparse label axis plus
# one dense multiplicity axis. Labels are grouped (a row key that meets several labels joins
# them into one group) and each group keeps its own basis, named by its smallest label;
# labels are never renumbered, so a group that keeps nothing just loses its blocks.
#   kept_counts                    the truncation rule (per group)
#   blocksparse_factorize          kernel on NewBlockSparseSorted, by storage axis positions
#   _external_factorize_storage    the `factorize` hook (orthogonalize!, truncate!)
#   _absorb_bond                   the neighbour's M * R (ITensorMPS `_absorb_bond` hook)

import ITensors
using LinearAlgebra: svd

"""
    kept_counts(spectra; maxdim, cutoff) -> (k::Vector{Int}, truncerr)

Kept values per spectrum (one per group of the bond), from each spectrum's singular values
(descending). Each keeps at most `maxdim` values and never an exact zero. `cutoff` bounds the
discarded weight pooled over all spectra, relative to the total: the aliased / dense rule
with every value in one spectrum. truncerr = discarded / total weight. Errors if nothing is
kept.
"""
function kept_counts(spectra::AbstractVector{<:AbstractVector{<:Real}};
                     maxdim::Int, cutoff::Float64)
    k = [min(maxdim, count(s -> s > 0, sv)) for sv in spectra]
    total = sum(sv -> sum(abs2, sv; init = 0.0), spectra; init = 0.0)
    if cutoff > 0
        vals = sort!([(Float64(s), l) for l in eachindex(spectra) for s in spectra[l]];
                     by = first, rev = true)
        running = 0.0
        for i in length(vals):-1:1
            running += first(vals[i])^2
            if running > cutoff * total
                kc = zeros(Int, length(spectra))
                for j in 1:i; kc[last(vals[j])] += 1; end
                k .= min.(k, kc); break
            end
        end
    end
    any(>(0), k) || error("kept_counts: every label would be truncated away")
    dropped = sum(l -> sum(abs2, @view(spectra[l][(k[l] + 1):end]); init = 0.0),
                  eachindex(spectra); init = 0.0)
    return k, dropped / total
end

"""
    blocksparse_factorize(A, lpos; maxdim, cutoff) -> (L, R, spec)

SVD of `A` across (storage axes `lpos`) x (bond), one per group of bond labels. The bond is
the complement of `lpos`: exactly one sparse (label) axis and one dense (multiplicity) axis,
anything else errors. A block's row key is its key without the label axis.

Groups: row key r and label b are connected when (r with b) is a key of A; a group G is a
connected set of row keys and labels, named by its smallest label b*. Each row key is in
exactly one group. Each block's dense tail is read as (pre, mult, post), so it enters its
group's matrix
    X_G[(row key, pre, post), (label, mult)]
(row keys in A's key order, labels ascending; absent (row key, label) pairs are zero) by
slice copies, with no permute; that copy is the only data movement. A group with one label
(a label-conserving bond) is exactly that label's matrix X_b[(block, pre, post), mult].

  L  keys (r with b*): one per row key of a kept group, in A's axis order, with the
     multiplicity axis -> n_new = max kept; U_G on the kept columns, zero padding after.
     Conserving bond: A's keys of the kept labels.
  R  (new label, old label, new mult, old mult), keys (b*, b) for every label b of a kept
     group, each block the S_G V_G' columns of b, padded. Conserving bond: keys (b, b).
  spec  Spectrum(kept s^2, discarded / total weight); the rule is `kept_counts` (maxdim per
     group, cutoff pooled).
"""
function blocksparse_factorize(A::NewBlockSparseSorted{T, N, N2, P, K}, lpos;
                               maxdim::Int, cutoff::Float64) where {T, N, N2, P, K}
    bpos = sort!(setdiff(1:N, lpos))
    (length(bpos) == 2 && bpos[1] <= P && bpos[2] > P) ||
        error("block-sparse factorize: the bond (axes outside Linds) must be one sparse " *
              "label axis and one dense multiplicity axis; got storage axes $bpos of $N " *
              "($P sparse)")
    jl, jm = bpos
    m    = A.dims[jm]
    pre  = prod(A.dims[(P + 1):(jm - 1)]; init = 1)
    post = A.blksize ÷ (pre * m)
    nrow = pre * post

    # ---- groups: union-find over labels; a row key joins every label it meets ---------
    rowkey(i) = ntuple(j -> A.keys[i][j < jl ? j : j + 1], Val(P - 1))
    root = collect(1:A.dims[jl])                  # root of a group = its smallest label b*
    findroot(b) = (while root[b] != b; b = root[b] = root[root[b]]; end; b)
    first_lab = Dict{NTuple{P - 1, K}, Int}()
    for i in eachindex(A.keys)
        b = Int(A.keys[i][jl])
        x, y = findroot(get!(first_lab, rowkey(i), b)), findroot(b)
        x == y || (x < y ? (root[y] = x) : (root[x] = y))
    end
    bins = [Int[] for _ in 1:A.dims[jl]]          # blocks of each group, indexed by b*
    for i in eachindex(A.keys); push!(bins[findroot(Int(A.keys[i][jl]))], i); end
    live = [g for g in eachindex(bins) if !isempty(bins[g])]

    # ---- one matrix and one SVD per group ------------------------------------------
    # per group: row slot / column slot of each block, the first block of each row slot,
    # the group's labels (ascending)
    grp = map(live) do g
        slots = Dict{NTuple{P - 1, K}, Int}()
        rix   = [get!(slots, rowkey(i), length(slots) + 1) for i in bins[g]]
        reps  = zeros(Int, length(slots))
        for (t, i) in enumerate(bins[g]); reps[rix[t]] == 0 && (reps[rix[t]] = i); end
        labs  = sort!(unique(Int(A.keys[i][jl]) for i in bins[g]))
        cix   = [searchsortedfirst(labs, Int(A.keys[i][jl])) for i in bins[g]]
        (; rix, cix, reps, labs)
    end
    Fs = map(eachindex(live)) do t
        G = grp[t]
        X = zeros(T, length(G.reps) * nrow, length(G.labs) * m)
        for (u, i) in enumerate(bins[live[t]])
            B  = reshape(_block_view(A, A.ids[i]), pre, m, post)
            r  = G.rix[u]
            cs = ((G.cix[u] - 1) * m + 1):(G.cix[u] * m)
            for q in 1:post
                X[((r - 1) * nrow + (q - 1) * pre + 1):((r - 1) * nrow + q * pre), cs] .=
                    @view(B[:, :, q])
            end
        end
        svd(X)
    end
    k, truncerr = kept_counts([F.S for F in Fs]; maxdim, cutoff)
    n_new = maximum(k)
    kept = [t for t in eachindex(live) if k[t] > 0]

    # ---- L: keys (r with b*) of the kept groups, U_G written straight into A's axis order
    pdims = ntuple(j -> A.dims[j], Val(P))
    Lent  = [(Base.setindex(A.keys[i], K(live[t]), jl), t, r)
             for t in kept for (r, i) in enumerate(grp[t].reps)]
    sort!(Lent; by = e -> _prefix_lin(e[1], pdims))
    blkL  = pre * n_new * post
    dataL = zeros(T, length(Lent) * blkL)
    for (s, (_, t, r)) in enumerate(Lent)
        U  = Fs[t].U
        BL = reshape(view(dataL, ((s - 1) * blkL + 1):(s * blkL)), pre, n_new, post)
        for q in 1:post
            BL[:, 1:k[t], q] .=
                @view(U[((r - 1) * nrow + (q - 1) * pre + 1):((r - 1) * nrow + q * pre), 1:k[t]])
        end
    end
    L = NewBlockSparseSorted{T, N, N2, P, K}(Base.setindex(A.dims, n_new, jm), blkL,
                                             [first(e) for e in Lent], collect(1:length(Lent)), dataL)

    # ---- R: (new label, old label, new mult, old mult), keys (b*, b), blocks S_G V_G' ----
    # each old label b is in one group, so keys sorted by b are in _prefix_lin order
    Rent = sort!([(b, t, c) for t in kept for (c, b) in enumerate(grp[t].labs)]; by = first)
    blkR  = n_new * m
    dataR = zeros(T, length(Rent) * blkR)
    for (s, (_, t, c)) in enumerate(Rent)
        F = Fs[t]
        BR = reshape(view(dataR, ((s - 1) * blkR + 1):(s * blkR)), n_new, m)
        BR[1:k[t], :] .= F.S[1:k[t]] .* @view(F.Vt[1:k[t], ((c - 1) * m + 1):(c * m)])
    end
    R = NewBlockSparseSorted{T, 4, 2, 2, K}((A.dims[jl], A.dims[jl], n_new, m), blkR,
                                            [(K(live[t]), K(b)) for (b, t, _) in Rent],
                                            collect(1:length(Rent)), dataR)

    eigs = sort!(reduce(vcat, [abs2.(Fs[t].S[1:k[t]]) for t in kept]); rev = true)
    return L, R, ITensors.Spectrum(eigs, truncerr)
end

# Single-site factorize of a block-sparse tensor (the `factorize` hook). Linds -> storage
# positions -> kernel. L keeps A's index order with the bond's label and multiplicity
# indices replaced (new Index, same tags); R is block-sparse over
# (new label, old label, new mult, old mult), keys (b*, b), for the neighbour's `_absorb_bond`.
# ortho = "left" and svd only; the other factorize options error.
function ITensors._external_factorize_storage(
    Aw :: WrappedBlockSparse,
    A  :: ITensors.ITensor,
    Linds...;
    mindim = nothing, maxdim = nothing, cutoff = nothing, ortho = nothing, tags = nothing,
    plev = nothing, which_decomp = nothing, eigen_perturbation = nothing, svd_alg = nothing,
    use_absolute_cutoff = nothing, use_relative_cutoff = nothing, min_blockdim = nothing,
    (singular_values!) = nothing, dir = nothing,
)
    something(ortho, "left") == "left" ||
        error("block-sparse factorize: only ortho=\"left\" is supported (got $ortho)")
    something(which_decomp, "svd") == "svd" ||
        error("block-sparse factorize: only which_decomp=\"svd\" is supported (got $which_decomp)")
    all(isnothing, (mindim, plev, eigen_perturbation, svd_alg, use_absolute_cutoff,
                    use_relative_cutoff, min_blockdim, singular_values!, dir)) ||
        error("block-sparse factorize: mindim, plev, eigen_perturbation, svd_alg, " *
              "use_absolute_cutoff, use_relative_cutoff, min_blockdim, singular_values! " *
              "and dir are not supported")

    Lis  = ITensors.commoninds(A, ITensors.indices(Linds...))
    lpos = [findfirst(==(I), Aw.inds) for I in Lis]
    Lbs, Rbs, spec = blocksparse_factorize(Aw.blocksparse, lpos;
                                           maxdim = something(maxdim, typemax(Int)),
                                           cutoff = Float64(something(cutoff, 0.0)))
    jl, jm = sort!(setdiff(1:length(Aw.inds), lpos))
    old_lab, old_mul = Aw.inds[jl], Aw.inds[jm]
    new_lab = ITensors.Index(ITensors.dim(old_lab); tags = ITensors.tags(old_lab))
    new_mul = ITensors.Index(Lbs.dims[jm]; tags = ITensors.tags(old_mul))
    swap(I) = I == old_lab ? new_lab : I == old_mul ? new_mul : I
    L = ITensors._itensor_from_external_storage(WrappedBlockSparse(Lbs, map(swap, Aw.inds)))
    R = ITensors._itensor_from_external_storage(
        WrappedBlockSparse(Rbs, (new_lab, old_lab, new_mul, old_mul)))
    return L, R, spec
end

# `_absorb_bond` for a block-sparse site: M * R with R from the neighbour's factorize (above).
# One `contract` call; R's new label and multiplicity indices take the old ones' slots in M
# (`_absorb_layout`), so the site keeps its index order and key/dense split whichever way the
# sweep runs. The contract joins M's old label with R's (b*, b) keys; M's blocks whose labels
# share a group b* sum into one output block (keys with b -> b*).
function _absorb_bond(::NewBlockSparseSorted, Mw::WrappedBlockSparse,
                      M::ITensors.ITensor, R::ITensors.ITensor)
    old, new = ITensors.commoninds(R, M), ITensors.uniqueinds(R, M)
    return ITensors.contract(M, R; preserve_bs_output = true,
                             layout = _absorb_layout(Mw.inds, _bs_head_len(Mw), old, new))
end
