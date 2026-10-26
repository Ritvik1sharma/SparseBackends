# aliased/sector_split/factorize.jl
#
# Single-site factorize of a SectorSplitAliasedBlockSparse site: the `factorize` hook's method
# for this storage (the hook itself, ITensors._external_factorize_storage, is in
# tensor_wrappers_aliased.jl and dispatches on the storage's child type).

import ITensors
using LinearAlgebra: svd

# Per-channel single-site factorize (the `factorize` hook for SectorSplitAliasedBlockSparse).
# The bond is the channel (key) index c and the multiplicity (dense) index m outside Linds.
# Per channel value c:
#     W_c = vcat_{a : w_{a,c} > 0} sqrt(w_{a,c}) T_a,   w_{a,c} = Σ_{keys: alias a, channel c} |scalar|²
# one SVD each; the kept counts k_c follow `kept_counts` (maxdim per channel, cutoff pooled).
#   L  every key of A kept (keys and scalars unchanged); one template per (template a, channel c)
#      pair present in the keys (a-major), U_{a,c} / sqrt(w_{a,c}) on the k_c kept columns and zero
#      after (all zero when w_{a,c} = 0 or k_c = 0); m -> K = max_c k_c. The channel index is kept
#      (it is P's link: the gauges of the next gate step refer to it). No numeric merge;
#      allocation exact.
#   R  aliased (Shared) on (c', c, m_new, m_old), c' = prime(c): diagonal keys (c, c) only, one
#      template per channel value, S_c V_c† (K x m_old, rows > k_c zero; zero for a channel with
#      no weight), scalars 1. contract(neighbour, R) sums c against the neighbour's channel and
#      leaves c', which this storage's `_absorb_bond` (sector_split/absorb.jl) renames back to c.
#   spec  Spectrum(kept s² over every channel, discarded / total weight).
function _aliased_factorize(X::SectorSplitAliasedBlockSparse{Tel,N,N2,Px,K,AI},
                            Aw::WrappedAliasedBlockSparse, A::ITensors.ITensor, Linds...;
                            mindim, maxdim, cutoff) where {Tel,N,N2,Px,K,AI}
    mindim === nothing || error("per-channel aliased factorize: mindim is not supported")
    maxdim = something(maxdim, typemax(Int))
    cutoff = Float64(something(cutoff, 0.0))

    Lis = ITensors.commoninds(A, ITensors.indices(Linds...))
    out = [p for p in 1:N if !(Aw.inds[p] in Lis)]
    (length(out) == 2 && out[1] <= Px && out[2] > Px) || error(
        "per-channel aliased factorize: the bond (indices outside Linds) must be one channel (key) " *
        "index and one multiplicity (dense) index; got storage axes $out of $N ($Px key axes)")
    cx, mx = out
    chan, old_mult = Aw.inds[cx], Aw.inds[mx]
    Dc = X.dims[cx]
    nt = X.n_templates
    nk = length(X.keys)

    # ---- weights per (template, channel); every (template, channel) pair of the keys --------
    tail  = Int[X.dims[Px + i] for i in 1:N2]
    jx    = mx - Px
    m_old = tail[jx]
    bs    = X.blksize
    nrest = bs ÷ m_old
    perm  = [setdiff(1:N2, jx); jx]                      # m axis last
    iperm = invperm(perm)
    wac     = zeros(real(Tel), nt, Dc)
    present = falses(nt, Dc)
    @inbounds for i in 1:nk
        t = Int(X.alias_ids[i]); c = Int(X.keys[i][cx])
        wac[t, c] += abs2(X.scalars[i]); present[t, c] = true
    end

    # ---- one weighted SVD per channel value with weight -----------------------------------
    tpl(t) = reshape(permutedims(reshape(@view(X.templates[((t - 1) * bs + 1):(t * bs)]), tail...),
                                 perm), nrest, m_old)
    live = [c for c in 1:Dc if any(>(0), @view wac[:, c])]
    rows = [findall(>(0), @view wac[:, c]) for c in live]
    rowof = zeros(Int, nt, Dc)                           # (template, channel) -> its row block in W_c
    for (q, c) in enumerate(live), (r, t) in enumerate(rows[q]); rowof[t, c] = r; end
    Fs = map(eachindex(live)) do q
        c = live[q]
        W = Matrix{Tel}(undef, length(rows[q]) * nrest, m_old)
        for (r, t) in enumerate(rows[q])
            W[((r - 1) * nrest + 1):(r * nrest), :] .= sqrt(wac[t, c]) .* tpl(t)
        end
        svd(W)
    end
    ks, truncerr = kept_counts([F.S for F in Fs]; maxdim, cutoff)
    kc = zeros(Int, Dc); fq = zeros(Int, Dc)
    for (q, c) in enumerate(live); kc[c] = ks[q]; fq[c] = q; end
    Kn = maximum(ks)

    # ---- L: one template per present (template, channel) pair, a-major --------------------
    newid = zeros(Int, nt, Dc); np = 0
    for t in 1:nt, c in 1:Dc
        present[t, c] && (np += 1; newid[t, c] = np)
    end
    ntail = copy(tail); ntail[jx] = Kn
    nbs   = prod(ntail)
    tdims = ntuple(i -> ntail[i], Val(N2))               # a template block, storage order
    udims = ntuple(i -> ntail[perm[i]], Val(N2))         # U's columns as (other tail..., m)
    tm    = Vector{Tel}(undef, np * nbs)
    U     = Matrix{Tel}(undef, nrest, Kn)
    for t in 1:nt, c in 1:Dc
        id = newid[t, c]; id == 0 && continue
        fill!(U, zero(Tel))
        if wac[t, c] > 0 && kc[c] > 0
            q = fq[c]; r = rowof[t, c]
            U[:, 1:kc[c]] .= @view(Fs[q].U[((r - 1) * nrest + 1):(r * nrest), 1:kc[c]]) ./ sqrt(wac[t, c])
        end
        Base.permutedims!(reshape(view(tm,((id - 1) * nbs + 1):(id * nbs)), tdims), reshape(U, udims), iperm)
    end
    AI2 = np <= typemax(UInt8) ? UInt8 : np <= typemax(UInt16) ? UInt16 : UInt32
    ids = Vector{AI2}(undef, nk)
    @inbounds for i in 1:nk; ids[i] = newid[Int(X.alias_ids[i]), Int(X.keys[i][cx])] % AI2; end
    dimsL = ntuple(p -> p <= Px ? X.dims[p] : ntail[p - Px], N)
    aL = SectorSplitAliasedBlockSparse{Tel,N,N2,Px,K,AI2}(dimsL, nbs, tm, np, copy(X.keys), ids,
                                                         copy(X.scalars))
    new_mult = ITensors.Index(Kn; tags = ITensors.tags(old_mult))
    indsL = ntuple(p -> p == mx ? new_mult : Aw.inds[p], N)
    L = ITensors._itensor_from_external_storage(WrappedAliasedBlockSparse(aL, indsL))

    # ---- R: (c', c, m_new, m_old), diagonal keys, one template per channel value ----------
    bsR = Kn * m_old
    tR  = zeros(Tel, Dc * bsR)
    for c in 1:Dc
        k = kc[c]; k == 0 && continue
        F = Fs[fq[c]]
        BR = reshape(view(tR, ((c - 1) * bsR + 1):(c * bsR)), Kn, m_old)
        BR[1:k, :] .= F.S[1:k] .* @view(F.Vt[1:k, :])
    end
    AIR = Dc <= typemax(UInt8) ? UInt8 : Dc <= typemax(UInt16) ? UInt16 : UInt32
    keysR, idsR = _diag_schema(Dc, AIR)
    aR = SharedAliasedBlockSparse{Tel,4,2,2,Int,AIR}((Dc, Dc, Kn, m_old), bsR, tR, Dc, keysR, idsR,
                                                     ones(Tel, Dc))
    R = ITensors._itensor_from_external_storage(
        WrappedAliasedBlockSparse(aR, (ITensors.prime(chan), chan, new_mult, old_mult)))

    eigs = Vector{real(Tel)}(undef, sum(ks))
    o = 0
    for q in eachindex(live), j in 1:ks[q]; o += 1; eigs[o] = abs2(Fs[q].S[j]); end
    sort!(eigs; rev = true)
    return L, R, ITensors.Spectrum(eigs, truncerr)
end

# R's key schema, diagonal keys (c, c) and alias ids 1:Dc, depends only on Dc and the alias-id
# type, so it is built once per (Dc, AIR) and shared by every R. Sharing is safe because no code
# mutates the keys or alias ids of an existing tensor in place: kernels write only into a fresh
# output C, and the merges reassign alias_ids instead of editing it.
const _DIAG_SCHEMA = Dict{Tuple{Int,DataType},Any}()
const _DIAG_SCHEMA_LOCK = ReentrantLock()
function _diag_schema(Dc::Int, ::Type{AIR}) where {AIR}
    s = lock(_DIAG_SCHEMA_LOCK) do
        get!(() -> ([(c, c) for c in 1:Dc], collect(AIR, 1:Dc)), _DIAG_SCHEMA, (Dc, AIR))
    end
    return s::Tuple{Vector{NTuple{2,Int}},Vector{AIR}}
end
