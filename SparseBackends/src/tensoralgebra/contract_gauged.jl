# tensoralgebra/contract_gauged.jl
#
# One site of a constrained TEBD layer applied to ψ = P·core, ψ aliased or block-sparse
# (tebd_utils `build_tebd_mpo_layers(...; P, datastructure = :aliased)`, applied by the ITensorMPS
# fork's `contract(layer, ψ, ...; layout)`):
#
#   ψ_i   key (c_l, s, c_r) in any slot order, keys sorted by (s, c_r, c_l); block T dense over ψ's tail
#   M_i   COO gate site stored (g_l, g_r, s, s'), keys sorted by (s', s, g_r, g_l)
#   X_l   = X_{i-1}  COO stored (c_l', g_l', g_l, c_l), keys sorted by (c_l, g_l, g_l', c_l')
#   Y     = X_i^-1   COO stored (g_r', c_r, g_r, c_r'), keys sorted by (c_r', g_r, c_r, g_r')
#   (orders written by tebd_utils _package_layer / _bond_gauges; absent gauges at the edge sites)
#
#   out[(c_l', s', c_r')][g_l', g_r'; ψ tail] = Σ_{s, c_l, c_r, g_l, g_r} X_l · M · Y · ψ[(c_l, s, c_r)]
#
# OUTPUT-STATIONARY, by rows, a walk over the operands' sorted entries (no search). c_l' must be the output
# key's fastest index; a row is (*, c_r', s'), the rows are visited in key order. Per row:
#   gate run s'  ->  its groups (s, g_r)  ->  Y run (c_r', g_r): entries (c_r, g_r')
#     ->  ψ run (s, c_r): blocks n with their c_l  ->  the gate group's entries: g_l, m
#       ->  X_l run (c_l, g_l): entries (c_l', g_l')
# Each run is found by its recorded start (ptr arrays, one walk over each operand's sorted keys), so every
# entry visited is one that contributes. There is no scatter: each output block is written only while the
# loop is on its row, its keys in increasing c_l', i.e. key order. Per output key the coefficients are summed
# per ψ block (one Dg_l × Dg_r matrix each) before the block is written, Σ C ⊗ T, in runs along ψ's first
# tail axis, by the layout's strides (no permute).
#
# This file: the index work shared by both ψ storages. _gauged_kernel has one method per ψ storage, next
# to that storage: aliased/gauged_kernel.jl, blocksparse/gauged_kernel.jl.
#
# Dispatch: top_level_contract wraps the COO gate and calls wrapped_contract(::NamedTuple, ψ, C) below.

# One site of a constrained TEBD layer: A = (gate = COO wrapped gate site, gauges = (X_left, X_right^-1)),
# each gauge nothing (edge) or a COO ITensor; B = ψ's site; C the output storage class (an aliased class
# for an aliased ψ, NewBlockSparseSorted for a block-sparse ψ); layout = the output's key and dense order.
# skip_non_psi_keys (aliased ψ; true for TEBD): the output keeps ψ's structure, its keys = ψ's keys, its
# template classes = ψ's templates and its scalars = ψ's; one template per class is computed, the other keys
# of a class are not, and a key outside ψ's key set must vanish to roundoff (else an error). false: every
# key that receives a nonzero coefficient is an output key with its own template, scalar 1. A block-sparse
# ψ's output is always the latter (each key its own block).
function wrapped_contract(
    A :: NamedTuple{(:gate,:gauges)},
    B :: Union{WrappedAliasedBlockSparse,WrappedBlockSparse},
    ::Type{C};
    in_position::Bool=false,
    layout::Union{Nothing,OutputLayout}=nothing,
    skip_non_psi_keys::Bool=true,
) where {C}
    layout === nothing && throw(OutputLayoutError("wrapped_contract: a gauged gate needs a layout"))
    Gw = A.gate
    Gw isa WrappedCOOTensor || throw(OutputLayoutError(
        "wrapped_contract: the gate of a gauged layer must be COO, got $(typeof(Gw).name.name)"))
    return _contract_gauged(C, B, Gw, A.gauges[1], A.gauges[2], layout, skip_non_psi_keys)
end

function _contract_gauged(::Type{CK}, ψw, Gw::WrappedCOOTensor, Xl, Yr, layout::OutputLayout,
                          skip_non_psi_keys::Bool) where {CK}
    kern  = "gauged gate"
    ψs    = rep(ψw)
    Pψ    = length(ψw.inds) - length(dense_inds(ψw))
    keyψ  = ψw.inds[1:Pψ]
    tailψ = ψw.inds[(Pψ + 1):end]
    ginds = Gw.inds
    # each gauge (nothing at an edge site): its COO storage, keys column-major sorted; g its index shared
    # with the gate, c ψ's channel. The left gauge is stored (c', g', g, c), the right one (g', c, g, c').
    function gauge(Z, left::Bool)
        Z === nothing && return (nothing, nothing, nothing)
        w = ITensors.has_external_storage(Z) ? ITensors.get_external_storage(Z) : nothing
        w isa WrappedCOOTensor || _layout_error(kern, "a gauge must be a COO ITensor")
        ix = w.inds
        length(ix) == 4 || _layout_error(kern, "a gauge has 4 indices, got $(length(ix))")
        c, c2, g2 = left ? (ix[4], ix[1], ix[2]) : (ix[2], ix[4], ix[1])
        (ix[3] in ginds && c in keyψ && g2 == ITensors.prime(ix[3]) && c2 == ITensors.prime(c)) ||
            _layout_error(kern, (left ? "the left gauge must be stored (c', g', g, c)" :
                                        "the right gauge must be stored (g', c, g, c')") *
                                ", g the gate's link and c ψ's channel (tebd_utils _bond_gauges)")
        (!w.coo.dirty && issorted(w.coo.keys; by = reverse)) || _layout_error(kern,
            "a gauge's COO keys must be column-major sorted")
        return (w.coo, ix[3], c)
    end
    Lcoo, Lg, Lc = gauge(Xl, true)
    Rcoo, Rg, Rc = gauge(Yr, false)
    TC = promote_type(eltype(ψs), eltype(Gw.coo), (C === nothing ? Bool : eltype(C) for C in (Lcoo, Rcoo))...)
    Dgl = Lg === nothing ? 1 : ITensors.dim(Lg)
    Dgr = Rg === nothing ? 1 : ITensors.dim(Rg)

    # the contracted site index s (in ψ's key), the gate's output site s', the gate links
    common = [I for I in ginds if I in ψw.inds]
    (length(common) == 1 && common[1] in keyψ) || _layout_error(kern,
        "the gate and ψ must share exactly one index, in ψ's key (the site index)")
    s = common[1]
    rest = [I for I in ginds if I != s && I != Lg && I != Rg]
    length(rest) == 1 || _layout_error(kern, "the gate must have s, s' and one gauged link per bond; " *
        "unmatched gate indices: $(length(rest))")
    sp = rest[1]
    all(I -> I == s || I == Lc || I == Rc, keyψ) || _layout_error(kern,
        "every key index of ψ other than the site index must be a gauged channel")
    ψs.dims[findfirst(==(s), keyψ)] == ITensors.dim(sp) || _layout_error(kern,
        "the gate's output site index s' must have the dimension of ψ's site index s")

    # gate: stored (g_l, g_r, s, s'), keys sorted by (s', s, g_r, g_l)
    NG = length(ginds)
    (ginds[NG] == sp && ginds[NG - 1] == s) || _layout_error(kern,
        "the gate must be stored in index order (g_l, g_r, s, s') (tebd_utils _package_layer)")
    gk = Gw.coo.keys
    pgl = Lg === nothing ? 0 : findfirst(==(Lg), ginds)
    pgr = Rg === nothing ? 0 : findfirst(==(Rg), ginds)
    (!Gw.coo.dirty && issorted(gk; by = k -> (k[NG], k[NG - 1], pgr == 0 ? 1 : k[pgr], pgl == 0 ? 1 : k[pgl]))) ||
        _layout_error(kern, "the gate's COO keys must be sorted by (s', s, g_r, g_l)")

    # ψ: positions of c_l, s, c_r in its key; keys sorted by (s, c_r, c_l)
    ψpos(I) = I === nothing ? 0 : findfirst(==(I), keyψ)
    ψpos3 = (ψpos(Lc), ψpos(s), ψpos(Rc))
    pcl, ps, pcr = ψpos3
    ψkey(k, p) = p == 0 ? 1 : Int(k[p])
    issorted(ψs.keys; by = k -> (ψkey(k, ps), ψkey(k, pcr), ψkey(k, pcl))) || _layout_error(kern,
        "ψ's keys must be sorted by (s, c_r, c_l)")
    ds  = ITensors.dim(s)
    Dcr = Rc === nothing ? 1 : ITensors.dim(Rc)
    Dcl = Lc === nothing ? 1 : ITensors.dim(Lc)

    # Recorded run starts, one walk over each operand's sorted keys: run b is ptr[b]:(ptr[b+1]-1).
    #   Gptr  gate, run s'               Yptr  Y, run (c_r', g_r):  b = g_r + Dg_r·(c_r' - 1)
    #   ψptr  ψ, run (s, c_r):  b = c_r + Dc_r·(s - 1)        Xptr  X_l, run (c_l, g_l):  b = g_l + Dg_l·(c_l - 1)
    # An absent gauge (edge site) is one entry, c = c' = g = g' = 1, coefficient 1.
    Gptr = fill(length(gk) + 1, ds + 1)
    @inbounds for e in length(gk):-1:1; Gptr[Int(gk[e][NG])] = e; end
    ψptr = fill(length(ψs.keys) + 1, ds * Dcr + 1)
    @inbounds for n in length(ψs.keys):-1:1
        k = ψs.keys[n]; ψptr[ψkey(k, pcr) + Dcr * (ψkey(k, ps) - 1)] = n
    end
    Yptr = [1, 2]
    if Rcoo !== nothing
        Yptr = fill(length(Rcoo.keys) + 1, Dcr * Dgr + 1)
        @inbounds for q in length(Rcoo.keys):-1:1
            k = Rcoo.keys[q]; Yptr[Int(k[3]) + Dgr * (Int(k[4]) - 1)] = q
        end
    end
    Xptr = [1, 2]
    if Lcoo !== nothing
        Xptr = fill(length(Lcoo.keys) + 1, Dcl * Dgl + 1)
        @inbounds for p in length(Lcoo.keys):-1:1
            k = Lcoo.keys[p]; Xptr[Int(k[3]) + Dgl * (Int(k[4]) - 1)] = p
        end
    end
    for P in (Gptr, ψptr, Yptr, Xptr), b in (length(P) - 1):-1:1   # an empty run starts where the next does
        P[b] = min(P[b], P[b + 1])
    end

    # output indices from the layout
    layout = resolve_layout(layout, Gw, ψw)
    validate_layout(layout, ψw.inds, Gw.inds)
    indsC = Tuple(ITensors.Index[I for I in layout_inds(layout)])
    nd = length(layout.dense)
    NC = length(indsC)

    # output key: (c_l', c_r', s') in the layout's order; slot of each in the key
    PC = NC - nd
    keyC, denseC = indsC[1:PC], indsC[(PC + 1):NC]
    dimsC = map(ITensors.dim, indsC)
    all(I -> I == sp || I == Lc || I == Rc, keyC) || _layout_error(kern,
        "the output key must be ψ's gauged channels and the gate's output site index")
    kpos(I) = I === nothing ? 0 : (j = findfirst(==(I), keyC); j === nothing ?
        _layout_error(kern, "the output key must hold every gauged channel and s'") : j)
    jcl, jcr, jsp = kpos(Lc), kpos(Rc), kpos(sp)
    jcl in (0, 1) || _layout_error(kern, "the left channel c_l' must be the output key's first (fastest) " *
        "index: the kernel takes c_l' from the left gauge within each output row")

    # output dense block: ψ's tail axes and the gate links, the layout's order; column-major strides
    ddC = map(ITensors.dim, denseC)
    dstC = ntuple(j -> prod(ddC[1:(j - 1)]; init = 1), nd)
    dslot(I) = I === nothing ? 0 : (j = findfirst(==(I), denseC); j === nothing ?
        _layout_error(kern, "the output dense block must hold ψ's dense indices and the gate links") : dstC[j])
    ost_gl, ost_gr = dslot(Lg), dslot(Rg)
    otail = Int[dslot(I) for I in tailψ]
    # runs along ψ's first tail axis; colOff = output offset of each block column
    tdims = Int[ITensors.dim(I) for I in tailψ]
    n1   = isempty(tdims) ? 1 : tdims[1]
    ost1 = isempty(tdims) ? 1 : otail[1]
    ncol = prod(tdims[2:end]; init = 1)
    colOff = zeros(Int, ncol)
    @inbounds for j in 1:ncol
        r = j - 1; o = 0
        for a in 2:length(tdims)
            o += (r % tdims[a]) * otail[a]; r ÷= tdims[a]
        end
        colOff[j] = o
    end

    return _gauged_kernel(CK, ψs, TC, indsC, Val(PC), dimsC, jcl, jcr, jsp, Dcl,
                          ψpos3, Dcr, Gptr, ψptr, Yptr, Xptr, gk, Gw.coo.vals, pgl, pgr, Lcoo, Rcoo, Dgl, Dgr,
                          ost_gl, ost_gr, n1, ost1, colOff, skip_non_psi_keys)
end

# Gather of one output row (*, c_r', s'), shared by both kernels, by walking the operands' runs (see the
# file header): for every output c_l' the row reaches and every ψ entry n that feeds key (c_l', c_r', s'),
# slot u = wpos[n, c_l'] holds W[(u-1)·Dg2 + g_l' + Dg_l·(g_r'-1)] = Σ X_l · M · Y; the n of each c_l' are
# listed in nlists[c_l'] (reset, with wpos, by the caller). Returns the number of slots used. Lcoo / Rcoo:
# the gauges' COO storages, stored (c', g', g, c) / (g', c, g, c'), or nothing at an edge site (one entry,
# coefficient 1). skipkey (or nothing): output keys not to gather, by linear index rowlin + c_l' - 1.
@inline function _gauged_row!(W::Vector{TC}, wpos::Matrix{Int}, nlists::Vector{Vector{Int}},
                              cr2::Int, spv::Int, ψkeys, pcl::Int, Dcr::Int, Gptr::Vector{Int},
                              ψptr::Vector{Int}, Yptr::Vector{Int}, Xptr::Vector{Int},
                              gk::Vector{NTuple{NG,KG}}, gv, pgl::Int, pgr::Int, Lcoo, Rcoo,
                              Dgl::Int, Dgr::Int, Dg2::Int, skipkey::Union{Nothing,Vector{Bool}},
                              rowlin::Int) where {TC,NG,KG}
    ns = 0
    e0 = Gptr[spv]; eend = Gptr[spv + 1] - 1
    @inbounds while e0 <= eend                                         # gate run s', group by group
        k0 = gk[e0]; sv = Int(k0[NG - 1]); gr = pgr == 0 ? 1 : Int(k0[pgr])
        e1 = e0                                                        # group (s, g_r) = e0:e1
        while e1 < eend && Int(gk[e1 + 1][NG - 1]) == sv && (pgr == 0 || Int(gk[e1 + 1][pgr]) == gr)
            e1 += 1
        end
        yb = gr + Dgr * (cr2 - 1)
        for q in Yptr[yb]:(Yptr[yb + 1] - 1)                           # Y run (c_r', g_r): (c_r, g_r')
            gr2, cr, yv = Rcoo === nothing ? (1, 1, one(TC)) :
                          (Int(Rcoo.keys[q][1]), Int(Rcoo.keys[q][2]), convert(TC, Rcoo.vals[q]))
            pb = cr + Dcr * (sv - 1)
            for n in ψptr[pb]:(ψptr[pb + 1] - 1)                       # ψ run (s, c_r): blocks n, their c_l
                cl = pcl == 0 ? 1 : Int(ψkeys[n][pcl])
                for e in e0:e1                                         # the group's entries: g_l, m
                    m = convert(TC, gv[e])
                    iszero(m) && continue
                    gl = pgl == 0 ? 1 : Int(gk[e][pgl])
                    xb = gl + Dgl * (cl - 1)
                    for p in Xptr[xb]:(Xptr[xb + 1] - 1)               # X_l run (c_l, g_l): (c_l', g_l')
                        cl2, gl2, xv = Lcoo === nothing ? (1, 1, one(TC)) :
                                       (Int(Lcoo.keys[p][1]), Int(Lcoo.keys[p][2]), convert(TC, Lcoo.vals[p]))
                        skipkey !== nothing && skipkey[rowlin + cl2 - 1] && continue
                        u = wpos[n, cl2]
                        if u == 0
                            ns += 1; u = ns; wpos[n, cl2] = u; push!(nlists[cl2], n)
                            length(W) < Dg2 * u && resize!(W, 2 * Dg2 * u)
                            wo = Dg2 * (u - 1)
                            for j in 1:Dg2; W[wo + j] = zero(TC); end
                        end
                        W[Dg2 * (u - 1) + gl2 + Dgl * (gr2 - 1)] += (xv * m) * yv
                    end
                end
            end
        end
        e0 = e1 + 1
    end
    return ns
end
