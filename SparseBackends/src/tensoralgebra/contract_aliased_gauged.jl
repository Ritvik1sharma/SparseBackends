# tensoralgebra/contract_aliased_gauged.jl
#
# One site of a constrained TEBD layer applied to an aliased ψ = P·core
# (tebd_utils `build_tebd_mpo_layers(...; P, datastructure = :aliased)`, applied by the ITensorMPS
# fork's `contract(layer, ψ, :dense, :aliased; layout)`):
#
#   ψ_i   aliased, key (c_l, s, c_r) in any order, block T (dense over ψ's tail, e.g. (m_l, m_r))
#   M_i   dense gate site on (s', s, g_l, g_r), any storage order; the builder stores it as
#         (g_l, g_r, s', s) so each (s', s) slice is one contiguous Dg_l × Dg_r block
#   X_l   = X_{i-1}  COO on (g_l, c_l, g_l', c_l')      (left bond; absent at site 1)
#   Y     = X_i^-1   COO on (g_r, c_r, g_r', c_r')      (right bond; absent at site N)
#
#   out[(c_l', s', c_r')][g_l', g_r'; ψ tail] = Σ_{(c_l, s, c_r) ∈ ψ} C[g_l', g_r'] · α · T
#   C[g_l', g_r'] = Σ_{g_l, g_r} X_l[g_l, c_l, g_l', c_l'] · M[s', s, g_l, g_r] · Y[g_r, c_r, g_r', c_r']
#
# STRUCTURAL. The output has ψ's own structure: output keys = ψ's keys (c_l, s, c_r) -> (c_l', s', c_r')
# with the same values (s' = s), template classes = ψ's templates (the keys that share a ψ template
# share an output template), and every key keeps ψ's scalar α. So the output allocation is known up
# front: one key per ψ key, one template per class. Per class the block of its representative key
# (the first key with α != 0) is computed, out_rep = Σ_{blocks n of ψ, s'} C · α_n · T_{t_n} /
# α_rep, accumulated as one Dg_l × Dg_r coefficient vector per (class, ψ template) and written once
# into the class's template, Σ_t C_t ⊗ T_t, in runs along ψ's first tail axis. Pass 1 touches keys
# and small coefficient matrices only: per ψ block and s' it forms every C (all (c_l', c_r') at
# once, from M's (s', s) slice and the gauges' per-(g, c) nonzero lists); exactly-zero C are
# skipped. A contribution that lands on a key outside ψ's key set is accumulated per (key, ψ
# template) and must vanish to roundoff (1e-10 of the largest class coefficient), else it is an
# error: the structure is not the one ψ has. The other keys of a class are not recomputed: they
# are ψ's α-multiples of the representative's block, which holds when M and the gauges map ψ's
# P-structure to itself. No numeric template merge is needed (nor wanted) after this kernel:
# nothing is divided by a computed coefficient, so the scalars neither drift nor pick up roundoff.
# No permute anywhere: M and the ψ templates are read in place by strides, and the output dense
# block is written directly in the layout's order.
#
# Dispatch: top_level_contract wraps the gate dense and calls wrapped_contract(::NamedTuple, ψ, C)
# below with C the requested aliased class; the kernel builds the output as C.

# Per-(g, c) nonzero lists of one bond gauge Z (COO ITensor on (g, c, g', c')): the entries with
# input (g, c) are p ∈ ptr[q] : ptr[q+1]-1, q = g + Dg·(c-1), output (g2[p], c2[p]), value val[p].
# g is Z's index shared with the gate, c the one shared with ψ's key, g' = prime(g), c' = prime(c).
function _gauge_lists(Z::ITensors.ITensor, gate_inds, key_inds, ::Type{TC}) where {TC}
    kern = "aliased × gauged gate"
    ITensors.has_external_storage(Z) || _layout_error(kern, "a gauge must be a COO ITensor")
    w = ITensors.get_external_storage(Z)
    w isa WrappedCOOTensor || _layout_error(kern, "a gauge must be a COO ITensor, got $(typeof(w).name.name)")
    ix = w.inds
    length(ix) == 4 || _layout_error(kern, "a gauge has 4 indices (g, c, g', c'), got $(length(ix))")
    gs = [I for I in ix if I in gate_inds]; cs = [I for I in ix if I in key_inds]
    (length(gs) == 1 && length(cs) == 1) || _layout_error(kern,
        "a gauge needs exactly one index of the gate (g) and one of ψ's key (c)")
    g, c = gs[1], cs[1]
    pg, pc = findfirst(==(g), ix), findfirst(==(c), ix)
    pg2, pc2 = findfirst(==(ITensors.prime(g)), ix), findfirst(==(ITensors.prime(c)), ix)
    (pg2 === nothing || pc2 === nothing) && _layout_error(kern, "a gauge's other indices must be g' and c'")
    Dg, Dc = ITensors.dim(g), ITensors.dim(c)
    ptr, g2, c2, val = _gauge_csr(w.coo, pg, pc, pg2, pc2, Dg, Dc, TC)
    return (g = g, c = c, Dg = Dg, Dc = Dc, ptr = ptr, g2 = g2, c2 = c2, val = val)
end

# Function barrier (w.coo is only partially typed): CSR of the COO entries by input (g, c).
function _gauge_csr(coo::COOTensor{T,4,K}, pg::Int, pc::Int, pg2::Int, pc2::Int, Dg::Int, Dc::Int,
                    ::Type{TC}) where {T,K,TC}
    ptr = zeros(Int, Dg * Dc + 1)
    @inbounds for (k, v) in zip(coo.keys, coo.vals)
        iszero(v) || (ptr[k[pg] + Dg * (k[pc] - 1) + 1] += 1)
    end
    ptr[1] = 1
    @inbounds for q in 2:length(ptr); ptr[q] += ptr[q - 1]; end
    nnz = ptr[end] - 1
    g2 = Vector{Int}(undef, nnz); c2 = Vector{Int}(undef, nnz); val = Vector{TC}(undef, nnz)
    fillp = ptr[1:end-1]
    @inbounds for (k, v) in zip(coo.keys, coo.vals)
        iszero(v) && continue
        q = k[pg] + Dg * (k[pc] - 1); p = fillp[q]; fillp[q] += 1
        g2[p] = Int(k[pg2]); c2[p] = Int(k[pc2]); val[p] = TC(v)
    end
    return ptr, g2, c2, val
end

# An absent gauge (edge site): one gate-link value, one channel, coefficient 1.
_unit_gauge(::Type{TC}) where {TC} =
    (g = nothing, c = nothing, Dg = 1, Dc = 1, ptr = [1, 2], g2 = [1], c2 = [1], val = TC[one(TC)])

# One site of a constrained TEBD layer: A = (gate = dense wrapped gate site, gauges = (X_left,
# X_right^-1)), each nothing (edge) or a COO ITensor on (g, c, g', c'); B = ψ's site; C the
# output's aliased class. The output has the layout's indices (key and dense block), written
# by _contract_aliased_gauged.
function wrapped_contract(
    A :: NamedTuple{(:gate,:gauges)},
    B :: WrappedAliasedBlockSparse,
    ::Type{C};
    in_position::Bool=false,
    layout::Union{Nothing,OutputLayout}=nothing,
) where {C}
    (C <: AliasedBlockSparse && layout !== nothing) || throw(OutputLayoutError(
        "wrapped_contract: a gauged gate needs an aliased output and a layout"))
    Gw = A.gate
    Gw isa WrappedTensor || throw(OutputLayoutError(
        "wrapped_contract: the gate of a gauged layer must be dense"))
    layout = resolve_layout(layout, Gw, B)
    validate_layout(layout, B.inds, Gw.inds)
    indsC = Tuple(ITensors.Index[I for I in layout_inds(layout)])
    return _contract_aliased_gauged(C, B, Gw, A.gauges[1], A.gauges[2], indsC, length(layout.dense))
end

function _contract_aliased_gauged(::Type{CK}, ψw::WrappedAliasedBlockSparse, Gw::WrappedTensor, Xl, Yr,
                                  indsC::NTuple{NC,ITensors.Index}, nd::Int) where {CK,NC}
    kern = "aliased × gauged gate"
    Pψ    = _abs_head_len(ψw)
    keyψ  = ψw.inds[1:Pψ]
    tailψ = ψw.inds[(Pψ + 1):end]
    ginds = Gw.inds
    TG = (Xl === nothing ? Bool : eltype(ITensors.get_external_storage(Xl).coo),
          Yr === nothing ? Bool : eltype(ITensors.get_external_storage(Yr).coo))
    TC = promote_type(eltype(ψw.aliased), eltype(Gw.data), TG...)
    L = Xl === nothing ? _unit_gauge(TC) : _gauge_lists(Xl, ginds, keyψ, TC)
    R = Yr === nothing ? _unit_gauge(TC) : _gauge_lists(Yr, ginds, keyψ, TC)

    # the contracted site index s (in ψ's key), the gate's output site s', the gate links
    common = [I for I in ginds if I in ψw.inds]
    (length(common) == 1 && common[1] in keyψ) || _layout_error(kern,
        "the gate and ψ must share exactly one index, in ψ's key (the site index)")
    s = common[1]
    rest = [I for I in ginds if I != s && I != L.g && I != R.g]
    length(rest) == 1 || _layout_error(kern, "the gate must have s, s' and one gauged link per bond; " *
        "unmatched gate indices: $(length(rest))")
    sp = rest[1]
    all(I -> I == s || I == L.c || I == R.c, keyψ) || _layout_error(kern,
        "every key index of ψ other than the site index must be a gauged channel")
    s_slot  = findfirst(==(s), keyψ)
    cl_slot = L.c === nothing ? 0 : findfirst(==(L.c), keyψ)
    cr_slot = R.c === nothing ? 0 : findfirst(==(R.c), keyψ)

    # gate: column-major strides of its stored order
    gdims = map(ITensors.dim, ginds)
    gst = ntuple(k -> prod(gdims[1:(k - 1)]; init = 1), length(gdims))
    gstride(I) = I === nothing ? 0 : gst[findfirst(==(I), ginds)]
    st_sp, st_s, st_gl, st_gr = gstride(sp), gstride(s), gstride(L.g), gstride(R.g)
    Mv = vec(Gw.data)

    # output key: (c_l', c_r', s') in the layout's order; linear index column-major
    PC = NC - nd
    keyC, denseC = indsC[1:PC], indsC[(PC + 1):NC]
    dimsC = map(ITensors.dim, indsC)
    pdC = ntuple(j -> dimsC[j], PC)
    kstC = ntuple(j -> prod(pdC[1:(j - 1)]; init = 1), PC)
    all(I -> I == sp || I == L.c || I == R.c, keyC) || _layout_error(kern,
        "the output key must be ψ's gauged channels and the gate's output site index")
    kslot(I) = I === nothing ? 0 : (j = findfirst(==(I), keyC); j === nothing ?
        _layout_error(kern, "the output key must hold every gauged channel and s'") : kstC[j])
    kst_cl, kst_cr, kst_sp = kslot(L.c), kslot(R.c), kslot(sp)

    # output dense block: ψ's tail axes and the gate links, any order; column-major strides
    ddC = map(ITensors.dim, denseC)
    dstC = ntuple(j -> prod(ddC[1:(j - 1)]; init = 1), nd)
    dslot(I) = I === nothing ? 0 : (j = findfirst(==(I), denseC); j === nothing ?
        _layout_error(kern, "the output dense block must hold ψ's dense indices and the gate links") : dstC[j])
    ost_gl, ost_gr = dslot(L.g), dslot(R.g)
    otail = Int[dslot(I) for I in tailψ]
    # runs along ψ's first tail axis; colOff = output offset of each template column
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

    ali = _gauged_kernel(CK, ψw.aliased, TC, Val(NC), Val(PC), dimsC, kstC,
                         s_slot, cl_slot, cr_slot, Mv, ITensors.dim(sp), st_sp, st_s, st_gl, st_gr,
                         L, R, kst_cl, kst_cr, kst_sp, ost_gl, ost_gr, n1, ost1, colOff)
    return WrappedAliasedBlockSparse(ali, indsC)
end

# Function barrier: everything below runs on concrete types. The output is built as class CK.
function _gauged_kernel(::Type{CK}, A::AliasedBlockSparse{TA,NA,N2A,PA,K,AI}, ::Type{TC}, ::Val{NC}, ::Val{PC},
                        dimsC::NTuple{NC,Int}, kstC::NTuple{PC,Int},
                        s_slot::Int, cl_slot::Int, cr_slot::Int,
                        Mv::AbstractVector, dsp::Int, st_sp::Int, st_s::Int, st_gl::Int, st_gr::Int,
                        L, R, kst_cl::Int, kst_cr::Int, kst_sp::Int, ost_gl::Int, ost_gr::Int,
                        n1::Int, ost1::Int, colOff::Vector{Int}) where {CK,TA,NA,N2A,PA,K,AI,TC,NC,PC}
    Dgl, Dcl, Dgr, Dcr = L.Dg, L.Dc, R.Dg, R.Dc
    Dg2 = Dgl * Dgr
    nch = Dcl * Dcr
    bsA = A.blksize
    bsC = bsA * Dg2
    ncol = length(colOff)
    Kc = prod(ntuple(j -> dimsC[j], PC); init = 1)

    # ── classes: ψ's templates, numbered by first key; representative = first key with α != 0 ──────
    nA = length(A.keys)
    A.dims[s_slot] == dsp || _layout_error("aliased × gauged gate",
        "the gate's output site index s' must have the dimension of ψ's site index s")
    cls = zeros(Int, A.n_templates); nc = 0
    @inbounds for n in 1:nA
        t = Int(A.alias_ids[n]); cls[t] == 0 && (nc += 1; cls[t] = nc)
    end
    rep = zeros(Int, nc)
    @inbounds for n in 1:nA
        u = cls[Int(A.alias_ids[n])]
        (rep[u] == 0 || (iszero(A.scalars[rep[u]]) && !iszero(A.scalars[n]))) && (rep[u] = n)
    end
    klin = Vector{Int}(undef, nA)       # linear index of the output key of ψ key n
    @inbounds for n in 1:nA
        k = A.keys[n]
        klin[n] = 1 + ((cl_slot == 0 ? 0 : Int(k[cl_slot]) - 1) * kst_cl) +
                      ((cr_slot == 0 ? 0 : Int(k[cr_slot]) - 1) * kst_cr) + (Int(k[s_slot]) - 1) * kst_sp
    end
    repmap = zeros(Int, Kc)             # output key -> class, for the representative keys
    @inbounds for u in 1:nc; repmap[klin[rep[u]]] = u; end
    inkeys = falses(Kc)                 # the output keys that are ψ's keys
    @inbounds for n in 1:nA; inkeys[klin[n]] = true; end

    # ── pass 1: per class, the recipe {ψ template t => Σ α·C} (linked entry lists) ───────────────
    head  = zeros(Int, nc)            # first recipe entry of each class, 0 = none
    tail  = zeros(Int, nc)            # last recipe entry of each class
    enext = Int[]                     # next entry of the same class
    etmpl = Int[]                     # ψ template of the entry
    ecoef = TC[]                      # Dgl × Dgr coefficients of the entry (column-major)
    outside = Dict{Tuple{Int,Int},Vector{TC}}()   # (output key, ψ template) off ψ's key set
    W     = Vector{TC}(undef, Dg2 * nch)     # C of every (c_l', c_r') for one (ψ block, s')
    tmask = falses(nch); tlist = Int[]
    Lptr, Lg2, Lc2, Lval = L.ptr, L.g2, L.c2, L.val
    Rptr, Rg2, Rc2, Rval = R.ptr, R.g2, R.c2, R.val
    @inbounds for n in 1:nA
        α = convert(TC, A.scalars[n]); iszero(α) && continue
        k = A.keys[n]; t = Int(A.alias_ids[n])
        s  = Int(k[s_slot])
        cl = cl_slot == 0 ? 1 : Int(k[cl_slot])
        cr = cr_slot == 0 ? 1 : Int(k[cr_slot])
        for spv in 1:dsp
            mo = 1 + (spv - 1) * st_sp + (s - 1) * st_s
            for gr in 1:Dgr
                rq = gr + Dgr * (cr - 1); r1 = Rptr[rq]; r2 = Rptr[rq + 1] - 1
                r1 > r2 && continue
                for gl in 1:Dgl
                    m = Mv[mo + (gl - 1) * st_gl + (gr - 1) * st_gr]
                    iszero(m) && continue
                    lq = gl + Dgl * (cl - 1)
                    for p in Lptr[lq]:(Lptr[lq + 1] - 1)
                        xm = Lval[p] * m; gl2 = Lg2[p]; cl2 = Lc2[p]
                        for q in r1:r2
                            cc = cl2 + Dcl * (Rc2[q] - 1)
                            if !tmask[cc]
                                tmask[cc] = true; push!(tlist, cc)
                                wo = Dg2 * (cc - 1)
                                for j in 1:Dg2; W[wo + j] = zero(TC); end
                            end
                            W[gl2 + Dgl * (Rg2[q] - 1) + Dg2 * (cc - 1)] += xm * Rval[q]
                        end
                    end
                end
            end
            for cc in tlist
                tmask[cc] = false
                wo = Dg2 * (cc - 1)
                nzc = false
                for j in 1:Dg2; nzc |= !iszero(W[wo + j]); end
                nzc || continue                                   # exactly-zero C: nothing to write
                cl2 = (cc - 1) % Dcl + 1; cr2 = (cc - 1) ÷ Dcl + 1
                lin = 1 + (cl2 - 1) * kst_cl + (cr2 - 1) * kst_cr + (spv - 1) * kst_sp
                u = repmap[lin]
                if u == 0
                    inkeys[lin] && continue                       # a non-representative key of a class: not recomputed
                    acc = get!(() -> zeros(TC, Dg2), outside, (lin, t))
                    for j in 1:Dg2; acc[j] += α * W[wo + j]; end
                    continue
                end
                e = head[u]
                while e != 0 && etmpl[e] != t; e = enext[e]; end
                if e == 0
                    push!(etmpl, t); push!(enext, 0); e = length(etmpl)
                    resize!(ecoef, Dg2 * e)
                    for j in 1:Dg2; ecoef[Dg2 * (e - 1) + j] = zero(TC); end
                    tail[u] == 0 ? (head[u] = e) : (enext[tail[u]] = e); tail[u] = e
                end
                eo = Dg2 * (e - 1)
                for j in 1:Dg2; ecoef[eo + j] += α * W[wo + j]; end
            end
            empty!(tlist)
        end
    end
    # contributions off ψ's key set must be roundoff
    cmax = zero(real(TC)); omax = zero(real(TC))
    for x in ecoef; cmax = max(cmax, abs(x)); end
    for v in values(outside), x in v; omax = max(omax, abs(x)); end
    omax <= 1e-10 * cmax || _layout_error("aliased × gauged gate",
        "the layer maps ψ to a key outside ψ's key set (largest coefficient $omax against $cmax in the " *
        "key set); the output cannot keep ψ's structure")

    # ── output keys = ψ's keys, ids = classes, scalars = ψ's; allocated once, exact sizes ─────────
    AI2 = nc <= typemax(UInt8) ? UInt8 : nc <= typemax(UInt16) ? UInt16 : UInt32
    pdC  = ntuple(j -> dimsC[j], PC)
    perm = sortperm(klin)
    keys = Vector{NTuple{PC,Int}}(undef, nA)
    ids  = Vector{AI2}(undef, nA)
    scal = Vector{TC}(undef, nA)
    @inbounds for i in 1:nA
        n = perm[i]; l0 = klin[n] - 1
        keys[i] = ntuple(j -> (l0 ÷ kstC[j]) % pdC[j] + 1, Val(PC))
        ids[i]  = cls[Int(A.alias_ids[n])] % AI2
        scal[i] = convert(TC, A.scalars[n])
    end
    tm = zeros(TC, nc * bsC)

    # ── pass 2: each class's template written once, in place ──────────────────────────────────────
    tA = A.templates
    @inbounds for u in 1:nc
        αr = convert(TC, A.scalars[rep[u]]); iszero(αr) && continue
        invα = inv(αr); ob = (u - 1) * bsC
        e = head[u]
        while e != 0
            ao = (etmpl[e] - 1) * bsA; co = Dg2 * (e - 1)
            for gr2 in 1:Dgr, gl2 in 1:Dgl
                c = ecoef[co + gl2 + Dgl * (gr2 - 1)] * invα
                iszero(c) && continue
                bo = ob + 1 + (gl2 - 1) * ost_gl + (gr2 - 1) * ost_gr
                for j in 1:ncol
                    o = bo + colOff[j]; a0 = ao + (j - 1) * n1
                    @simd for a in 1:n1
                        tm[o + (a - 1) * ost1] += c * tA[a0 + a]
                    end
                end
            end
            e = enext[e]
        end
    end
    return CK{TC,NC,NC-PC,PC,Int,AI2}(dimsC, bsC, tm, nc, keys, ids, scal)
end
