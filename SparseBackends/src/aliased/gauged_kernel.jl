# aliased/gauged_kernel.jl — _gauged_kernel for an aliased ψ (setup: tensoralgebra/contract_gauged.jl).
#
# The dense data of every output block comes from ψ's templates only: the gate and the gauges give, per
# output key, one Dg_l × Dg_r coefficient matrix C_n per ψ entry n that feeds it, and the block is
# Σ_n C_n · α_n · T_{t_n} = Σ_t V_t ⊗ T_t with V_t = Σ_{n: t_n = t} α_n C_n (one write per ψ template).
#
# skip_non_psi_keys = true (TEBD: the layer maps ψ = P·core to P·core'): the output has ψ's own structure,
# output keys = ψ's keys (c_l, s, c_r) -> (c_l', s', c_r') with the same values (s' = s), template classes =
# ψ's templates, every key keeps ψ's scalar α. Per class one key is computed, the first with α != 0; its
# block divided by its α is the class template. The other keys of a class are not gathered. A key outside
# ψ's key set is gathered and must vanish to roundoff (1e-10 of the largest class coefficient), else it is
# an error: the structure is not the one ψ has.
# skip_non_psi_keys = false: every key that receives a nonzero coefficient is an output key, in key order,
# with its own template (the block itself) and scalar 1.
#
# Output-stationary by rows (*, c_r', s') in key order (contract_gauged.jl); within a row the keys in
# increasing c_l'. Nothing is divided by a computed coefficient.

# Function barrier: everything below runs on concrete types.
function _gauged_kernel(::Type{CK}, A::AliasedBlockSparse{TA,NA,N2A,PA,K,AI}, ::Type{TC}, indsC, ::Val{PC},
                        dimsC::NTuple{NC,Int}, jcl::Int, jcr::Int, jsp::Int, Dcl::Int,
                        ψpos::NTuple{3,Int}, Dcr::Int, Gptr::Vector{Int}, ψptr::Vector{Int},
                        Yptr::Vector{Int}, Xptr::Vector{Int}, gk::Vector{NTuple{NG,KG}}, gv::AbstractVector, pgl::Int, pgr::Int, Lcoo, Rcoo, Dgl::Int, Dgr::Int,
                        ost_gl::Int, ost_gr::Int, n1::Int, ost1::Int,
                        colOff::Vector{Int}, skip_non_psi_keys::Bool) where {CK,TA,NA,N2A,PA,K,AI,TC,NC,PC,NG,KG}
    Dg2 = Dgl * Dgr
    bsA = A.blksize
    bsC = bsA * Dg2
    ncol = length(colOff)
    pdC = ntuple(j -> dimsC[j], PC)
    kstC = ntuple(j -> prod(pdC[1:(j - 1)]; init = 1), PC)
    Kc = prod(pdC; init = 1)
    nA = length(A.keys)

    # ── skip_non_psi_keys: classes = ψ's templates, numbered by first key; computed key = first with α != 0 ──
    cls = zeros(Int, A.n_templates); nc = 0
    outn = Int[]                                              # output key (linear index) -> ψ entry
    skipkey = nothing
    if skip_non_psi_keys
        @inbounds for n in 1:nA
            t = Int(A.alias_ids[n]); cls[t] == 0 && (nc += 1; cls[t] = nc)
        end
        rep = zeros(Int, nc)
        @inbounds for n in 1:nA
            u = cls[Int(A.alias_ids[n])]
            (rep[u] == 0 || (iszero(A.scalars[rep[u]]) && !iszero(A.scalars[n]))) && (rep[u] = n)
        end
        outn = zeros(Int, Kc)
        pcl, ps, pcr = ψpos
        @inbounds for n in 1:nA
            k = A.keys[n]
            outn[1 + (pcl == 0 ? 0 : (Int(k[pcl]) - 1) * kstC[jcl]) + (pcr == 0 ? 0 : (Int(k[pcr]) - 1) * kstC[jcr]) +
                 (Int(k[ps]) - 1) * kstC[jsp]] = n
        end
        isrep = falses(nA); @inbounds for u in 1:nc; isrep[rep[u]] = true; end
        skipkey = Bool[outn[l] != 0 && !isrep[outn[l]] for l in 1:Kc]   # a key of a class that is not computed
    end

    tm = zeros(TC, nc * bsC)
    tA = A.templates
    keysF = NTuple{PC,Int}[]                                  # skip_non_psi_keys = false: the output keys
    W = Vector{TC}(undef, Dg2 * 16); wpos = zeros(Int, nA, Dcl); nlists = [Int[] for _ in 1:Dcl]
    V = Vector{TC}(undef, Dg2 * 16); tpos = zeros(Int, A.n_templates); tlist = Int[]
    cmax = zero(real(TC)); omax = zero(real(TC))
    # rows (*, c_r', s') in key order: the faster of c_r', s' in the key is the inner loop
    dsp = pdC[jsp]
    crfast = jcr != 0 && jcr < jsp
    @inbounds for a in 1:(crfast ? dsp : Dcr), b in 1:(crfast ? Dcr : dsp)
        cr2, spv = crfast ? (b, a) : (a, b)
        rowlin = 1 + (jcr == 0 ? 0 : (cr2 - 1) * kstC[jcr]) + (spv - 1) * kstC[jsp]
        _gauged_row!(W, wpos, nlists, cr2, spv, A.keys, ψpos[1], Dcr, Gptr, ψptr, Yptr, Xptr, gk, gv, pgl, pgr,
                     Lcoo, Rcoo, Dgl, Dgr, Dg2, skipkey, rowlin)
        for cl2 in 1:Dcl                                      # the row's keys in increasing c_l'
            lst = nlists[cl2]
            isempty(lst) && continue
            # per ψ template: V_t = Σ_{n: t_n = t} α_n C_n
            for n in lst
                α = convert(TC, A.scalars[n]); wo = Dg2 * (wpos[n, cl2] - 1); wpos[n, cl2] = 0
                iszero(α) && continue
                t = Int(A.alias_ids[n])
                if tpos[t] == 0
                    push!(tlist, t); tpos[t] = length(tlist)
                    length(V) < Dg2 * length(tlist) && resize!(V, 2 * Dg2 * length(tlist))
                    vo = Dg2 * (tpos[t] - 1)
                    for j in 1:Dg2; V[vo + j] = zero(TC); end
                end
                vo = Dg2 * (tpos[t] - 1)
                for j in 1:Dg2; V[vo + j] += α * W[wo + j]; end
            end
            empty!(lst)
            # where the key's template goes (ob, -1 = not written) and its scale
            ob = -1; sc = one(TC)
            if skip_non_psi_keys
                no = outn[rowlin + cl2 - 1]
                if no == 0                                    # off ψ's key set: must be roundoff
                    for j in 1:(Dg2 * length(tlist)); omax = max(omax, abs(V[j])); end
                else                                          # the computed key of its class
                    for j in 1:(Dg2 * length(tlist)); cmax = max(cmax, abs(V[j])); end
                    αr = convert(TC, A.scalars[no])
                    iszero(αr) || (ob = (cls[Int(A.alias_ids[no])] - 1) * bsC; sc = inv(αr))
                end
            else
                nz = false
                for j in 1:(Dg2 * length(tlist)); nz |= !iszero(V[j]); end
                if nz                                         # exactly-zero output: no key
                    push!(keysF, ntuple(j -> j == jcl ? cl2 : (j == jcr ? cr2 : spv), Val(PC)))
                    ob = length(tm); resize!(tm, ob + bsC); fill!(view(tm, (ob + 1):(ob + bsC)), zero(TC))
                end
            end
            if ob >= 0
                for t in tlist
                    vo = Dg2 * (tpos[t] - 1); ao = (t - 1) * bsA
                    for gr2 in 1:Dgr, gl2 in 1:Dgl
                        c = V[vo + gl2 + Dgl * (gr2 - 1)] * sc
                        iszero(c) && continue
                        bo = ob + 1 + (gl2 - 1) * ost_gl + (gr2 - 1) * ost_gr
                        for j in 1:ncol
                            o = bo + colOff[j]; a0 = ao + (j - 1) * n1
                            @simd for a1 in 1:n1
                                tm[o + (a1 - 1) * ost1] += c * tA[a0 + a1]
                            end
                        end
                    end
                end
            end
            for t in tlist; tpos[t] = 0; end
            empty!(tlist)
        end
    end
    omax <= 1e-10 * cmax || _layout_error("aliased × gauged gate",
        "the layer maps ψ to a key outside ψ's key set (largest coefficient $omax against $cmax in the " *
        "key set); the output cannot keep ψ's structure")

    if skip_non_psi_keys
        # ── output keys = ψ's keys in the output's key order, ids = classes, scalars = ψ's ──────────────
        AI2 = nc <= typemax(UInt8) ? UInt8 : nc <= typemax(UInt16) ? UInt16 : UInt32
        keys = NTuple{PC,Int}[]; ids = AI2[]; scal = TC[]
        @inbounds for ck in CartesianIndices(pdC)
            n = outn[LinearIndices(pdC)[ck]]; n == 0 && continue
            push!(keys, Tuple(ck)); push!(ids, cls[Int(A.alias_ids[n])] % AI2); push!(scal, convert(TC, A.scalars[n]))
        end
        return WrappedAliasedBlockSparse(CK{TC,NC,NC-PC,PC,Int,AI2}(dimsC, bsC, tm, nc, keys, ids, scal), indsC)
    end
    # ── each key its own template, scalar 1 ─────────────────────────────────────────────────────────────
    nt = length(keysF)
    AI2 = nt <= typemax(UInt8) ? UInt8 : nt <= typemax(UInt16) ? UInt16 : UInt32
    return WrappedAliasedBlockSparse(CK{TC,NC,NC-PC,PC,Int,AI2}(dimsC, bsC, tm, nt, keysF,
                                     AI2[i % AI2 for i in 1:nt], ones(TC, nt)), indsC)
end
