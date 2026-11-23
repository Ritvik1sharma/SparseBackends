# blocksparse/gauged_kernel.jl — _gauged_kernel for a block-sparse ψ (setup: tensoralgebra/contract_gauged.jl).
#
# Output-stationary by rows (*, c_r', s') in key order (contract_gauged.jl); within a row the keys in
# increasing c_l'. A key that receives a nonzero coefficient gets the next block, so the blocks are packed in
# key order (ids[i] == i) as they are written. Each block is Σ_n C_n ⊗ T_n over the ψ blocks n that feed it,
# written in place by the layout's strides. skip_non_psi_keys is not used: every key is its own block.

# Function barrier: everything below runs on concrete types.
function _gauged_kernel(::Type{<:NewBlockSparseSorted}, A::NewBlockSparseSorted{TA,NA,N2A,PA,K}, ::Type{TC},
                        indsC, ::Val{PC}, dimsC::NTuple{NC,Int}, jcl::Int, jcr::Int, jsp::Int, Dcl::Int,
                        ψpos::NTuple{3,Int}, Dcr::Int, Gptr::Vector{Int}, ψptr::Vector{Int},
                        Yptr::Vector{Int}, Xptr::Vector{Int}, gk::Vector{NTuple{NG,KG}}, gv::AbstractVector, pgl::Int, pgr::Int, Lcoo, Rcoo, Dgl::Int, Dgr::Int,
                        ost_gl::Int, ost_gr::Int, n1::Int, ost1::Int,
                        colOff::Vector{Int}, ::Bool) where {TA,NA,N2A,PA,K,TC,NC,PC,NG,KG}
    Dg2 = Dgl * Dgr
    bsA = A.blksize
    bsC = bsA * Dg2
    ncol = length(colOff)
    pdC = ntuple(j -> dimsC[j], PC)
    tA = A.data

    keys = NTuple{PC,Int}[]
    data = TC[]
    W = Vector{TC}(undef, Dg2 * 16); wpos = zeros(Int, length(A.keys), Dcl); nlists = [Int[] for _ in 1:Dcl]
    # rows (*, c_r', s') in key order: the faster of c_r', s' in the key is the inner loop
    dsp = pdC[jsp]
    crfast = jcr != 0 && jcr < jsp
    @inbounds for a in 1:(crfast ? dsp : Dcr), b in 1:(crfast ? Dcr : dsp)
        cr2, spv = crfast ? (b, a) : (a, b)
        _gauged_row!(W, wpos, nlists, cr2, spv, A.keys, ψpos[1], Dcr, Gptr, ψptr, Yptr, Xptr, gk, gv, pgl, pgr,
                     Lcoo, Rcoo, Dgl, Dgr, Dg2, nothing, 0)
        for cl2 in 1:Dcl                                      # the row's keys in increasing c_l'
            lst = nlists[cl2]
            isempty(lst) && continue
            nz = false
            for n in lst
                wo = Dg2 * (wpos[n, cl2] - 1)
                for j in 1:Dg2; nz |= !iszero(W[wo + j]); end
            end
            if nz                                             # exactly-zero output: no block
                push!(keys, ntuple(j -> j == jcl ? cl2 : (j == jcr ? cr2 : spv), Val(PC)))
                ob = length(data); resize!(data, ob + bsC); fill!(view(data, (ob + 1):(ob + bsC)), zero(TC))
                for n in lst
                    wo = Dg2 * (wpos[n, cl2] - 1); ao = (A.ids[n] - 1) * bsA
                    for gr2 in 1:Dgr, gl2 in 1:Dgl
                        c = W[wo + gl2 + Dgl * (gr2 - 1)]
                        iszero(c) && continue
                        bo = ob + 1 + (gl2 - 1) * ost_gl + (gr2 - 1) * ost_gr
                        for j in 1:ncol
                            o = bo + colOff[j]; a0 = ao + (j - 1) * n1
                            @simd for a1 in 1:n1
                                data[o + (a1 - 1) * ost1] += c * tA[a0 + a1]
                            end
                        end
                    end
                end
            end
            for n in lst; wpos[n, cl2] = 0; end
            empty!(lst)
        end
    end
    nk = length(keys)
    return WrappedBlockSparse(NewBlockSparseSorted{TC,NC,NC - PC,PC,Int}(dimsC, bsC, keys, collect(1:nk), data),
                              indsC)
end
