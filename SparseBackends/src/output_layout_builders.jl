# ─────────────────────────────────────────────────────────────────────────────
# OutputLayout builders for the known callers. Each returns exactly the layout the
# caller's old keyword arguments produced inside wrapped_contract_aliased, so moving
# a caller from the old keywords to `layout=` changes no result.
# (Included after tensor_wrappers.jl, tensor_contraction.jl, tensor_wrappers_aliased.jl.)
# ─────────────────────────────────────────────────────────────────────────────

"""
    dense_from_aliased_operand(Aw, Bw) -> OutputLayout

Layout used by the DMRG environment/matvec products (`_mul_preserve_aliased`) and
`mul_aliased`: the dense block holds only the aliased operand(s)' dense indices; every
other output index (e.g. the dense operand's site/link indices) goes in the key.
Same as passing `output_inds_hint = dense_inds(aliased operand(s))`.
"""
function dense_from_aliased_operand(Aw::WrappedTensorTypes, Bw::WrappedTensorTypes)
    a_ali = Aw isa WrappedAliasedBlockSparse
    b_ali = Bw isa WrappedAliasedBlockSparse
    (a_ali || b_ali) || throw(OutputLayoutError(
        "dense_from_aliased_operand: neither operand has aliased storage"))
    hint = Set{ITensors.Index}()
    a_ali && union!(hint, dense_inds(Aw))
    b_ali && union!(hint, dense_inds(Bw))
    indsC, denseC = output_inds(Aw.inds, Bw.inds, dense_inds(Aw), dense_inds(Bw);
                                output_inds_hint=hint)
    return _layout_from_output_inds(indsC, denseC)
end

# The aliased-first operand order contract_and_fuse_links uses with a layout policy.
_aliased_first(Aw, Bw) =
    (Bw isa WrappedAliasedBlockSparse && !(Aw isa WrappedAliasedBlockSparse)) ? (Bw, Aw) : (Aw, Bw)

"""
    preserve_prefix(Aw, Bw; aLeft, aRight, bLeft, bRight) -> OutputLayout

Layout policy for MPO × MPS on an aliased ψ (TEBD; pass as `layout_policy=preserve_prefix`):
keep the aliased operand's key and move the dense operand's site index into it, last.
Same as the old `alias_hint = :preserve_prefix`.
"""
function preserve_prefix(Aw::WrappedTensorTypes, Bw::WrappedTensorTypes;
                         aLeft=nothing, aRight=nothing, bLeft=nothing, bRight=nothing)
    hint = _preserve_prefix_hint(Aw, Bw, aLeft, aRight, bLeft, bRight)
    Xw, Yw = _aliased_first(Aw, Bw)
    pref = _preserve_prefix_order(Xw, Yw, hint)
    indsC, denseC = output_inds(Xw.inds, Yw.inds, dense_inds(Xw), dense_inds(Yw);
                                output_inds_hint=hint)
    # the order the old preferred_output_labels produced (same acceptance test)
    want = collect(ITensors.Index, pref)
    if length(want) == length(indsC) && Set(ITensors.id.(want)) == Set(ITensors.id.(indsC))
        dset = Set(ITensors.id(I) for I in denseC)
        ntail = count(I -> ITensors.id(I) in dset, want)
        if all(I -> ITensors.id(I) in dset, want[(end - ntail + 1):end])
            indsC = Tuple(want)
        end
    end
    nd = length(denseC)
    return OutputLayout(collect(ITensors.Index, indsC[1:end-nd]),
                        collect(ITensors.Index, indsC[end-nd+1:end]))
end

"""
    matvec_dense_layout(bondtype, step, A, B) -> Union{Nothing,OutputLayout}

Fixed, per-position output order for the DMRG dense-ψ × aliased-H matvec (steps 2 and 3;
bond type `:left`, `:bulk` or `:right`), taken from the precomputed table
`STATIC_OUTPUT_PERM_DENSE`. `A` is the aliased operator site, `B` the dense
intermediate, as in the matvec call. Returns `nothing` where the table has no entry
(steps 1 and 4), which means the stock order. Same as the old
`output_perm = static_output_perm_dense(bondtype, step)`.
"""
function matvec_dense_layout(bondtype::Symbol, step::Int, A::ITensors.ITensor, B::ITensors.ITensor)
    perm = static_output_perm_dense(bondtype, step)
    perm === nothing && return nothing
    Aw = wrap_itensor_aliased(A)
    Bw = wrap_itensor(B; backend=:dense)
    indsC, _ = output_inds(Aw.inds, Bw.inds, dense_inds(Aw), dense_inds(Bw))
    # the old output_perm was only applied when its length matched (stock order otherwise)
    length(perm) == length(indsC) || return nothing
    return OutputLayout(ITensors.Index[], collect(ITensors.Index, indsC)[perm])
end
