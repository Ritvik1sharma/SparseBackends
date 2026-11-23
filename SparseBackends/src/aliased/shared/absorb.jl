# aliased/shared/absorb.jl
#
# `_absorb_bond` for a SharedAliasedBlockSparse site: M * R, where R comes from `factorize` on
# M's neighbour (ITensorMPS orthogonalize! in both directions and truncate! call this hook
# through their generic `_absorb_bond`). One `contract` call: R's new bond indices take the old
# ones' slots in M (`_absorb_layout`), so the site keeps its index order and key/dense split
# whichever way the sweep runs.
function _absorb_bond(::SharedAliasedBlockSparse, Mw::WrappedAliasedBlockSparse,
                      M::ITensors.ITensor, R::ITensors.ITensor)
    old, new = ITensors.commoninds(R, M), ITensors.uniqueinds(R, M)
    Rw = ITensors.has_external_storage(R) ? ITensors.get_external_storage(R) : nothing
    alias_id_type = Rw isa WrappedAliasedBlockSparse ?
        promote_type(eltype(Mw.aliased.alias_ids), eltype(Rw.aliased.alias_ids)) : eltype(Mw.aliased.alias_ids)
    return ITensors.contract(M, R; preserve_bs_output = true,
                             layout = _absorb_layout(Mw.inds, _abs_head_len(Mw), old, new), alias_id_type)
end
