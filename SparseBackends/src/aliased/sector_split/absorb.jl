# aliased/sector_split/absorb.jl
#
# `_absorb_bond` for a SectorSplitAliasedBlockSparse site: M * R with R from the neighbour's
# per-channel factorize (sector_split/factorize.jl), aliased on (c', c, m_new, m_old). One
# `contract` call with R's new indices in the old ones' slots (`_absorb_layout`): c is summed
# against M's channel and c' takes its slot. c' is then renamed back to c by relabelling the
# Index entry of the product's own (fresh) wrapper, so the bond keeps its index: P's link,
# which a constrained gate's gauges refer to. Any new index that is its old index primed is
# renamed this way.
function _absorb_bond(::SectorSplitAliasedBlockSparse, Mw::WrappedAliasedBlockSparse,
                      M::ITensors.ITensor, R::ITensors.ITensor)
    old, new = ITensors.commoninds(R, M), ITensors.uniqueinds(R, M)
    C = ITensors.contract(M, R; preserve_bs_output = true,
                          layout = _absorb_layout(Mw.inds, _abs_head_len(Mw), old, new))
    Cw = ITensors.get_external_storage(C)::WrappedAliasedBlockSparse
    for i in eachindex(new)
        i <= length(old) && new[i] == ITensors.prime(old[i]) && relabel_ind!(Cw, new[i], old[i])
    end
    return C
end
