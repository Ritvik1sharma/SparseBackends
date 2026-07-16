# core_helpers/window_map.jl — factor-core window-map EMISSION for the ψ[b]·ψ[b+1]
# AliasedBS×AliasedBS contraction. Produces `AliasedBlockSparse.window_slice_map`, the
# (rv_b, rv_{b+1}) → template-id routing that CANNOT be recovered post-hoc for a flip P
# (φ sums over the internal FSM bond). The consumer side (window_write_map /
# write_core_window!) lives in core_helpers/factor_core.jl; the storage field is on the
# AliasedBlockSparse struct in aliased/storage.jl.
#
# Two entry points, both used ONLY on the factor-core path (emit_window_map=true):
#   • window_emit_init / window_emit_record! / window_emit_finalize! — the three hooks
#     the AliasedBS×AliasedBS `contract_shared!` calls (gated by its emit_window_map kwarg).
#   • contract_shared_emit_window! — a front-end that builds the label→axis maps + the
#     reduced-shared set and runs `contract_shared!` with emission on, so the general
#     `wrapped_contract_aliased` need not re-derive that dispatch state itself.

# Emission state: per-input inverse rv maps (template-id → pre-P slice) + the per-output-
# key source-pair accumulator. PC = C's sparse-prefix rank.
struct WindowEmit{PC}
    tidA2rv     :: Dict{Int,Int}
    tidB2rv     :: Dict{Int,Int}
    ckey_rvpair :: Dict{NTuple{PC,Int},NTuple{2,Int}}
end

# Called ONCE at the top of contract_shared! (BEFORE the permute — raw A/B slice_to_template;
# template ids are permute-invariant). Returns `nothing` when not emitting (the hot path).
function window_emit_init(emit::Bool, A::AliasedBlockSparse, B::AliasedBlockSparse, ::Val{PC}) where {PC}
    (emit && !isempty(A.slice_to_template) && !isempty(B.slice_to_template)) || return nothing
    tidA2rv = Dict{Int,Int}(); tidB2rv = Dict{Int,Int}()
    @inbounds for rv in eachindex(A.slice_to_template)
        t = A.slice_to_template[rv]; t == 0 && continue; tidA2rv[t] = rv
    end
    @inbounds for rv in eachindex(B.slice_to_template)
        t = B.slice_to_template[rv]; t == 0 && continue; tidB2rv[t] = rv
    end
    return WindowEmit{PC}(tidA2rv, tidB2rv, Dict{NTuple{PC,Int},NTuple{2,Int}}())
end

# Called per output-key contribution in the merge-join: record the pre-P (rv_b, rv_{b+1})
# the key is sourced from. No in-loop conflict check — a double contribution is caught
# by the demotion guard in window_emit_finalize! (which subsumes it).
@inline function window_emit_record!(st::WindowEmit{PC}, ckey, tidA::Integer, tidB::Integer) where {PC}
    rvA = get(st.tidA2rv, Int(tidA), 0); rvA == 0 && return
    rvB = get(st.tidB2rv, Int(tidB), 0); rvB == 0 && return
    st.ckey_rvpair[ntuple(j -> Int(ckey[j]), Val(PC))] = (rvA, rvB)
    return
end

# Called after the lazy commit. Guard: no output key demoted to an accumulator (a
# multi-contribution key ⇒ ψ = P·core not clean). Then build the final (rv_b,rv_{b+1}) →
# FINAL-template-id map via C's committed keys, asserting it is a function (one template
# per source pair — the invariant write_core_window! relies on). Sets C.window_slice_map.
function window_emit_finalize!(C::AliasedBlockSparse, st::WindowEmit{PC}, key_to_accum) where {PC}
    isempty(key_to_accum) ||
        error("emit_window_map: $(length(key_to_accum)) output key(s) demoted to accumulators — ψ = P·core not clean")
    wm = Dict{NTuple{2,Int},Int}()
    @inbounds for i in eachindex(C.keys)
        ik  = ntuple(j -> Int(C.keys[i][j]), Val(PC))
        rvp = get(st.ckey_rvpair, ik, (0, 0))
        rvp == (0, 0) && continue
        ftid = Int(C.alias_ids[i])
        prev = get(wm, rvp, 0)
        if prev == 0
            wm[rvp] = ftid
        elseif prev != ftid
            error("emit_window_map: rv-pair $rvp maps to templates $prev and $ftid")
        end
    end
    C.window_slice_map = wm
    return C
end

# Front-end for wrapped_contract_aliased: build the label→axis maps + reduced-shared set
# (the dispatch state contract! would otherwise compute) and run the AliasedBS×AliasedBS
# contract_shared! with emission ON. Keeps that map-building out of the general wrapper.
function contract_shared_emit_window!(C::AliasedBlockSparse, labelsC,
                                      A::AliasedBlockSparse, labelsA,
                                      B::AliasedBlockSparse, labelsB)
    mapA   = Dict(l => i for (i, l) in enumerate(labelsA))
    mapB   = Dict(l => i for (i, l) in enumerate(labelsB))
    cset   = Set(labelsC)
    shared = [l for l in labelsA if haskey(mapB, l) && !(l in cset)]
    return contract_shared!(C, labelsC, A, labelsA, B, labelsB, mapA, mapB, shared;
                            emit_window_map=true)
end
