# php.jl — the P†HP sandwich. Model-independent; kl.jl and pxp.jl both use it.
#
# The dense sandwich and the MPO × MPO product are the shared
# experiments/utils/cotenn_utils.jl ones. The dense operator (:sb_dense) is
#     sandwich_mpo(Ps, H; cutoff1 = 0, cutoff2 = 0)
# Both cutoffs 0 are load-bearing: they make cotenn_utils MERGE all plaquettes
# into one P (exactly) and sandwich ONCE with is_ctn_compression = true (no SVD),
# matching the SparseBackends test drivers. Its defaults (1e-12 / 1e-32) instead
# sandwich SEQUENTIALLY with SVD truncation, which left the PXP dense reference at
# chi_PHP = 12 instead of the exact 16 and silently deletes zero-padded bond slots.
#
# The projector argument is EITHER a single MPO (PXP builds one directly via
# NotEqlsLoop_R1) OR a vector of them (KL has one per plaquette). A 1-element
# vector and a bare MPO mean the same thing.

include(joinpath(@__DIR__, "..", "..", "experiments", "utils", "cotenn_utils.jl"))

"""
    merge_projectors(Ps; backend = :dense)

Collapse per-plaquette constraint MPOs into the single P the sandwich uses.
A bare MPO passes straight through. Exact products (cotenn_utils
`multiply_mpos` at cutoff 0): `backend = :coo` uses the COO kernel, which is what
`sandwich_mpo_aliased` needs; `:dense` is is_ctn_compression (no SVD).
"""
merge_projectors(P::MPO; backend::Symbol = :dense) = P
function merge_projectors(Ps::Vector{MPO}; backend::Symbol = :dense)
    result = Ps[1]
    for j in 2:length(Ps)
        result = multiply_mpos(result, Ps[j]; cutoff = 0, Abackend = backend, Bbackend = backend)
    end
    return result
end

"""
    sandwich_mpo_aliased(Ps, H)

Aliased P†HP (the `:sb_aliased` / `:sb_fused` operator), including the two
transforms the aliased matvec relies on: sparse-link fusion and the one-shot
dense-tail prepermute. Both are unconditional in the test drivers.

Uses the MPO-level `contract(A, B, Abackend, Bbackend; Cbackend)`, which threads a
`BondMap` across sites and passes each site's aLeft/aRight/bLeft/bRight link
indices so output bonds are identified consistently between neighbours. The
original PXP driver instead looped per site over `contract_aliased_itensor`,
which does neither; `probe_pxp_php.jl` verifies the two agree (identical
structure on every site, DMRG energy difference exactly 0).
"""
function sandwich_mpo_aliased(Ps::Union{MPO,Vector{MPO}}, H::MPO)
    P     = merge_projectors(Ps; backend = :coo)
    H1    = contract(P'', H', :coo, :dense;   Cbackend = :aliased)
    H_eff = contract(P,   H1, :coo, :aliased; Cbackend = :aliased)
    Hphp  = replaceprime(H_eff, 3 => 1)
    fuse_sparse_links!(Hphp)
    prepermute_aliased_mpo!(Hphp)
    return Hphp
end
