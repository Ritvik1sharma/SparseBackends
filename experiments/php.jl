# php.jl — the P†HP sandwich. Model-independent; kl.jl and pxp.jl both use it.
#
# NAME COLLISION WARNING. `sandwich_mpo` is also defined in
# experiments/utils/cotenn_utils.jl, and that one is NOT interchangeable with
# this. If a file ever includes both, whichever loads last wins and the
# difference is silent. Neither benchmark tree includes cotenn_utils.jl today —
# keep it that way. The two differ in exactly two respects:
#
#   1. cotenn_utils sandwiches SEQUENTIALLY, once per element of the P vector
#        for i in 1:length(P);  H_eff = sandwich_mpo(P[i], H_eff);  end
#      whereas this MERGES all plaquettes into a single P and sandwiches ONCE,
#      matching the SparseBackends test drivers.
#   2. cotenn_utils truncates: `contract(P'', H'; cutoff=1e-12)` then `1e-32`.
#      That is what left the PXP dense reference at chi_PHP = 12 instead of the
#      exact 16 and flattered dense by 15-25%. It also silently deletes
#      zero-padded bond slots (they have zero singular values), which would make
#      the chi_H padding experiment a no-op.
#
# The projector argument is EITHER a single MPO (PXP builds one directly via
# NotEqlsLoop_R1) OR a vector of them (KL has one per plaquette). A 1-element
# vector and a bare MPO mean the same thing.

"""
    multiply_mpos(A, B; backend = :dense)

MPO × MPO product: prime B's site indices, contract, unprime. Same operation and
same name as `cotenn_utils.jl::multiply_mpos`, defined locally rather than
included so this tree stays self-contained.

`backend = :coo` routes through the COO kernel, which is what the aliased
sandwich needs; `:dense` uses `is_ctn_compression = true` (no SVD, exact).

NOTE the cotenn_utils version silently ignores its `cutoff` kwarg on the
non-ctn branch and hardcodes 1e-12, so it is not a drop-in for a caller that
needs an exact dense product.
"""
function multiply_mpos(A::MPO, B::MPO; backend::Symbol = :dense)
    Bp = prime(B, "Site")
    C  = backend === :coo ? contract(A, Bp, :coo, :coo) :
                            contract(A, Bp; is_ctn_compression = true)
    return replaceprime(C, 2 => 1)
end

"""
    merge_projectors(Ps; backend = :dense)

Collapse per-plaquette constraint MPOs into the single P the sandwich uses.
A bare MPO passes straight through. The backend must match the sandwich that
will consume the result — a COO-merged P feeds `sandwich_mpo_aliased`, a
dense-merged P feeds `sandwich_mpo`.
"""
merge_projectors(P::MPO; backend::Symbol = :dense) = P
function merge_projectors(Ps::Vector{MPO}; backend::Symbol = :dense)
    result = Ps[1]
    for j in 2:length(Ps)
        result = multiply_mpos(result, Ps[j]; backend = backend)
    end
    return result
end

"""
    sandwich_mpo(Ps, H)

Dense P†HP (the `:sb_dense` operator). `Ps` is one MPO or a vector to be merged.

`is_ctn_compression = true` means NO SVD: no orthogonalize, no factorize, no
truncate!, then `collapse_all_bonds!` — an EXACT MPO product.
"""
function sandwich_mpo(Ps::Union{MPO,Vector{MPO}}, H::MPO)
    P     = merge_projectors(Ps; backend = :dense)
    H1    = contract(P'', H'; is_ctn_compression = true)
    H_eff = contract(P, H1;   is_ctn_compression = true)
    return replaceprime(H_eff, 3 => 1)
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
