# probe_pxp_php.jl — is the MPO-level PXP aliased sandwich equivalent to the
# per-site loop it replaced?
#
# `sandwich_mpo_aliased` on PXP used to loop per site over
# `SparseBackends.contract_aliased_itensor`, copied from
# test_sparse_ham/test_pxp_aliased.jl. It now calls the MPO-level
# `contract(A, B, Abackend, Bbackend; Cbackend)` that KL always used. The two are
# NOT obviously identical: the MPO-level path threads a BondMap across sites and
# passes per-site aLeft/aRight/bLeft/bRight link indices, so output bonds are
# identified consistently between neighbours; the per-site loop does neither.
#
# Checks, in increasing strength:
#   1. structure  — bond dims, block/template counts per site
#   2. action     — DMRG energy from an identical psi0 and sweep schedule
#
# Usage: julia --project=SparseBackends SparseBackends/experiments/probe_pxp_php.jl [config_id]

include(joinpath(@__DIR__, "bench_common.jl"))
using SparseBackends: WrappedAliasedBlockSparse

# The OLD construction, kept here (and only here) so the comparison is possible.
function sandwich_mpo_aliased_persite(P::MPO, H::MPO)
    new_H = MPO(length(H))
    for i in 1:length(H)
        H1  = SparseBackends.contract_aliased_itensor(P[i]'', H[i]', :coo, :dense)
        Hi  = SparseBackends.contract_aliased_itensor(P[i],   H1,    :coo, :aliased)
        new_H[i] = replaceprime(Hi, 3 => 1)
    end
    fuse_sparse_links!(new_H)
    prepermute_aliased_mpo!(new_H)
    return new_H
end

function structure(H::MPO)
    rows = NamedTuple[]
    for i in 1:length(H)
        d = H[i].tensor.data
        if d isa WrappedAliasedBlockSparse
            a = d.aliased
            push!(rows, (site = i, dims = a.dims, blksize = a.blksize,
                         nblocks = length(a.keys), ntempl = a.n_templates))
        else
            push!(rows, (site = i, dims = size(H[i]), blksize = 0, nblocks = 0, ntempl = 0))
        end
    end
    return rows
end

cfg_id = get(ARGS, 1, "pxp_nopad_v16_bd20")
cfg    = config_or_die(cfg_id)
println("="^78); println("PXP aliased sandwich: per-site loop  vs  MPO-level contract")
println("config = $cfg_id   nsites = $(cfg.nsites)   pad_h_chi = $(cfg.pad_h_chi)")
println("="^78)

sites, H_raw, P = pxp_operators(cfg.nsites; pad_h_chi = cfg.pad_h_chi)
t_old = @elapsed H_old = sandwich_mpo_aliased_persite(P, H_raw)
t_new = @elapsed H_new = sandwich_mpo_aliased(P, H_raw)
@printf("build: per-site %.3f s   MPO-level %.3f s\n\n", t_old, t_new)

# ── 1. structure ─────────────────────────────────────────────────────────────
so, sn = structure(H_old), structure(H_new)
bad = 0
for (a, b) in zip(so, sn)
    if a != b
        bad += 1
        bad <= 5 && println("  site $(a.site) DIFFERS\n    per-site : $a\n    MPO-level: $b")
    end
end
println(bad == 0 ? "1. structure : IDENTICAL on all $(length(so)) sites" :
                   "1. structure : $bad / $(length(so)) sites differ")
println("   bonds per-site : ", true_bond_dims(H_old))
println("   bonds MPO-level: ", true_bond_dims(H_new))
println("   bytes per-site  = ", mpo_bytes(H_old), "   MPO-level = ", mpo_bytes(H_new))

# ── 2. action: same psi0, same schedule, compare energies ────────────────────
md = [cfg.maxdim]
common = (nsweeps = 3, maxdim = md, mindim = md, cutoff = cfg.cutoff,
          outputlevel = 0, use_early_exit = false)
E_old, _ = dmrg(H_old, pxp_psi0(sites, P, 0); common...)
E_new, _ = dmrg(H_new, pxp_psi0(sites, P, 0); common...)
@printf("\n2. action    : E(per-site) = %.12f\n", real(E_old))
@printf("               E(MPO-level)= %.12f\n", real(E_new))
@printf("               |diff|      = %.3e\n", abs(real(E_old) - real(E_new)))

ok = bad == 0 && abs(real(E_old) - real(E_new)) < 1e-9
println("\n", ok ? "==> EQUIVALENT — safe to keep the MPO-level call" :
                   "==> NOT EQUIVALENT — revert to the per-site loop")

# ── 3. dense sandwich: does the second contraction's OPERAND ORDER matter? ───
# KL always wrote contract(P, P''H'); the PXP driver wrote contract(P''H', P)
# with P on the right. The contracted index is fixed by the priming (s') in both,
# so with no truncation they should be the same product — checked here because
# every orig_dense PXP row on disk came from the driver's order.
println("\n", "-"^78)
println("3. dense sandwich operand order")
H_pd    = sandwich_mpo(P, H_raw)                                    # contract(P, H1)
H_drv   = replaceprime(contract(contract(P'', H_raw'; is_ctn_compression = true), P;
                                is_ctn_compression = true), 3 => 1)  # contract(H1, P)
println("   contract(P, H1) bonds : ", true_bond_dims(H_pd))
println("   contract(H1, P) bonds : ", true_bond_dims(H_drv))
println("   bytes                 : ", mpo_bytes(H_pd), "  vs  ", mpo_bytes(H_drv))
E_pd,  _ = dmrg(H_pd,  pxp_psi0(sites, P, 0); common...)
E_drv, _ = dmrg(H_drv, pxp_psi0(sites, P, 0); common...)
@printf("   E contract(P, H1)     = %.12f\n", real(E_pd))
@printf("   E contract(H1, P)     = %.12f\n", real(E_drv))
@printf("   |diff|                = %.3e\n", abs(real(E_pd) - real(E_drv)))
println(abs(real(E_pd) - real(E_drv)) < 1e-9 ?
        "   ==> operand order does NOT matter — safe to unify" :
        "   ==> operand order MATTERS — keep the driver's order for orig_dense")
