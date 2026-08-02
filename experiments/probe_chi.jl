# probe_chi.jl — what IS the PHP bond dimension of the aliased operator?
#
# mpo_stats reports `maxlinkdim(H)`, which returns the largest "Link"-tagged
# index. For the aliased PHP that came out 20x smaller than for the dense PHP
# (exactly chi_H), so either
#   (a) the bond really is factorized across two axes and maxlinkdim sees one, or
#   (b) maxlinkdim is simply the wrong probe for this storage.
# Distinguish by dumping the ACTUAL indices of both operators side by side, plus
# the AliasedBlockSparse `dims` tuple and its prefix/dense split.
#
# Usage: julia --project=SparseBackends SparseBackends/experiments/probe_chi.jl [config_id]

include(joinpath(@__DIR__, "bench_common.jl"))
using SparseBackends: WrappedAliasedBlockSparse

function describe(T::ITensor, label)
    println("  $label")
    println("    ndims=", ndims(T), "  maxdim over ALL inds=", maximum(dim.(inds(T))),
            "  prod(all dims)=", prod(dim.(inds(T))))
    for I in inds(T)
        println("      dim=", lpad(dim(I), 4), "  plev=", plev(I),
                "  tags=", tags(I))
    end
    if ITensors.has_external_storage(T) && T.tensor.data isa WrappedAliasedBlockSparse
        a = T.tensor.data.aliased
        N = length(a.dims)
        P = N - Int(log(1))  # placeholder; real P read from the type below
        Pt = typeof(a).parameters[4]
        println("    AliasedBlockSparse: dims=", a.dims,
                "  P(sparse prefix)=", Pt, "  blksize=", a.blksize)
        println("      prefix dims (sparse axes) = ", a.dims[1:Pt],
                "   -> prod = ", prod(a.dims[1:Pt]))
        println("      dense tail dims           = ", a.dims[Pt+1:end])
        println("      n_blocks=", length(a.keys), "  n_templates=", a.n_templates)
    end
end

cfg_id = get(ARGS, 1, "kl_padh20_v18_bd120")
cfg = config_or_die(cfg_id)
println("="^78); println("config = $cfg_id   pad_h_chi = ", get(cfg, :pad_h_chi, 0)); println("="^78)

_, H_d, _, _, _, _ = build_problem(cfg, :sb_dense,   0)
_, H_a, _, _, _, _ = build_problem(cfg, :sb_aliased, 0)

j = length(H_d) ÷ 2   # a bulk site, away from the edges
println("\nBulk site j=$j\n")
describe(H_d[j], "DENSE PHP  H_d[$j]")
println()
describe(H_a[j], "ALIASED PHP  H_a[$j]")

println("\n", "-"^78)
println("maxlinkdim(dense)   = ", maxlinkdim(H_d))
println("maxlinkdim(aliased) = ", maxlinkdim(H_a))
println("\nlinkinds at bond $j:")
println("  dense   : ", [(dim(I), string(tags(I))) for I in commoninds(H_d[j], H_d[j+1])])
println("  aliased : ", [(dim(I), string(tags(I))) for I in commoninds(H_a[j], H_a[j+1])])
println("-"^78)
