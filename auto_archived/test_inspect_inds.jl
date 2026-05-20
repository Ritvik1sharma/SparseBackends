# Inspect actual index structure of (it, Hv, output) for matvec calls.
# Prints, for each record:
#   - inds(it):  ordered indices of H[j]
#   - inds(Hv):  ordered indices of the running env
#   - commoninds(it, Hv):  contracted indices
#   - uniqueinds(it, Hv):  free on it (kept in output)
#   - uniqueinds(Hv, it):  free on Hv (kept in output)
#   - inds(out): order of the result tensor
#
# Goal: understand the actual contract pattern so we can choose a good
# canonical output order for H_new at construction time.

using Serialization: deserialize
using SparseBackends
using ITensors

const PATH = length(ARGS) >= 1 ? ARGS[1] :
    "temp/edited_packages/results/sd_pairs_md160.jls"
const MAX_RECS = parse(Int, get(ENV, "MAX_RECS", "5"))

# Compact view of an index: shows (id-tag, dim, prime-level, tags-set)
function ind_repr(i)
    return "$(dim(i))-$(tags(i))-p$(plev(i))"
end
inds_repr(is) = "[" * join(ind_repr.(is), ", ") * "]"

recs = Any[]
open(PATH, "r") do io
    while !eof(io); push!(recs, deserialize(io)); end
end
println("Loaded $(length(recs)) records from $PATH")
println()

# Show a handful of records (early, middle, late) to see if structure changes
sample_idxs = unique([1, length(recs)÷4, length(recs)÷2, 3*length(recs)÷4, length(recs)])[1:min(MAX_RECS, end)]

for k in sample_idxs
    r = recs[k]
    println("=" ^ 100)
    println("Record $k: H[j] external storage? ", ITensors.has_external_storage(r.it),
            "  | Hv external storage? ", ITensors.has_external_storage(r.Hv))
    println("inds(it) : ", inds_repr(inds(r.it)))
    println("inds(Hv) : ", inds_repr(inds(r.Hv)))
    shared = commoninds(r.it, r.Hv)
    only_it = uniqueinds(r.it, r.Hv)
    only_Hv = uniqueinds(r.Hv, r.it)
    println("  shared (contracted): ", inds_repr(shared))
    println("  only_it (kept)     : ", inds_repr(only_it))
    println("  only_Hv (kept)     : ", inds_repr(only_Hv))
    # Compute actual output
    out = r.it * r.Hv
    println("inds(out): ", inds_repr(inds(out)))
    # Index positions: where are the shared indices placed in `it` and `Hv`?
    pos_shared_in_it = [findfirst(==(s), inds(r.it)) for s in shared]
    pos_shared_in_Hv = [findfirst(==(s), inds(r.Hv)) for s in shared]
    pos_shared_in_out = [findfirst(==(s), inds(out)) for s in shared]
    pos_only_it_in_it = [findfirst(==(s), inds(r.it)) for s in only_it]
    pos_only_Hv_in_Hv = [findfirst(==(s), inds(r.Hv)) for s in only_Hv]
    println("  positions:")
    println("    shared in it : ", pos_shared_in_it, "    (rank=", ndims(r.it), ")")
    println("    shared in Hv : ", pos_shared_in_Hv, "    (rank=", ndims(r.Hv), ")")
    println("    only_it in it: ", pos_only_it_in_it)
    println("    only_Hv in Hv: ", pos_only_Hv_in_Hv)
    println()
end
