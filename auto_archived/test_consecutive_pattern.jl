# Find the common pattern between consecutive matvec calls.
# For each consecutive (record_k, record_{k+1}) pair:
#   - record k: contracts (it_k, Hv_k) → produces Hv_out
#   - record k+1: starts with Hv_{k+1} (= Hv_out, possibly) and contracts with it_{k+1}
# We ask: where (in inds(Hv_out)) sit the indices that record k+1 will contract?
# If those positions are consistent, we can choose canonical_indsC_for_bd to put
# them there reliably.

using Serialization: deserialize
using SparseBackends
using ITensors
using Printf: @printf

const PATH = length(ARGS) >= 1 ? ARGS[1] :
    "temp/edited_packages/results/sd_pairs_md160.jls"

recs = Any[]
open(PATH, "r") do io
    while !eof(io); push!(recs, deserialize(io)); end
end
println("Loaded $(length(recs)) records from $PATH")

# Walk consecutive pairs. For each k where Hv_{k+1} could plausibly be the
# output of record k (same inds), compute the shared positions.
# We DON'T assume they're related (could be reset between matvecs); we just
# check ind-equality.
n_pairs_checked = 0
n_pairs_continue = 0   # where Hv_{k+1} actually equals output of record k
shared_positions = Vector{Vector{Int}}()  # positions of next-call shared inds in Hv_{k+1}
shared_position_normalized = Vector{Vector{Float64}}()  # positions as fraction of rank
shared_in_front_count = 0
shared_in_back_count = 0
shared_scattered_count = 0

# We can't easily verify "Hv_{k+1} came from record k's contract" without
# running the contracts. Instead, just analyze where shared inds (with next
# H[j]) sit in EACH Hv. Each record's Hv came from SOME previous contract.

# So for every record k (looking at Hv_k and the *next* matvec's H[j]):
# but actually we want: for record k, what positions in Hv_k do the shared
# inds with it_k occupy? That tells us "given the order Hv_k arrives in,
# where are the contracted inds?". If they're consistently in the back/front,
# we know what to aim for.

println("\nAnalyzing position of shared inds in inds(Hv) for each record:")
println("(if these are consistent, we know where canonical output should put them)")
println()

pos_first = Int[]   # min position of any shared ind
pos_last = Int[]    # max position
n_shared = Int[]
rank_Hv = Int[]
hv_consecutive_shared = Int[]    # whether shared inds are consecutive

for r in recs
    shared = commoninds(r.it, r.Hv)
    if isempty(shared); continue; end
    pos = sort([findfirst(==(s), inds(r.Hv)) for s in shared])
    push!(pos_first, pos[1])
    push!(pos_last, pos[end])
    push!(n_shared, length(shared))
    push!(rank_Hv, ndims(r.Hv))
    push!(hv_consecutive_shared, (pos[end] - pos[1] + 1 == length(pos)) ? 1 : 0)
end

@printf("Records analyzed:    %d\n", length(pos_first))
@printf("Mean rank of Hv:     %.2f\n", sum(rank_Hv) / length(rank_Hv))
@printf("Mean n_shared:       %.2f\n", sum(n_shared) / length(n_shared))
@printf("Shared inds CONSECUTIVE in Hv: %d / %d (%.1f%%)\n",
        sum(hv_consecutive_shared), length(hv_consecutive_shared),
        100*sum(hv_consecutive_shared)/length(hv_consecutive_shared))

# Histogram: where does the first shared ind sit (as a fraction of rank)?
println("\nPosition of first shared ind in Hv (count):")
bins = Dict{Int,Int}()
for (pos, n) in zip(pos_first, rank_Hv); bins[pos] = get(bins,pos,0) + 1; end
for k in sort(collect(keys(bins)))
    println("  pos $k: $(bins[k])")
end

println("\nPosition of last shared ind in Hv (count):")
bins = Dict{Int,Int}()
for (pos, n) in zip(pos_last, rank_Hv); bins[pos] = get(bins,pos,0) + 1; end
for k in sort(collect(keys(bins)))
    println("  pos $k: $(bins[k])")
end

# Run consecutive-pair check (where Hv_{k+1} could be output of record k)
println("\n--- Consecutive pair analysis ---")
n_match = 0
n_check = 0
for k in 1:length(recs)-1
    # Did record k's contract produce a tensor whose inds match Hv_{k+1}?
    try
        out_k = recs[k].it * recs[k].Hv
        if Set(inds(out_k)) == Set(inds(recs[k+1].Hv))
            n_match += 1
        end
        n_check += 1
    catch
        # skip (could be wrapped storage issue)
    end
end
@printf("Consecutive pairs where out(k) inds == Hv(k+1) inds: %d / %d\n", n_match, n_check)
