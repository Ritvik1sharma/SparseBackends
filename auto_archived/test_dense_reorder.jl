# Offline replay: for each dumped (it, Hv) pair from a sparse-DMRG run's
# denseH_denseV calls, time:
#   A) it * Hv                       (current order — what production does)
#   B) (permute(it, …)) * Hv         (move shared inds to the back of it)
#   C) it * permute(Hv, …)           (move shared inds to the front of Hv)
#
# If any alternative is meaningfully faster, then reordering the indices upstream
# (i.e., changing the sparse kernel's output index order) could close the gap.
#
# Usage:
#   julia --project=temp/edited_packages temp/edited_packages/test_dense_reorder.jl \
#       /home/ritvik/temp/temp/edited_packages/results/dd_pairs_md160.jls

using Serialization: deserialize
using LinearAlgebra: BLAS
using Printf: @printf
using SparseBackends
using ITensors

BLAS.set_num_threads(1)

const PATH = length(ARGS) >= 1 ? ARGS[1] :
    "temp/edited_packages/results/dd_pairs_md160.jls"
const NREPEAT = parse(Int, get(ENV, "NREPEAT", "20"))

recs = Any[]
open(PATH, "r") do io
    while !eof(io); push!(recs, deserialize(io)); end
end
println("Loaded $(length(recs)) (it, Hv) pairs")

# Bench helper. If `f` throws on any record, returns NaN.
function bench(f, recs, nrepeat)
    # warm up
    try
        for r in recs[1:min(5, length(recs))]; _ = f(r); end
    catch
        return NaN
    end
    t = @elapsed begin
        for _ in 1:nrepeat
            for r in recs
                _ = f(r)
            end
        end
    end
    return t
end

# A: baseline
bench_A = bench(r -> r.it * r.Hv, recs, NREPEAT)

# B: pre-permute `it` so shared inds are at the end (canonical for GEMM)
function call_B(r)
    sh = commoninds(r.it, r.Hv)
    not_sh = uniqueinds(r.it, r.Hv)
    return permute(r.it, [not_sh..., sh...]) * r.Hv
end
bench_B = bench(call_B, recs, NREPEAT)

# C: pre-permute `Hv` so shared inds are at the front
function call_C(r)
    sh = commoninds(r.it, r.Hv)
    not_sh = uniqueinds(r.Hv, r.it)
    return r.it * permute(r.Hv, [sh..., not_sh...])
end
bench_C = bench(call_C, recs, NREPEAT)

# D: pre-permute BOTH (canonical: it = [non-shared..., shared...], Hv = [shared..., non-shared...])
function call_D(r)
    sh = commoninds(r.it, r.Hv)
    not_it = uniqueinds(r.it, r.Hv)
    not_Hv = uniqueinds(r.Hv, r.it)
    it_p = permute(r.it, [not_it..., sh...])
    Hv_p = permute(r.Hv, [sh..., not_Hv...])
    return it_p * Hv_p
end
bench_D = bench(call_D, recs, NREPEAT)

us(t) = 1e6 * t / (NREPEAT * length(recs))
@printf("\nNREPEAT=%d, %d records, single-thread BLAS\n", NREPEAT, length(recs))
@printf("A: it * Hv (baseline):                   %.3f s  (%.1f μs/call)\n", bench_A, us(bench_A))
@printf("B: permute(it) so shared at end * Hv:    %.3f s  (%.1f μs/call)  speedup %.2fx\n",
        bench_B, us(bench_B), isfinite(bench_A/bench_B) ? bench_A/bench_B : NaN)
@printf("C: it * permute(Hv) so shared at front:  %.3f s  (%.1f μs/call)  speedup %.2fx\n",
        bench_C, us(bench_C), bench_A/bench_C)
@printf("D: permute both (canonical):             %.3f s  (%.1f μs/call)  speedup %.2fx\n",
        bench_D, us(bench_D), isfinite(bench_A/bench_D) ? bench_A/bench_D : NaN)

# Also: what would the absolute best look like if ITensors had zero permute cost?
# Approximate: do a raw mul! on the post-permute matrices and call that the floor.
# For now, just report what we see — the differences above already tell us if
# there's meaningful headroom.
