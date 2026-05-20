# Offline estimate: how much does pooling wc.alloc_dense save?
#
# Reads canon_dims from a SB_GEMM_DUMP file and times two strategies on those
# exact shapes (replayed in DMRG order):
#   - kernel-A: zeros(T, dims...)              — current behavior
#   - kernel-B: pooled buffer + fill!(0)       — proposed
# Also reports shape-reuse stats to gauge how large the pool needs to be.

using Serialization: deserialize
using LinearAlgebra: BLAS
using Printf: @printf

BLAS.set_num_threads(1)

const PATH = length(ARGS) >= 1 ? ARGS[1] :
    "temp/edited_packages/results/gemm_dump_md80_v2.jls"
const NREPEAT = parse(Int, get(ENV, "NREPEAT", "5"))

records = Any[]
open(PATH, "r") do io
    while !eof(io); push!(records, deserialize(io)); end
end
println("Loaded $(length(records)) records from $PATH")

# Shape stats
shapes = [r.canon_dims for r in records]
unique_shapes = unique(shapes)
@printf("Total alloc calls per pass: %d\n", length(shapes))
@printf("Unique canon_dims shapes:   %d\n", length(unique_shapes))
sizes_bytes = [prod(s) * sizeof(records[i].TC) for (i,s) in enumerate(shapes)]
@printf("Total bytes allocated/pass: %.2f GiB\n", sum(sizes_bytes) / 2^30)
@printf("Per-call alloc bytes (med): %.1f KiB\n", sort(sizes_bytes)[end÷2] / 1024)
@printf("Per-call alloc bytes (max): %.1f MiB\n", maximum(sizes_bytes) / 2^20)

# kernel-A: zeros(T, dims...) per call
function kernA(records)
    bufs = Vector{Array}()
    for r in records
        push!(bufs, zeros(r.TC, r.canon_dims...))
    end
    return bufs
end

# kernel-B: pool keyed on (TC, dims). Reuses buffer; zeroes on each checkout.
function kernB(records, pool)
    bufs = Vector{Array}()
    for r in records
        key = (r.TC, r.canon_dims)
        buf = get(pool, key, nothing)
        if buf === nothing
            buf = Array{r.TC}(undef, r.canon_dims...)
            pool[key] = buf
        end
        fill!(buf, zero(r.TC))
        push!(bufs, buf)
    end
    return bufs
end

# warmup
let pool = Dict{Tuple{DataType,Tuple}, Array}()
    kernA(records[1:10]); kernB(records[1:10], pool)
end

# Time them
tA = @elapsed begin
    for _ in 1:NREPEAT
        kernA(records)
    end
end

pool = Dict{Tuple{DataType,Tuple}, Array}()
# Prime the pool once so first-iteration alloc doesn't dominate
kernB(records, pool)
tB = @elapsed begin
    for _ in 1:NREPEAT
        kernB(records, pool)
    end
end

@printf("\nzeros(T, dims...) per call:     %.3f s  (%.1f μs/call)\n",
        tA, 1e6*tA/(NREPEAT*length(records)))
@printf("pool + fill!(buf, 0) per call:  %.3f s  (%.1f μs/call)\n",
        tB, 1e6*tB/(NREPEAT*length(records)))
@printf("Speedup B/A:                    %.2fx\n", tA/tB)
@printf("Time saved per pass:            %.3f s  (~%.1f%% of A)\n",
        (tA-tB)/NREPEAT, 100*(tA-tB)/tA)
