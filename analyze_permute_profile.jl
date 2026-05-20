# Analyze SB_PERMUTE_PROFILE TSV. Reports per-(site, kernel) totals and the
# total permute-vs-gemm breakdown. Goal: find the upper bound on what static
# layout reordering can save.
#
# TSV columns (defined in abstractprojmpo.jl and contract_bs_dense.jl):
#   site  kernel  rank_A  rank_B  rank_out  shared
#   permute_A_s  permute_B_s  gemm_s  total_s
#   shared_pos_A  shared_pos_B  shared_pos_out  inds_in  inds_out

using Printf: @printf
using Statistics: mean, median

const PATH = length(ARGS) >= 1 ? ARGS[1] :
    "temp/edited_packages/results/permute_profile_md160.tsv"

struct Row
    site::String
    kernel::String
    rank_A::Int
    rank_B::Int
    rank_out::Int
    shared::Int
    permute_A_s::Float64
    permute_B_s::Float64
    gemm_s::Float64
    total_s::Float64
end

rows = Row[]
for ln in eachline(PATH)
    parts = split(ln, '\t')
    length(parts) < 10 && continue
    try
        push!(rows, Row(parts[1], parts[2],
            parse(Int, parts[3]), parse(Int, parts[4]), parse(Int, parts[5]),
            parse(Int, parts[6]),
            parse(Float64, parts[7]), parse(Float64, parts[8]),
            parse(Float64, parts[9]), parse(Float64, parts[10])))
    catch e
        # skip malformed
    end
end

println("Loaded $(length(rows)) records from $PATH")
println()

# Aggregate per (site, kernel)
function agg(site, kernel)
    ms = filter(r -> r.site == site && r.kernel == kernel, rows)
    isempty(ms) && return nothing
    return (
        count = length(ms),
        permute_A = sum(r.permute_A_s for r in ms),
        permute_B = sum(r.permute_B_s for r in ms),
        gemm      = sum(r.gemm_s      for r in ms),
        total     = sum(r.total_s     for r in ms),
    )
end

println("Aggregate by (site, kernel):")
@printf("%-10s %-8s %8s %12s %12s %12s %12s\n",
        "site", "kernel", "count", "permute_A", "permute_B", "gemm", "total")
for site in ("matvec", "position")
    for kernel in ("sparseH", "denseH")
        a = agg(site, kernel)
        a === nothing && continue
        @printf("%-10s %-8s %8d %12.3f %12.3f %12.3f %12.3f\n",
                site, kernel, a.count, a.permute_A, a.permute_B, a.gemm, a.total)
    end
end

# Grand totals
total_pA = sum(r.permute_A_s for r in rows)
total_pB = sum(r.permute_B_s for r in rows)
total_gemm = sum(r.gemm_s for r in rows)
total_all = sum(r.total_s for r in rows)
println()
@printf("Grand totals across all logged calls:\n")
@printf("  total                   = %.3f s\n", total_all)
@printf("  bdd.permute_A           = %.3f s (%.1f%%)\n", total_pA, 100*total_pA/total_all)
@printf("  bdd.permute_B           = %.3f s (%.1f%%)\n", total_pB, 100*total_pB/total_all)
@printf("  bdd.gemm                = %.3f s (%.1f%%)\n", total_gemm, 100*total_gemm/total_all)
@printf("  unaccounted (overhead, denseH internals, etc.) = %.3f s (%.1f%%)\n",
        total_all - total_pA - total_pB - total_gemm,
        100*(total_all - total_pA - total_pB - total_gemm) / total_all)

# Permute fraction within sparse-kernel calls only
sparse_rows = filter(r -> r.kernel == "sparseH", rows)
if !isempty(sparse_rows)
    sp_pA = sum(r.permute_A_s for r in sparse_rows)
    sp_pB = sum(r.permute_B_s for r in sparse_rows)
    sp_gemm = sum(r.gemm_s for r in sparse_rows)
    sp_total = sum(r.total_s for r in sparse_rows)
    println()
    @printf("Sparse kernel only (n=%d):\n", length(sparse_rows))
    @printf("  permute_A / total      = %.1f%%\n", 100*sp_pA/sp_total)
    @printf("  permute_B / total      = %.1f%%\n", 100*sp_pB/sp_total)
    @printf("  gemm / total           = %.1f%%\n", 100*sp_gemm/sp_total)
end

# How much of permute_B is in matvec vs position?
mv_pB = sum(r.permute_B_s for r in rows if r.site == "matvec" && r.kernel == "sparseH")
ps_pB = sum(r.permute_B_s for r in rows if r.site == "position" && r.kernel == "sparseH")
println()
@printf("bdd.permute_B split: matvec=%.3f s   position!=%.3f s\n", mv_pB, ps_pB)

# Calls where permute_B == 0 (already in canonical layout) vs > 0
sparse_no_pB = count(r -> r.kernel == "sparseH" && r.permute_B_s < 1e-7, rows)
sparse_yes_pB = count(r -> r.kernel == "sparseH" && r.permute_B_s >= 1e-7, rows)
@printf("Sparse calls with no permute_B (already canonical): %d / %d (%.1f%%)\n",
        sparse_no_pB, sparse_no_pB + sparse_yes_pB,
        100*sparse_no_pB / max(1, sparse_no_pB + sparse_yes_pB))

# NDTensors dense internals: kernel == "denseH_internal"
dh_rows = filter(r -> r.kernel == "denseH_internal", rows)
if !isempty(dh_rows)
    println()
    println("=== NDTensors dense internal breakdown (denseH_internal) ===")
    @printf("Calls: %d\n", length(dh_rows))
    pA_total = sum(r.permute_A_s for r in dh_rows)
    pB_total = sum(r.permute_B_s for r in dh_rows)  # this column holds permuteB+permuteC_in+permuteC_out
    gemm_total = sum(r.gemm_s for r in dh_rows)
    tot_total = sum(r.total_s for r in dh_rows)
    @printf("  total:                 %.3f s\n", tot_total)
    @printf("  permute_A:             %.3f s  (%.1f%%)\n", pA_total, 100*pA_total/tot_total)
    @printf("  permute_B+C_in+C_out:  %.3f s  (%.1f%%)\n", pB_total, 100*pB_total/tot_total)
    @printf("  gemm (mul!):           %.3f s  (%.1f%%)\n", gemm_total, 100*gemm_total/tot_total)
    @printf("  permute fraction:      %.1f%%\n", 100*(pA_total + pB_total)/tot_total)

    # Split by site
    mv_dh = filter(r -> r.site == "matvec", dh_rows)
    ps_dh = filter(r -> r.site == "position", dh_rows)
    @printf("\n  matvec calls: %d, permutes=%.3f s, gemm=%.3f s\n",
        length(mv_dh), sum(r.permute_A_s + r.permute_B_s for r in mv_dh), sum(r.gemm_s for r in mv_dh))
    @printf("  position calls: %d, permutes=%.3f s, gemm=%.3f s\n",
        length(ps_dh), sum(r.permute_A_s + r.permute_B_s for r in ps_dh), sum(r.gemm_s for r in ps_dh))
end
