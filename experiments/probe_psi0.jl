# probe_psi0.jl — SparseBackends (sparse-edits) side of the divergence probe.
# Uses the DENSE sandwich so it is the direct counterpart of the baseline's
# kl_php_dense: any difference is the packages, not the aliased backend.
#   julia --project=. SparseBackends/experiments/probe_psi0.jl [config_id] [seed]
include(joinpath(@__DIR__, "bench_common.jl"))
build_php(ConsOps1, H_raw) = kl_php_dense(ConsOps1, H_raw)
include(joinpath(@__DIR__, "..", "..", "experiments", "psi0_probe_body.jl"))
probe(length(ARGS) >= 1 ? ARGS[1] : "kl_min1_v26_bd40", "sb_dense",
      length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0)
nothing
