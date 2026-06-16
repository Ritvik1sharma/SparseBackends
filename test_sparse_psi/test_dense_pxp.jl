# Dense reference: same PXP H, dense psi, dense PHP. Apples-to-apples vs
# test_sparse_pxp.jl.
using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using Random
using ArgParse
using Printf
include("utils.jl")

ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]

function parse_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N"
            arg_type = Int
            default = 12
        "--n-sweeps"
            arg_type = Int
            default = 6
        "--maxdim"
            arg_type = Int
            default = 40
        "--mindim"
            arg_type = Int
            default = 1
        "--target-energy"
            arg_type = Float64
            default = NaN
    end
    return parse_args(s)
end

function itensor_from_nonzeros(dims::NTuple{N,Int}, coords::Vector; left::Bool=false) where {N}
    A = zeros(Float64, dims...)
    for c in coords
        @assert length(c) == N
        A[(c .+ 1)...] = 1.0
    end
    return A
end
bind_to_idx(A::AbstractArray, idxs::Index...) = ITensor(A, idxs...)

function NotEqlsLoop_R1(sites)
    N = length(sites)
    R1_first = itensor_from_nonzeros((3, 3, 2), [(0,0,0), (1,1,1), (2,2,0)])
    R1_bulk  = itensor_from_nonzeros((3, 3, 2, 2),
        [(0,0,0,0), (0,0,1,0), (1,1,0,1), (1,1,1,0), (2,2,0,0)])
    R1_last  = itensor_from_nonzeros((3, 3, 2),
        [(0,0,0), (0,0,1), (1,1,0), (1,1,1), (2,2,0)]; left=true)
    bonds = [Index(2, "Link,l=$(i)") for i in 1:N-1]
    Wvec = Vector{ITensor}(undef, N)
    Wvec[1] = bind_to_idx(R1_first, sites[1], sites[1]', bonds[1])
    for j in 2:N-1
        Wvec[j] = bind_to_idx(R1_bulk, sites[j], sites[j]', bonds[j-1], bonds[j])
    end
    Wvec[N] = bind_to_idx(R1_last, sites[N], sites[N]', bonds[N-1])
    return MPO(Wvec)
end

function build_setup(N::Int)
    Random.seed!(42)
    sites = siteinds("S=1", N)
    HT = OpSum()
    HT += 1, "Xp", 1
    for j in 0:N-2; HT += 1, "Px", j+1, "LP", j+2; end
    for j in 0:N-2; HT += 1, "RP", j+1, "Xp", j+2; end
    HT += 1, "Px", N
    H = MPO(HT, sites)
    P = NotEqlsLoop_R1(sites)
    # Build PHP via dense apply (same as reference test_pxp_check.jl's sandwich_mpo).
    H_eff = contract(contract(P'', H'; cutoff=1e-12), P; cutoff=1e-32)
    H_eff = replaceprime(H_eff, 3 => 1)
    psi0     = random_mps(sites)
    # Project via dense apply (multiply_mpo_mps in the reference).
    psi_dense = contract(P, psi0; cutoff=1e-12)
    psi_dense = replaceprime(psi_dense, 1 => 0)
    normalize!(psi_dense)
    return H_eff, psi_dense
end

mps_footprint_bytes(psi) = Base.summarysize(psi)
linkdims_of(psi) = [ITensors.dim(commonind(psi[i], psi[i+1])) for i in 1:length(psi)-1]
function honest_linkdims(psi)
    [(cis = commoninds(psi[i], psi[i+1]); isempty(cis) ? 0 : prod(ITensors.dim, cis))
     for i in 1:length(psi)-1]
end

function report_state(label, psi, E=nothing)
    bytes_total = mps_footprint_bytes(psi)
    mb = round(bytes_total / 2^20, digits=3)
    lds  = linkdims_of(psi)
    hlds = honest_linkdims(psi)
    mx  = isempty(lds) ? 0 : maximum(lds)
    mxh = isempty(hlds) ? 0 : maximum(hlds)
    println("  $label: footprint=$(mb) MiB  reported_maxlinkdim=$mx  honest_maxlinkdim=$mxh")
end

function run_sweeps(H, psi0, n_sweeps::Int, maxdim::Int;
                     cutoff=1e-10, mindim=1, target_E=NaN,
                     orthogonal_states=nothing, weight=20.0)
    psi = psi0
    E = NaN
    sweep_times = Float64[]
    cum = 0.0; cum_excl1 = 0.0
    target_reached_sweep = 0
    target_reached_cum = NaN
    target_reached_cum_excl1 = NaN
    println("  using maxdim=$maxdim  mindim=$mindim  cutoff=$cutoff")
    for i in 1:n_sweeps
        sw = Sweeps(1)
        setmaxdim!(sw, maxdim); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        t = if orthogonal_states === nothing
            @elapsed (E, psi) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false)
        else
            @elapsed (E, psi) = dmrg(H, orthogonal_states, psi, sw;
                outputlevel=0, use_early_exit=false, weight=weight)
        end
        push!(sweep_times, t)
        cum += t
        if i > 1; cum_excl1 += t; end
        @printf("  [sweep %2d] t=%7.3fs  E=%.12f\n", i, t, E)
        if target_reached_sweep == 0 && !isnan(target_E) && E <= target_E
            target_reached_sweep = i
            target_reached_cum = cum
            target_reached_cum_excl1 = cum_excl1
        end
    end
    return (; E, psi, total=cum, total_excl1=cum_excl1,
            target_reached_sweep, target_reached_cum, target_reached_cum_excl1)
end

let
    parsed_args = parse_command_line()
    N = parsed_args["N"]
    n_sweeps = parsed_args["n-sweeps"]
    maxdim_target = parsed_args["maxdim"]
    mindim_target = parsed_args["mindim"]
    target_E = parsed_args["target-energy"]

    println("PXP DENSE  N=$N  n_sweeps=$n_sweeps  maxdim=$maxdim_target  target_E=$(isnan(target_E) ? "—" : target_E)")
    println("Building dense setup ...")
    t_setup = @elapsed (H_d, psi_d) = build_setup(N)
    println("Setup time: $(round(t_setup, digits=1))s")
    report_state("initial psi_d", psi_d)

    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    println("\n=== GROUND STATE ($n_sweeps sweeps at maxdim=$maxdim_target, mindim=$mindim_target; sweep 1 = JIT) ===")
    res = run_sweeps(H_d, psi_d, n_sweeps, maxdim_target; mindim=mindim_target, target_E=target_E)
    E_gs = res.E; psi_gs = res.psi
    avg_excl1 = n_sweeps > 1 ? res.total_excl1 / (n_sweeps - 1) : NaN
    @printf("[ground]    total=%.3fs  excl1=%.3fs  avg/sw=%.3fs  E=%.12f\n",
            res.total, res.total_excl1, avg_excl1, E_gs)
    report_state("final psi_gs", psi_gs, E_gs)

    println("\n=== EXCITED STATE (orthogonal to ground; $n_sweeps sweeps; weight=20) ===")
    psi_init = deepcopy(psi_d)
    sw = Sweeps(n_sweeps)
    setmaxdim!(sw, maxdim_target); setmindim!(sw, mindim_target); setcutoff!(sw, 1e-10)
    t_ex = @elapsed (E_ex, psi_ex) = dmrg(H_d, [psi_gs], psi_init, sw;
        outputlevel=0, use_early_exit=false, weight=20.0)
    @printf("[excited]   total=%.3fs  E=%.12f\n", t_ex, E_ex)
    report_state("final psi_ex", psi_ex, E_ex)

    println("\n=========== SUMMARY (dense PXP N=$N md=$maxdim_target mindim=$mindim_target) ===========")
    @printf("ground E:   %.12f\n", E_gs)
    @printf("excited E:  %.12f\n", E_ex)
    @printf("gap (E_ex - E_gs): %.12f\n", E_ex - E_gs)
    println("\n========== ITensorMPS.PROJMPO_TIMER ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
end
nothing
