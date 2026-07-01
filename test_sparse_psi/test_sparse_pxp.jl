# Sparse Path A DMRG for the PXP / Rydberg-blockade-constrained model.
#
# Hamiltonian: H = sum_j Px_j ⊗ LP_{j+1} + RP_j ⊗ Xp_{j+1} + Xp at boundaries.
# Projector:   P = NotEqlsLoop_R1 (no two adjacent state-1 sites).
# Sparse setup:
#   - psi_sp = P · psi_random (block-sparse storage inherited from P)
#   - DMRG on BARE H (since [H,P] = 0, psi stays in image(P))
#
# Path A: run_mode=:iso (strict iso eigsolve, see dmrg calls), SB_USE_QR=1,
# SB_BALANCED_OWNERSHIP=1, SB_ADAPTIVE_RANK=1 (cleanest config from our optimization work).
using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using Random
using ArgParse
using Printf
include("utils.jl")

# PXP / Rydberg-blockade site operators on S=1 (states 0, 1, 2; "1" is the
# Rydberg-excited / blockaded state).
ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0]
ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0]
ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0]
ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1]

function parse_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N"
            help = "Chain length (number of sites). PXP is single-site, so total dim = N."
            arg_type = Int
            default = 12
        "--n-sweeps"
            help = "Total number of DMRG sweeps. Sweep 1 is JIT warmup, excluded from post-JIT totals."
            arg_type = Int
            default = 6
        "--maxdim"
            help = "DMRG maxdim cap."
            arg_type = Int
            default = 40
        "--mindim"
            arg_type = Int
            default = 1
        "--target-energy"
            help = "If set, log the first sweep at which E ≤ target."
            arg_type = Float64
            default = NaN
    end
    return parse_args(s)
end

# Helper from test_pxp.jl — write a small dense Array with 1.0 at the given coords.
function itensor_from_nonzeros(dims::NTuple{N,Int}, coords::Vector; left::Bool=false) where {N}
    A = zeros(Float64, dims...)
    for c in coords
        @assert length(c) == N
        A[(c .+ 1)...] = 1.0
    end
    return A
end
bind_to_idx(A::AbstractArray, idxs::Index...) = ITensor(A, idxs...)

# Rydberg / Fibonacci constraint MPO: no two adjacent state-1 sites.
# bond=0: "previous site was 0 or 2"; bond=1: "previous site was 1, current must not be 1"
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
    for j in 0:N-2
        HT += 1, "Px", j+1, "LP", j+2
    end
    for j in 0:N-2
        HT += 1, "RP", j+1, "Xp", j+2
    end
    HT += 1, "Px", N
    H = MPO(HT, sites)

    P_sparse = NotEqlsLoop_R1(sites)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    normalize!(psi_sp)
    return H, psi_sp
end

mps_footprint_bytes(psi) = Base.summarysize(psi)
linkdims(psi) = [ITensors.dim(commonind(psi[i], psi[i+1])) for i in 1:length(psi)-1]
function honest_linkdims(psi)
    [(cis = commoninds(psi[i], psi[i+1]); isempty(cis) ? 0 : prod(ITensors.dim, cis))
     for i in 1:length(psi)-1]
end

function report_state(label, psi, E=nothing)
    bytes_total = mps_footprint_bytes(psi)
    mb = round(bytes_total / 2^20, digits=3)
    lds  = linkdims(psi)
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
    sweep_energies = Float64[]
    cum = 0.0; cum_excl1 = 0.0
    target_reached_sweep = 0
    target_reached_cum = NaN
    target_reached_cum_excl1 = NaN
    for i in 1:n_sweeps
        sw = Sweeps(1)
        setmaxdim!(sw, maxdim); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        t = if orthogonal_states === nothing
            @elapsed (E, psi) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false, run_mode=:iso)
        else
            @elapsed (E, psi) = dmrg(H, orthogonal_states, psi, sw;
                outputlevel=0, use_early_exit=false, weight=weight, run_mode=:iso)
        end
        push!(sweep_times, t); push!(sweep_energies, E)
        cum += t
        if i > 1; cum_excl1 += t; end
        @printf("  [sweep %2d] t=%7.3fs  E=%.12f\n", i, t, E)
        if target_reached_sweep == 0 && !isnan(target_E) && E <= target_E
            target_reached_sweep = i
            target_reached_cum = cum
            target_reached_cum_excl1 = cum_excl1
        end
    end
    return (; E, psi, sweep_times, sweep_energies, total=cum, total_excl1=cum_excl1,
            target_reached_sweep, target_reached_cum, target_reached_cum_excl1)
end

let
    parsed_args = parse_command_line()
    N = parsed_args["N"]
    n_sweeps = parsed_args["n-sweeps"]
    maxdim_target = parsed_args["maxdim"]
    mindim_target = parsed_args["mindim"]
    target_E = parsed_args["target-energy"]

    println("PXP Path A sparse  run_mode=:iso")
    println("N=$N  n_sweeps=$n_sweeps  maxdim=$maxdim_target  target_E=$(isnan(target_E) ? "—" : target_E)")
    println("Building setup ...")
    H, psi_sp = build_setup(N)
    println("System: $(length(psi_sp)) sites")
    report_state("initial psi_sp", psi_sp)

    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)
    println("\n=== GROUND STATE ($n_sweeps sweeps at maxdim=$maxdim_target, mindim=$mindim_target; sweep 1 = JIT) ===")
    res = run_sweeps(H, psi_sp, n_sweeps, maxdim_target; mindim=mindim_target, target_E=target_E)
    E_gs = res.E; psi_gs = res.psi
    avg_excl1 = n_sweeps > 1 ? res.total_excl1 / (n_sweeps - 1) : NaN
    @printf("[ground]    total=%.3fs  excl1=%.3fs  avg/sw=%.3fs  E=%.12f\n",
            res.total, res.total_excl1, avg_excl1, E_gs)
    report_state("final psi_gs", psi_gs, E_gs)

    println("\n=== EXCITED STATE (orthogonal to ground; $n_sweeps sweeps; weight=20) ===")
    psi_init = deepcopy(psi_sp)
    sw = Sweeps(n_sweeps)
    setmaxdim!(sw, maxdim_target); setmindim!(sw, mindim_target); setcutoff!(sw, 1e-10)
    t_ex = @elapsed (E_ex, psi_ex) = dmrg(H, [psi_gs], psi_init, sw;
        outputlevel=0, use_early_exit=false, weight=20.0, run_mode=:iso)
    @printf("[excited]   total=%.3fs  E=%.12f\n", t_ex, E_ex)
    report_state("final psi_ex", psi_ex, E_ex)

    println("\n=========== SUMMARY (sparse PXP N=$N md=$maxdim_target mindim=$mindim_target) ===========")
    @printf("ground E:   %.12f\n", E_gs)
    @printf("excited E:  %.12f\n", E_ex)
    @printf("gap (E_ex - E_gs): %.12f\n", E_ex - E_gs)
    println("\n========== ITensorMPS.PROJMPO_TIMER ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
    println("\n========== SparseBackends.TIMER ==========")
    print_timer(SparseBackends.TIMER)
end
nothing
