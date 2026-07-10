# Dense reference: same Hamiltonian, dense psi, dense MPO. Apples-to-apples
# vs test_sparse_kl.jl — same sweep schedule, same args.
using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using Random
using ArgParse
using Printf

include("utils.jl")

function parse_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N-plaq"
            help = "Number of plaquettes (N) → system size = 2N+2 sites"
            arg_type = Int
            default = 16
        "--eignv"
            help = "Whether to use eigenvalue +1/-1 (true → +1, false → -1)."
            arg_type = Bool
            default = true
        "--spin"
            help = "Whether to D=2 or D=3 encoding."
            arg_type = Int
            default = 3
        "--n-sweeps"
            help = "Total number of DMRG sweeps at maxdim. Sweep 1 is the JIT warmup and is excluded from post-JIT totals."
            arg_type = Int
            default = 6
        "--maxdim"
            help = "DMRG maxdim cap."
            arg_type = Int
            default = 40
        "--target-energy"
            help = "If set, log the first sweep at which E ≤ target (does not early-exit)."
            arg_type = Float64
            default = NaN
        "--roofline"
            help = "Print the PROJMPO_TIMER breakdown (eigsolve/matvec/replacebond/position), steady-state (sweeps 2..n)."
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end

function build_setup(N::Int, psign::Int, spin::Int)
    Random.seed!(42)
    if spin == 2
        sites = siteinds("S=1/2", 2*N+2)
    elseif spin == 3
        sites = siteinds("S=1", 2*N+2)
    else
        error("Unsupported spin: $spin")
    end
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += 0.5*psign, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    H_dense   = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in H_sparse])
    psi_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp])
    return H_dense, psi_dense
end

mps_footprint_bytes(psi) = Base.summarysize(psi)
linkdims_of(psi) = [ITensors.dim(commonind(psi[i], psi[i+1])) for i in 1:length(psi)-1]
function honest_linkdims(psi)
    [(cis = commoninds(psi[i], psi[i+1]); isempty(cis) ? 0 : prod(ITensors.dim, cis))
     for i in 1:length(psi)-1]
end

function inspect_storage(psi)
    payload_total = 0
    keys_total    = 0
    site_total    = 0
    full_total    = 0
    rows = String[]
    for (i, T) in enumerate(psi)
        s = try ITensors.get_external_storage(T) catch _ nothing end
        ss = Base.summarysize(T)
        site_total += ss
        if s isa SparseBackends.WrappedBlockSparse
            bs = s.blocksparse
            nblocks  = length(bs.keys)
            data_b   = Base.summarysize(bs.data)
            keys_b   = Base.summarysize(bs.keys)
            ids_b    = Base.summarysize(bs.ids)
            full_sz  = prod(bs.dims)
            payload_total += data_b
            keys_total    += keys_b + ids_b
            full_total    += full_sz * sizeof(eltype(bs.data))
            push!(rows, "site $i [BS] nblocks=$nblocks blksize=$(bs.blksize)  data=$(round(data_b/1024,digits=2))KiB  keys=$(round(keys_b/1024,digits=2))KiB  ids=$(round(ids_b/1024,digits=2))KiB  total=$(round(ss/1024,digits=2))KiB  overhead/data=$(round((ss-data_b)/max(data_b,1)*100,digits=1))%")
        else
            a = ITensors.array(T)
            data_b = Base.summarysize(a)
            payload_total += data_b
            full_total    += data_b
            push!(rows, "site $i [dense] data=$(round(data_b/1024,digits=2))KiB  total=$(round(ss/1024,digits=2))KiB  overhead/data=$(round((ss-data_b)/max(data_b,1)*100,digits=1))%")
        end
    end
    return rows, payload_total, keys_total, site_total, full_total
end

function report_state(label, psi, E=nothing; verbose=false)
    bytes_total = mps_footprint_bytes(psi)
    mb = round(bytes_total / 2^20, digits=3)
    lds  = linkdims_of(psi)
    hlds = honest_linkdims(psi)
    mx  = isempty(lds) ? 0 : maximum(lds)
    mxh = isempty(hlds) ? 0 : maximum(hlds)
    println("  $label: footprint=$(mb) MiB  reported_maxlinkdim=$mx  honest_maxlinkdim=$mxh")
    println("    linkdims(reported)=$lds")
    println("    linkdims(honest, ∏shared)=$hlds" * (E === nothing ? "" : "  E=$E"))
    if verbose
        rows, payload, keys_b, site_total, full = inspect_storage(psi)
        for r in rows; println("    $r"); end
        non_data = site_total - payload
        println("    --- MPS totals ---")
        println("    data(numeric)=$(round(payload/1024,digits=2)) KiB   keys+ids=$(round(keys_b/1024,digits=2)) KiB   non-data=$(round(non_data/1024,digits=2)) KiB")
        println("    sum-of-sites=$(round(site_total/1024,digits=2)) KiB   summarysize(psi)=$(round(bytes_total/1024,digits=2)) KiB   overhead/data=$(round(non_data/max(payload,1)*100, digits=1))%")
    end
end

function run_sweeps(H, psi0, n_sweeps::Int, maxdim::Int;
                     cutoff=1e-10, mindim=1, target_E=NaN, roofline::Bool=false)
    psi = psi0
    E = NaN
    sweep_times = Float64[]
    sweep_energies = Float64[]
    cum = 0.0
    cum_excl1 = 0.0
    target_reached_sweep = 0
    target_reached_cum = NaN
    target_reached_cum_excl1 = NaN
    for i in 1:n_sweeps
        sw = Sweeps(1)
        setmaxdim!(sw, maxdim); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        t = @elapsed (E, psi, _esw, terr) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false)
        push!(sweep_times, t); push!(sweep_energies, E)
        cum += t
        if i > 1; cum_excl1 += t; end
        @printf("  [sweep %2d] t=%7.3fs  E=%.12f  maxtruncerr=%.3e\n", i, t, E, terr)
        # Reset AFTER sweep 1 (JIT) so the printed breakdown is steady-state only.
        if i == 1 && roofline
            reset_timer!(ITensorMPS.PROJMPO_TIMER)
        end
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
    N_plaq = parsed_args["N-plaq"]
    psign  = parsed_args["eignv"] ? +1 : -1
    spin   = parsed_args["spin"]
    n_sweeps = parsed_args["n-sweeps"]
    maxdim_target = parsed_args["maxdim"]
    target_E = parsed_args["target-energy"]
    roofline = parsed_args["roofline"]

    println("Projector sign = $psign  (P = ∏(I", psign > 0 ? "+" : "-", "C)/2)")
    println("N_plaq=$N_plaq  n_sweeps=$n_sweeps  maxdim=$maxdim_target  target_E=$(isnan(target_E) ? "—" : target_E)")
    println("Building dense setup for N=$N_plaq plaquettes, spin=$spin ...")
    t_setup = @elapsed (H_d, psi_d) = build_setup(N_plaq, psign, spin)
    println("Setup time: $(round(t_setup, digits=1))s")
    report_state("initial psi_d", psi_d; verbose=false)

    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    println("\n=== RUN ($n_sweeps sweeps at maxdim=$maxdim_target; sweep 1 = JIT) ===")
    res = run_sweeps(H_d, psi_d, n_sweeps, maxdim_target; target_E=target_E, roofline=roofline)
    E_prof = res.E; psi_prof = res.psi

    avg_excl1 = n_sweeps > 1 ? res.total_excl1 / (n_sweeps - 1) : NaN
    println("\n=========== SUMMARY (dense N=$N_plaq md=$maxdim_target) ===========")
    @printf("total time (incl sweep 1, JIT): %8.3fs\n", res.total)
    @printf("total time (excl sweep 1):      %8.3fs\n", res.total_excl1)
    @printf("avg per sweep (excl sweep 1):   %8.3fs\n", avg_excl1)
    @printf("final E: %.12f\n", E_prof)
    if !isnan(target_E)
        if res.target_reached_sweep > 0
            @printf("reached target E=%.12f by sweep %d  (cum=%.3fs  cum_excl1=%.3fs)\n",
                    target_E, res.target_reached_sweep,
                    res.target_reached_cum, res.target_reached_cum_excl1)
        else
            @printf("did NOT reach target E=%.12f within %d sweeps (final %.12f)\n",
                    target_E, n_sweeps, E_prof)
        end
    end
    println("\n--- final state ---")
    report_state("final psi_d", psi_prof, E_prof; verbose=true)

    if roofline
        println("\n========== ITensorMPS.PROJMPO_TIMER (dense, steady-state sweeps 2..$n_sweeps) ==========")
        print_timer(ITensorMPS.PROJMPO_TIMER)
    end
end
nothing
