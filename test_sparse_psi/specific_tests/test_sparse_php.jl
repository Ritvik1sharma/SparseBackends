# Dense reference: same Hamiltonian, dense psi, dense MPO. Apples-to-apples
# vs test_profile_bareh.jl — same sweep schedule, same args.
using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using Random
using ArgParse

include("../utils.jl")

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
        "--n-profile-sweeps"
            help = "Number of sweeps in the profile (post-warmup) phase."
            arg_type = Int
            default = 8
        "--maxdim"
            help = "DMRG maxdim cap."
            arg_type = Int
            default = 40
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
    H_dense   = H_sparse  # KEEP SPARSE (BS-stored)
    psi_dense = psi_sp  # KEEP SPARSE
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

function dmrg_with_per_sweep_report(H, psi0, maxdim_schedule::Vector{Int};
                                      cutoff=1e-10, mindim=1, label="")
    psi = psi0
    E = NaN
    for (i, md) in enumerate(maxdim_schedule)
        sw = Sweeps(1)
        setmaxdim!(sw, md); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        t = @elapsed (E, psi) = dmrg(H, psi, sw; outputlevel=1, use_early_exit=false)
        println("  [$label sweep $i / md=$md] $(round(t, digits=2))s  E=$E")
        report_state("after sweep $i", psi, E; verbose=(i == length(maxdim_schedule)))
    end
    return E, psi
end

let
    parsed_args = parse_command_line()
    N_plaq = parsed_args["N-plaq"]
    psign  = parsed_args["eignv"] ? +1 : -1
    spin   = parsed_args["spin"]
    n_prof = parsed_args["n-profile-sweeps"]
    maxdim_target = parsed_args["maxdim"]

    println("Projector sign = $psign  (P = ∏(I", psign > 0 ? "+" : "-", "C)/2)")
    println("Building dense setup for N=$N_plaq plaquettes, spin=$spin ...")
    t_setup = @elapsed (H_d, psi_d) = build_setup(N_plaq, psign, spin)
    println("Setup time: $(round(t_setup, digits=1))s")
    report_state("initial psi_d", psi_d; verbose=true)

    md_lo = max(2, div(maxdim_target, 2))
    println("\n=== WARMUP (2 sweeps, ramp $md_lo → $maxdim_target) ===")
    t_warm = @elapsed (E_warm, psi_warm) = dmrg_with_per_sweep_report(
        H_d, psi_d, [md_lo, maxdim_target]; label="warm")
    println("Warmup done in $(round(t_warm, digits=1))s. E_after_warm = $E_warm")

    println("\n=== PROFILE ($n_prof sweeps at maxdim=$maxdim_target) — timers reset ===")
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    t_prof = @elapsed (E_prof, psi_prof) = dmrg_with_per_sweep_report(
        H_d, psi_warm, fill(maxdim_target, n_prof); label="prof")
    println("\nDense profile ($n_prof sweeps at maxdim=$maxdim_target) done in $(round(t_prof, digits=1))s.")
    println("Dense final E = $E_prof")

    println("\n========== ITensorMPS.PROJMPO_TIMER (dense) ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
end
nothing
