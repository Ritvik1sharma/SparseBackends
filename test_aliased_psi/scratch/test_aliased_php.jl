# Aliased-psi DMRG: mirror of ../test_sparse_psi/test_sparse_php.jl with psi
# stored as WrappedAliasedBlockSparse (alias structure frozen across sweeps;
# only template numeric data updates). Compared against the bare-H dense
# baseline run in ../test_sparse_psi/test_profile_bareh.jl.
#
# Other workflows are untouched.
using SparseBackends, ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using Random
using ArgParse
using Printf

include("../test_sparse_psi/utils.jl")

function parse_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--N-plaq"
            arg_type = Int
            default = 4
        "--eignv"
            arg_type = Bool
            default = true
        "--spin"
            arg_type = Int
            default = 3
        "--n-profile-sweeps"
            arg_type = Int
            default = 2
        "--maxdim"
            arg_type = Int
            default = 20
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
    psi0     = random_mps(sites)
    # KEY DIFFERENCE vs test_sparse_php.jl: build psi as aliased (not BS).
    # Pass denseLinksB=0 so site + both bond axes live in the sparse prefix.
    psi_ali  = replaceprime(
        contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0),
        1 => 0)
    # Bare H (dense MPO) — matches test_profile_bareh.jl baseline.
    return H, psi_ali
end

mps_footprint_bytes(psi) = Base.summarysize(psi)
linkdims_of(psi) = [ITensors.dim(commonind(psi[i], psi[i+1])) for i in 1:length(psi)-1]
function honest_linkdims(psi)
    [(cis = commoninds(psi[i], psi[i+1]); isempty(cis) ? 0 : prod(ITensors.dim, cis))
     for i in 1:length(psi)-1]
end

function inspect_storage(psi)
    payload_total = 0   # numeric data (templates / bs.data / dense)
    keys_total    = 0   # bs.keys + bs.ids OR aliased keys+alias_ids+scalars
    site_total    = 0
    full_total    = 0   # what fully dense would cost
    bs_total      = 0   # what BS would cost (n_blocks * blksize * sizeof eltype)
    rows = String[]
    for (i, T) in enumerate(psi)
        s = try ITensors.get_external_storage(T) catch _ nothing end
        ss = Base.summarysize(T)
        site_total += ss
        if s isa SparseBackends.WrappedAliasedBlockSparse
            ali = s.aliased
            nb       = length(ali.keys)
            nt       = ali.n_templates
            blksz    = ali.blksize
            data_b   = Base.summarysize(ali.templates)
            keys_b   = Base.summarysize(ali.keys) + Base.summarysize(ali.alias_ids) + Base.summarysize(ali.scalars)
            full_sz  = prod(ali.dims)
            payload_total += data_b
            keys_total    += keys_b
            full_total    += full_sz * sizeof(eltype(ali.templates))
            bs_total      += nb * blksz * sizeof(eltype(ali.templates))
            push!(rows, "site $i [ALIASED] nb=$nb ntmpl=$nt blksize=$blksz  templates=$(round(data_b/1024,digits=2))KiB  keys+ids+scalars=$(round(keys_b/1024,digits=2))KiB  total=$(round(ss/1024,digits=2))KiB  alias_compression(nb/nt)=$(round(nb/max(nt,1),digits=2))x")
        elseif s isa SparseBackends.WrappedBlockSparse
            bs = s.blocksparse
            nblocks  = length(bs.keys)
            data_b   = Base.summarysize(bs.data)
            keys_b   = Base.summarysize(bs.keys) + Base.summarysize(bs.ids)
            full_sz  = prod(bs.dims)
            payload_total += data_b
            keys_total    += keys_b
            full_total    += full_sz * sizeof(eltype(bs.data))
            bs_total      += data_b
            push!(rows, "site $i [BS] nblocks=$nblocks blksize=$(bs.blksize)  data=$(round(data_b/1024,digits=2))KiB  keys=$(round(keys_b/1024,digits=2))KiB  total=$(round(ss/1024,digits=2))KiB")
        else
            a = ITensors.array(T)
            data_b = Base.summarysize(a)
            payload_total += data_b
            full_total    += data_b
            bs_total      += data_b
            push!(rows, "site $i [dense] data=$(round(data_b/1024,digits=2))KiB  total=$(round(ss/1024,digits=2))KiB")
        end
    end
    return rows, payload_total, keys_total, site_total, full_total, bs_total
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
        rows, payload, keys_b, site_total, full, bs_eq = inspect_storage(psi)
        for r in rows; println("    $r"); end
        non_data = site_total - payload
        println("    --- MPS totals ---")
        println("    data(numeric)=$(round(payload/1024,digits=2)) KiB   keys+ids+scalars=$(round(keys_b/1024,digits=2)) KiB   non-data=$(round(non_data/1024,digits=2)) KiB")
        println("    sum-of-sites=$(round(site_total/1024,digits=2)) KiB   if_BS=$(round(bs_eq/1024,digits=2)) KiB   if_dense=$(round(full/1024,digits=2)) KiB")
        if bs_eq > 0; println("    vs-BS compression: $(round(bs_eq/max(payload,1), digits=2))x  vs-dense compression: $(round(full/max(payload,1), digits=2))x"); end
    end
end

# Per-sweep invariant check: every psi site must remain WrappedAliasedBlockSparse.
function check_aliased_invariant(psi; label="")
    bad = Int[]
    for (i, T) in enumerate(psi)
        if !(ITensors.has_external_storage(T) &&
             T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse)
            push!(bad, i)
        end
    end
    if !isempty(bad)
        @warn "[$label] psi sites NOT aliased: $bad (alias invariant broken)"
    else
        println("  [$label] psi storage invariant ✓ (all $(length(psi)) sites aliased)")
    end
    return isempty(bad)
end

function dmrg_with_per_sweep_report(H, psi0, maxdim_schedule::Vector{Int};
                                      cutoff=1e-10, mindim=1, label="")
    psi = psi0
    E = NaN
    for (i, md) in enumerate(maxdim_schedule)
        sw = Sweeps(1)
        setmaxdim!(sw, md); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        t = @elapsed (E, psi) = dmrg(H, psi, sw; outputlevel=1, use_early_exit=false, run_mode=:iso)
        println("  [$label sweep $i / md=$md] $(round(t, digits=2))s  E=$E")
        report_state("after sweep $i", psi, E; verbose=(i == length(maxdim_schedule)))
        check_aliased_invariant(psi; label="after sweep $i")
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

    println("Projector sign = $psign")
    println("Building aliased-psi setup for N=$N_plaq plaquettes, spin=$spin ...")
    t_setup = @elapsed (H_d, psi_d) = build_setup(N_plaq, psign, spin)
    println("Setup time: $(round(t_setup, digits=1))s")
    report_state("initial psi_ali", psi_d; verbose=true)
    check_aliased_invariant(psi_d; label="initial")

    md_lo = max(2, div(maxdim_target, 2))
    println("\n=== WARMUP (2 sweeps, ramp $md_lo → $maxdim_target) ===")
    t_warm = @elapsed (E_warm, psi_warm) = dmrg_with_per_sweep_report(
        H_d, psi_d, [md_lo, maxdim_target]; label="warm")
    println("Warmup done in $(round(t_warm, digits=1))s. E_after_warm = $E_warm")

    println("\n=== PROFILE ($n_prof sweeps at maxdim=$maxdim_target) — timers reset ===")
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)
    t_prof = @elapsed (E_prof, psi_prof) = dmrg_with_per_sweep_report(
        H_d, psi_warm, fill(maxdim_target, n_prof); label="prof")
    println("\nAliased-psi profile ($n_prof sweeps at maxdim=$maxdim_target) done in $(round(t_prof, digits=1))s.")
    println("Aliased final E = $E_prof")

    println("\n========== ITensorMPS.PROJMPO_TIMER ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
    println("\n========== SparseBackends.TIMER ==========")
    print_timer(SparseBackends.TIMER)
end
nothing
