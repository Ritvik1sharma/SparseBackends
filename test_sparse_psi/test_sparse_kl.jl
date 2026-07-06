# Profile sparse DMRG at given N (plaquettes) and projector sign in production
# regime: warmup sweeps first (JIT compile + ramp to maxdim), then reset timers
# and profile subsequent sweeps where mult is fully grown. Reports both
# ITensorMPS.PROJMPO_TIMER (DMRG-level) and SparseBackends.TIMER (kernel-level).
# Uses run_mode=:iso (see dmrg call) so dmrg runs the standard eigsolve path (no
# M^{-1/2} wrap), since iso is preserved by strict-cap SVD in this run.
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
            default = 4
        "--eignv"
            help = "Whether to use eigenvalue +1/-1 (true → +1, false → -1)."
            arg_type = Bool
            default = true
        "--n-sweeps"
            help = "Total number of DMRG sweeps at maxdim. Sweep 1 is the JIT warmup and is excluded from post-JIT totals."
            arg_type = Int
            default = 6
        "--maxdim"
            help = "DMRG maxdim cap (per-bond effective total dim target)."
            arg_type = Int
            default = 40
        "--target-energy"
            help = "If set, log the first sweep at which E ≤ target (does not early-exit)."
            arg_type = Float64
            default = NaN
        "--roofline"
            help = "Enable the consolidated roofline/flop-count/env-footprint/perm-capture/GEMM-histogram instrumentation."
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end

function build_setup(N::Int, psign::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    # psign = +1 → P = ∏(I+C)/2 ; psign = -1 → P = ∏(I-C)/2 (larger bond dims).
    cs = 0.5 * psign
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += cs,  "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    # Bare H (dense MPO) — assumes [H, P] = 0 so ψ stays in image-of-P.
    return H, psi_sp
end

mps_footprint_bytes(psi) = Base.summarysize(psi)
linkdims(psi) = [ITensors.dim(commonind(psi[i], psi[i+1])) for i in 1:length(psi)-1]

# Walk the raw storage of each site. For BS we read bs.dims (per-axis size),
# bs.blksize (scalars per block), length(bs.keys) (number of stored blocks).
# For dense we read size(array). This is the ground truth — does NOT trust
# `dim(commonind)`.
# For a sparse site at position i, find which BS axes correspond to the
# right-link indices, then count unique values present in stored block keys
# at those axes. Returns (real_bond_dim_R, possible_bond_dim_R).
function _real_bond_dim(s::SparseBackends.WrappedBlockSparse, link_inds)
    bs = s.blocksparse
    P  = length(bs.keys) > 0 ? length(bs.keys[1]) : 0
    inds_tuple = s.inds
    # Map each link index to its axis position in bs.dims.
    axis_positions = Int[]
    for li in link_inds
        for (j, idx) in enumerate(inds_tuple)
            if idx == li
                push!(axis_positions, j); break
            end
        end
    end
    if isempty(axis_positions); return (0, 0); end
    # Channel axes (j ≤ P) contribute "# unique values across stored blocks".
    # Dense axes (j > P) contribute their full dim (multiplicity is dense).
    real_dim = 1
    poss_dim = 1
    for j in axis_positions
        poss_dim *= bs.dims[j]
        if j <= P
            vals = Set{Int}()
            for k in bs.keys
                push!(vals, k[j])
            end
            real_dim *= length(vals)
        else
            real_dim *= bs.dims[j]
        end
    end
    return (real_dim, poss_dim)
end

function inspect_storage(psi)
    payload_total = 0   # bytes of pure numeric data (bs.data or dense array)
    keys_total    = 0   # bytes of bs.keys + bs.ids (block-index metadata)
    site_total    = 0   # bytes from summarysize per site (incl. wrappers)
    full_total    = 0   # what dense storage WOULD cost (if all schema slots filled)
    rows = String[]
    for (i, T) in enumerate(psi)
        s = try ITensors.get_external_storage(T) catch _ nothing end
        ss = Base.summarysize(T)
        site_total += ss
        if s isa SparseBackends.WrappedBlockSparse
            bs = s.blocksparse
            nblocks  = length(bs.keys)
            stored_entries = nblocks * bs.blksize
            full_sz  = prod(bs.dims)
            data_b   = Base.summarysize(bs.data)
            keys_b   = Base.summarysize(bs.keys)
            ids_b    = Base.summarysize(bs.ids)
            other_b  = ss - data_b - keys_b - ids_b
            payload_total += data_b
            keys_total    += keys_b + ids_b
            full_total    += full_sz * sizeof(eltype(bs.data))
            push!(rows, "site $i [BS] nblocks=$nblocks blksize=$(bs.blksize)  data=$(round(data_b/1024,digits=2))KiB  keys=$(round(keys_b/1024,digits=2))KiB  ids=$(round(ids_b/1024,digits=2))KiB  other(wrappers/dims)=$(round(other_b/1024,digits=2))KiB  total=$(round(ss/1024,digits=2))KiB  overhead/data=$(round((ss-data_b)/max(data_b,1)*100, digits=1))%")
        else
            a = ITensors.array(T)
            data_b = Base.summarysize(a)
            other_b = ss - data_b
            payload_total += data_b
            full_total    += data_b
            push!(rows, "site $i [dense] data=$(round(data_b/1024,digits=2))KiB  other(wrappers)=$(round(other_b/1024,digits=2))KiB  total=$(round(ss/1024,digits=2))KiB  overhead/data=$(round((ss-data_b)/max(data_b,1)*100, digits=1))%")
        end
    end
    return rows, payload_total, keys_total, site_total, full_total
end

# Honest per-bond dim: ALL indices shared between psi[i] and psi[i+1]
# (channel + multiplicity), multiplied. ITensors' `dim(commonind)` returns
# only one shared index — for BS storage that's typically just the channel.
function honest_linkdims(psi)
    out = Int[]
    for i in 1:length(psi)-1
        cis = commoninds(psi[i], psi[i+1])
        push!(out, isempty(cis) ? 0 : prod(ITensors.dim, cis))
    end
    return out
end

# Iso check. For an MPS in mixed canonical form with ortho center c, sites
# 1..c-1 should be left-iso (L†L = I on right link) and sites c+1..N right-iso
# (R R† = I on left link). We compute BOTH "violations" per site and report
# both columns — the smaller one tells you which side is canonical for that
# site. ε(i) = ‖A_i†_link A_i_link - I_link‖_F / √dim(link).
#
# For sparse psi with channel structure: this is the literal iso definition,
# summing over all left-side indices when projecting on right link (left-iso)
# or vice versa. Cross-channel contributions show up as nonzero off-diagonal
# blocks in (link, link') — pure within-channel breaks show up as nonzero
# diagonal blocks (not = I).
using LinearAlgebra: I as eye

# Pure-sparse iso check. T stays sparse; we contract T * dag(prime(T, link))
# via sparse machinery → small result on (link, link'). Densify ONLY that small
# result (a single matrix on the link indices) for the norm-vs-identity check.
function _link_gram(T::ITensors.ITensor, link_inds)
    Tp = prime(T, link_inds)
    G  = T * dag(Tp)
    # G is small (link × link'). Convert via to_dense_itensors_unfused which
    # walks the BS payload manually — doesn't go through Array/_permute path
    # that's missing methods for WrappedBlockSparse external storage.
    Gd = ITensors.has_external_storage(G) ?
            SparseBackends.to_dense_itensors_unfused(G) : G
    link_pr = [prime(I) for I in link_inds]
    G_arr = Array(Gd, link_inds..., link_pr...)
    n = prod(ITensors.dim, link_inds)
    return reshape(G_arr, n, n), n
end

function iso_violations(psi)
    N = length(psi)
    rows = NamedTuple{(:site, :left_iso_err, :right_iso_err, :right_link_dim, :left_link_dim), Tuple{Int, Float64, Float64, Int, Int}}[]
    for i in 1:N
        T = psi[i]
        # right-link iso (left-canonical): G = T†T on right link should be I
        le = NaN; r_dim = 0
        if i < N
            ri = commoninds(psi[i], psi[i+1])
            if !isempty(ri)
                G, n = _link_gram(T, ri)
                r_dim = n
                le = norm(G - eye(n)) / sqrt(n)
            end
        end
        # left-link iso (right-canonical): G = T T† on left link should be I
        re = NaN; l_dim = 0
        if i > 1
            li = commoninds(psi[i], psi[i-1])
            if !isempty(li)
                G, n = _link_gram(T, li)
                l_dim = n
                re = norm(G - eye(n)) / sqrt(n)
            end
        end
        push!(rows, (site=i, left_iso_err=le, right_iso_err=re, right_link_dim=r_dim, left_link_dim=l_dim))
    end
    return rows
end

function print_iso(label, psi)
    println("  iso check ($label):")
    println("    site | left-iso(L†L=I rt)  right-iso(R R†=I lt) | rt-dim   lt-dim")
    rows = iso_violations(psi)
    for r in rows
        lstr = isnan(r.left_iso_err)  ? "   -  " : @sprintf("%.2e", r.left_iso_err)
        rstr = isnan(r.right_iso_err) ? "   -  " : @sprintf("%.2e", r.right_iso_err)
        println(@sprintf("    %4d | %s            %s        | %5d    %5d", r.site, lstr, rstr, r.right_link_dim, r.left_link_dim))
    end
end

function report_state(label, psi, E=nothing; verbose=false)
    bytes_total = mps_footprint_bytes(psi)
    mb = round(bytes_total / 2^20, digits=3)
    lds  = linkdims(psi)
    hlds = honest_linkdims(psi)
    mx  = isempty(lds) ? 0 : maximum(lds)
    mxh = isempty(hlds) ? 0 : maximum(hlds)
    println("  $label: footprint=$(mb) MiB  reported_maxlinkdim=$mx  honest_maxlinkdim=$mxh")
    println("    linkdims(reported, single shared idx)=$lds")
    println("    linkdims(honest, ∏all shared)=$hlds" * (E === nothing ? "" : "  E=$E"))
    if verbose
        rows, payload, keys_b, site_total, full = inspect_storage(psi)
        for r in rows; println("    $r"); end
        occ = payload / max(full, 1)
        non_data = site_total - payload
        println("    --- MPS totals ---")
        println("    data(numeric)=$(round(payload/1024,digits=2)) KiB   keys+ids(BS only)=$(round(keys_b/1024,digits=2)) KiB   non-data(keys+ids+wrappers+dims)=$(round(non_data/1024,digits=2)) KiB")
        println("    sum-of-sites=$(round(site_total/1024,digits=2)) KiB   if_fully_dense=$(round(full/1024,digits=2)) KiB   summarysize(psi)=$(round(bytes_total/1024,digits=2)) KiB   overhead/data=$(round(non_data/max(payload,1)*100, digits=1))%")
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
        t = @elapsed (E, psi, _esw, terr) = dmrg(H, psi, sw; outputlevel=0, use_early_exit=false, run_mode=:iso, roofline=roofline)
        push!(sweep_times, t); push!(sweep_energies, E)
        cum += t
        if i > 1; cum_excl1 += t; end
        @printf("  [sweep %2d] t=%7.3fs  E=%.12f  maxtruncerr=%.3e\n", i, t, E, terr)
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
    n_sweeps = parsed_args["n-sweeps"]
    maxdim_target = parsed_args["maxdim"]
    target_E = parsed_args["target-energy"]
    roofline = parsed_args["roofline"]

    println("run_mode = :iso  (standard eigsolve, no M⁻¹ wrap)")
    println("Projector sign = $psign  (P = ∏(I", psign > 0 ? "+" : "-", "C)/2)")
    println("N_plaq=$N_plaq  n_sweeps=$n_sweeps  maxdim=$maxdim_target  target_E=$(isnan(target_E) ? "—" : target_E)")
    println("Building setup for N=$N_plaq plaquettes ...")
    H_sp, psi_sp = build_setup(N_plaq, psign)
    println("System: $(length(psi_sp)) sites")
    report_state("initial psi_sp", psi_sp; verbose=false)

    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)
    println("\n=== RUN ($n_sweeps sweeps at maxdim=$maxdim_target; sweep 1 = JIT) ===")
    res = run_sweeps(H_sp, psi_sp, n_sweeps, maxdim_target; target_E=target_E, roofline=roofline)
    E_prof = res.E; psi_prof = res.psi

    avg_excl1 = n_sweeps > 1 ? res.total_excl1 / (n_sweeps - 1) : NaN
    println("\n=========== SUMMARY (sparse N=$N_plaq md=$maxdim_target) ===========")
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
    report_state("final psi_sp", psi_prof, E_prof; verbose=true)
    print_iso("final", psi_prof)

    println("\n========== ITensorMPS.PROJMPO_TIMER ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
    println("\n========== SparseBackends.TIMER ==========")
    print_timer(SparseBackends.TIMER)
    if roofline
        SparseBackends.show_gemm_dims_hist()
    end
end
nothing
