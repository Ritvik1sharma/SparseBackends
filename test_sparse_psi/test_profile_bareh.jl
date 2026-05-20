# Profile sparse DMRG at given N (plaquettes) and projector sign in production
# regime: warmup sweeps first (JIT compile + ramp to maxdim), then reset timers
# and profile subsequent sweeps where mult is fully grown. Reports both
# ITensorMPS.PROJMPO_TIMER (DMRG-level) and SparseBackends.TIMER (kernel-level).
# Forces BMF_ISO_PATH=1 so dmrg uses the standard eigsolve path (no M^{-1/2}
# wrap), since iso is preserved by strict-cap SVD in this run.
ENV["BMF_ISO_PATH"] = "1"
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
        "--n-profile-sweeps"
            help = "Number of sweeps in the profile (post-warmup) phase."
            arg_type = Int
            default = 5
        "--maxdim"
            help = "DMRG maxdim cap (per-bond effective total dim target)."
            arg_type = Int
            default = 40
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
function inspect_storage(psi)
    payload_total = 0
    rows = String[]
    for (i, T) in enumerate(psi)
        s = try ITensors.get_external_storage(T) catch _ nothing end
        if s isa SparseBackends.WrappedBlockSparse
            bs = s.blocksparse
            nblocks = length(bs.keys)
            payload = nblocks * bs.blksize * sizeof(eltype(bs.data))
            payload_total += payload
            push!(rows, "site $i [BS] axis_dims=$(bs.dims)  blksize=$(bs.blksize)  nblocks=$nblocks  payload=$(round(payload/1024,digits=2))KiB")
        else
            a = ITensors.array(T)
            payload = sizeof(a)
            payload_total += payload
            push!(rows, "site $i [dense] size=$(size(a))  payload=$(round(payload/1024,digits=2))KiB")
        end
    end
    return rows, payload_total
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
        rows, payload = inspect_storage(psi)
        for r in rows; println("    $r"); end
        println("    payload(numeric only)=$(round(payload/1024,digits=2)) KiB   summarysize=$(round(bytes_total/1024,digits=2)) KiB  overhead_ratio=$(round(bytes_total/payload, digits=2))×")
    end
end

# Run sweeps one-at-a-time so we can dump per-sweep state.
function dmrg_with_per_sweep_report(H, psi0, maxdim_schedule::Vector{Int};
                                      cutoff=1e-10, mindim=1, label="")
    psi = psi0
    E = NaN
    for (i, md) in enumerate(maxdim_schedule)
        sw = Sweeps(1)
        setmaxdim!(sw, md); setmindim!(sw, mindim); setcutoff!(sw, cutoff)
        t = @elapsed (E, psi) = dmrg(H, psi, sw; outputlevel=1, use_early_exit=false)
        println("  [$label sweep $i / md=$md] $(round(t, digits=2))s  E=$E")
        # Verbose dump only on last sweep to keep log readable.
        report_state("after sweep $i", psi, E; verbose=(i == length(maxdim_schedule)))
        print_iso("after sweep $i", psi)
    end
    return E, psi
end

let
    parsed_args = parse_command_line()
    N_plaq = parsed_args["N-plaq"]
    psign  = parsed_args["eignv"] ? +1 : -1
    n_prof = parsed_args["n-profile-sweeps"]
    maxdim_target = parsed_args["maxdim"]

    println("BMF_ISO_PATH = ", ENV["BMF_ISO_PATH"], "  (1 ⇒ standard eigsolve, no M⁻¹ wrap)")
    println("Projector sign = $psign  (P = ∏(I", psign > 0 ? "+" : "-", "C)/2)")
    println("Building setup for N=$N_plaq plaquettes ...")
    H_sp, psi_sp = build_setup(N_plaq, psign)
    println("System: $(length(psi_sp)) sites")
    report_state("initial psi_sp", psi_sp; verbose=true)

    # === Phase 1: WARMUP. Ramp from maxdim/2 to maxdim so mult fully grows. ===
    md_lo = max(2, div(maxdim_target, 2))
    println("\n=== WARMUP (2 sweeps, ramp $md_lo → $maxdim_target) ===")
    t_warm = @elapsed (E_warm, psi_warm) = dmrg_with_per_sweep_report(
        H_sp, psi_sp, [md_lo, maxdim_target]; label="warm")
    println("Warmup done in $(round(t_warm, digits=1))s.")

    # === Phase 2: PROFILE n_prof production-regime sweeps with timers reset. ===
    println("\n=== PROFILE ($n_prof sweeps at maxdim=$maxdim_target with reset timers) ===")
    reset_timer!(ITensorMPS.PROJMPO_TIMER)
    reset_timer!(SparseBackends.TIMER)
    t_prof = @elapsed (E_prof, psi_prof) = dmrg_with_per_sweep_report(
        H_sp, psi_warm, fill(maxdim_target, n_prof); label="prof")
    println("Profile sweeps done in $(round(t_prof, digits=1))s. E = $E_prof")

    println("\n========== ITensorMPS.PROJMPO_TIMER ==========")
    print_timer(ITensorMPS.PROJMPO_TIMER)
    println("\n========== SparseBackends.TIMER ==========")
    print_timer(SparseBackends.TIMER)
    if get(ENV, "GEMM_DIMS_HIST", "0") == "1"
        SparseBackends.show_gemm_dims_hist()
    end
end
nothing
