# Per-channel QR factorization test.
#
# At each bond of a sparse-projected psi, we factorize phi = psi[b] * psi[b+1]
# back into L * R via a per-channel QR (one QR per value c of the shared
# sparse bond between psi[b] and psi[b+1]).
#
# What this validates:
#   1. Reconstruction:    L^c · R^c = phi^c   per channel c
#   2. Per-channel iso:   (Q^c)† Q^c = I       per channel c   (exact, by QR)
#   3. Cross-channel iso: (Q^c)† Q^c' for c ≠ c'.  EXPECTED to be:
#        - 0 at commuting bonds (single-plaquette).
#        - non-zero at non-commuting overlap bonds (where Fourier rotation alone
#          can't fix it). This tells us the "BS-structural" iso convention
#          (View B) and the "densified" iso convention (View A) diverge here.
#
# Usage:  julia --project=.. test_qr_channel.jl [N_plaq]  [linkdim]

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra
using Random
include("../utils.jl")

# Reproduce verify_iso.jl's setup. N plaquettes of (I + exp(iπSy)·exp(iπSx)·
# exp(iπSx)·exp(iπSy)) on 4 sites each, overlapping at every other site.
function build_setup(N::Int; linkdim::Int=1)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += 0.5, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]
    for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    psi0   = random_mps(sites; linkdims=linkdim)
    psi_sp = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return psi_sp
end

# The shared bond between psi[b] and psi[b+1] has TWO ITensor indices:
#   - bond_sparse: the sparse axis (carries channel labels c_M)
#   - bond_mult:   the dense multiplicity axis (within-channel index)
# We use SparseBackends' dense_inds() to tell them apart.
function shared_bonds(t1::ITensor, t2::ITensor)
    common = collect(commoninds(t1, t2))
    # If t1 has external BS storage, use its dense_inds to split.
    if ITensors.has_external_storage(t1)
        w1 = ITensors.get_external_storage(t1)
        dense = Set(SparseBackends.dense_inds(w1))
        sparse_axis = nothing
        mult_axis   = nothing
        for I in common
            if I in dense
                mult_axis = I
            else
                sparse_axis = I
            end
        end
        @assert sparse_axis !== nothing "no sparse bond found among $(common)"
        return sparse_axis, mult_axis
    else
        # No BS info; assume the larger-dim common ind is the sparse one.
        @assert length(common) >= 1
        sorted = sort(common; by=dim, rev=true)
        return sorted[1], length(sorted) > 1 ? sorted[2] : nothing
    end
end

# Slice an ITensor at one value of a given (single) sparse axis, keeping
# all other axes. Returns a dense Array in `other_inds` order.
function slice_at(t::ITensor, bond::Index, c::Int, other_inds::Vector{<:Index})
    arr = Array(t, other_inds..., bond)
    return copy(selectdim(arr, ndims(arr), c))
end

# Per-channel QR of phi at the bond between psi[b] and psi[b+1].
#
# Returns:
#   bond_dim    — dim of the shared sparse bond between psi[b] and psi[b+1].
#   per_channel — a NamedTuple with vectors of length bond_dim, one entry per c:
#                   :iso_err   — ‖Q^c† Q^c − I‖
#                   :recon_err — ‖Q^c R^c − M^c‖
#                   :chi       — kept rank for c (0 if M^c was numerically zero)
#                   :norm_Mc   — ‖M^c‖ (sanity: should be > 0 for active channels)
#   cross_err   — max ‖(Q^c)† Q^c'‖ over c ≠ c' (after padding to common chi).
#                  Tells us whether cross-channel iso holds densely.
function check_per_channel_qr(psi_b::ITensor, psi_b1::ITensor; debug::Bool=false)
    bond_sp, bond_mult = shared_bonds(psi_b, psi_b1)
    D = dim(bond_sp)
    if debug
        println("    DEBUG: bond_sp dim=$(dim(bond_sp)) tags=$(tags(bond_sp))")
        println("    DEBUG: bond_mult dim=$(bond_mult === nothing ? "(none)" : dim(bond_mult)) tags=$(bond_mult === nothing ? "" : tags(bond_mult))")
        println("    DEBUG: psi_b inds = ", [(string(tags(i)), dim(i)) for i in inds(psi_b)])
        println("    DEBUG: psi_b1 inds = ", [(string(tags(i)), dim(i)) for i in inds(psi_b1)])
    end

    # Densify BS storage so we can use ITensor ops with named indices. Use the
    # UNFUSED variant so bond_sparse and bond_mult stay as separate indices.
    pb  = ITensors.has_external_storage(psi_b)  ? SparseBackends.to_dense_itensors_unfused(psi_b)  : psi_b
    pb1 = ITensors.has_external_storage(psi_b1) ? SparseBackends.to_dense_itensors_unfused(psi_b1) : psi_b1

    # "Outside" indices (rows on left side, cols on right side): everything
    # except the two shared bond axes (bond_sp will be projected; bond_mult
    # will be contracted).
    left_inds  = filter(i -> i != bond_sp && (bond_mult === nothing || i != bond_mult), collect(inds(pb)))
    right_inds = filter(i -> i != bond_sp && (bond_mult === nothing || i != bond_mult), collect(inds(pb1)))

    T = ComplexF64
    iso_errs     = fill(NaN, D)
    recon_errs   = fill(NaN, D)
    iso_t_errs   = fill(NaN, D)
    recon_t_errs = fill(NaN, D)
    chis         = zeros(Int, D)
    chis_t       = zeros(Int, D)
    norm_Mcs     = zeros(Float64, D)
    Q_blocks     = Dict{Int, Matrix{T}}()
    Q_t_blocks   = Dict{Int, Matrix{T}}()
    Mcs          = Dict{Int, Matrix{T}}()
    n_left_g     = 0
    n_right_g    = 0

    for c in 1:D
        # Project both tensors to bond_sp=c. bond_mult remains free and gets
        # contracted in the subsequent product. This gives phi_c, the
        # contribution to phi from channel c.
        proj = onehot(bond_sp => c)
        a_it = pb  * proj          # has (left_inds..., bond_mult)
        b_it = pb1 * proj          # has (bond_mult, right_inds...)
        phi_c_it = a_it * b_it      # contracts bond_mult → (left_inds..., right_inds...)

        # Reshape to a (n_left × n_right) matrix.
        phi_c_arr = Array(phi_c_it, left_inds..., right_inds...)
        n_left  = isempty(left_inds)  ? 1 : prod(dim.(left_inds))
        n_right = isempty(right_inds) ? 1 : prod(dim.(right_inds))
        M_c = convert(Matrix{T}, reshape(phi_c_arr, n_left, n_right))
        Mcs[c]    = M_c
        n_left_g  = n_left
        n_right_g = n_right

        nM = opnorm(M_c)
        norm_Mcs[c] = nM
        if nM < 1e-14
            iso_errs[c]   = 0.0
            recon_errs[c] = 0.0
            chis[c]       = 0
            continue
        end

        # (a) FULL per-channel QR (no truncation). Sanity check: must give
        # exact iso and exact reconstruction.
        F = qr(M_c)
        chi = min(n_left, n_right)
        Q_c = Matrix(F.Q)[:, 1:chi]
        R_c = F.R[1:chi, :]
        iso_errs[c]   = opnorm(Q_c' * Q_c - I)
        recon_errs[c] = opnorm(Q_c * R_c - M_c)
        chis[c]       = chi
        Q_blocks[c]   = Q_c

        # (b) TRUNCATED via QR + SVD-on-R. Drop singular values below 1e-10·max.
        # Q_trunc = Q · U_r[:, 1:k]   (still iso: Q iso, U_r unitary)
        # R_trunc = Σ_r[1:k] · V_r†[1:k, :]
        Fr = svd(R_c)
        thr = 1e-10 * max(Fr.S[1], 1.0)
        k = count(>(thr), Fr.S)
        k = max(k, 1)
        Q_ct = Q_c * Fr.U[:, 1:k]
        R_ct = Diagonal(Fr.S[1:k]) * Fr.Vt[1:k, :]
        iso_t_errs[c]   = opnorm(Q_ct' * Q_ct - I)
        recon_t_errs[c] = opnorm(Q_ct * R_ct - M_c)
        chis_t[c]       = k
        Q_t_blocks[c]   = Q_ct

        if debug
            sv_M = svdvals(M_c)
            println("    DEBUG c=$c: n_left=$n_left n_right=$n_right ‖M_c‖=$(round(nM, sigdigits=3)) chi_full=$chi chi_trunc=$k sv(M)[1:min(end,3)]=$(round.(sv_M[1:min(end,3)], sigdigits=3)) recon_full=$(round(recon_errs[c], sigdigits=3)) recon_trunc=$(round(recon_t_errs[c], sigdigits=3)) iso_full=$(round(iso_errs[c], sigdigits=3)) iso_trunc=$(round(iso_t_errs[c], sigdigits=3))")
        end
    end

    # Cross-channel iso check on PER-CHANNEL Q's: should fail at bonds where
    # different c_M's have non-orthogonal row data (e.g. (I+C) bonds).
    cross_err = 0.0
    cs = sort!(collect(keys(Q_t_blocks)))
    for i in 1:length(cs), j in (i+1):length(cs)
        c1 = cs[i]; c2 = cs[j]
        cross = Q_t_blocks[c1]' * Q_t_blocks[c2]
        cross_err = max(cross_err, opnorm(cross))
    end

    # ── GROUPED QR ──────────────────────────────────────────────────────────
    # Stack all channels' M_c column-wise into a single (n_left × D·n_right)
    # matrix M_G, then do one QR. Q_G's columns are orthonormal globally over
    # the row space → cross-channel iso is automatic. Scatter Q_G's columns
    # back to per-channel L blocks based on column ranges.
    #
    # For this test we put ALL channels in one group. For the real BS impl,
    # grouping should follow the projector's fA/fB factor structure (channels
    # sharing row support → same group).
    iso_g_err   = NaN
    recon_g_err = NaN
    cross_g_err = NaN
    chi_g       = 0
    active_cs = [c for c in 1:D if haskey(Mcs, c)]
    if !isempty(active_cs)
        M_G = hcat([Mcs[c] for c in active_cs]...)   # n_left × (k_active · n_right)
        F_G = qr(M_G)
        chi_g = min(size(M_G)...)
        Q_G = Matrix(F_G.Q)[:, 1:chi_g]
        R_G = F_G.R[1:chi_g, :]
        iso_g_err = opnorm(Q_G' * Q_G - I)

        # Per-channel reconstruction: Q_G * R_G[:, col_range_of_c] should = M_c
        recon_g = 0.0
        Q_g_blocks = Dict{Int, Matrix{T}}()
        for (idx, c) in enumerate(active_cs)
            c_lo = (idx - 1) * n_right_g + 1
            c_hi =  idx      * n_right_g
            R_c_in_G = R_G[:, c_lo:c_hi]
            M_c_recon = Q_G * R_c_in_G
            recon_g = max(recon_g, opnorm(M_c_recon - Mcs[c]))
            # For the grouped path, the L tensor's c-th channel block is
            # Q_G (the SAME matrix for all c — the bond label only distinguishes
            # which R block sits next to it). We still record per-channel Q
            # for cross-check below.
            Q_g_blocks[c] = Q_G
        end
        recon_g_err = recon_g

        # In grouped QR, Q_G is shared across channels, so "cross-channel iso"
        # in the densified sense is just Q_G' Q_G = I (already checked above).
        # We compute it explicitly here too as a paranoia check.
        cross_g_max = 0.0
        for i in 1:length(active_cs), j in (i+1):length(active_cs)
            cross_g_max = max(cross_g_max, opnorm(Q_g_blocks[active_cs[i]]' * Q_g_blocks[active_cs[j]] - I))
        end
        cross_g_err = cross_g_max
    end

    return (
        bond_dim     = D,
        iso_errs     = iso_errs,
        recon_errs   = recon_errs,
        iso_t_errs   = iso_t_errs,
        recon_t_errs = recon_t_errs,
        iso_g_err    = iso_g_err,
        recon_g_err  = recon_g_err,
        chi_g        = chi_g,
        chis         = chis,
        chis_t       = chis_t,
        norm_Mcs    = norm_Mcs,
        cross_err   = cross_err,
        active_chans = count(>(0), chis),
    )
end

# Classify a bond by reading the actual BlockSparse keys of psi[b] and psi[b+1].
# Returns the (n_L, n_M, n_R) pattern plus fan-out structure that determines
# which factorization algorithm applies:
#   - max_cMs_per_cL  == 1  → "clean" per-c_M QR works (with row-stacking if max_cLs_per_cM > 1)
#   - max_cMs_per_cL  > 1   → (2,4,2)-pattern, needs Gram-Schmidt within c_L group
function classify_bond(psi_b::ITensor, psi_b1::ITensor)
    @assert ITensors.has_external_storage(psi_b)
    @assert ITensors.has_external_storage(psi_b1)
    w_b  = ITensors.get_external_storage(psi_b)
    w_b1 = ITensors.get_external_storage(psi_b1)
    bond_sp, _ = shared_bonds(psi_b, psi_b1)

    dense_b  = SparseBackends.dense_inds(w_b)
    dense_b1 = SparseBackends.dense_inds(w_b1)
    M_b_inds  = collect(w_b.inds)
    M_b1_inds = collect(w_b1.inds)
    M_b_sparse_pos  = [i for i in 1:length(M_b_inds)  if !(M_b_inds[i]  in dense_b)]
    M_b1_sparse_pos = [i for i in 1:length(M_b1_inds) if !(M_b1_inds[i] in dense_b1)]
    bond_pos_in_b  = findfirst(p -> M_b_inds[p]  == bond_sp, M_b_sparse_pos)
    bond_pos_in_b1 = findfirst(p -> M_b1_inds[p] == bond_sp, M_b1_sparse_pos)

    cL_to_cMs = Dict{Tuple, Set{Int}}()
    cM_to_cLs = Dict{Int, Set{Tuple}}()
    for key in w_b.blocksparse.keys
        vals = [Int(key[p]) for p in M_b_sparse_pos]
        cM = vals[bond_pos_in_b]
        lk = Tuple([vals[i] for i in 1:length(vals) if i != bond_pos_in_b])
        push!(get!(cL_to_cMs, lk, Set{Int}()), cM)
        push!(get!(cM_to_cLs, cM, Set{Tuple}()), lk)
    end
    cM_to_cRs = Dict{Int, Set{Tuple}}()
    cR_to_cMs = Dict{Tuple, Set{Int}}()
    for key in w_b1.blocksparse.keys
        vals = [Int(key[p]) for p in M_b1_sparse_pos]
        cM = vals[bond_pos_in_b1]
        rk = Tuple([vals[i] for i in 1:length(vals) if i != bond_pos_in_b1])
        push!(get!(cM_to_cRs, cM, Set{Tuple}()), rk)
        push!(get!(cR_to_cMs, rk, Set{Int}()), cM)
    end

    n_L = length(cL_to_cMs)
    n_M = dim(bond_sp)
    n_R = length(cR_to_cMs)
    max_cMs_per_cL = isempty(cL_to_cMs) ? 0 : maximum(length, values(cL_to_cMs))
    max_cLs_per_cM = isempty(cM_to_cLs) ? 0 : maximum(length, values(cM_to_cLs))
    max_cMs_per_cR = isempty(cR_to_cMs) ? 0 : maximum(length, values(cR_to_cMs))
    max_cRs_per_cM = isempty(cM_to_cRs) ? 0 : maximum(length, values(cM_to_cRs))

    case = if max_cMs_per_cL > 1
        "(n,$(n_M),$(n_R))-needs-GS"
    elseif max_cLs_per_cM > 1
        "($(n_L),$(n_M),$(n_R))-row-stack"
    else
        "($(n_L),$(n_M),$(n_R))-clean"
    end

    return (
        n_L = n_L, n_M = n_M, n_R = n_R,
        max_cMs_per_cL = max_cMs_per_cL,
        max_cLs_per_cM = max_cLs_per_cM,
        max_cMs_per_cR = max_cMs_per_cR,
        max_cRs_per_cM = max_cRs_per_cM,
        case = case,
    )
end

# Existing-SVD comparison: factorize via the channel_aware SVD wrapper and
# compute its densified iso. We have to find the new bond by inspecting the
# common ind between L and R after un-fused densification.
function check_existing_svd(psi_b::ITensor, psi_b1::ITensor)
    L, R, spec = SparseBackends.itensor_blocksparse_svd_channel_aware(
        psi_b * psi_b1, psi_b, psi_b1;
        ortho="left", maxdim=typemax(Int), mindim=1, cutoff=0.0,
        relax_iso_cap=false,
    )
    # Densify L for the iso check.
    L_d = ITensors.has_external_storage(L) ? SparseBackends.to_dense_itensors_unfused(L) : L
    common = collect(commoninds(L, R))   # the new bond — could be 1 or 2 inds (sparse+mult)
    other = filter(i -> !(i in common), collect(inds(L_d)))
    # Build dense matrix L_mat with rows = "other" inds, cols = "common" (new bond) inds.
    L_arr = Array(L_d, other..., common...)
    n_bond = isempty(common) ? 1 : prod(dim.(common))
    n_rows = div(length(L_arr), n_bond)
    L_mat = reshape(L_arr, n_rows, n_bond)
    diff = L_mat' * L_mat - I
    return opnorm(diff), n_bond
end

# Mark whether a bond is single- or double-plaquette in the verify_iso.jl
# setup (sites 1..2N+2, plaquettes at sites [2j-1..2j+2] for j=1..N).
function bond_overlap_label(b::Int, N_plaq::Int)
    n_plaq = 0
    for j in 1:N_plaq
        lo, hi = 2*j-1, 2*j+2
        if lo <= b && b+1 <= hi
            n_plaq += 1
        end
    end
    return n_plaq == 1 ? "single" : (n_plaq == 2 ? "double" : "other($n_plaq)")
end

let
    N_plaq  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
    linkdim = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1

    println("=== Per-channel QR test ===")
    println("  N_plaq=$N_plaq  linkdim=$linkdim")
    psi = build_setup(N_plaq; linkdim=linkdim)
    println("  $(length(psi)) sites; orthogonalizing to bond 1 …")
    psi = ITensorMPS.orthogonalize!(psi, 1)

    println("\n", repeat("-", 130))
    println("bond | type   | dim(act) | per-ch iso | per-ch recon | per-ch cross | GROUPED iso | GROUPED recon | EXISTING-SVD iso")
    println(repeat("-", 130))

    bad_iso = bad_recon = bad_cross = bad_g_iso = bad_g_recon = bad_g_cross = 0
    for b in 1:length(psi)-1
        local tag = bond_overlap_label(b, N_plaq)
        try
            r = check_per_channel_qr(psi[b], psi[b+1]; debug = (b == 1))
            max_iso     = maximum(filter(!isnan, r.iso_t_errs);   init=0.0)
            max_recon   = maximum(filter(!isnan, r.recon_t_errs); init=0.0)
            iso_ok      = max_iso       < 1e-8
            recon_ok    = max_recon     < 1e-8
            cross_ok    = r.cross_err   < 1e-8
            g_iso_ok    = !isnan(r.iso_g_err)   && r.iso_g_err   < 1e-8
            g_recon_ok  = !isnan(r.recon_g_err) && r.recon_g_err < 1e-8
            g_cross_ok  = !isnan(r.iso_g_err)   && r.iso_g_err   < 1e-8
            iso_ok     || (bad_iso     += 1)
            recon_ok   || (bad_recon   += 1)
            cross_ok   || (bad_cross   += 1)
            g_iso_ok   || (bad_g_iso   += 1)
            g_recon_ok || (bad_g_recon += 1)
            g_cross_ok || (bad_g_cross += 1)
            # Existing-SVD comparison
            svd_iso = NaN
            try
                svd_iso, _ = check_existing_svd(psi[b], psi[b+1])
            catch e
                svd_iso = NaN
            end
            println(
                lpad(b, 4), " | ", rpad(tag, 6), " | ",
                lpad("$(r.bond_dim)($(r.active_chans))", 8), " | ",
                lpad(round(max_iso;       sigdigits=3), 10), " | ",
                lpad(round(max_recon;     sigdigits=3), 12), " | ",
                lpad(round(r.cross_err;   sigdigits=3), 12), " | ",
                lpad(round(r.iso_g_err;   sigdigits=3), 11), " | ",
                lpad(round(r.recon_g_err; sigdigits=3), 13), " | ",
                lpad(round(svd_iso;       sigdigits=3), 16),
            )
        catch e
            println(lpad(b, 4), " | $tag | FAILED: ", sprint(showerror, e))
        end
    end

    println(repeat("-", 120))
    println("Per-channel: bad_iso=$bad_iso  bad_recon=$bad_recon  bad_cross=$bad_cross")
    println("Grouped:     bad_iso=$bad_g_iso  bad_recon=$bad_g_recon  bad_cross=$bad_g_cross")
    println()
    println("Notes:")
    println("  - per-channel iso ≈ machine eps is the expected behavior (QR property).")
    println("  - per-channel recon ≈ machine eps confirms L·R = phi block-by-block.")
    println("  - cross_err > 0 at 'double' bonds is EXPECTED — non-commuting C's")
    println("    can't be diagonalized simultaneously. View-B iso (sparse-axis-")
    println("    structural orthogonality) still holds; densified iso check fails.")
end
nothing
