using ITensors, ITensorMPS, SparseBackends, Random, KrylovKit, LinearAlgebra
include("utils.jl")

function align_links(A::ITensors.OneITensor, B::ITensors.OneITensor, dir::String)
    align_links(ITensor(A), ITensor(B), dir)
end

# Helpers copied from test.jl
function align_links(t1::ITensor, t2::ITensor, label::String)
  links1 = filter(i -> hastags(i, "Link"), inds(t1))
  links2 = filter(i -> hastags(i, "Link"), inds(t2))
  old_inds = Index{Int64}[]
  new_inds = Index{Int64}[]
  for l2 in links2
    matches = filter(l1 -> tags(l1) == tags(l2) && dim(l1) == dim(l2) && l1 ∉ new_inds, links1)
    isempty(matches) && (println("  [$label] no match for $(tags(l2)) dim=$(dim(l2))"); return nothing)
    push!(old_inds, l2); push!(new_inds, first(matches))
  end
  return replaceinds(t2, old_inds, new_inds)
end

function multiplyVecMPOtoMPO(vec::Vector{MPO})
  result = vec[1]
  for j in 2:length(vec)
    Bp = prime(vec[j], "Site")
    result = replaceprime(contract(result, Bp, :coo, :coo), 2 => 1)
  end
  return result
end

function multiplydense(vec::Vector{MPO})
  result = vec[1]
  for j in 2:length(vec)
    Bp = prime(vec[j], "Site")
    result = replaceprime(contract(result, Bp; is_ctn_compression=true), 2 => 1)
  end
  return result
end

function sandwich_mpo(P::MPO, H::MPO)
  H1 = contract(P'', H', :coo, :dense)
  H_eff = contract(P, H1, :coo, :blocksparse)
  return replaceprime(H_eff, 3 => 1)
end

function sandwich_mpo_dense(P::MPO, H::MPO)
  H1 = contract(P'', H'; is_ctn_compression=true)
  H_eff = contract(P, H1; is_ctn_compression=true)
  return replaceprime(H_eff, 3 => 1)
end

# ── System setup (identical to test.jl) ──────────────────────────────────────
let
  N = 1
  spin_sector = 1.0
  sites = siteinds("S=1", 2*N+2)

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
    t += 0.5*spin_sector, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                          "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
    push!(os2, t)
  end

  ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1,2*j,2*j+1,2*j+2])) for j in 1:N]
  ConsOps2 = [MPO(os2[j], sites) for j in 1:N]

  ConsOpsCombined  = multiplyVecMPOtoMPO(ConsOps1)
  ConsOpsCombined2 = multiplydense(ConsOps1)

  H = MPO(os, sites)
  H_sparse = sandwich_mpo(ConsOpsCombined,  copy(H))
  H_dense  = sandwich_mpo_dense(ConsOpsCombined2, copy(H))

  # Build psi0 — same seed and projection as test.jl
  Random.seed!(42)
  psi0 = random_mps(sites)
  for j in 1:N
    psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0)
    normalize!(psi0)
  end

  # ── Hermiticity check via inner products ────────────────────────────────
  # H is Hermitian iff <psi|H|psi> ∈ ℝ and <a|H|b> = conj(<b|H|a>) for all a, b.
  # This avoids dag(MPO)/swapprime which lacks methods for the sparse storage.
  println("\n=== Hermiticity check via inner products ===")
  Random.seed!(43); psi_a = normalize!(random_mps(sites; linkdims=4))
  Random.seed!(44); psi_b = normalize!(random_mps(sites; linkdims=4))
  for (label, Hop) in (("dense ", H_dense), ("sparse", H_sparse))
    e_aa = inner(psi_a', Hop, psi_a)
    e_bb = inner(psi_b', Hop, psi_b)
    e_ab = inner(psi_a', Hop, psi_b)
    e_ba = inner(psi_b', Hop, psi_a)
    im_aa = abs(imag(e_aa)) / max(abs(real(e_aa)), 1e-300)
    im_bb = abs(imag(e_bb)) / max(abs(real(e_bb)), 1e-300)
    cross = abs(e_ab - conj(e_ba)) / max(abs(e_ab), 1e-300)
    println("  $label  <a|H|a> = $e_aa  |Im|/|Re| = $im_aa")
    println("  $label  <b|H|b> = $e_bb  |Im|/|Re| = $im_bb")
    println("  $label  |<a|H|b> - conj(<b|H|a>)| / |<a|H|b>| = $cross")
  end

  # ── Focused test: compare product(PH, phi) at every bond ─────────────────
  println("\n=== Testing product(PH, phi): dense vs sparse ===")
  println("(same phi, same psi0; only H differs in storage format)\n")

  PH_dense  = ProjMPO(H_dense)
  PH_sparse = ProjMPO(H_sparse)

  # psi_test = orthogonalize(psi0, 1)

  psi_test = random_mps(sites; linkdims=9)
  orthogonalize!(psi_test, 1)

  for b in 1:length(psi_test)-1
    PH_dense  = position!(PH_dense,  psi_test, b)
    PH_sparse = position!(PH_sparse, psi_test, b)

    phi = psi_test[b] * psi_test[b+1]

    hphi_dense  = product(PH_dense,  phi)
    hphi_sparse = product(PH_sparse, phi)

    # align link indices before comparing
    hphi_sparse_aligned = align_links(hphi_dense, hphi_sparse, "hphi[b=$b]")
    if isnothing(hphi_sparse_aligned)
      println("  bond $b: ✗ align_links failed")
      continue
    end

    diff      = hphi_dense - hphi_sparse_aligned
    diff_norm = norm(diff)
    max_diff  = maximum(abs, array(diff))

    # Ratio of max element to norm: ~1/sqrt(N) for uniform noise, >>1 means structured error
    n_elems   = prod(dims(hphi_dense))
    noise_ratio = max_diff / (diff_norm / sqrt(n_elems) + 1e-300)

    if diff_norm < 1e-12
      println("  bond $b: ✓  ‖Δ‖ = $diff_norm")
    else
      println("  bond $b: ✗  ‖Δ‖ = $diff_norm  max|Δ| = $max_diff  noise_ratio = $(round(noise_ratio; digits=1))")
      println("            (noise_ratio ≈ 1 → uniform float noise; >> 1 → structured/logical error)")

      # Show which elements of hphi_dense vs hphi_sparse_aligned are large
      arr_dense  = array(hphi_dense)
      arr_sparse = array(hphi_sparse_aligned)
      arr_diff   = arr_dense .- arr_sparse
      threshold  = max_diff * 0.1
      println("    Top differing elements (|Δ| > $(round(threshold; sigdigits=3))):")
      for I in CartesianIndices(arr_diff)
        abs(arr_diff[I]) > threshold || continue
        println("      idx=$I  dense=$(round(arr_dense[I]; sigdigits=6))  sparse=$(round(arr_sparse[I]; sigdigits=6))  Δ=$(round(arr_diff[I]; sigdigits=4))")
      end

      # Drill into which term in the contraction chain differs
      println("    Checking individual contraction terms:")
      Lp = ITensorMPS.lproj(PH_dense)
      Rp = ITensorMPS.rproj(PH_dense)
      Lp_sp = ITensorMPS.lproj(PH_sparse)
      Rp_sp = ITensorMPS.rproj(PH_sparse)

      Lp_sp_aligned = align_links(Lp, Lp_sp, "Lp[b=$b]")
      Rp_sp_aligned = align_links(Rp, Rp_sp, "Rp[b=$b]")

      if !isnothing(Lp_sp_aligned)
        Ldiff = norm(Lp - Lp_sp_aligned)
        println("    lproj  ‖Δ‖ = $Ldiff")
      end
      if !isnothing(Rp_sp_aligned)
        Rdiff = norm(Rp - Rp_sp_aligned)
        println("    rproj  ‖Δ‖ = $Rdiff")
      end
      for (i, site) in enumerate(ITensorMPS.site_range(PH_dense))
        Hd = H_dense[site]
        Hs = H_sparse[site]
        Hs_dense = ITensors.has_external_storage(Hs) ? SparseBackends.to_dense_itensors(Hs) : Hs
        Hs_aligned = align_links(Hd, Hs_dense, "H[$site]")
        if !isnothing(Hs_aligned)
          Hdiff = norm(Hd - Hs_aligned)
          println("    H[$site]   ‖Δ‖ = $Hdiff")
        end
      end
    end
  end

  # ── Mini-DMRG: run sparse and dense in parallel from identical start ──────
  # Manually do eigsolve + replacebond! at every bond and diff after each step.
  # This is the smallest test that can reproduce trajectory divergence.
  println("\n=== Mini-DMRG: parallel sparse/dense trajectories ===")
  println("(identical psi_init, eigsolve+replacebond! per bond, diff after each step)\n")

  psi_d = deepcopy(psi_test)
  psi_s = deepcopy(psi_test)

  PH_d = ProjMPO(H_dense)
  PH_s = ProjMPO(H_sparse)

  # Initial positioning (mirrors what DMRG does at startup)
  position!(PH_d, psi_d, 1)
  position!(PH_s, psi_s, 1)

  Nb = length(psi_d) - 1
  nsweeps = 3

  for sw in 1:nsweeps
    # forward sweep
    for b in 1:Nb
      PH_d = position!(PH_d, psi_d, b)
      PH_s = position!(PH_s, psi_s, b)

      # Compare environments BEFORE eigsolve (psi already diverged from prev step)
      Lp_d = ITensorMPS.lproj(PH_d); Rp_d = ITensorMPS.rproj(PH_d)
      Lp_s = ITensorMPS.lproj(PH_s); Rp_s = ITensorMPS.rproj(PH_s)
      Lp_s_a = align_links(Lp_d, Lp_s, "Lp[sw$sw,b=$b]")
      Rp_s_a = align_links(Rp_d, Rp_s, "Rp[sw$sw,b=$b]")
      L_diff = isnothing(Lp_s_a) ? NaN : norm(Lp_d - Lp_s_a)
      R_diff = isnothing(Rp_s_a) ? NaN : norm(Rp_d - Rp_s_a)

      phi_d = psi_d[b] * psi_d[b+1]
      phi_s = psi_s[b] * psi_s[b+1]

      phi_s_a = align_links(phi_d, phi_s, "phi[sw$sw,b=$b]")
      phi_diff = isnothing(phi_s_a) ? NaN : norm(phi_d - phi_s_a)

      # eigsolve directly through PH
      vals_d, vecs_d, _ = eigsolve(x -> product(PH_d, x), phi_d, 1, :SR;
                                    ishermitian=true, krylovdim=8, tol=1e-12)
      vals_s, vecs_s, _ = eigsolve(x -> product(PH_s, x), phi_s, 1, :SR;
                                    ishermitian=true, krylovdim=8, tol=1e-12)
      e_d = real(vals_d[1]); e_s = real(vals_s[1])
      psi_phi_d = vecs_d[1]
      psi_phi_s = vecs_s[1]

      # Compare eigenvectors after eigsolve
      psi_phi_s_a = align_links(psi_phi_d, psi_phi_s, "phi_new[sw$sw,b=$b]")
      eig_diff = isnothing(psi_phi_s_a) ? NaN : norm(psi_phi_d - psi_phi_s_a)

      # Replace bond (SVD update)
      spec_d = ITensorMPS.replacebond!(psi_d, b, psi_phi_d; ortho="left",
                                        normalize=true, maxdim=20, cutoff=1e-12)
      spec_s = ITensorMPS.replacebond!(psi_s, b, psi_phi_s; ortho="left",
                                        normalize=true, maxdim=20, cutoff=1e-12)

      println("  sw=$sw b=$b  E_d=$(round(e_d; digits=10))  E_s=$(round(e_s; digits=10))  " *
              "ΔE=$(round(e_d - e_s; sigdigits=3))  " *
              "‖Δlproj‖=$(round(L_diff; sigdigits=3))  ‖Δrproj‖=$(round(R_diff; sigdigits=3))  " *
              "‖Δphi_in‖=$(round(phi_diff; sigdigits=3))  ‖Δphi_out‖=$(round(eig_diff; sigdigits=3))")
    end
    # backward sweep
    for b in (Nb-1):-1:1
      PH_d = position!(PH_d, psi_d, b)
      PH_s = position!(PH_s, psi_s, b)

      phi_d = psi_d[b] * psi_d[b+1]
      phi_s = psi_s[b] * psi_s[b+1]

      vals_d, vecs_d, _ = eigsolve(x -> product(PH_d, x), phi_d, 1, :SR;
                                    ishermitian=true, krylovdim=8, tol=1e-12)
      vals_s, vecs_s, _ = eigsolve(x -> product(PH_s, x), phi_s, 1, :SR;
                                    ishermitian=true, krylovdim=8, tol=1e-12)
      e_d = real(vals_d[1]); e_s = real(vals_s[1])

      ITensorMPS.replacebond!(psi_d, b, vecs_d[1]; ortho="right",
                              normalize=true, maxdim=20, cutoff=1e-12)
      ITensorMPS.replacebond!(psi_s, b, vecs_s[1]; ortho="right",
                              normalize=true, maxdim=20, cutoff=1e-12)

      println("  sw=$sw b=$b←  E_d=$(round(e_d; digits=10))  E_s=$(round(e_s; digits=10))  " *
              "ΔE=$(round(e_d - e_s; sigdigits=3))")
    end

    # End-of-sweep summary
    energy_d = real(inner(psi_d', H_dense, psi_d))
    energy_s = real(inner(psi_s', H_sparse, psi_s))
    psi_norm_d = norm(psi_d); psi_norm_s = norm(psi_s)
    # Overlap |<psi_d|psi_s>| using site-by-site contraction with link alignment
    overlap_sq = 0.0
    try
      ovlp = 1.0 + 0.0im
      ITensorMPS.orthogonalize!(psi_d, 1)
      ITensorMPS.orthogonalize!(psi_s, 1)
      ovlp = inner(psi_d, psi_s)  # may fail if link IDs differ
      overlap_sq = abs(ovlp)
    catch err
      overlap_sq = NaN
    end
    println(">> Sweep $sw end:  E_dense=$(round(energy_d; digits=10))  " *
            "E_sparse=$(round(energy_s; digits=10))  ΔE=$(round(energy_d - energy_s; sigdigits=3))  " *
            "‖psi_d‖=$(round(psi_norm_d; digits=10))  ‖psi_s‖=$(round(psi_norm_s; digits=10))  " *
            "|<psi_d|psi_s>|=$overlap_sq\n")
  end

  println("\nDone.")
end
