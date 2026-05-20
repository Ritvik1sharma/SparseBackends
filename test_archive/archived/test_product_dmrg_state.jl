using ITensors, ITensorMPS, SparseBackends, Random
using KrylovKit: eigsolve
include("utils.jl")

# ---- helpers reused from test_product.jl --------------------------------
# function align_links(t1::ITensor, t2::ITensor, label::String)
#   links1 = filter(i -> hastags(i, "Link"), inds(t1))
#   links2 = filter(i -> hastags(i, "Link"), inds(t2))
#   old_inds = Index{Int64}[]
#   new_inds = Index{Int64}[]
#   for l2 in links2
#     matches = filter(l1 -> tags(l1) == tags(l2) && dim(l1) == dim(l2) && l1 ∉ new_inds, links1)
#     if isempty(matches)
#       println("  [$label] no match for $(tags(l2)) dim=$(dim(l2))")
#       println("links for dense ", links1, " and for sparse ", links2)
#       return nothing
#     end
#     push!(old_inds, l2); push!(new_inds, first(matches))
#   end
#   return replaceinds(t2, old_inds, new_inds)
# end

function align_links(t1::ITensor, t2::ITensor, label="")
    links1 = filter(i -> hastags(i, "Link"), inds(t1))
    matched_dense = Index[]

    for l1 in links1
        links2 = filter(i -> hastags(i, "Link"), inds(t2))

        # remaining sparse links not already mapped
        remaining2 = [l2 for l2 in links2 if !(l2 in matched_dense)]

        # -----------------------------------------
        # 1) exact same dimension + tag + plev match
        # (plev is needed to disambiguate two same-id different-plev links;
        #  e.g. an L env tensor that contains both psi-link and dag(prime(psi))-link
        #  carries the SAME Index id at plev=0 and plev=1.)
        # -----------------------------------------
        exact = nothing
        for l2 in remaining2
            if tags(l1) == tags(l2) && dim(l1) == dim(l2) && plev(l1) == plev(l2)
                exact = l2
                break
            end
        end

        if exact !== nothing
            if exact != l1
                t2 = replaceinds(t2, exact => l1)
            end
            push!(matched_dense, l1)
            continue
        end

        # -----------------------------------------
        # 2) find subset whose dims multiply to dim(l1)
        # -----------------------------------------
        target = dim(l1)

        function find_subset(v, target, start=1, acc=Index[], prod_so_far=1)
            prod_so_far == target && return copy(acc)
            prod_so_far > target && return nothing

            for k in start:length(v)
                l = v[k]
                newprod = prod_so_far * dim(l)
                target % newprod == 0 || continue
                push!(acc, l)
                r = find_subset(v, target, k+1, acc, newprod)
                r !== nothing && return r
                pop!(acc)
            end
            return nothing
        end

        subset = find_subset(remaining2, target)

        if subset === nothing
            println("[$label] no match for ", l1)
            return nothing
        end

        C = combiner(subset...)
        t2 = t2 * C
        fused = commonind(t2, C)

        t2 = replaceinds(t2, fused => l1)

        push!(matched_dense, l1)
    end

    return t2
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

function compare_hphi(hphi_dense::ITensor, hphi_sparse::ITensor, b::Int, label::String)
  t1 = ITensors.has_external_storage(hphi_dense)  ? SparseBackends.to_dense_itensors(hphi_dense)  : hphi_dense
  t2 = ITensors.has_external_storage(hphi_sparse) ? SparseBackends.to_dense_itensors(hphi_sparse) : hphi_sparse
  t2_aligned = align_links(t1, t2, "$label[b=$b]")
  # println("     inds are ", inds(t1), ", ", inds(t2))
  if isnothing(t2_aligned)
    println("  bond $b ($label): align_links failed")
    println("inds for the original tensors: ")
    println("\t Dense- ", inds(hphi_dense))
    println("\t Sparse- ", inds(hphi_sparse))
    println("inds for tensors post-densification")
    println("\t Dense-", inds(t1))
    println("\t Sparse-", inds(t2))
    return
  end
  i1 = collect(inds(t1))
  a1 = Array(t1, i1...)
  a2 = Array(t2_aligned, i1...)
  d  = a1 .- a2
  println("  bond $b ($label): ‖Δ‖=", norm(d), "  max|Δ|=", maximum(abs, d))
end

# -------------------------------------------------------------------------
# Build the same system as test.jl / test_product.jl, then *manually run
# one DMRG half-step with the dense H* so psi[1], psi[2] become the post-SVD
# truncated tensors. Then run product(PH, phi) at every bond and compare
# sparse vs dense — the question is whether the bond-2 diff jumps from
# ~5.9e-16 (clean roundoff seen in test_product.jl) to ~5.8e-14 (the
# DMRG-level error seen in e3.txt).
# -------------------------------------------------------------------------
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

  H        = MPO(os, sites)
  H_sparse = sandwich_mpo(ConsOpsCombined, copy(H))
  H_dense  = sandwich_mpo_dense(ConsOpsCombined2, copy(H))

  # Same psi0 construction as test.jl
  Random.seed!(42)
  psi0 = random_mps(sites)
  for j in 1:N
    psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0)
    normalize!(psi0)
  end


  # for j in 1:length(H_dense)
  #   compare_hphi(H_dense[j], H_sparse[j], j, "H tensor")
  # end

  # ---------- (1) baseline test: original (pre-DMRG) psi --------------
  println("\n=== baseline: original psi (matches test_product.jl) ===")
  let psi_base = orthogonalize(psi0, 1)
    PH_dense  = ProjMPO(H_dense)
    PH_sparse = ProjMPO(H_sparse)
    for b in 1:length(psi_base)-1
      PH_dense  = position!(PH_dense,  psi_base, b)
      PH_sparse = position!(PH_sparse, psi_base, b)
      phi = psi_base[b] * psi_base[b+1]
      hphi_dense  = product(PH_dense,  phi)
      hphi_sparse = product(PH_sparse, phi)
      compare_hphi(hphi_dense, hphi_sparse, b, "baseline")
    end
  end

  # ---------- (2) one DMRG half-step with dense H, then re-test -------
  println("\n=== after one DMRG half-step (dense H) at bond 1: post-SVD psi ===")
  let psi = orthogonalize(psi0, 1)
    PH_init = ProjMPO(H_dense)
    PH_init = position!(PH_init, psi, 1)
    phi1    = psi[1] * psi[2]
    println("  bond 1 phi inds: ", inds(phi1))
    vals, vecs = eigsolve(PH_init, phi1, 1, :SR;
                          ishermitian=true,
                          tol=1e-14,
                          krylovdim=3,
                          maxiter=1,
                          verbosity=0)
    println("  eigsolve at bond 1 → energy = ", vals[1])

    spec = replacebond!(psi, 1, vecs[1];
                        ortho="left",
                        maxdim=81,
                        mindim=81,
                        cutoff=1e-12,
                        normalize=true)
    println("  replacebond! at bond 1 done; truncerr=", spec.truncerr,
            "  new psi[1] inds=", inds(psi[1]),
            "  new psi[2] inds=", inds(psi[2]))

    # Fresh PH on each side, positioned at every bond, identical psi
    PH_dense  = ProjMPO(H_dense)
    PH_sparse = ProjMPO(H_sparse)
    for b in 1:length(psi)-1
      PH_dense  = position!(PH_dense,  psi, b)
      PH_sparse = position!(PH_sparse, psi, b)
      phi = psi[b] * psi[b+1]
      hphi_dense  = product(PH_dense,  phi)
      hphi_sparse = product(PH_sparse, phi)
      compare_hphi(hphi_dense, hphi_sparse, b, "post-SVD")
    end
  end

  # ---------- (3) drill into bond-2 intermediates with post-SVD psi ---
  println("\n=== bond-2 intermediate breakdown (post-SVD psi) ===")
  let psi = orthogonalize(psi0, 1)
    PH_init = ProjMPO(H_dense)
    PH_init = position!(PH_init, psi, 1)
    phi1    = psi[1] * psi[2]
    vals, vecs = eigsolve(PH_init, phi1, 1, :SR;
                          ishermitian=true, tol=1e-14, krylovdim=3, maxiter=1, verbosity=0)
    replacebond!(psi, 1, vecs[1]; ortho="left", maxdim=81, mindim=81, cutoff=1e-12, normalize=true)

    PH_dense  = ProjMPO(H_dense);  PH_dense  = position!(PH_dense,  psi, 2)
    PH_sparse = ProjMPO(H_sparse); PH_sparse = position!(PH_sparse, psi, 2)

    # LR[1] — built by makeL! during position!(PH, psi, 2)
    L_dense  = PH_dense.LR[1]
    L_sparse = PH_sparse.LR[1]
    compare_hphi(L_dense,  L_sparse,  2, "LR[1]")

    # LR[3] — built during initial position!(PH, psi, 1) by makeR!
    if isassigned(PH_dense.LR, 3)
      compare_hphi(PH_dense.LR[3], PH_sparse.LR[3], 2, "LR[3]")
    end
    if isassigned(PH_dense.LR, 4)
      compare_hphi(PH_dense.LR[4], PH_sparse.LR[4], 2, "LR[4]")
    end

    println("------------------------------------")
    # Step-by-step inside product(PH, phi) at bond 2
    phi   = psi[2] * psi[3]
    L_d   = L_dense
    L_s   = L_sparse
    H2_d  = H_dense[2]
    H2_s  = H_sparse[2]
    H3_d  = H_dense[3]
    H3_s  = H_sparse[3]
    R_d   = PH_dense.LR[4]
    R_s   = PH_sparse.LR[4]

    s1_d = L_d  * H2_d
    s1_s = L_s  * H2_s
    # println("inds of the dense * dense case is ", inds(s1_d))
    # println("inds of the dense * blocksparse case is ", inds(s1_s))
    compare_hphi(s1_d, s1_s, 2, "L*H[2]")

    s2_d = s1_d * H3_d
    s2_s = s1_s * H3_s
    compare_hphi(s2_d, s2_s, 2, "L*H[2]*H[3]")

    s3_d = s2_d * R_d
    s3_s = s2_s * R_s
    compare_hphi(s3_d, s3_s, 2, "L*H[2]*H[3]*R")

    hphi_d = noprime(s3_d * phi)
    hphi_s = noprime(s3_s * phi)
    compare_hphi(hphi_d, hphi_s, 2, "full hphi (manual)")

    hphi_d2 = product(PH_dense,  phi)
    hphi_s2 = product(PH_sparse, phi)
    compare_hphi(hphi_d2, hphi_s2, 2, "full hphi (product())")
  end

  # ---------- (4) makeL! 3-step build of LR[1] — sparse vs dense -----
  # diag_a1 confirmed LR[1] has ‖Δ‖ ≈ 5.4e-14 while LR[4] is clean.
  # makeL! builds  L = H[1] * dag(prime(psi[1])) * psi[1]  in 3 sequential `*`.
  # Replicate that here with both H_dense and H_sparse so we see which step
  # first introduces the e-14, then run a "to-dense control" replacing
  # H_sparse[1] with its `to_dense_itensors` materialization.
  println("\n=== makeL! 3-step build of LR[1] (post-SVD psi[1]) ===")
  let psi = orthogonalize(psi0, 1)
    PH_init = ProjMPO(H_dense)
    PH_init = position!(PH_init, psi, 1)
    phi1    = psi[1] * psi[2]
    vals, vecs = eigsolve(PH_init, phi1, 1, :SR;
                          ishermitian=true, tol=1e-14, krylovdim=3, maxiter=1, verbosity=0)
    replacebond!(psi, 1, vecs[1]; ortho="left", maxdim=81, mindim=81, cutoff=1e-12, normalize=true)

    psi1     = psi[1]
    psi1_dag = dag(prime(psi1))

    println("\n--- variant A/B: H_sparse[1] direct contract ---")
    # step 2: L = H[1] * dag(prime(psi[1]))
    L_d_step2 = H_dense[1]  * psi1_dag
    L_s_step2 = H_sparse[1] * psi1_dag
    compare_hphi(L_d_step2, L_s_step2, 1, "step2: H * psi'_dag")

    # step 3: L = L * psi[1]
    L_d_step3 = L_d_step2 * psi1
    L_s_step3 = L_s_step2 * psi1
    compare_hphi(L_d_step3, L_s_step3, 1, "step3: (H*psi'_dag) * psi  [LR[1]]")

    println("\n--- variant C: control — H_sparse[1] materialized to dense first ---")
    # If the bug is in the BS×Dense contract kernel itself, replacing
    # H_sparse[1] with its dense materialization should make this match.
    H1_sparse_materialized = SparseBackends.to_dense_itensors(H_sparse[1])
    println("  H_sparse[1] inds:               ", inds(H_sparse[1]))
    println("  to_dense_itensors(H_sparse[1]): ", inds(H1_sparse_materialized))
    println("  has_external_storage after:     ", ITensors.has_external_storage(H1_sparse_materialized))

    # H_dense[1] vs the materialized version: should be exact
    compare_hphi(H_dense[1], H1_sparse_materialized, 1, "H_dense[1] vs materialized H_sparse[1]")

    L_c_step2 = H1_sparse_materialized * psi1_dag
    compare_hphi(L_d_step2, L_c_step2, 1, "control step2 (dense H1_materialized)")

    L_c_step3 = L_c_step2 * psi1
    compare_hphi(L_d_step3, L_c_step3, 1, "control step3 (dense H1_materialized)")

    println("\n--- variant D: associativity test ---")
    # makeL!'s order: ((H * psi'_dag) * psi).  Try the other association:
    inner_dense  = psi1_dag * psi1                # rho-like, dense × dense
    inner_sparse = psi1_dag * psi1                # identical (psi dense both runs)
    Ld_alt = H_dense[1]  * inner_dense
    Ls_alt = H_sparse[1] * inner_sparse
    compare_hphi(Ld_alt, Ls_alt, 1, "alt assoc: H * (psi'_dag * psi)")

    # And: dense LR[1] computed two ways must match each other to ~e-16
    compare_hphi(L_d_step3, Ld_alt, 1, "dense self-check: ((H*psi')*psi) vs H*(psi'*psi)")
    # Sparse LR[1] computed two ways: difference here exposes ordering-dependent
    # accumulation in the BS contract kernel.
    compare_hphi(L_s_step3, Ls_alt, 1, "sparse self-check: ((H*psi')*psi) vs H*(psi'*psi)")

    println("\n--- bonus: reverse step order  L = H * psi * psi'_dag ---")
    Ld_rev = (H_dense[1]  * psi1) * psi1_dag
    Ls_rev = (H_sparse[1] * psi1) * psi1_dag
    compare_hphi(L_d_step3, Ld_rev, 1, "dense self-check: forward vs reverse step order")
    compare_hphi(L_s_step3, Ls_rev, 1, "sparse self-check: forward vs reverse step order")
  end

  println("\nDone.")
end
