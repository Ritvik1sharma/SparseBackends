# test_check_working_aliased.jl
#
# Variant of ../test_check_working.jl that stores the projected Hamiltonian
# (H_new in the original) using `AliasedBlockSparse` instead of
# `NewBlockSparseSorted`. Both intermediates of `sandwich_mpo` are routed
# through `contract_aliased_itensor` when `output_hint = :aliasedblocksparse`.
#
# Original behaviour is preserved when no output_hint is supplied.

using SparseBackends, Random
using ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
println("[BLAS threads pinned to ", BLAS.get_num_threads(), "]")

const _DIAG = get(ENV, "DMRG_DIAG", "0") == "1"


function run_dmrg_with_timers(label, H, psi0; kwargs...)
  do_warmup = get(ENV, "DMRG_NO_WARMUP", "0") != "1"
  if do_warmup
    println("[warmup pass: 1 JIT sweep, results discarded]")
    kw = NamedTuple(kwargs)
    md = haskey(kw, :maxdim) ? kw.maxdim : [20]
    mn = haskey(kw, :mindim) ? kw.mindim : md
    co = haskey(kw, :cutoff) ? kw.cutoff : 1e-10
    try
      dmrg(H, deepcopy(psi0);
           nsweeps = 1, maxdim = md, mindim = mn, cutoff = co,
           outputlevel = 0, use_early_exit = false)
    catch e
      println("  (warmup failed: ", sprint(showerror, e), " — continuing without warmup)")
    end
  end
  reset_timer!(SparseBackends.TIMER)
  reset_timer!(ITensorMPS.PROJMPO_TIMER)
  GC.gc(); GC.gc()

  wall = @elapsed result = dmrg(H, psi0; kwargs...)
  global LAST_DMRG_WALL = wall
  println("\n========== TIMER REPORT: $label  (wall = $(round(wall; digits=3)) s, JIT excluded) ==========")
  println("\n--- ProjMPO matvec breakdown ---")
  print_timer(ITensorMPS.PROJMPO_TIMER; sortby=:firstexec)
  println("\n--- SparseBackends contract dispatch breakdown ---")
  print_timer(SparseBackends.TIMER; sortby=:firstexec)
  println("==========\n")
  return result
end

LAST_DMRG_WALL = 0.0

include("../test_sparse_psi/utils.jl")

function calcInner(Oper::Vector{MPO}, state::MPS)
  diff = 0
  for (i, P) in enumerate(Oper)
    Pψ = apply(P, state)
    norm_Pψ = norm(Pψ)
    if isapprox(norm_Pψ, 0.0; atol=1e-12)
        println("⟨ψ|P|ψ⟩ [i=$i]: norm ≈ 0 → skipping normalization")
        continue
    end
    Pψ_norm = replace_siteinds(Pψ / norm_Pψ, siteinds(state))
    overlap = inner(state, Pψ_norm)
    diff += 1 - overlap
  end
  println("Overall error is $diff")
  return diff
end

function reindex_mpo_siteinds(mpo::MPO, index_map::Vector{Pair{Index{Int64}, Index{Int64}}})
  new_mpo = MPO(length(mpo))
  for i in 1:length(mpo)
    new_mpo[i] = replaceinds(mpo[i], index_map)
  end
  return new_mpo
end

# ─────────────────────────────────────────────────────────────────────────────
# sandwich_mpo — projected Hamiltonian P H P.
#
# `output_hint` controls the storage of intermediate / output tensors:
#   :default              — original behaviour (NewBlockSparseSorted via
#                           SparseBackends.contract with :coo / :blocksparse).
#   :aliasedblocksparse   — both contractions go through
#                           SparseBackends.contract_aliased_itensor so the
#                           result is backed by AliasedBlockSparse.
#   :dense                — plain ITensors.contract (no sparse storage).
# ─────────────────────────────────────────────────────────────────────────────
function sandwich_mpo(P::MPO, H::MPO; output_hint::Symbol = :default)
  if output_hint === :aliasedblocksparse
    new_H = MPO(length(H))
    for i in 1:length(H)
      H1 = SparseBackends.contract_aliased_itensor(P[i]'', H[i]', :coo, :dense)
      H_eff_i = SparseBackends.contract_aliased_itensor(P[i], H1, :coo, :aliased)
      new_H[i] = replaceprime(H_eff_i, 3 => 1)
    end
    return new_H
  elseif output_hint === :dense
    H1 = contract(P'', H')
    H_eff = contract(P, H1)
    return replaceprime(H_eff, 3 => 1)
  else
    # :default — original behaviour (NewBlockSparseSorted)
    H1 = contract(P'', H', :coo, :dense)
    H_eff = contract(P, H1, :coo, :blocksparse)
    return replaceprime(H_eff, 3 => 1)
  end
end

function sandwich_mpo_dense(P::MPO, H::MPO)
  H1 = contract(P'', H'; is_ctn_compression=true)
  H_eff = contract(P, H1; is_ctn_compression=true)
  H_eff = replaceprime(H_eff, 3 => 1)
  return H_eff
end

# ──────────────────────────────────────────────────────────────────────────────
# Fuse multi-strand sparse links in a PHP-style MPO.
#
# After P·H·P each bond between sites k and (k+1) carries multiple Link,l=k
# "strands" (one per projector layer).  These are the sparse-prefix indices
# that aren't sites.  Each strand has matching IDs on H[k] and H[k+1], so we
# can apply the SAME combiner to both sides to fuse them into a single Index.
#
# Sites (e.g. "S=1,Site,n=k") and dense-tail indices are left untouched.
# Aliased storage is preserved because the contraction Aliased × Dense → Aliased
# is the existing kernel (preserve_bs_output=true).
# ──────────────────────────────────────────────────────────────────────────────
# Return the set of sparse-prefix indices of an aliased H[k] (first PA inds).
function sparse_prefix_inds(T::ITensor)
    if ITensors.has_external_storage(T) && T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse
        w = T.tensor.data
        PA = SparseBackends._abs_head_len(w)
        return Set(w.inds[1:PA])
    end
    return Set(inds(T))
end

# Materialise a combiner ITensor to a fully-stored dense ITensor (no NoData).
# combiner(a, b, ...) is a sparse "virtual" tensor in ITensors; we rebuild it
# as a plain dense array with 1.0 at column-major linearised positions.
function _materialize_combiner_dense(strands::Vector{<:Index}, fused::Index)
    dims_in = Tuple(ITensors.dim(i) for i in strands)
    Nin     = length(strands)
    dout    = ITensors.dim(fused)
    @assert prod(dims_in) == dout
    data = zeros(ComplexF64, dims_in..., dout)
    # Walk all multi-indices in column-major order over dims_in.
    strides = ones(Int, Nin)
    for d in 2:Nin; strides[d] = strides[d-1] * dims_in[d-1]; end
    for cart in CartesianIndices(dims_in)
        lin = 1
        for d in 1:Nin
            lin += (cart[d] - 1) * strides[d]
        end
        data[cart, lin] = 1.0 + 0.0im
    end
    return ITensors.ITensor(data, strands..., fused)
end

function fuse_sparse_links!(H::MPO)
    L = length(H)
    for k in 1:(L-1)
        common = commoninds(H[k], H[k+1])
        # Only sparse-prefix indices on BOTH sides are candidates for fusion.
        sp_k   = sparse_prefix_inds(H[k])
        sp_k1  = sparse_prefix_inds(H[k+1])
        # Group by tag-string, restricted to sparse-prefix on both sides.
        by_tag = Dict{String, Vector{Index}}()
        for I in common
            (I in sp_k) && (I in sp_k1) || continue
            t = string(tags(I))
            push!(get!(by_tag, t, Index[]), I)
        end
        for (t, strand_list) in by_tag
            length(strand_list) <= 1 && continue
            # Mark the fused index with "FusedSparse" so _makeL!/_makeR! can
            # identify it later and arrange it last in the env layout.
            cmb_raw = combiner(strand_list...; tags="Link,FusedSparse,bond=$(k)")
            fused   = ITensors.combinedind(cmb_raw)
            cmb     = _materialize_combiner_dense(strand_list, fused)
            H[k]   = SparseBackends.contract_aliased_itensor(
                        H[k],   cmb,
                        ITensors.has_external_storage(H[k])   ? :aliased : :dense,
                        :dense; preserve_bs_output=true)
            H[k+1] = SparseBackends.contract_aliased_itensor(
                        H[k+1], cmb,
                        ITensors.has_external_storage(H[k+1]) ? :aliased : :dense,
                        :dense; preserve_bs_output=true)
        end
    end
    return H
end

function mulMPO(A::MPO, B::MPO; is_ctn_compression=false, cutoff=0.0, maxdim=typemax(Int), debug=false)
  Bp = prime(B, "Site")
  C = contract(A, Bp, :coo, :coo)
  return replaceprime(C, 2 => 1)
end

function multiplyVecMPOtoMPO(vec::Vector{MPO}; is_ctn_compression=false, cutoff=0.0, maxdim=typemax(Int))
  result = vec[1]
  for j in 2:length(vec)
    result = mulMPO(result, vec[j]; is_ctn_compression=is_ctn_compression, cutoff=cutoff, maxdim=maxdim)
  end
  return result
end

function multiplydense(vec::Vector{MPO}; is_ctn_compression=false, cutoff=0.0, maxdim=typemax(Int))
  result = vec[1]
  for j in 2:length(vec)
    Bp = prime(vec[j], "Site")
    result2 = contract(result, Bp; is_ctn_compression=true)
    result = result2
    result = replaceprime(result, 2 => 1)
  end
  return result
end


function _maxabs(T::ITensor)
  is = inds(T)
  if isempty(is)
      return abs(scalar(T))
  else
      return maximum(abs, Array(T, is...))
  end
end

maxabs(H::MPO) = maximum(_maxabs, H)

function clean!(op::MPO; tol=1e-12)
  for j in 1:length(op)
      T = op[j]
      A = array(T)
      for i in eachindex(A)
          if abs(A[i]) < tol
              A[i] = 0.0
          elseif abs(A[i] - 1.0) < tol
              A[i] = 1.0
          elseif abs(A[i] + 1.0) < tol
              A[i] = -1.0
          elseif abs(A[i] - 0.5) < tol
              A[i] = 0.5
          elseif abs(A[i] + 0.5) < tol
              A[i] = -0.5
          elseif abs(A[i] - 1.0im) < tol
              A[i] = 1.0im
          elseif abs(A[i] + 1.0im) < tol
              A[i] = -1.0im
          elseif abs(A[i] - 0.5im) < tol
              A[i] = 0.5im
          elseif abs(A[i] + 0.5im) < tol
              A[i] = -0.5im
          elseif abs(A[i] + 2.0) < tol
              A[i] = -2.0
          elseif abs(A[i] - 2.0) < tol
              A[i] = 2.0
          end
      end
      op[j] = ITensor(A, inds(T)...)
  end
  return op
end


function mpo_memory_bytes(H::MPO)
  total_bytes = 0
  for (i, W) in enumerate(H)
      total_bytes += Base.summarysize(W)
  end
  println("Total MPO memory: $(total_bytes/1e6) MB")
  return total_bytes
end


function align_links(t1::ITensor, t2::ITensor, label::String; debug=false)
  links1 = filter(i -> hastags(i, "Link"), inds(t1))
  links2 = filter(i -> hastags(i, "Link"), inds(t2))
  old_inds = Index{Int64}[]
  new_inds = Index{Int64}[]
  for l2 in links2
    base_tag = tags(l2)
    matches = filter(l1 -> tags(l1) == base_tag && dim(l1) == dim(l2) && l1 ∉ new_inds, links1)
    if isempty(matches)
      println("  [$label] no matching link for tag $base_tag (dim=$(dim(l2)))")
      return nothing
    end
    l1 = first(matches)
    push!(old_inds, l2)
    push!(new_inds, l1)
  end
  return replaceinds(t2, old_inds, new_inds)
end



length(ARGS) < 1 && error("Usage: julia test_check_working_aliased.jl <N_plaq>")

# Hard gate: this test exercises the experimental AliasedBlockSparse code path.
# It is OFF by default to avoid interfering with any other workflows that
# happen to import SparseBackends.  Opt in explicitly:
#
#   SB_ALIASED_ENABLE=1 julia --project=. test_check_working_aliased.jl <N>
#
# When the gate is off the script exits with a no-op message.
const _ALIASED_ENABLE = get(ENV, "SB_ALIASED_ENABLE", "0") == "1"
if !_ALIASED_ENABLE
    println("[SB_ALIASED_ENABLE != 1] Aliased path is gated off — set SB_ALIASED_ENABLE=1 to run.")
    exit(0)
end

let
    spin = parse(Int, get(ENV, "BENCH_SPIN", "3"))
    spin_sector = 1.0

    N = parse(Int, ARGS[1])

    states = 2*N+2
    if spin == 2
        sites = siteinds("S=1/2", states)
    elseif spin == 3
        sites = siteinds("S=1", states)
    else
        error("Not supported spin case")
    end
    os = OpSum()
    os_reg = OpSum()
    for j in 1:N+1
        os += "Sz", 2*j-1, "Sz", 2*j
        os_reg += "Sz", 2*j - 1, "Sz", 2*j
    end
    for j in 1:N
        os += "Sx", 2*j - 1, "Sx", 2*j + 2
        os += "Sy", 2*j, "Sy", 2*j + 1

        os_reg += "Sx", 2*j - 1, "Sx", 2*j + 2
        os_reg += "Sy", 2*j, "Sy", 2*j + 1
    end
    os2 = OpSum[]
    os3 = OpSum[]

    for j in 1:N
        coeff = 0.5
        temp = OpSum()
        temp += coeff, "Id", 2*j - 1, "Id", 2*j, "Id", 2*j + 1, "Id", 2*j + 2
        temp += spin_sector*coeff, "exp(i*pi*Sy)", 2*j - 1, "exp(i*pi*Sx)", 2*j, "exp(i*pi*Sx)", 2*j + 1, "exp(i*pi*Sy)", 2*j + 2
        push!(os2, temp)
        temp = OpSum()
        temp += spin_sector, "exp(i*pi*Sy)", 2*j - 1, "exp(i*pi*Sx)", 2*j, "exp(i*pi*Sx)", 2*j + 1, "exp(i*pi*Sy)", 2*j + 2
        push!(os3, temp)
    end

    cutoff = 10.0^(-12)
    ConsOps1 = MPO[]
    ConsOps2 = MPO[]

    for j in 1:N
        operator = MPO(os2[j], sites, [2*j - 1, 2*j, 2*j + 1, 2*j + 2])
        clean!(operator; tol=1e-12)
        push!(ConsOps1, operator)
        operator = MPO(os2[j], sites)
        push!(ConsOps2, operator)
    end


    ConsOpsCombined = multiplyVecMPOtoMPO(ConsOps1)
    ConsOpsCombined2 = multiplydense(ConsOps1)

    H = MPO(os, sites)

    # ── Aliased path ─────────────────────────────────────────────────────────
    println("\n[building H_new with output_hint = :aliasedblocksparse]")
    H_new_aliased = sandwich_mpo(ConsOpsCombined, copy(H); output_hint = :aliasedblocksparse)

    # Fuse multi-strand sparse links so each bond carries ONE sparse link
    # instead of multiple strands. Gate behind SB_FUSE_LINKS env var so we can
    # A/B test perf.
    if get(ENV, "SB_FUSE_LINKS", "0") == "1"
        println("\n[fusing multi-strand sparse links in H_new_aliased]")
        # Print pre-fuse axis counts
        println("  pre-fuse  H[3] inds: ", inds(H_new_aliased[3]))
        fuse_sparse_links!(H_new_aliased)
        println("  post-fuse H[3] inds: ", inds(H_new_aliased[3]))
    end

    # ── Reference dense path ─────────────────────────────────────────────────
    H_new2 = sandwich_mpo_dense(ConsOpsCombined2, copy(H))

    println("\n── Memory footprint of PHP ──")
    print("ALIASED  PHP:  "); mpo_memory_bytes(H_new_aliased)
    print("DENSE    PHP:  "); mpo_memory_bytes(H_new2)
    # Per-site aliasing stats — three sizes side-by-side:
    #   dense_size       = prod(dims)            (no compression)
    #   bs_size          = n_blocks × blksize    (block sparsity only)
    #   aliased_size     = n_templates × blksize + n_blocks  (alias dedup)
    # Plus the two ratios that matter:
    #   vs-dense : aliased's win vs full dense storage (the "real" win)
    #   vs-BS    : aliased's extra win on top of block sparsity (alias dedup factor)
    println("  ", rpad("site", 5),
                rpad("nb",  6), rpad("ntmpl", 7), rpad("blksize", 9),
                rpad("dense", 10), rpad("BS", 10), rpad("aliased", 10),
                rpad("vs-dense", 10), "vs-BS")
    tot_dense = 0; tot_bs = 0; tot_ali = 0
    for i in 1:length(H_new_aliased)
        if ITensors.has_external_storage(H_new_aliased[i])
            es = H_new_aliased[i].tensor
            if es.data isa SparseBackends.WrappedAliasedBlockSparse
                ali = es.data.aliased
                nb       = length(ali.keys)
                nt       = ali.n_templates
                bksz     = ali.blksize
                dense_sz = prod(ali.dims)
                bs_sz    = nb * bksz
                ali_sz   = nt * bksz + nb
                vd       = dense_sz / max(ali_sz, 1)
                vb       = bs_sz    / max(ali_sz, 1)
                tot_dense += dense_sz; tot_bs += bs_sz; tot_ali += ali_sz
                println("  ", rpad(string(i), 5),
                          rpad(string(nb),     6),
                          rpad(string(nt),     7),
                          rpad(string(bksz),   9),
                          rpad(string(dense_sz), 10),
                          rpad(string(bs_sz),    10),
                          rpad(string(ali_sz),   10),
                          rpad(string(round(vd; digits=2)), 10),
                          round(vb; digits=2))
            end
        end
    end
    println("  ", rpad("TOT",   5),
              rpad("",        6), rpad("",       7), rpad("",       9),
              rpad(string(tot_dense), 10),
              rpad(string(tot_bs),    10),
              rpad(string(tot_ali),   10),
              rpad(string(round(tot_dense/max(tot_ali,1); digits=2)), 10),
              round(tot_bs/max(tot_ali,1); digits=2))

    # ── Pointwise comparison (sanity) ────────────────────────────────────────
    for i in 1:length(H_new_aliased)
      t1 = ITensors.has_external_storage(H_new_aliased[i]) ?
            SparseBackends.to_dense_itensors(H_new_aliased[i]) : H_new_aliased[i]
      t2 = ITensors.has_external_storage(H_new2[i]) ?
            SparseBackends.to_dense_itensors(H_new2[i]) : H_new2[i]
      t2_aligned = align_links(t1, t2, "H[$i]")
      if !isnothing(t2_aligned)
        if isapprox(t1, t2_aligned)
          println("  H[$i] ✓ match (aliased vs dense)")
        else
          println("  H[$i] ✗ values differ")
        end
      end
    end

    # ── DMRG ─────────────────────────────────────────────────────────────────
    Random.seed!(42)
    psi_old = random_mps(sites)

    psi0 = copy(psi_old)
    for j in 1:N
      psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0)
      normalize!(psi0)
    end

    psi1, psi2 = copy(psi0), copy(psi0)

    _md_default = _DIAG ? 20 : 40
    _ns_default = _DIAG ? 3 : 25
    _md = parse(Int, get(ENV, "BENCH_MAXDIM", string(_md_default)))
    _ns = parse(Int, get(ENV, "BENCH_NSWEEPS", string(_ns_default)))
    maxdim = [_md]; mindim = [_md]
    nsweeps = _ns
    target_energy = nothing
    last_sweep_energy = nothing

    # ── DENSE reference run ──────────────────────────────────────────────────
    println("\n[diag=$_DIAG] Running DENSE DMRG (H_new2) with nsweeps=$nsweeps, maxdim=$_md ...")
    ENV["SB_RUN_LABEL"] = "DENSE"
    t_dense = @elapsed begin
        energy_d, psi_dense, sweeps_d, t_err_d = run_dmrg_with_timers("DENSE H_new2",
            H_new2, psi1; nsweeps, maxdim, mindim, cutoff, target_energy,
            use_early_exit=false, last_sweep_energy=last_sweep_energy,
            outputlevel=1)
    end
    E_0 = inner(copy(psi0)', H, copy(psi0))
    E_1_dense = inner(psi_dense', H, psi_dense)
    println("\n\t Energy at start ", E_0, " and at end ", E_1_dense,
            " in sweeps ", sweeps_d, " and truncation error ", t_err_d)
    println("[DENSE]   Energy: $E_1_dense in sweeps $sweeps_d and terr $t_err_d and total time $t_dense seconds")
    dense_wall = t_dense
    dense_wall_jit_excluded = LAST_DMRG_WALL

    # ── ALIASED run ──────────────────────────────────────────────────────────
    println("\n[diag=$_DIAG] Running ALIASED DMRG (H_new_aliased) with nsweeps=$nsweeps, maxdim=$_md ...")
    ENV["SB_RUN_LABEL"] = "ALIASED"
    t_ali = @elapsed begin
        energy_a, psi_ali, sweeps_a, t_err_a = run_dmrg_with_timers("ALIASED H_new_aliased",
            H_new_aliased, psi2; nsweeps, maxdim, mindim, cutoff, target_energy,
            use_early_exit=false, last_sweep_energy=last_sweep_energy,
            outputlevel=1)
    end
    E_1_ali = inner(psi_ali', H, psi_ali)
    println("\n\t Energy at start ", E_0, " and at end ", E_1_ali,
            " in sweeps ", sweeps_a, " and truncation error ", t_err_a)
    println("[ALIASED] Energy: $E_1_ali in sweeps $sweeps_a and terr $t_err_a and total time $t_ali seconds")
    aliased_wall = t_ali
    aliased_wall_jit_excluded = LAST_DMRG_WALL

    println("\n========== HEAD-TO-HEAD WALL TIME ==========")
    println("  Includes-warmup totals (for reference):")
    println("    DENSE   total: $(round(dense_wall;   digits=3)) s")
    println("    ALIASED total: $(round(aliased_wall; digits=3)) s")
    println("    ratio aliased/dense = $(round(aliased_wall/dense_wall; digits=3))")
    println("\n  JIT-EXCLUDED (post-warmup) totals — the fair comparison:")
    println("    DENSE   measured: $(round(dense_wall_jit_excluded;   digits=3)) s")
    println("    ALIASED measured: $(round(aliased_wall_jit_excluded; digits=3)) s")
    println("    ratio aliased/dense = $(round(aliased_wall_jit_excluded/dense_wall_jit_excluded; digits=3))  (want < 1)")
    println("\n  Energy comparison:")
    println("    DENSE   final: $E_1_dense")
    println("    ALIASED final: $E_1_ali")
    println("    |ΔE|         : $(abs(E_1_ali - E_1_dense))")
end
nothing
