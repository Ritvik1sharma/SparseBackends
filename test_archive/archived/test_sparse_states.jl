# using CSV, DataFrames
using SparseBackends
using ITensors, ITensorMPS

# const SBX = let m = Base.get_extension(SparseBackends, :SparseBackendsITensorsExt)
#   m === nothing && error("SparseBackendsITensorsExt did not load. Did you `using ITensors` and set up [extensions]/[weakdeps]?")
#   m
# end

function calcInner(Oper::Vector{MPO}, state::MPS)
  diff = 0
  for (i, P) in enumerate(Oper)
    # Apply P to state (this handles primes safely)
    Pψ = apply(P, state)
    norm_Pψ = norm(Pψ)
    if isapprox(norm_Pψ, 0.0; atol=1e-12)
        println("⟨ψ|P|ψ⟩ [i=$i]: norm ≈ 0 → skipping normalization")
        continue
    end
    # Normalize the projected state
    Pψ_norm = replace_siteinds(Pψ / norm_Pψ, siteinds(state))
    # Overlap ⟨ψ|P|ψ⟩
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

function commutator_mpo(A::MPO, B::MPO)
  # Ensure site indices are matched
  s = siteinds(A) |> Iterators.flatten |> collect
  index_map = s .=> prime.(s, 1)
  Aprime = reindex_mpo_siteinds(A, index_map)
  AB = Aprime * B
  AB = replaceprime(AB, 1 => 0)
  BA = reindex_mpo_siteinds(B, index_map) * A
  BA = replaceprime(BA, 1 => 0)
  comm = AB - BA
  return comm
end

function max_element_norm(mpo::MPO)
  return maximum(abs, [norm(t) for t in mpo])
end

function check_commute(A::Vector{MPO}, B::MPO; tol=1e-10)
  commute_error = 0
  for j in 1:length(A)
    err = norm(commutator_mpo(A[j], B))
    println(j, " ", err)
    commute_error += err
  end
  println("Max element norm of commutator: ", commute_error/length(A))
  # return C # norm(C) < tol
end

function sandwich_mpo(P::MPO, H::MPO)
  H1 = contract(P'', H', :coo, :dense)
  H_eff = contract(P, H1, :coo, :blocksparse)
  H_eff = replaceprime(H_eff, 3 => 1)
  return H_eff
end

function sandwich_mpo_dense(P::MPO, H::MPO)
  H1 = contract(P'', H'; is_ctn_compression=true)
  H_eff = contract(P, H1; is_ctn_compression=true)
  H_eff = replaceprime(H_eff, 3 => 1)
  return H_eff
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
    for k in 1:length(result)
    end
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

function normalize_by_max(H::MPO)
  m = maxabs(H)
  println("max value of the MPO is ", m)
  @assert m > 0 "All entries are zero; cannot normalize."
  return (1/m) * H
end


_mpsnorm(x::MPS) = norm(x)  # sqrt(real(inner(x', x)))

# Estimate operator norm ||P|| by power iteration on MPS space
function opnorm_estimate(P::MPO, sites::Vector{Index{Int64}}; iters::Int=8, linkdim::Int=2)
  s = sites
  v = randomMPS(s; linkdims=linkdim)
  for _ in 1:iters
    w = P * v
    n = _mpsnorm(w)
    n == 0 && return 0.0
    v = w / n
  end
  # Rayleigh-like estimate of ||P||
  return _mpsnorm(P * v) / _mpsnorm(v)
end


function isometric_check(P::MPO, sites::Vector{Index{Int64}}; trials::Int=2, power_iters::Int=6, tol::Float64=1e-10, linkdim::Int=2)
  s = sites
  dev_max = 0.0

  # Power iteration to estimate ||B|| where B := P'P - I
  for _ in 1:trials
    v = randomMPS(s; linkdims=linkdim)
    v = ITensorMPS.orthogonalize(v, 1)

    for _ in 1:power_iters
      w = dag(P) * (P * v) - v   # B*v
      n = _mpsnorm(w)
      n == 0 && break
      v = w / n
    end

    w = dag(P) * (P * v) - v
    dev = _mpsnorm(w) / max(_mpsnorm(v), 1e-300)
    dev_max = max(dev_max, dev)
  end

  pop = opnorm_estimate(P, sites; iters=power_iters, linkdim)

  if dev_max < tol
    println("✅ P is (approximately) isometric: ||P'P - I|| ≈ $dev_max  ≤ tol=$tol")
    println("   Approx operator norm ||P|| ≈ $pop")
    return true, dev_max, pop
  else
    println("❌ P is NOT isometric: ||P'P - I|| ≈ $dev_max  (tol=$tol)")
    println("   Approx operator norm ||P|| ≈ $pop")
    return false, dev_max, pop
  end
end


function clean!(op::MPO; tol=1e-12)
  for j in 1:length(op)
      T = op[j]
      A = array(T)  # convert to dense Julia array
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
      op[j] = ITensor(A, inds(T)...)  # rebuild tensor with same indices
  end
  return op
end


function mpo_memory_bytes(H::MPO)
  total_bytes = 0
  for (i, W) in enumerate(H)
      bytes_i = Base.summarysize(W)
      total_bytes += bytes_i
  end
  println("Total MPO memory: $(total_bytes/1e6) MB")
  return total_bytes
end

function mps_memory_bytes(H::MPS)
  total_bytes = 0
  for (i, W) in enumerate(H)
      bytes_i = Base.summarysize(W)
      total_bytes += bytes_i
  end
  println("Total MPO memory: $(total_bytes/1e6) MB")
  return total_bytes
end


let
    spin = 3
    error = 10
    lambda = 0.0
    spin_sector = 1.0
    maxdim_list = [10]

    is_ctn_compression = false # Let's start with no compression to verify correctness first
    N = 2 # 2 plaquettes = 4 sites, so 8 total spins

    states = 2*N+2
    if spin == 2
        sites = siteinds("S=1/2", states)
    elseif spin == 3
        sites = siteinds("S=1", states)
    else
        error("Not supported spin case")
    end
    os = OpSum()
    os3 = OpSum[]
    os_reg = OpSum()
    lamb = 0
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
        if lambda > 0.0
          os_reg += lambda, "Id", 2*j - 1, "Id", 2*j, "Id", 2*j + 1, "Id", 2*j + 2
          os_reg += -1.0*spin_sector*lambda, "exp(i*pi*Sy)", 2*j - 1, "exp(i*pi*Sx)", 2*j, "exp(i*pi*Sx)", 2*j + 1, "exp(i*pi*Sy)", 2*j + 2
        end
    end

    cutoff = 10.0^(-12)
    Comm = MPO[]
    ConsOps1 = MPO[]
    ConsOps2 = MPO[]
    
    for j in 1:N
        operator = MPO(os2[j], sites, [2*j - 1, 2*j, 2*j + 1, 2*j + 2])
        clean!(operator; tol=1e-12)
        push!(ConsOps1, operator)
        operator = MPO(os2[j], sites)
        push!(ConsOps2, operator)
        push!(Comm, MPO(os3[j], sites)) 
    end


    ConsOpsCombined = multiplyVecMPOtoMPO(ConsOps1, is_ctn_compression=is_ctn_compression)
    ConsOpsCombined2 = multiplydense(ConsOps1)

    # for i in 1:length(ConsOpsCombined)
    #   tensor1 = SparseBackends.to_dense(ConsOpsCombined[i])
    #   tensor2 = SparseBackends.to_dense(ConsOpsCombined2[i])
    #   # print(inds(ConsOpsCombined[i]), " vs ", inds(ConsOpsCombined2[i]), "  ")
    #   if !isapprox(tensor1, tensor2; atol=1e-12)
    #     println("Tensor $i differs between COO and dense multiplication!")
    #     println("COO result: ", tensor1)
    #     println("Dense result: ", tensor2)
    #     # else
    #       # println("Tensor $i matches between COO and dense multiplication.")
    #   end
    # end


    H = MPO(os, sites)
    H_new = copy(H)
    H_new = sandwich_mpo(ConsOpsCombined, H_new)
    H_new2 = copy(H)
    H_new2 = sandwich_mpo_dense(ConsOpsCombined2, H_new2)

    # for i in 1:length(H_new)
    #   tensor1 = SparseBackends.to_dense(H_new[i])
    #   tensor2 = SparseBackends.to_dense(H_new2[i])
    #   if !isapprox(tensor1, tensor2; atol=1e-12)
    #     println("Tensor $i differs between Blocksparse and dense sandwiching!")
    #     println("COO result: ", tensor1)
    #     println("Dense result: ", tensor2)
    #     # else
    #     # println("Tensor $i matches between Blocksparse and dense sandwiching.")
    #   end
    # end


    mem_bytes_projected = mpo_memory_bytes(H_new)
    mem_bytes_projected2 = mpo_memory_bytes(H_new2)
    mem_bytes_original = mpo_memory_bytes(H)
    println("Mem comparison ", mem_bytes_projected, " ", mem_bytes_projected2, " ", mem_bytes_original)


    psi_old = random_mps(sites)
    psi = copy(psi_old)

    psi0 = copy(psi_old)
    for j in 1:N
      psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0)
      normalize!(psi0)
    end

    psi_ = copy(psi0)
    psi_ = replaceprime(contract(ConsOpsCombined, psi_, :coo, :dense), 1 => 0)
    println("Memory usage of different states: ", mps_memory_bytes(psi_), " ", mps_memory_bytes(psi0), " ", mps_memory_bytes(psi_old))
    for i in 1:length(psi_)
      println("inds of psi_[$i] are ", inds(psi_[i]), " and inds of psi0[$i] are ", inds(psi0[i]))
      println(psi_[i])
    end


    maxdim = [80]
    mindim = [80]
    target_energy = nothing
    nsweeps = 0
    last_sweep_energy = nothing
    t = @elapsed begin
        energy, psi, sweeps, t_err = dmrg(H, psi_; nsweeps, maxdim, mindim, cutoff, target_energy, use_early_exit=false, last_sweep_energy=last_sweep_energy, outputlevel=1)
    end
    # E_0 = inner(copy(psi0)', H, copy(psi0))
    # E_1 = inner(psi', H, psi)
    # println("\n\t Energy at start ", E_0, " and at end ", E_1, " in sweeps ", sweeps, " and truncation error ", t_err)
    # println("\n\t\tEnergy under Hamiltonian: $E_1 in sweeps $sweeps and terr $t_err and total time $t seconds")


    maxdim = [80]
    mindim = [80]
    target_energy = nothing
    nsweeps = 0
    last_sweep_energy = nothing
    t = @elapsed begin
        energy, psi, sweeps, t_err = dmrg(H_new2, psi0; nsweeps, maxdim, mindim, cutoff, target_energy, use_early_exit=false, last_sweep_energy=last_sweep_energy, outputlevel=1)
    end
    E_0 = inner(copy(psi0)', H, copy(psi0))
    E_1 = inner(psi', H, psi)
    println("\n\t Energy at start ", E_0, " and at end ", E_1, " in sweeps ", sweeps, " and truncation error ", t_err)
    println("\n\t\tEnergy under Hamiltonian: $E_1 in sweeps $sweeps and terr $t_err and total time $t seconds")


    # maxdim = [80]
    # mindim = [80]
    # target_energy = nothing
    # nsweeps = 0
    # last_sweep_energy = nothing
    # t = @elapsed begin
    #     energy, psi, sweeps, t_err = dmrg(H, psi0; nsweeps, maxdim, mindim, cutoff, target_energy, use_early_exit=false, last_sweep_energy=last_sweep_energy, outputlevel=1)
    # end
    # E_0 = inner(copy(psi0)', H, copy(psi0))
    # E_1 = inner(psi', H, psi)
    # println("\n\t Energy at start ", E_0, " and at end ", E_1, " in sweeps ", sweeps, " and truncation error ", t_err)
    # println("\n\t\tEnergy under Hamiltonian: $E_1 in sweeps $sweeps and terr $t_err and total time $t seconds")
end
nothing