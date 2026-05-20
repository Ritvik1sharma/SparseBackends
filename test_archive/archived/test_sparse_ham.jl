# using CSV, DataFrames
using SparseBackends
using ITensors, ITensorMPS

# const SBX = let m = Base.get_extension(SparseBackends, :SparseBackendsITensorsExt)
#   m === nothing && error("SparseBackendsITensorsExt did not load. Did you `using ITensors` and set up [extensions]/[weakdeps]?")
#   m
# end

include("utils.jl")


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

function multiplyVecMPOtoMPO(vec::Vector{MPO}; 
                             is_ctn_compression=false,
                             cutoff=0.0, maxdim=typemax(Int),
                             target_len=length(vec))
  result = vec[1]
  result
  for j in 2:target_len
    result = mulMPO(result, vec[j]; is_ctn_compression=is_ctn_compression, cutoff=cutoff, maxdim=maxdim)
  end
  return result
end

function multiplydense(vec::Vector{MPO};
                       is_ctn_compression=false,
                       cutoff=0.0, maxdim=typemax(Int),
                       target_len=length(vec))
  result = vec[1]
  for j in 2:target_len
    Bp = prime(vec[j], "Site")
    result2 = contract(result, Bp; is_ctn_compression=true) 
    result = replaceprime(result2, 2 => 1)
  end
  return result
end

function multiplyVecAndCompare(vec::Vector{MPO})
  result1, result2 = vec[1], vec[1]
  for i in 2:length(vec)
    result1_ = mulMPO(result1, vec[i])
    result2_ = replaceprime(contract(result2, prime(vec[i], "Site"); is_ctn_compression=true), 2 => 1)
    for j in 1:length(result1_)
      a, b, c = compare_mpo_tensors(result1_[j], result2_[j])
      if !a
        println("Difference found ")
        println("Result1 input 1: ", result1[j])
        println("Result1 input 2: ", vec[i][j])
        println("Result1 output: ", result1_[j])
        println("Result2 input 1: ", result2[j])
        println("Result2 output: ", result2_[j])
        error("Tensors differ at step $i tensor $j")
      end
    end
    result1, result2 = result1_, result2_
  end
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
    # multiplyVecAndCompare(ConsOps1)
    
    ConsOpsCombined = multiplyVecMPOtoMPO(ConsOps1, is_ctn_compression=is_ctn_compression)
    ConsOpsCombined2 = multiplydense(ConsOps1)
    for i in 1:length(ConsOpsCombined)
      print("Comparing tensor $i")
      ok, max_diff, c = compare_mpo_tensors(ConsOpsCombined[i], ConsOpsCombined2[i])
      if ok
        println("  Tensor matches.")
      else
        println("  Tensor DIFFERS! Max abs diff: $max_diff")
      end
    end


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
    #   println("Comparing tensor $i of H_new and H_new2")
    #   ok, max_diff = compare_mpo_tensors(H_new[i], H_new2[i])
    #   if ok
    #     println("  Tensor $i matches.")
    #   else
    #     println("  Tensor $i DIFFERS! Max abs diff: $max_diff")
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

    tensor_tracker = []

    maxdim = [80]
    mindim = [80]
    target_energy = nothing
    nsweeps = 1
    last_sweep_energy = nothing
    t = @elapsed begin
        energy, psi, sweeps, t_err = dmrg(H_new, psi0; nsweeps, maxdim, mindim, cutoff, target_energy, use_early_exit=false, last_sweep_energy=last_sweep_energy, outputlevel=1, tensor_tracker=tensor_tracker)
    end
    E_0 = inner(copy(psi0)', H, copy(psi0))
    E_1 = inner(psi', H, psi)
    println("\n\t Energy at start ", E_0, " and at end ", E_1, " in sweeps ", sweeps, " and truncation error ", t_err)
    println("\n\t\tEnergy under Hamiltonian: $E_1 in sweeps $sweeps and terr $t_err and total time $t seconds")


    maxdim = [80]
    mindim = [80]
    target_energy = nothing
    nsweeps = 1
    last_sweep_energy = nothing
    t = @elapsed begin
        energy, psi, sweeps, t_err = dmrg(H_new2, psi0; nsweeps, maxdim, mindim, cutoff, target_energy, use_early_exit=false, last_sweep_energy=last_sweep_energy, outputlevel=1, tensor_tracker=tensor_tracker)
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