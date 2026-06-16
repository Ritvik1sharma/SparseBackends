using ITensors, ITensorMPS, Random, LinearAlgebra



function sandwich_mpo(P::MPO, H::MPO)
  H_eff = contract(contract(P'', H'; cutoff=1e-12), P; cutoff=1e-32)
  H_eff = replaceprime(H_eff, 3 => 1)
  return H_eff
end

function multiply_mpos(A::MPO, B::MPO; is_ctn_compression=false, cutoff=0.0, maxdim=typemax(Int), debug=false)
  Bp = prime(B, "Site")
  if is_ctn_compression
    C  = contract(A, Bp; is_ctn_compression=is_ctn_compression, debug=debug, cutoff=cutoff, maxdim=maxdim)
  else
    C  = contract(A, Bp; cutoff=1e-12)
  end
  return replaceprime(C, 2 => 1)
end

function multiply_mpo_mps(A::MPO, B::MPS; is_ctn_compression=false, cutoff=0.0, maxdim=typemax(Int), debug=false)
  C  = contract(A, B; cutoff=cutoff)
  C = replaceprime(C, 1 => 0)
  return C
end

function multiply_mpo_mps(A::Vector{MPO}, B::MPS)
  for i in 1:length(A)
    B = multiply_mpo_mps(A[i], B; cutoff=1e-12)
  end
  return B
end

function sandwich_mpo(P::Vector{MPO}, H::MPO)
  H_eff = H
  for i in 1:length(P)
    H_eff = sandwich_mpo(P[i], H_eff)
  end
  return H_eff
end

function get_energy(H::MPO, psi::MPS)
  return inner(psi', H, psi)
end

function add_to_constr!(P::Vector{MPO}, constr::MPO)
    push!(P, constr)
    return P
end

function add_to_constr!(P::Vector{MPO}, constr::Vector{MPO})
    for c in constr
        add_to_constr!(P, c)
    end
    return P
end

let
	
	seed = 42
	cpus = 1
    N = 100
	ITensors.op(::OpName"Xp", ::SiteType"S=1") = [0 0 1 ; 0 0 0 ; 1 0 0 ]
	ITensors.op(::OpName"Px", ::SiteType"S=1") = [0 1 0 ; 1 0 0 ; 0 0 0 ]
	ITensors.op(::OpName"LP", ::SiteType"S=1") = [1 0 0 ; 0 1 0 ; 0 0 0 ]
	ITensors.op(::OpName"RP", ::SiteType"S=1") = [1 0 0 ; 0 0 0 ; 0 0 1 ]
	
	sites = siteinds("S=1", 100)
	Random.seed!(seed)
	
	psi = random_mps(sites)
	
	HamiltonianTerms = OpSum()
	HamiltonianTerms += 1, "Xp", 1
	for j in 0:98
	  HamiltonianTerms += 1, "Px", 1*j+1, "LP", 1*j+2
	end
	for j in 0:98
	  HamiltonianTerms += 1, "RP", 1*j+1, "Xp", 1*j+2
	end
	HamiltonianTerms += 1, "Px", 100
	
	H = MPO(HamiltonianTerms, sites)
	
	
	function NotEqlsLoop_R1(sites)
		R1_first = itensor_from_nonzeros((3, 3, 2), [(0, 0, 0), (1, 1, 1), (2, 2, 0)])
		R1_bulk = itensor_from_nonzeros((3, 3, 2, 2), [(0, 0, 0, 0), (0, 0, 1, 0), (1, 1, 0, 1), (1, 1, 1, 0), (2, 2, 0, 0)])
		R1_last = itensor_from_nonzeros((3, 3, 2), [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (2, 2, 0)], left=true)
		N = length(sites)
		Wvec = Vector{ITensor}(undef, N)
		bonds = [Index(2, "Link,l=$(i)") for i in 1:100]
		Wvec[1] = bind_to_idx(R1_first, sites[1], sites[1]', bonds[1])
		for j in 2:99
			Wvec[j] = bind_to_idx(R1_bulk, sites[j], sites[j]', bonds[j-1], bonds[j])
		end
		Wvec[100] = bind_to_idx(R1_last, sites[100], sites[100]', bonds[99])
		return MPO(Wvec)
	end
	R1 = NotEqlsLoop_R1(sites)
	
	P = MPO[]
	P = add_to_constr!(P, R1)
	
	H_cstr = H
	
	H_cstr = sandwich_mpo(P, H_cstr)
	psi = multiply_mpo_mps(P, psi)
	
	time_1 = @elapsed begin
	nsweeps = 10
	maxdim = [20]
	mindim = [20]
	cutoff = [1e-10]
	outputlevel = 1
		
	energy1, psi1, num_sweeps1, trucerr1 = dmrg(H_cstr, psi; nsweeps, maxdim, mindim, cutoff, outputlevel)
	end
	
	time_2 = @elapsed begin
	nsweeps = 10
	maxdim = [20]
	mindim = [20]
	cutoff = [1e-10]
	weight = 20
	outputlevel = 1
		
	energy2, psi2, num_sweeps2, trucerr2 = dmrg(H_cstr, [psi1], psi; nsweeps, maxdim, mindim, cutoff, weight, outputlevel)
	end
	
	println("Number of sweeps are: $num_sweeps1, $num_sweeps2")
	
	system_energy1 = get_energy(H, psi1)
	
	system_energy2 = get_energy(H, psi2)
	
	println("System time is $time_1, $time_2")
	
	println("The energy is: $system_energy1, $system_energy2")
	
end