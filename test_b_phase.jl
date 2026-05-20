# B-phase DMRG integration test: sparse DMRG on N plaquettes, optionally vs dense.
# Usage: julia test_b_phase.jl <N> [nsweeps] [maxdim] [sparse_only]
#   N           — number of plaquettes (required)
#   nsweeps     — number of sweeps (default 8)
#   maxdim      — max bond dimension (default 40)
#   sparse_only — "true" to skip dense DMRG (default "false")
# Reports: timing, real bond dims, energy gap, fidelity vs dense.
using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

length(ARGS) < 1 && error("Usage: julia test_b_phase.jl <N> [nsweeps] [maxdim] [sparse_only]")
const N_PLAQ     = parse(Int,  ARGS[1])
const N_SWEEPS   = length(ARGS) >= 2 ? parse(Int,   ARGS[2]) : 8
const MAXDIM     = length(ARGS) >= 3 ? parse(Int,   ARGS[3]) : 40
const SPARSE_ONLY = length(ARGS) >= 4 ? (ARGS[4] == "true") : false

function build_setup(N::Int)
    Random.seed!(42)
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
        t += 0.5, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    H_dense   = MPO([SparseBackends.to_dense_itensors_unfused(T) for T in H_sparse])
    psi_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp])
    return H_sparse, H_dense, psi_sp, psi_dense
end

function real_linkdims(psi)
    [prod(ITensors.dim(I) for I in commoninds(psi[i], psi[i+1]); init=1) for i in 1:length(psi)-1]
end

function shared_inds_summary(psi, i)
    parts = String[]
    for I in collect(commoninds(psi[i], psi[i+1]))
        push!(parts, string(ITensors.dim(I), ITensors.hastags(I, "Link") ? "L" : "M"))
    end
    "[" * join(parts, "x") * "]"
end

let
    println("############################################################")
    println("##  N_plaq = $N_PLAQ   sites = $(2*N_PLAQ+2)   nsweeps = $N_SWEEPS   maxdim = $MAXDIM   sparse_only = $SPARSE_ONLY")
    println("############################################################")

    t_setup = @elapsed (H_sp, H_d, psi_sp, psi_d) = build_setup(N_PLAQ)
    println("Setup time: $(round(t_setup, digits=1))s")
    println("psi_sp initial REAL linkdims = ", real_linkdims(psi_sp))
    println("psi_d  initial linkdims      = ", real_linkdims(psi_d))
    println("psi_sp is_sparse_mps? ", SparseBackends.is_sparse_mps(psi_sp))

    sweeps = Sweeps(N_SWEEPS)
    setmaxdim!(sweeps, [min(10*k, MAXDIM) for k in 1:N_SWEEPS]...)
    setmindim!(sweeps, 1)
    setcutoff!(sweeps, 1e-10)

    E_d = nothing
    psi_d_out = nothing
    if !SPARSE_ONLY
        println("\n--- Dense DMRG (N=$N_PLAQ, maxdim=$MAXDIM, sweeps=$N_SWEEPS) ---")
        t_d = @elapsed (E_d, psi_d_out) = dmrg(H_d, psi_d, sweeps; outputlevel=1, use_early_exit=false)
        println("Dense final E = $E_d  (time: $(round(t_d, digits=1))s)")
        println("Dense final REAL linkdims = ", real_linkdims(psi_d_out))
    end

    println("\n--- Sparse DMRG (Path B, N=$N_PLAQ, maxdim=$MAXDIM, sweeps=$N_SWEEPS) ---")
    t_s = @elapsed (E_s, psi_sp_out) = dmrg(H_sp, psi_sp, sweeps; outputlevel=1, use_early_exit=false)
    println("Sparse final E = $E_s  (time: $(round(t_s, digits=1))s)")
    println("Sparse final REAL linkdims  = ", real_linkdims(psi_sp_out))
    println("Sparse final Link-only dims = ", [ITensors.dim(commonind(psi_sp_out[i], psi_sp_out[i+1])) for i in 1:length(psi_sp_out)-1])
    println("Sparse per-bond shared-ind breakdown (L=Link, M=Mult):")
    for i in 1:length(psi_sp_out)-1
        println("   bond $i: $(shared_inds_summary(psi_sp_out, i))")
    end

    if !SPARSE_ONLY && !isnothing(E_d)
        println("\n|E_dense - E_sparse| = ", abs(E_d - E_s))
        println("\n--- Overlap check ---")
        psi_sp_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp_out])
        nrm_sp = sqrt(real(inner(psi_sp_dense, psi_sp_dense)))
        nrm_d  = sqrt(real(inner(psi_d_out,   psi_d_out)))
        fid    = abs(inner(psi_sp_dense, psi_d_out)) / (nrm_sp * nrm_d)
        println("  ||psi_sparse|| = $nrm_sp   ||psi_dense|| = $nrm_d")
        println("  fidelity = $fid   1 - fidelity = $(1 - fid)")
    end
end
nothing
