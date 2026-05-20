# Multi-N verification: run sparse + dense DMRG at N=4, 6, 8.
# Reports REAL bond dims (product of all shared indices, not just Link-tagged)
# to distinguish "sparse mult is starved" from "linkdims reporter is misleading".
using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

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

# REAL bond dim across two adjacent tensors: product over ALL shared indices.
# For sparse psi the shared inds are (new_sp_link, new_d_mult) — product = full
# bond capacity. For dense psi typically a single index.
function real_linkdims(psi)
    return [prod(ITensors.dim(I) for I in commoninds(psi[i], psi[i+1]); init=1)
            for i in 1:length(psi)-1]
end

function shared_inds_summary(psi, i)
    shared = collect(commoninds(psi[i], psi[i+1]))
    parts = String[]
    for I in shared
        tag = ITensors.hastags(I, "Link") ? "L" : "M"
        push!(parts, string(ITensors.dim(I), tag))
    end
    return "[" * join(parts, "x") * "]"
end

function run_one(N::Int; nsweeps_::Int = 8, maxdim_::Int = 40, cutoff_::Float64 = 1e-10)
    println("\n############################################################")
    println("##  N_plaq = $N   sites = $(2*N+2)")
    println("############################################################")
    t_setup = @elapsed (H_sp, H_d, psi_sp, psi_d) = build_setup(N)
    println("Setup time: $(round(t_setup, digits=1))s")
    println("psi_sp initial linkdims (Link-only) = ", [ITensors.dim(commonind(psi_sp[i], psi_sp[i+1])) for i in 1:length(psi_sp)-1])
    println("psi_sp initial REAL linkdims        = ", real_linkdims(psi_sp))
    println("psi_d  initial linkdims              = ", real_linkdims(psi_d))

    sweeps = Sweeps(nsweeps_)
    setmaxdim!(sweeps, [min(10*k, maxdim_) for k in 1:nsweeps_]...)
    setmindim!(sweeps, 1)
    setcutoff!(sweeps, cutoff_)

    println("\n--- Dense DMRG (N=$N, maxdim=$maxdim_, sweeps=$nsweeps_) ---")
    t_d = @elapsed (E_d, psi_d_out) = dmrg(H_d, psi_d, sweeps; outputlevel=1)
    println("Dense final E = $E_d  (time: $(round(t_d, digits=1))s)")
    println("Dense final REAL linkdims = ", real_linkdims(psi_d_out))

    println("\n--- Sparse DMRG (Path B, N=$N, maxdim=$maxdim_, sweeps=$nsweeps_) ---")
    t_s = @elapsed (E_s, psi_sp_out) = dmrg(H_sp, psi_sp, sweeps; outputlevel=1)
    println("Sparse final E = $E_s  (time: $(round(t_s, digits=1))s)")
    println("Sparse final REAL linkdims  = ", real_linkdims(psi_sp_out))
    println("Sparse final Link-only dims = ", [ITensors.dim(commonind(psi_sp_out[i], psi_sp_out[i+1])) for i in 1:length(psi_sp_out)-1])
    println("Sparse per-bond shared-ind breakdown (L=Link, M=Mult):")
    for i in 1:length(psi_sp_out)-1
        println("   bond $i: $(shared_inds_summary(psi_sp_out, i))")
    end

    println("\n|E_dense - E_sparse| = ", abs(E_d - E_s))

    println("--- Overlap check ---")
    psi_sp_dense = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in psi_sp_out])
    nrm_sp = sqrt(real(inner(psi_sp_dense, psi_sp_dense)))
    nrm_d  = sqrt(real(inner(psi_d_out,  psi_d_out)))
    ov     = inner(psi_sp_dense, psi_d_out)
    fid    = abs(ov) / (nrm_sp * nrm_d)
    println("  ||psi_sparse|| = $nrm_sp   ||psi_dense|| = $nrm_d")
    println("  fidelity = $fid   1 - fidelity = $(1 - fid)")

    return (N=N, E_d=E_d, E_s=E_s, gap=abs(E_d - E_s), fid=fid,
            sp_dims=real_linkdims(psi_sp_out), d_dims=real_linkdims(psi_d_out),
            t_d=t_d, t_s=t_s)
end

let
    results = []
    for N in (4, 6, 8)
        push!(results, run_one(N; nsweeps_=8, maxdim_=40, cutoff_=1e-10))
    end

    println("\n############################################################")
    println("##  SUMMARY")
    println("############################################################")
    println("N     E_dense              E_sparse             |gap|        fid                  max_real_sp_dim  max_real_d_dim")
    for r in results
        println(rpad(string(r.N), 6), rpad(string(r.E_d), 21), rpad(string(r.E_s), 21),
                rpad(string(round(r.gap, sigdigits=4)), 13),
                rpad(string(round(r.fid, sigdigits=10)), 21),
                rpad(string(maximum(r.sp_dims)), 17),
                string(maximum(r.d_dims)))
    end
end
nothing
