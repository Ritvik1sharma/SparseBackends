using SparseBackends, Random
using ITensors, ITensorMPS
using TimerOutputs: reset_timer!

# Sweep N, spin, maxdim and time DMRG sweeps for both sparse and dense projected H.
# Lifted from test_check_working.jl utilities and inlined for self-containment.

function _build_H_and_P(N::Int, spin::Int; lambda=0.0, spin_sector=1.0)
    states = 2*N + 2
    if spin == 2
        sites = siteinds("S=1/2", states)
    elseif spin == 3
        sites = siteinds("S=1", states)
    else
        error("Not supported spin case")
    end

    os = OpSum()
    for j in 1:N+1
        os += "Sz", 2*j-1, "Sz", 2*j
    end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end

    os2 = OpSum[]
    for j in 1:N
        coeff = 0.5
        temp = OpSum()
        temp += coeff,                "Id",            2*j-1, "Id",            2*j, "Id",            2*j+1, "Id",            2*j+2
        temp += spin_sector*coeff,    "exp(i*pi*Sy)",  2*j-1, "exp(i*pi*Sx)",  2*j, "exp(i*pi*Sx)",  2*j+1, "exp(i*pi*Sy)",  2*j+2
        push!(os2, temp)
    end

    ConsOps1 = MPO[]
    ConsOps2 = MPO[]
    for j in 1:N
        op1 = MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])
        push!(ConsOps1, op1)
        op2 = MPO(os2[j], sites)
        push!(ConsOps2, op2)
    end

    return sites, MPO(os, sites), ConsOps1, ConsOps2
end

function _mulMPO(A::MPO, B::MPO)
    Bp = prime(B, "Site")
    C = contract(A, Bp, :coo, :coo)
    return replaceprime(C, 2 => 1)
end

function _multiplyVec(vec::Vector{MPO})
    result = vec[1]
    for j in 2:length(vec)
        result = _mulMPO(result, vec[j])
    end
    return result
end

function _multiplydense(vec::Vector{MPO})
    result = vec[1]
    for j in 2:length(vec)
        Bp = prime(vec[j], "Site")
        result = contract(result, Bp; is_ctn_compression=true)
        result = replaceprime(result, 2 => 1)
    end
    return result
end

function _sandwich_sparse(P, H)
    H1 = contract(P'', H', :coo, :dense)
    Heff = contract(P, H1, :coo, :blocksparse)
    return replaceprime(Heff, 3 => 1)
end

function _sandwich_dense(P, H)
    H1 = contract(P'', H'; is_ctn_compression=true)
    Heff = contract(P, H1; is_ctn_compression=true)
    return replaceprime(Heff, 3 => 1)
end

function _sparsity(H::MPO)
    total_zero = 0
    total_size = 0
    for t in H
        A = Array(t, inds(t)...)
        total_size += length(A)
        total_zero += count(x -> abs(x) < 1e-12, A)
    end
    return total_zero / max(total_size, 1)
end

function bench_one(N::Int, spin::Int, mdim::Int; nsweeps=2, cutoff=1e-12, seed=42)
    Random.seed!(seed)
    sites, H, ConsOps1, ConsOps2 = _build_H_and_P(N, spin)

    P_sparse = _multiplyVec(ConsOps1)
    P_dense  = _multiplydense(ConsOps1)

    H_sparse = _sandwich_sparse(P_sparse, H)
    H_dense  = _sandwich_dense(P_dense, H)

    sp_frac = _sparsity(H_sparse)

    psi0 = random_mps(sites)
    for j in 1:N
        psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0)
        normalize!(psi0)
    end
    psi1 = copy(psi0)
    psi2 = copy(psi0)

    maxdim = [mdim]
    mindim = [mdim]

    # Warmup with 1 sweep (ignored). Both ops to JIT both code paths.
    GC.gc()
    _ = dmrg(H_dense,  copy(psi0); nsweeps=1, maxdim, mindim, cutoff,
             use_early_exit=false, outputlevel=0)
    _ = dmrg(H_sparse, copy(psi0); nsweeps=1, maxdim, mindim, cutoff,
             use_early_exit=false, outputlevel=0)

    GC.gc()
    t_dense  = @elapsed dmrg(H_dense,  psi1; nsweeps, maxdim, mindim, cutoff,
                             use_early_exit=false, outputlevel=0)
    GC.gc()
    t_sparse = @elapsed dmrg(H_sparse, psi2; nsweeps, maxdim, mindim, cutoff,
                             use_early_exit=false, outputlevel=0)

    return (; N, spin, mdim, nsweeps,
            sparse_zero_frac=sp_frac,
            t_dense, t_sparse,
            ratio_sp_over_dn = t_sparse / t_dense)
end

function main()
    Ns      = [6, 10]
    spins   = [2, 3]
    mdims   = [20, 40, 80]
    nsweeps = parse(Int, get(ENV, "BENCH_SWEEPS", "2"))

    println("# sparse-vs-dense DMRG benchmark (1 warmup + $nsweeps timed sweeps each)")
    println("# columns: N, spin, mdim, sparsity(zero-frac of H_sparse), t_dense(s), t_sparse(s), sparse/dense")
    println("N\tspin\tmdim\tsparsity\tt_dense\tt_sparse\tratio")
    for N in Ns, spin in spins, mdim in mdims
        try
            r = bench_one(N, spin, mdim; nsweeps)
            println(rpad(r.N, 4), "\t", r.spin, "\t", r.mdim, "\t",
                    round(r.sparse_zero_frac; digits=3), "\t",
                    round(r.t_dense;   digits=3), "\t",
                    round(r.t_sparse;  digits=3), "\t",
                    round(r.ratio_sp_over_dn; digits=3),
                    r.ratio_sp_over_dn < 1.0 ? "  ← SPARSE WINS" : "")
            flush(stdout)
        catch e
            println(N, "\t", spin, "\t", mdim, "\tERROR\t", sprint(showerror, e))
            flush(stdout)
        end
    end
end

main()
