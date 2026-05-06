using SparseBackends, Random
using ITensors, ITensorMPS
include("utils.jl")

# ─── helpers copied from test_check_working.jl ────────────────────────────────
function sandwich_mpo(P::MPO, H::MPO)
    H1   = contract(P'', H', :coo, :dense)
    H_eff = contract(P, H1, :coo, :blocksparse)
    replaceprime(H_eff, 3 => 1)
end

function sandwich_mpo_dense(P::MPO, H::MPO)
    H1   = contract(P'', H'; is_ctn_compression=true)
    H_eff = contract(P, H1; is_ctn_compression=true)
    replaceprime(H_eff, 3 => 1)
end

function mulMPO(A::MPO, B::MPO)
    Bp = prime(B, "Site")
    C  = contract(A, Bp, :coo, :coo)
    replaceprime(C, 2 => 1)
end

function multiplyVecMPOtoMPO(vec::Vector{MPO})
    result = vec[1]
    for j in 2:length(vec)
        result = mulMPO(result, vec[j])
    end
    result
end

function multiplydense(vec::Vector{MPO})
    result = vec[1]
    for j in 2:length(vec)
        Bp = prime(vec[j], "Site")
        result = replaceprime(contract(result, Bp; is_ctn_compression=true), 2 => 1)
    end
    result
end

function clean!(op::MPO; tol=1e-12)
    for j in 1:length(op)
        T = op[j]
        A = array(T)
        for i in eachindex(A)
            abs(A[i])         < tol && (A[i] = 0.0)
            abs(A[i] - 1.0)  < tol && (A[i] = 1.0)
            abs(A[i] + 1.0)  < tol && (A[i] = -1.0)
            abs(A[i] - 0.5)  < tol && (A[i] = 0.5)
            abs(A[i] + 0.5)  < tol && (A[i] = -0.5)
        end
        op[j] = ITensor(A, inds(T)...)
    end
    op
end
# ──────────────────────────────────────────────────────────────────────────────

let
    Random.seed!(42)
    N          = 2            # small: 2*2+2 = 6 sites
    spin       = 3            # spin-1
    spin_sector = 1.0
    states     = 2 * N + 2
    sites      = siteinds("S=1", states)

    # Heisenberg Hamiltonian
    os = OpSum()
    for j in 1:N+1
        os += "Sz", 2*j-1, "Sz", 2*j
    end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end

    # Constraint MPOs
    os2 = OpSum[]
    for j in 1:N
        coeff = 0.5
        temp  = OpSum()
        temp += coeff, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        temp += spin_sector*coeff, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j, "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, temp)
    end

    ConsOps1 = [begin
        op = MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])
        clean!(op)
    end for j in 1:N]

    ConsOps2      = [MPO(os2[j], sites) for j in 1:N]
    ConsOpsCombined  = multiplyVecMPOtoMPO(ConsOps1)
    ConsOpsCombined2 = multiplydense(ConsOps1)

    H       = MPO(os, sites)
    H_sparse = sandwich_mpo(ConsOpsCombined, copy(H))
    H_dense  = sandwich_mpo_dense(ConsOpsCombined2, copy(H))

    # Initial state
    psi_old = random_mps(sites)
    psi0    = copy(psi_old)
    for j in 1:N
        psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0)
        normalize!(psi0)
    end

    # ── Dense DMRG reference ──────────────────────────────────────────────────
    cutoff  = 1e-12
    maxdim  = [20]
    mindim  = [20]
    nsweeps = 5
    psi1 = copy(psi0)
    t_dense = @elapsed begin
        E_dense, psi_dense, sw_dense, terr_dense = dmrg(
            H_dense, psi1; nsweeps, maxdim, mindim, cutoff,
            target_energy=nothing, use_early_exit=false, outputlevel=1,
        )
    end
    E_dense_final = inner(psi_dense', H, psi_dense)
    println("\n[Dense DMRG]  E = $E_dense_final  sweeps=$sw_dense  terr=$terr_dense  time=$(round(t_dense,digits=2))s")

    # ── Build sparse psi from projection ─────────────────────────────────────
    psi_sp = replaceprime(contract(ConsOpsCombined, copy(psi0), :coo, :dense), 1 => 0)
    any(ITensors.has_external_storage, psi_sp) || println("WARNING: psi_sp has no external storage tensors!")
    println("psi_sp tensor types: ", [ITensors.has_external_storage(psi_sp[i]) ? "BS" : "dense" for i in 1:length(psi_sp)])

    # ── Pre-DMRG diagnostics (H1: orthogonalize is skipped for sparse psi) ───
    println("[pre-DMRG dense]  norm(psi1)   = ", norm(psi1),
            "  isortho = ", isortho(psi1),
            "  <psi|psi> = ", inner(psi1, psi1),
            "  <psi|H|psi> = ", inner(psi1', H, psi1))
    println("[pre-DMRG sparse] norm(psi_sp) = ", norm(psi_sp),
            "  isortho = ", isortho(psi_sp),
            "  <psi|psi> = ", inner(psi_sp, psi_sp),
            "  <psi|H|psi> = ", inner(psi_sp', H, psi_sp),
            "  <psi|H_sparse|psi> = ", inner(psi_sp', H_sparse, psi_sp))

    # ── Sparse DMRG ──────────────────────────────────────────────────────────
    t_sparse = @elapsed begin
        E_sparse, psi_sparse, sw_sparse, terr_sparse = dmrg(
            H_sparse, psi_sp; nsweeps, maxdim, mindim, cutoff,
            target_energy=nothing, use_early_exit=false, outputlevel=1,
        )
    end
    E_sparse_final = inner(psi_sparse', H, psi_sparse)
    println("\n[Sparse DMRG] E = $E_sparse_final  sweeps=$sw_sparse  terr=$terr_sparse  time=$(round(t_sparse,digits=2))s")

    # ── Verify ────────────────────────────────────────────────────────────────
    delta = abs(E_sparse_final - E_dense_final)
    println("\n|E_sparse - E_dense| = $delta")
    if delta < 1e-4
        println("✓ Sparse psi DMRG matches dense within tolerance")
    else
        println("✗ Energy mismatch: $delta (expected < 1e-4)")
    end
end
nothing
