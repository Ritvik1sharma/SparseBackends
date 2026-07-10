# test_check_working_aliased.jl
#
# Variant of ../test_check_working.jl that stores the projected Hamiltonian
# (H_new in the original) using `AliasedBlockSparse` instead of
# `NewBlockSparseSorted`. Both intermediates of `sandwich_mpo` are routed
# through `contract_aliased_itensor` when `output_hint = :aliased`.
#
# Original behaviour is preserved when no output_hint is supplied.

using SparseBackends, Random
using ITensors, ITensorMPS
using TimerOutputs: reset_timer!, print_timer
using LinearAlgebra: BLAS
BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_BLAS_THREADS", "1")))
println("[BLAS threads pinned to ", BLAS.get_num_threads(), "]")

const _DIAG = get(ENV, "DMRG_DIAG", "0") == "1"

include("aliased_helpers.jl")
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
#   :aliased              — both contractions go through
#                           SparseBackends.contract_aliased_itensor so the
#                           result is backed by AliasedBlockSparse.
#   :dense                — plain ITensors.contract (no sparse storage).
# ─────────────────────────────────────────────────────────────────────────────
function sandwich_mpo(P::MPO, H::MPO; output_hint::Symbol = :default)
    H1    = contract(P'', H', :coo, :dense; Cbackend=output_hint)
    H_eff = contract(P, H1, :coo, output_hint; Cbackend=output_hint)
    return replaceprime(H_eff, 3 => 1)
    # end
end

function sandwich_mpo_dense(P::MPO, H::MPO)
    H1    = contract(P'', H'; is_ctn_compression=true)
    H_eff = contract(P, H1; is_ctn_compression=true)
    return replaceprime(H_eff, 3 => 1)
end

function mulMPO(A::MPO, B::MPO; is_ctn_compression=false, cutoff=0.0, maxdim=typemax(Int), debug=false)
    Bp = prime(B, "Site")
    C  = contract(A, Bp, :coo, :coo)
    return replaceprime(C, 2 => 1)
end

function multiplyVecMPOtoMPO(vec::Vector{MPO}; is_ctn_compression=false, cutoff=0.0, maxdim=typemax(Int))
    result = vec[1]
    for j in 2:length(vec)
        result = mulMPO(result, vec[j]; is_ctn_compression, cutoff, maxdim)
    end
    return result
end

function multiplydense(vec::Vector{MPO}; is_ctn_compression=false, cutoff=0.0, maxdim=typemax(Int))
    result = vec[1]
    for j in 2:length(vec)
        Bp     = prime(vec[j], "Site")
        result = replaceprime(contract(result, Bp; is_ctn_compression=true), 2 => 1)
    end
    return result
end

function _maxabs(T::ITensor)
    is = inds(T)
    isempty(is) ? abs(scalar(T)) : maximum(abs, Array(T, is...))
end

maxabs(H::MPO) = maximum(_maxabs, H)

function clean!(op::MPO; tol=1e-12)
    for j in 1:length(op)
        T = op[j]
        A = array(T)
        for i in eachindex(A)
            v = A[i]
            if     abs(v)       < tol; A[i] =  0.0
            elseif abs(v - 1.0) < tol; A[i] =  1.0
            elseif abs(v + 1.0) < tol; A[i] = -1.0
            elseif abs(v - 0.5) < tol; A[i] =  0.5
            elseif abs(v + 0.5) < tol; A[i] = -0.5
            elseif abs(v - 1.0im) < tol; A[i] =  1.0im
            elseif abs(v + 1.0im) < tol; A[i] = -1.0im
            elseif abs(v - 0.5im) < tol; A[i] =  0.5im
            elseif abs(v + 0.5im) < tol; A[i] = -0.5im
            elseif abs(v + 2.0)   < tol; A[i] = -2.0
            elseif abs(v - 2.0)   < tol; A[i] =  2.0
            end
        end
        op[j] = ITensor(A, inds(T)...)
    end
    return op
end

function mpo_memory_bytes(H::MPO)
    total_bytes = sum(Base.summarysize(W) for W in H)
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
        matches  = filter(l1 -> tags(l1) == base_tag && dim(l1) == dim(l2) && l1 ∉ new_inds, links1)
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

# ─────────────────────────────────────────────────────────────────────────────

length(ARGS) < 1 && error("Usage: julia test_check_working_aliased.jl <N_plaq>")

let
    spin        = parse(Int, get(ENV, "BENCH_SPIN", "3"))
    spin_sector = parse(Float64, get(ENV, "BENCH_PSIGN", "1.0"))   # projector eigenvalue sector (+1/-1)
    N           = parse(Int, ARGS[1])
    states      = 2*N + 2

    sites = spin == 2 ? siteinds("S=1/2", states) :
            spin == 3 ? siteinds("S=1",   states) :
            error("Not supported spin case")

    os = OpSum(); os_reg = OpSum()
    for j in 1:N+1
        os     += "Sz", 2*j-1, "Sz", 2*j
        os_reg += "Sz", 2*j-1, "Sz", 2*j
    end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
        os_reg += "Sx", 2*j-1, "Sx", 2*j+2
        os_reg += "Sy", 2*j,   "Sy", 2*j+1
    end

    os2 = OpSum[]; os3 = OpSum[]
    for j in 1:N
        coeff = 0.5
        temp  = OpSum()
        temp += coeff,              "Id",           2*j-1, "Id",           2*j, "Id",           2*j+1, "Id",           2*j+2
        temp += spin_sector*coeff, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j, "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, temp)
        temp  = OpSum()
        temp += spin_sector, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j, "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os3, temp)
    end

    cutoff   = 10.0^(-12)
    ConsOps1 = MPO[]; ConsOps2 = MPO[]
    for j in 1:N
        operator = MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])
        clean!(operator; tol=1e-12)
        push!(ConsOps1, operator)
        push!(ConsOps2, MPO(os2[j], sites))
    end

    ConsOpsCombined  = multiplyVecMPOtoMPO(ConsOps1)
    ConsOpsCombined2 = multiplydense(ConsOps1)
    H                = MPO(os, sites)

    # ── Aliased path ──────────────────────────────────────────────────────────
    println("\n[building H_new with output_hint = :aliased]")
    H_new_aliased = sandwich_mpo(ConsOpsCombined, copy(H); output_hint = :aliased)

    # Link fusion is always applied for the aliased path (was gated by
    # SB_FUSE_LINKS): it produces the compact "FusedSparse"-tagged links the
    # aliased matvec + env canonicalization rely on to skip permutations.
    println("\n[fusing multi-strand sparse links in H_new_aliased]")
    println("  pre-fuse  H[3] inds: ", inds(H_new_aliased[3]))
    fuse_sparse_links!(H_new_aliased)
    println("  post-fuse H[3] inds: ", inds(H_new_aliased[3]))

    # HARDENED: prepermute the aliased H dense tails unconditionally (was gated by
    # SB_PREPERMUTE_H). Bakes the operand tail reorder into the constant H once so
    # the per-matvec permute_A is ~identity (measured: add.permute_A alloc −81%,
    # bit-identical E). No env knob.
    println("\n[prepermuting aliased H tails]")
    prepermute_aliased_mpo!(H_new_aliased)

    # ── Reference dense path ──────────────────────────────────────────────────
    H_new2 = sandwich_mpo_dense(ConsOpsCombined2, copy(H))

    # ── Memory footprint ──────────────────────────────────────────────────────
    println("\n── Memory footprint of PHP ──")
    print("DENSE    PHP:  "); mpo_memory_bytes(H_new2)
    report_aliased_footprint(H_new_aliased, "ALIASED PHP:")

    # ── Pointwise comparison (sanity) ─────────────────────────────────────────
    for i in 1:length(H_new_aliased)
        t1 = ITensors.has_external_storage(H_new_aliased[i]) ?
             SparseBackends.to_dense_itensors(H_new_aliased[i]) : H_new_aliased[i]
        t2 = ITensors.has_external_storage(H_new2[i]) ?
             SparseBackends.to_dense_itensors(H_new2[i]) : H_new2[i]
        t2_aligned = align_links(t1, t2, "H[$i]")
        if !isnothing(t2_aligned)
            println(isapprox(t1, t2_aligned) ? "  H[$i] ✓ match (aliased vs dense)" :
                                               "  H[$i] ✗ values differ")
        end
    end

    # ── DMRG ─────────────────────────────────────────────────────────────────
    Random.seed!(parse(Int, get(ENV, "BENCH_SEED", "42")))   # initial-state seed (benchmark knob)
    psi_old = random_mps(sites)
    psi0    = copy(psi_old)
    for j in 1:N
        psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0)
        normalize!(psi0)
    end
    psi1, psi2 = copy(psi0), copy(psi0)

    _md_default = _DIAG ? 20 : 40
    _ns_default = _DIAG ? 3  : 25
    _md = parse(Int, get(ENV, "BENCH_MAXDIM", string(_md_default)))
    _ns = parse(Int, get(ENV, "BENCH_NSWEEPS", string(_ns_default)))
    maxdim = [_md]; mindim = [_md]; nsweeps = _ns
    target_energy = nothing; last_sweep_energy = nothing

    # ── DENSE reference ───────────────────────────────────────────────────────
    println("\n[diag=$_DIAG] Running DENSE DMRG (H_new2) with nsweeps=$nsweeps, maxdim=$_md ...")
    (energy_d, psi_dense, sweeps_d, t_err_d), dense_wall =
        run_dmrg_ground("DENSE H_new2", H_new2, psi1;
                        nsweeps, maxdim, mindim, cutoff, target_energy,
                        last_sweep_energy, outputlevel=1, run_label="DENSE")
    E_0       = inner(copy(psi0)', H, copy(psi0))
    E_1_dense = inner(psi_dense', H, psi_dense)
    println("\n\t Energy at start $E_0 and at end $E_1_dense",
            " in sweeps $sweeps_d and truncation error $t_err_d")
    println("[DENSE]   Energy: $E_1_dense in sweeps $sweeps_d and terr $t_err_d",
            " and wall $dense_wall seconds")

    # ── ALIASED run ───────────────────────────────────────────────────────────
    println("\n[H_new_aliased storage at DMRG entry]")
    n_alias = 0; n_bs = 0; n_dense_t = 0
    for i in 1:length(H_new_aliased)
        T = H_new_aliased[i]
        if ITensors.has_external_storage(T)
            T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse ? (n_alias += 1) : (n_bs += 1)
        else
            n_dense_t += 1
        end
    end
    println("  sites = $(length(H_new_aliased)):  aliased=$n_alias  BS=$n_bs  dense=$n_dense_t")

    println("[diag=$_DIAG] Running ALIASED DMRG (H_new_aliased) with nsweeps=$nsweeps, maxdim=$_md ...")
    (energy_a, psi_ali, sweeps_a, t_err_a), aliased_wall =
        run_dmrg_ground("ALIASED H_new_aliased", H_new_aliased, psi2;
                        nsweeps, maxdim, mindim, cutoff, target_energy,
                        last_sweep_energy, outputlevel=1, run_label="ALIASED")
    E_1_ali = inner(psi_ali', H, psi_ali)
    println("\n\t Energy at start $E_0 and at end $E_1_ali",
            " in sweeps $sweeps_a and truncation error $t_err_a")
    println("[ALIASED] Energy: $E_1_ali in sweeps $sweeps_a and terr $t_err_a",
            " and wall $aliased_wall seconds")

    println("\n========== HEAD-TO-HEAD WALL TIME ==========")
    println("  JIT-excluded wall (warmup done internally by runner):")
    println("    DENSE   wall: $(round(dense_wall;   digits=3)) s")
    println("    ALIASED wall: $(round(aliased_wall; digits=3)) s")
    println("    ratio aliased/dense = $(round(aliased_wall/dense_wall; digits=3))  (want < 1)")
    println("\n  Energy comparison:")
    println("    DENSE   final: $E_1_dense")
    println("    ALIASED final: $E_1_ali")
    println("    |ΔE|         : $(abs(E_1_ali - E_1_dense))")
end
nothing
