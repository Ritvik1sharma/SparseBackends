# Does DMRG preserve the per-tensor block-sparse key set across all sweeps?
# Captures initial ψ key sets per tensor, runs sparse DMRG, then checks that
# final ψ's key set at each site is a subset of the INITIAL one.

using SparseBackends, ITensors, ITensorMPS
using Random
include("utils.jl")

function build_psi_and_H(N::Int)
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
    return H_sparse, psi_sp
end

# Pull the block-key set off a sparse tensor.
function bs_keyset(T::ITensor)
    w = ITensors.get_external_storage(T)
    return Set(collect(w.blocksparse.keys))
end

let
    N = 3
    println("=== Sparsity-preservation across DMRG (N=$N) ===")
    H_sp, psi_sp = build_psi_and_H(N)

    # Capture initial keys per site.
    initial_keys = [bs_keyset(T) for T in psi_sp]
    println("Initial per-site |keyset|: ", length.(initial_keys))

    # Run DMRG.
    nsweeps_ = 2
    sweeps = Sweeps(nsweeps_)
    setmaxdim!(sweeps, 10, 20)
    setmindim!(sweeps, 1)
    setcutoff!(sweeps, 1e-10)
    E, psi_out = dmrg(H_sp, psi_sp, sweeps; outputlevel=0)
    println("\nDMRG completed. E = $E")

    # Capture final keys per site.
    final_keys = [bs_keyset(T) for T in psi_out]
    println("Final per-site |keyset|:   ", length.(final_keys))

    # Subset check per site.
    println("\nPer-site subset check (final ⊆ initial?):")
    all_ok = true
    for i in 1:length(psi_out)
        ok = issubset(final_keys[i], initial_keys[i])
        extra = setdiff(final_keys[i], initial_keys[i])
        missing_ = setdiff(initial_keys[i], final_keys[i])
        flag = ok ? "✓" : "✗"
        println("  site $i:  init=$(length(initial_keys[i]))  final=$(length(final_keys[i]))  $flag  (extra=$(length(extra)), missing=$(length(missing_)))")
        if !ok
            all_ok = false
            println("    EXTRA keys (in final but not initial): ", collect(extra))
        end
    end
    println()
    println(all_ok ? "RESULT: sparsity pattern PRESERVED (final ⊆ initial at every site)" :
                     "RESULT: sparsity pattern BROKEN at some sites")
end
nothing
