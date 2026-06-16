# Verify post-DMRG sparse ψ has the expected physical properties:
#   (1a) Memory cells:  BlockSparse data length vs dense prod(dims)
#   (1b) Memory bytes:  Base.summarysize on sparse ψ vs densified ψ
#        (cells understate compression because they ignore metadata; bytes is
#         the deployable number.)
#   (2)  ⟨ψ|C_j|ψ⟩ / ⟨ψ|ψ⟩ = 1 for every constraint j (true +1 eigenvector of every C_j).
# Usage: julia test_psi_properties.jl <N_plaq>

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
    # Per-constraint MPOs.  Each factor (I+C_j)/2 acts on 4 sites.
    os2 = OpSum[]    # for the projector
    osC = OpSum[]    # for the bare C_j (so we can measure ⟨ψ|C_j|ψ⟩ directly)
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += 0.5, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
        # bare C_j alone (the exp(i pi ...) tensor product without the I+ part).
        c = OpSum()
        c += 1.0, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(osC, c)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    Cj_MPOs  = [MPO(osC[j], sites) for j in 1:N]
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    H1       = contract(P_sparse'', H', :coo, :dense)
    H_sparse = replaceprime(contract(P_sparse, H1, :coo, :blocksparse), 3 => 1)
    psi0     = random_mps(sites)
    psi_sp   = replaceprime(contract(P_sparse, copy(psi0), :coo, :dense), 1 => 0)
    return H_sparse, psi_sp, Cj_MPOs, sites
end

# Count storage cells of a tensor.
function tensor_cells(T::ITensor)
    if ITensors.has_external_storage(T)
        w = ITensors.get_external_storage(T)
        return length(w.blocksparse.data), prod(w.blocksparse.dims)
    else
        return length(T), length(T)
    end
end

densify_mps(M) = MPS([SparseBackends.to_dense_itensors_unfused(T) for T in M])

length(ARGS) < 1 && error("Usage: julia test_psi_properties.jl <N_plaq>")

let
    N = parse(Int, ARGS[1])
    println("=== ψ-properties check after DMRG (N=$N) ===")
    H_sp, psi_sp, Cj_MPOs, sites = build_setup(N)

    sweeps = Sweeps(2)
    setmaxdim!(sweeps, 10, 20)
    setmindim!(sweeps, 1)
    setcutoff!(sweeps, 1e-10)
    E, psi = dmrg(H_sp, psi_sp, sweeps; outputlevel=0)
    println("DMRG E = $E")

    # ---- (1a) Memory cells: stored data length vs dense prod(dims) ----
    println("\n--- (1a) Memory CELLS (sparse stored / dense cells) ---")
    total_sp = 0; total_de = 0
    for i in 1:length(psi)
        sp, de = tensor_cells(psi[i])
        total_sp += sp; total_de += de
        println("  site $i:  stored=$sp  dense_cells=$de  ratio=$(round(sp/de; digits=4))")
    end
    println("  TOTAL    stored=$total_sp  dense_cells=$total_de  overall_ratio=$(round(total_sp/total_de; digits=4))")

    # ---- (1b) Memory BYTES: Base.summarysize (includes metadata) ----
    println("\n--- (1b) Memory BYTES via Base.summarysize ---")
    psi_d = densify_mps(psi)
    human(n) = n < 1024 ? "$(n) B" :
              n < 1024^2 ? "$(round(n/1024; digits=1)) KiB" :
              "$(round(n/1024^2; digits=2)) MiB"
    tot_sp_b = 0; tot_d_b = 0
    for i in 1:length(psi)
        bs = Base.summarysize(psi[i])
        bd = Base.summarysize(psi_d[i])
        tot_sp_b += bs; tot_d_b += bd
        println("  site $i:  sparse=$(human(bs))  dense=$(human(bd))  ratio=$(round(bs/bd; digits=4))")
    end
    println("  TOTAL    sparse=$(human(tot_sp_b))  dense=$(human(tot_d_b))  overall_ratio=$(round(tot_sp_b/tot_d_b; digits=4))  fold=$(round(tot_d_b/tot_sp_b; digits=2))x")

    # ---- (2) +1 eigenvector check: ⟨ψ|C_j|ψ⟩ / ⟨ψ|ψ⟩ for each j ----
    println("\n--- (2) +1 eigenvector: ⟨ψ|C_j|ψ⟩ / ⟨ψ|ψ⟩ should be 1.0 for ψ ∈ image(P) ---")
    nrm2 = real(inner(psi_d, psi_d))
    println("  ||ψ||² = $nrm2")
    for j in 1:length(Cj_MPOs)
        Cj = Cj_MPOs[j]
        Cpsi = contract(Cj, copy(psi_d); cutoff=1e-14)
        ev   = inner(psi_d, Cpsi) / nrm2
        println("  j=$j:  ⟨ψ|C_$j|ψ⟩ / ||ψ||² = $ev   |.-1| = $(abs(ev - 1))")
    end
end
nothing
