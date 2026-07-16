# env_channel_audit_kl.jl — read-only audit of the KL factor-core matvec operands.
#
# Goal: find WHICH operand in product() = Lenv·φ·PH·Renv carries the same index
# TWICE at one prime level (the malformed permB the aliased×dense kernel chokes on),
# and confirm φ (built by the emit-window-map path) is NOT the culprit. Prints the
# (dim, plev, tags, id) anatomy of PH[b], Lenv, Renv, φ and flags any (id,plev) that
# repeats within a single tensor.

using SparseBackends, ITensors, ITensorMPS, Random, Printf, LinearAlgebra
include("../../test_sparse_psi/utils.jl")
include("../../test_sparse_ham/aliased_helpers.jl")

function build_models(N::Int, psign::Int)
    Random.seed!(42)
    sites = siteinds("S=1", 2*N+2)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2*j-1, "Sz", 2*j; end
    for j in 1:N
        os += "Sx", 2*j-1, "Sx", 2*j+2
        os += "Sy", 2*j,   "Sy", 2*j+1
    end
    cs = 0.5 * psign
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
        t += cs,  "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    mulMPO(A, B) = (Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1))
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H = MPO(os, sites)
    psi0 = random_mps(sites)
    psi_ali = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)
    return H, P_sparse, psi_ali
end

function showinds(tag, T)
    println("  $tag  (ndims=$(length(inds(T)))):")
    seen = Dict{Tuple{UInt64,Int},Int}()
    for i in inds(T)
        key = (ITensors.id(i), ITensors.plev(i)); seen[key] = get(seen, key, 0) + 1
        @printf("     dim=%-4d plev=%d  tags=%-30s  id=%s\n",
                ITensors.dim(i), ITensors.plev(i), string(ITensors.tags(i)), string(ITensors.id(i)%100000))
    end
    dups = [(k, v) for (k, v) in seen if v > 1]
    if isempty(dups)
        println("     [no repeated (id,plev) — well-formed]")
    else
        for (k, v) in dups
            @printf("     >>> REPEATED (id=%s, plev=%d) appears %d times  <<<\n", string(k[1]%100000), k[2], v)
        end
    end
end

let
    N_plaq = 5
    H, P, psi = build_models(N_plaq, +1)
    N = length(psi); b = div(N, 2)
    println("===== KL env-channel audit  N=$N  bond b=$b =====\n")

    cpm = ITensorMPS.CoreProjMPO(H, P; nsite=2)     # builds PH = build_ph_output(P,H)
    println("--- PH[b], PH[b+1] = build_ph_output(P,H) ---")
    showinds("PH[$b]", cpm.PH[b]); showinds("PH[$(b+1)]", cpm.PH[b+1])

    ITensorMPS.position!(cpm, psi, b)
    Lenv = ITensorMPS.lproj(cpm.Hbare); Renv = ITensorMPS.rproj(cpm.Hbare)
    println("\n--- ENVIRONMENTS (bare H, aliased ψ) ---")
    Lenv === nothing ? println("  Lenv: (none)") : showinds("Lenv", Lenv)
    Renv === nothing ? println("  Renv: (none)") : showinds("Renv", Renv)

    phi = SparseBackends.contract(psi[b], psi[b+1], :aliased, :aliased, :aliased;
                                  preserve_bs_output=true, emit_window_map=true)
    println("\n--- φ = ψ[b]·ψ[b+1]  (the operand MY change builds) ---")
    showinds("phi", phi)
end
nothing
