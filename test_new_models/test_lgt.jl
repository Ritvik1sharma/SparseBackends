using ITensors
using ITensorMPS
using LinearAlgebra

# Helper: one-site Pauli gates
function one_site_gate(sites, i, opstring, angle)
    O = op(opstring, sites[i])
    return exp(-1im*angle*O)
end

# Build U_mu: exp(-i mu dt sum X_m)
function build_mu_gates(sites,L,dt,mu)
    gates = ITensor[]
    for j in 1:L
        i = matter(j)
        push!(gates, one_site_gate(sites, i, "Sx", mu*dt))
    end
    return gates
end

# Build U_h: exp(-i h dt sum X_g)
function build_h_gates(sites,L,dt,h)
    gates = ITensor[]
    for j in 1:L-1
        i = gauge(j)
        push!(gates, one_site_gate(sites, i, "Sx", h*dt))
    end
    return gates
end

# Three-site ZZZ gate: exp(-i J dt Z_i Z_j Z_k)
function zzz_gate(sites,i,j,k,angle)
    inds = (sites[i],sites[j],sites[k])
    U = ITensor(inds..., prime.(inds)...)
    for a in 0:1, b in 0:1, c in 0:1
        z =
        (a==0 ? 1 : -1) *
        (b==0 ? 1 : -1) *
        (c==0 ? 1 : -1)
        val = exp(-1im*angle*z)
        U[
            inds[1]=>a,
            inds[2]=>b,
            inds[3]=>c,
            prime(inds[1])=>a,
            prime(inds[2])=>b,
            prime(inds[3])=>c
        ] = val
    end
    return U
end

# Build exp(-i J dt sum Zm Zg Zm)
function build_J_gates(sites,L,dt,J)
    gates = ITensor[]
    for j in 1:L-1
        push!(gates,
            zzz_gate(sites, matter(j), gauge(j), matter(j+1), J*dt))
    end
    return gates
end

function apply_layer!(psi,gates;cutoff,maxdim)
    for g in gates
        psi = apply(g, psi; cutoff=cutoff, maxdim=maxdim)
    end
    return psi
end


# Lattice indexing
function matter(j)
    return 2j - 1
end

function gauge(j)
    return 2j
end

# Measurements
function expectation_local(psi,sites,opname,idx)
    O = op(opname,sites[idx])
    return real(inner(psi, O*psi))
end

function gauge_polarization(psi,sites,L)
    vals = Float64[]
    for j in 1:L-1
        push!(vals, expectation_local(psi, sites, "Sx", gauge(j)))
    end
    return mean(vals)
end


let
    # Parameters
    L = 10                  # number of matter sites
    Nsites = 2L - 1         # matter + gauge qubits
    J  = 1.0
    h  = 1.3
    mu = 1.5
    dt = 0.25
    nsteps = 20
    cutoff = 1e-10
    maxdim = 200

    # Sites
    sites = siteinds("S=1/2", Nsites)

    ############################################################
    # Initial state
    # matter: |+x> # gauge:  |+x>
    state = String[]
    for n in 1:Nsites
        push!(state,"Up")
    end
    psi = productMPS(sites,state)

    # Construct layers
    J_half = build_J_gates(sites, L, dt/2,J)
    J_full = build_J_gates(sites, L, dt, J)
    h_layer = build_h_gates(sites, L, dt, h)
    mu_layer = build_mu_gates(sites, L, dt, mu)
    
    
    # TEBD evolution
    println("Starting TEBD")
    for step in 1:nsteps
        # second order Suzuki
        psi = apply_layer!(
            psi,
            J_half;
            cutoff=cutoff,
            maxdim=maxdim
        )
        psi = apply_layer!(
            psi,
            h_layer;
            cutoff=cutoff,
            maxdim=maxdim
        )
        psi = apply_layer!(
            psi,
            mu_layer;
            cutoff=cutoff,
            maxdim=maxdim
        )
        psi = apply_layer!(
            psi,
            J_half;
            cutoff=cutoff,
            maxdim=maxdim
        )

        orthogonalize!(psi,1)
        pol = gauge_polarization(
            psi,
            sites,
            L
        )
        println("step = $step   gauge polarization = $pol") 
    end
end