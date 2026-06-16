# Dump the actual MPO tensor entries of one (I+C)/2 projector and check
# whether bond states correspond to literal I/C operators.
using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra
using Printf
include("../utils.jl")

let
    N = 4
    sites = siteinds("S=1", 2*N+2)
    j = 1
    cs = 0.5
    t = OpSum()
    t += 0.5, "Id", 2*j-1, "Id", 2*j, "Id", 2*j+1, "Id", 2*j+2
    t += cs,  "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
               "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
    P_j = clean!(MPO(t, sites, [2*j-1, 2*j, 2*j+1, 2*j+2]))

    println("=== MPO tensor at site 1 (support of P_1 = sites 1-4) ===")
    T1 = P_j[1]
    println("inds = ", inds(T1))
    println("dim = ", [dim(i) for i in inds(T1)])
    println()
    a = Array(T1, inds(T1)...)
    println("size(array) = ", size(a))

    # Site 1 tensor: should have shape (bond_right, site_top, site_bot) for left edge
    # We expect bond_right dim 2 (I-track and C-track).
    # For each bond_right value k, the matrix element T1[k, s_top, s_bot] is the operator at site 1 for that bond state.
    for k in 1:size(a, 1)
        println("\n  -- bond_right = $k --")
        op_mat = a[k, :, :]
        @printf("  operator matrix at site 1 (3x3):\n")
        for r in 1:size(op_mat, 1)
            row_str = join([@sprintf("%+.4f%+.4fim", real(op_mat[r,c]), imag(op_mat[r,c])) for c in 1:size(op_mat,2)], "  ")
            println("    ", row_str)
        end
    end

    # Compare to the operator I (Identity) and C_1 = exp(iπSy) at site 1.
    println("\n=== Reference: site-1 operators ===")
    println("  Identity (3x3):")
    Id = Matrix{ComplexF64}(I, 3, 3)
    for r in 1:3
        println("    " * join([@sprintf("%+.4f%+.4fim", real(Id[r,c]), imag(Id[r,c])) for c in 1:3], "  "))
    end
    # exp(iπSy) for spin-1: Sy in canonical basis. Compute via op.
    Sy_op = op("Sy", sites[1])
    Sy_arr = Array(Sy_op, sites[1]', sites[1])
    # eigen-decompose for exp(iπ Sy)
    using_eig = eigen(Sy_arr)
    expSy = using_eig.vectors * Diagonal(exp.(im*pi.*using_eig.values)) * inv(using_eig.vectors)
    println("  exp(i*pi*Sy) (3x3):")
    for r in 1:3
        println("    " * join([@sprintf("%+.4f%+.4fim", real(expSy[r,c]), imag(expSy[r,c])) for c in 1:3], "  "))
    end
end
nothing
