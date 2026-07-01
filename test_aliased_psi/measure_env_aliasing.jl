# Standalone experiment: build the FIRST aliased env step
# (H × dag(prime(psi)) where psi is the aliased input). Then count how much
# alias dedup the output COULD have if axes were classified differently.
#
# Method: compute the dense result, then for each candidate "sparse prefix"
# choice (set of axes), partition the dense tensor into slices indexed by
# those axes and count unique slices (within tolerance). The ratio
# n_blocks/n_unique_slices = achievable aliased dedup under that
# classification.

using SparseBackends, ITensors, ITensorMPS
using Random
using Printf

include("../test_sparse_psi/utils.jl")

function build_setup(N::Int, psign::Int)
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
        t += 0.5*psign, "exp(i*pi*Sy)", 2*j-1, "exp(i*pi*Sx)", 2*j,
                   "exp(i*pi*Sx)", 2*j+1, "exp(i*pi*Sy)", 2*j+2
        push!(os2, t)
    end
    ConsOps1 = [clean!(MPO(os2[j], sites, [2*j-1, 2*j, 2*j+1, 2*j+2])) for j in 1:N]
    function mulMPO(A, B); Bp = prime(B, "Site"); replaceprime(contract(A, Bp, :coo, :coo), 2 => 1); end
    P_sparse = ConsOps1[1]; for j in 2:length(ConsOps1); P_sparse = mulMPO(P_sparse, ConsOps1[j]); end
    H        = MPO(os, sites)
    psi0     = random_mps(sites)
    psi_ali  = replaceprime(contract(P_sparse, copy(psi0), :coo, :aliased; denseLinksB=0), 1 => 0)
    return H, psi_ali, P_sparse
end

# Count unique slices of a dense Array along the dimensions given by `prefix_axes`.
# Returns (n_keys, n_unique, max_compression_ratio).
function count_unique_slices(A::AbstractArray, prefix_axes::Vector{Int}; atol::Float64=1e-10)
    sz = size(A)
    N = ndims(A)
    tail_axes = setdiff(1:N, prefix_axes)
    # Permute so prefix dims come first.
    perm = vcat(prefix_axes, tail_axes)
    Aperm = permutedims(A, perm)
    prefix_dim = prod(sz[i] for i in prefix_axes)
    tail_dim = prod(sz[i] for i in tail_axes)
    A2 = reshape(Aperm, prefix_dim, tail_dim)

    # Find unique rows (each row = one slice indexed by prefix coords).
    n_keys = 0  # nonzero slices
    seen_slices = Vector{Vector{eltype(A)}}()
    seen_ids = Int[]  # which existing slice (if any) each new slice maps to
    for r in 1:prefix_dim
        slice = A2[r, :]
        nrm = sum(abs2, slice)
        if nrm < atol^2
            continue  # all-zero slice
        end
        n_keys += 1
        # Check if this slice matches any seen one (up to a scalar multiple).
        matched = 0
        for (j, s_other) in enumerate(seen_slices)
            # Check parallel: slice = α * s_other for some scalar α.
            # Find the largest entry of s_other; compute α; check residual.
            i0 = argmax(abs.(s_other))
            if abs(s_other[i0]) < atol
                continue
            end
            α = slice[i0] / s_other[i0]
            resid = sum(abs2, slice .- α .* s_other)
            if resid < atol^2 * sum(abs2, slice)
                matched = j
                break
            end
        end
        if matched == 0
            push!(seen_slices, slice)
            push!(seen_ids, length(seen_slices))
        else
            push!(seen_ids, matched)
        end
    end
    n_unique = length(seen_slices)
    return n_keys, n_unique, n_keys / max(n_unique, 1)
end

function densify(T::ITensor)
    if ITensors.has_external_storage(T)
        s = ITensors.get_external_storage(T)
        if s isa SparseBackends.WrappedAliasedBlockSparse
            return ITensors.itensor(SparseBackends.to_dense(s.aliased), s.inds...)
        elseif s isa SparseBackends.WrappedBlockSparse
            return ITensors.itensor(SparseBackends.to_dense(s.blocksparse), s.inds...)
        end
    end
    return T
end

function main()
    N = 2
    H, psi, _ = build_setup(N, +1)

    println("=== psi[1] structure ===")
    ali = psi[1].tensor.data.aliased
    @printf("psi[1]: ndims=%d  dims=%s  nb=%d  nt=%d  blksize=%d\n",
        ndims(ali), ali.dims, length(ali.keys), ali.n_templates, ali.blksize)
    @printf("keys = %s\n", ali.keys)
    @printf("alias_ids = %s\n", ali.alias_ids)
    @printf("scalars = %s\n", round.(ali.scalars; digits=4))

    println("\n=== Compute H[1] * dag(prime(psi[1])) (DENSIFYING psi first) ===")
    # Bypass the aliased kernel entirely: densify psi[1] before contract.
    H1 = H[1]
    psi1_dense = densify(psi[1])
    psi1_p_dag = dag(prime(psi1_dense))
    R = H1 * psi1_p_dag
    R_dense = densify(R)
    R_arr = Array(R_dense, inds(R_dense)...)
    println("R has inds:")
    for (i, I) in enumerate(inds(R_dense))
        @printf("  axis %d: dim=%d  tags=%s  plev=%d\n", i, dim(I), tags(I), plev(I))
    end

    println("\n=== Sparse-prefix classifications (only non-degenerate: blksize > 1) ===")
    N_ax = ndims(R_arr)
    println("Each entry: prefix=(axes...)  blksize=...  → (nb=keys, nt=unique templates up-to-scalar, dedup=nb/nt)")
    println("(Filtering out blksize==1 cases: those are COO-of-scalars, not real template aliasing)")
    sz = size(R_arr)
    best = (axes=Int[], nb=0, nt=0, ratio=0.0, blksize=0)
    for prefix_axes_set in 1:(2^N_ax - 2)
        prefix_axes = [i for i in 1:N_ax if (prefix_axes_set >> (i-1)) & 1 == 1]
        tail_axes = setdiff(1:N_ax, prefix_axes)
        blksize = prod(sz[i] for i in tail_axes)
        if blksize <= 1
            continue  # degenerate: blksize==1 is COO-scalar storage, no real template dedup
        end
        nb, nt, ratio = count_unique_slices(R_arr, prefix_axes)
        prefix_descr = join(["axis$(i)(dim=$(sz[i]))" for i in prefix_axes], ", ")
        marker = ratio > 1.0 ? "  ★" : ""
        @printf("  prefix=[%s]  blksize=%d  nb=%d  nt=%d  dedup=%.2fx%s\n",
                prefix_descr, blksize, nb, nt, ratio, marker)
        if ratio > best.ratio
            best = (axes=prefix_axes, nb=nb, nt=nt, ratio=ratio, blksize=blksize)
        end
    end
    println()
    if best.ratio > 1.0
        println("Best non-degenerate dedup: prefix=$(best.axes)  blksize=$(best.blksize)  dedup=$(round(best.ratio,digits=2))x")
    else
        println("No non-degenerate classification yields >1x dedup. Output is intrinsically dense at the template level.")
    end
end

main()
nothing
