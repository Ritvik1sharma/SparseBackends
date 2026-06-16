# Wrapper to make test_pxp_check.jl runnable: define the helpers it depends on.
using ITensors, ITensorMPS, Random, LinearAlgebra

function itensor_from_nonzeros(dims::NTuple{N,Int}, coords::Vector; left::Bool=false) where {N}
    A = zeros(Float64, dims...)
    for c in coords
        @assert length(c) == N
        A[(c .+ 1)...] = 1.0
    end
    return A
end
bind_to_idx(A::AbstractArray, idxs::Index...) = ITensor(A, idxs...)

include("test_pxp_check.jl")
