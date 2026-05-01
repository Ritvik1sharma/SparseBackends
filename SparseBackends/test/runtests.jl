using Test
using SparseBackends

@testset "SparseBackends" begin
  # include("blocksparse/test_convertor.jl")
  # include("blocksparse/test_storage.jl")
  # include("coo/test.jl")
  # include("tensoralgebra/coo.jl")
  # include("tensoralgebra/coo_bs.jl")
  # include("tensoralgebra/bs_bs.jl")
  # include("tensoralgebra/coo_dense.jl")
  # include("tensoralgebra/bs_bs_multi_labels.jl")
  include("tensoralgebra/bs_dense.jl")
  # edited_packages/SparseBackends/test/tensoralgebra/bs_bs_multi_labels.jl
end