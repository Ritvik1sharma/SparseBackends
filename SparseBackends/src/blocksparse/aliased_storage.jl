# blocksparse/aliased_storage.jl
#
# DEPRECATED: contents have been reorganised into:
#
#   src/aliased/storage.jl      — AliasedBlockSparse struct, helpers, permutedims
#   src/aliased/conversions.jl  — to_blocksparse, to_dense, Base.Array
#
# The COO × Dense → AliasedBlockSparse contraction kernel has moved to:
#   src/tensoralgebra/contract_aliased_coo_dense.jl
#
# This file is intentionally empty; all includes are now in SparseBackends.jl.