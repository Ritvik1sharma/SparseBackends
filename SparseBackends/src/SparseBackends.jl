module SparseBackends

# export whatever should be public:
export NewBlockSparseSorted, blocksparse_from_dense, to_dense
export COOTensor, coo_from_dense, to_dense
export AliasedBlockSparse, to_blocksparse, to_dense, contract_aliased!, compression_ratio
export WrappedAliasedBlockSparse, contract_aliased_itensor,
       contract_and_fuse_links_aliased, contract_coo_dense_aliased

include("base.jl")

# ── BlockSparse ──────────────────────────────────────────────────────────────
include("blocksparse/storage.jl")
include("blocksparse/conversions.jl")

# ── COO ──────────────────────────────────────────────────────────────────────
include("coo/storage.jl")
include("coo/conversions.jl")

# ── Contraction dispatch + BlockSparse/COO kernels ───────────────────────────
# contract.jl also defines: Label, _dims, _prefix_lin, aligned_A_to_Cprefix,
# _cdense_grouping_and_orders, _advance_run, _cmp_join_tuple (via sub-includes)
include("tensoralgebra/contract.jl")

# ── AliasedBlockSparse ───────────────────────────────────────────────────────
# Must come after blocksparse/storage.jl (needs _check_no_cross_perm, _prefix_lin,
# NewBlockSparseSorted) and after contract.jl (needs Label, _dims).
include("aliased/storage.jl")
include("aliased/conversions.jl")

# ── AliasedBlockSparse contraction kernels ───────────────────────────────────
# Must come after aliased/storage.jl and after contract.jl helpers.
include("tensoralgebra/contract_aliased_coo_dense.jl")   # COOTensor × Dense → AliasedBS
include("tensoralgebra/contract_aliased.jl")              # AliasedBS × Dense / AliasedBS × AliasedBS (single label)
include("tensoralgebra/contract_aliased_shared.jl")       # AliasedBS × Dense / AliasedBS × AliasedBS (multi-label)

include("ops_factorize.jl")
include("tensor_wrappers.jl")
include("tensor_index.jl")
include("tensor_contraction.jl")
include("tensor_wrappers_aliased.jl")   # WrappedAliasedBlockSparse + top-level aliased contract functions
end
