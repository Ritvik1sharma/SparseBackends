# In-Kernel Image-of-P Filter — Design + Scaffolding

## Goal

Eliminate the per-step recast in Path B DMRG by making the BS×Dense contraction
kernel write only to output keys present in a pre-supplied allowed-key set
(template's keys = image-of-P). This:

1. Prevents intermediate state blowup (kernel doesn't compute non-image-of-P
   outputs).
2. Eliminates the dense recast cost (output already aligned to template).
3. Makes B_op exactly Hermitian on image-of-P → Lanczos becomes valid.

Estimated impact (extrapolating from N=4 / N=16 profiles):
- Recast cost: 5.8% → ~0% (eliminated).
- cpb.contract cost: 8.9% → ~3-5% (smaller intermediates).
- Krylov iter count: ~30 → ~10 via Lanczos.
- **Net**: sparse may approach dense performance at N=16.

## Mismatch with current architecture

The kernel currently produces output BS with a structure determined by the
contraction's label algebra (which axes are reduced/kept/shared). Template's
structure may have:

- Different sparse/dense axis split (P_template ≠ P_kernel_output)
- Different axis ordering
- Subset of keys (image-of-P-only vs all combinatorially-possible)

A bare HashSet filter requires the FIRST TWO to already match. Otherwise we
also need axis-classification conversion (fission/fusion).

## Proposed implementation, three layers

### Layer 1: BS-key filter (independent of axis matching)

```julia
# Drop blocks of `bs` whose keys are not in `allowed`. In-place re-key.
# Cost: O(nblocks).
function filter_bs_keys!(bs::NewBlockSparseSorted, allowed::AbstractSet)
    nblocks = length(bs.keys)
    new_keys = similar(bs.keys, 0)
    new_data = similar(bs.data, 0)
    bsz = bs.blksize
    sizehint!(new_keys, length(allowed))
    sizehint!(new_data, length(allowed) * bsz)
    @inbounds for i in 1:nblocks
        k = bs.keys[i]
        if k in allowed
            push!(new_keys, k)
            base = (bs.ids[i] - 1) * bsz
            append!(new_data, view(bs.data, (base+1):(base+bsz)))
        end
    end
    bs.keys = new_keys
    bs.ids  = collect(1:length(new_keys))
    bs.data = new_data
    return bs
end
```

Use when: Cw and Tw have **same** axis split + ordering. Drops non-allowed
keys efficiently. **Doesn't help our concrete case (blksize differs).**

### Layer 2: Pre-allocated output with template structure (controls kernel output shape)

Modify `wc.alloc_bs` (`tensor_wrappers.jl:1017`) to take a hint:

```julia
C = if template_keys !== nothing
    # Allocate Cw with template's structure: blksize, P, dims_perm from template.
    WrappedBlockSparse(TC, dimsC_perm, denseLinksC_perm, indsC_perm;
                       initial_keys=template_keys)
else
    WrappedBlockSparse(TC, dimsC, denseLinksC, indsC)
end
```

The kernel then writes into a pre-keyed buffer. **Requires kernel to do key
lookup at each write site** (Layer 3).

### Layer 3: Kernel-side key lookup

Modify `contract!` in `contract_bs_dense.jl:340` (and the prefix/dense_bd
variants) to use `C`'s existing keys, skipping output blocks for keys not in
`C`.

Concretely in `contract_dense_bd!`: instead of generating output keys from
A's keys, look up each candidate output key in C's keys map. If absent, skip.

```julia
# At the per-block write site:
out_key = compute_output_prefix(A_block.keys[i], ...)
out_idx = get(C_key_map, out_key, 0)
if out_idx == 0
    continue  # skipped — not in image-of-P
end
# write to C.data[(out_idx-1)*blksize+1 ... out_idx*blksize]
```

`C_key_map = Dict{NTuple,Int}` built once per call.

## Caveats / risks

1. **Kernel correctness**: existing kernels currently mutate C's keys/data
   dynamically. Layer 3 requires they respect a pre-keyed C. Need careful
   audit of `contract_prefix_outer_bd!` and `contract_dense_bd!` to see
   whether such hooks already exist or must be added.
2. **Axis ordering**: template's axis order may differ from kernel's natural
   output order. Either: (a) permute template to match, OR (b) permute
   kernel output post-contraction (cheap if just BS permute). Pick (a) for
   simpler kernel logic.
3. **Reduction labels**: when multiple A blocks contribute to one output
   block (reduction), skipping the output block requires skipping ALL its
   contributors. The check at the write site naturally handles this — but
   may inflate A-iteration cost slightly.
4. **`product(PH, phi)` chain**: each step in ProjMPO.contract would need
   its own filter. The template at each step is different (mid-loop tensors
   have transient indices). Practical scheme: only filter at the END of the
   H × phi chain, not internally — matches current strategy.

## Estimated work

| Task | Time |
|---|---|
| Layer 1 (`filter_bs_keys!`) | 1 hour |
| Layer 2 (template-structured alloc) | 4 hours (WrappedBlockSparse constructor variant + plumbing through `wrapped_contract`) |
| Layer 3 (kernel key lookup) | 1-2 days (audit + modify two kernel functions, regression tests) |
| Integration with `contract_preserve_bs` (template_keys kwarg) | 2 hours |
| B_op restructure: no-internal-recasts + final filter-only fast path | 1 hour |
| Lanczos toggle + correctness validation | 2 hours |
| N=4 + N=16 profile | 1 hour |
| **Total** | **~3-4 days** |

## Phased rollout

**Phase A** (today, ~2 hours): Layer 1 only. Use as a faster recast fallback
when axis split happens to match. Modest gain, low risk.

**Phase B** (next, 1 day): Layer 2 + Layer 3 for the dominant kernel path
(`contract_dense_bd!`). Test correctness at N=2-4. Profile.

**Phase C** (final, 1-2 days): Cover all kernel paths, full Path B
restructure, Lanczos toggle, N=16 validation.
