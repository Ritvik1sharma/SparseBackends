# Aliased matvec permute handoff

**Goal for the next session:** find a way to remove the permute overheads in the
aliased-PHP × dense-ψ DMRG matvec. This doc gives the concrete per-step index
trace, the kernel's default output order at each step, and every lever the kernel
already has for reshaping the output (strided/direct write, permB, permute_back,
NRED output ordering, offline prepermute) — plus what has already been ruled out.

## Setup that produced this trace
- Runner: `test_sparse_ham/test_pxp_aliased.jl` (PXP, aliased PHP, **dense ψ**), N=8, bd=10, seed 0.
- Trace facility: `do_trace` in `ITensorMPS.jl/src/abstractprojmpo/abstractprojmpo.jl:453`
  (manually re-enabled; fires once on the first aliased **bulk** bond — lands in the JIT warmup, so dims are small but index STRUCTURE = measured run).
- `SB_PREPERMUTE_H` is now **hardcoded ON** (prepermute of H dense tails; see §Levers).
- permB dump: `SB_PERM_PROFILE=1` (`--profile`). Logs: `results/bench_single_N8_bd10_20260709_114503/`.

Notation: each index printed as `(dim, "tags", plev)`. Physical labels for this
bond (2-site block on sites n=2,3): `l1`=left link, `l2`=middle link (internal),
`l3`=right link; `s2,s3`=S=1 physical sites; `FS.bondk`=fused-sparse H-MPO bond k.
Aliased H sites store a **sparse prefix** (site ket/bra + fused bonds) + a **dense
tail** (extra bra strands from the P″HP′ sandwich).

## The operands
```
φ (v)  : [ l1(3,p0),  s2(3,p0),  s3(3,p0),  l3(2,p0) ]                            dense ket
Lenv   : [ l1(3,p0 ket),  l1(4,p1 bra),  l1(3,p1 bra'),  FS.bond1(4,p0) ]         dense
H_b(2) : sparse-prefix[ s2(p0), s2(p1), FS.bond1, FS.bond2 ] + dense-tail[ l2(4,p1), l1(4,p1) ]   aliased (PA=4)
H_{b+1}(3): sparse-prefix[ s3(p0), s3(p1), FS.bond2, FS.bond3 ] + dense-tail[ l3(4,p1), l2(4,p1) ] aliased (PA=4)
Renv   : [ l3(2,p0 ket),  l3(4,p1 bra),  l3(2,p1 bra'),  FS.bond3(4,p0) ]         dense
```
(Note the H dense tails are in **descending link order** `[l2,l1]`, `[l3,l2]` — that
is the effect of the hardcoded prepermute; without it they were `[l1,l2]`,`[l2,l3]`.)

## Per-step index trace + default output + levers

The matvec is the left-fold `((((φ·Lenv)·H_b)·H_{b+1})·Renv)`, 4 steps.
`CURRENT_STEP` (`abstractprojmpo.jl:500`) tags each. Steps 1,4 are dense `Hv*it`
(plain ITensors, no SB kernel). Steps 2,3 are the aliased kernel
`SparseBackends.contract_aliased_itensor(it, Hv, :aliased, :dense; next_op=…)`
→ `contract_aliased_dense_to_dense!`.

### Step 1 — φ · Lenv  (dense `*`, contract `l1`)
```
IN  Hv = [ l1, s2, s3, l3 ]
OUT Hv = [ s2, s3, l3 | l1(4,p1), l1(3,p1), FS.bond1 ]     ← [φ survivors | Lenv survivors]
```
- **Default output:** ITensors' own contraction order = uncontracted-of-A then
  uncontracted-of-B. **NOT controllable** — a dense `*` re-derives its output order
  regardless of operand leg order. This is where B's interleaving is *born*.
- **Levers:** none in-kernel (it's ITensors `*`). The only way to control this
  output is to route it through a layout-aware kernel (i.e. make Lenv aliased →
  aliased×dense with `output_perm`), which is a **net loss** (see §Ruled out).

### Step 2 — · H_b  (aliased kernel)  — permB fires 100%
```
IN  Hv (=B) = [ s2, s3, l3, l1(4,p1), l1(3,p1), FS.bond1 ]
shared/contracted = { s2 (sparse key, from φ), FS.bond1 (sparse key, from Lenv), l1(4,p1) (dense, from Lenv) }
classification:  red_dense={ l1(4,p1) }  keepB={ s3, l3, l1(3,p1) }  shared_prefix={ s2, FS.bond1 }
permB = [4,2,3,5,1,6]   (reorder B → [red_dense ; keepB ; shared_prefix])
OUT Hv = [ l2(4,p1), s2(p1), l3, l1(3,p1), s3, FS.bond2 ]
```
- **Default (natural) output** = `canon_labels = [keepA ; keepB ; c_prefix]`
  (`contract_aliased_dense_to_dense.jl:263`) — H's kept legs first (BLAS-contiguous).
- **Contracted keys straddle both operands:** one sparse key from φ (`s2`), one from
  Lenv (`FS.bond1`), reduced dense leg from Lenv (`l1(4,p1)`). This is why no
  single-operand reorder makes permB identity.
- **Levers at this step:** permB (input reshape) + output-write mode (below).

### Step 3 — · H_{b+1}  (aliased kernel)  — permB identity (0%), permute_back FIRES
```
IN  Hv (=B) = [ l2(4,p1), s2(p1), l3, l1(3,p1), s3, FS.bond2 ]
shared/contracted = { s3 (key), FS.bond2 (key), l2(4,p1) (dense red) }
permB = identity   (step-2 output already canonical for step-3, via next_op/NRED)
OUT Hv = [ s2(p1), l3, l1(3,p1), l3(4,p1), s3(p1), FS.bond3 ]   ← desired labelsC
```
- **This is where `permute_back` fires** (560 calls in this run). Natural kernel
  order `[keepA ; keepB]` = `[s3(p1), FS.bond3, l3(4,p1) | s2(p1), l3, l1(3,p1)]`,
  but the desired `labelsC` (above) puts **keepB / ψ-legs first** and pulls the
  keepA dense-tail `l3(4,p1)` in front of `s3/FS.bond3` → **keepA is internally
  permuted** → the strided/direct-write path is rejected → GEMM into a canonical
  scratch then `permutedims!` to labelsC. See §Ruled out (strided write).

### Step 4 — · Renv  (dense `*`, contract `l3`,`FS.bond3`)
```
IN  Hv = [ s2(p1), l3, l1(3,p1), l3(4,p1), s3(p1), FS.bond3 ]
OUT Hv = [ s2(p1), l1(3,p1), s3(p1), l3(2,p1) ]     ← finished 2-site result (bra sites + outer links)
```
- **Default output:** ITensors `*` order; not controllable. Feeds `replacebond!` (SVD).

## The kernel's output-shaping options (what a solver can actually tune)

Location: `SparseBackends/src/tensoralgebra/contract_aliased_dense_to_dense.jl`.

1. **permB — input B reshape** (`:197`). Reorders B into `[red_dense ; keepB ; shared_prefix]`
   for the batched GEMM. Cost: `permutedims!` of the whole B (`:230`) — OR **free**
   when it's already in that order (`Bp = B`, `:205`). **Binary: identity (free) or
   full copy.** No strided/partial fast path for B.

2. **Output write mode** (`:277–347`):
   - **Direct / strided write (skips permute_back):** allowed when `perm_C == identity`
     OR one kept group is the unit-stride **leading run `[1..r]`** of labelsC and the
     other kept group is a **contiguous run `[p..p+c-1]`** (kernel order), everything
     else a c_prefix selector. Then each block's GEMM writes straight into a strided
     slice `view(C4,:,g,:,t)` (uniform `ldc = reshR·reshGAP`); keepB-leading served by
     transposed GEMM `Cᵀ=BᵀAᵀ`. Handles **block PLACEMENT** (gaps/strides), incl. a
     c_prefix axis sitting between kept groups.
   - **permute_back fallback** (`:340–347` + write-back): used ONLY when a kept group
     is **INTERNALLY PERMUTED** (its column run is not contiguous-ascending in kernel
     order ⇒ non-uniform stride ⇒ no single GEMM can scatter it). GEMM → canonical
     `[keepA;keepB;c_prefix]` scratch → `permutedims!` to labelsC.
   - **Key limit:** strided write does block placement, **not intra-block permutation.**

3. **Choosing labelsC / perm_C — the output order the kernel targets**
   (`tensor_wrappers_aliased.jl`): NRED hints derive it from `next_op`:
   - `_canon_keepB_red` (`:930`): sorts sparse-prefix + keepB-dense by `is_reduced_next`
     (reduced-next LAST, cheap key/permB reorder), **keeps keepA-dense in A-order** to
     minimize THIS step's input permA — accepting the keepA "terminal ×Renv residual"
     (the permute_back above).
   - `_canon_labels_for_next` (`:806`) / `_canon_inds_for_next_A` (`:846`): place the
     axes the NEXT step contracts to make the next step's permB/permA identity.
   - Cost asymmetry (critical): reordering the **sparse prefix** = cheap key-tuple
     remap (no data move); reordering the **dense tail** = `permutedims` of templates
     (expensive). NRED exploits this.

4. **Offline prepermute** (`test_sparse_ham/aliased_helpers.jl:181`,
   `prepermute_aliased_mpo!`, now hardcoded ON): permutes the **constant H's dense
   tails once** (descending link order) so the per-matvec **permute_A ≈ identity**.
   Measured: `add.permute_A` alloc −81% (5.34→0.99 MiB/matvec), **bit-identical E**.
   Does NOT affect permB or permute_back (different tensors).

5. **Env reorder** (`abstractprojmpo.jl:966`, `_reorder_env_for_aliased`): reorders the
   dense Lenv/Renv to `[ket, dense-H, sparse-H, bra]`. Intended to make step-2 permB
   identity — **measured ineffective** (its own comment: "the fire-counter says it does
   not") because step-1's dense `*` re-sorts its output (see §Ruled out).

## Overhead sizes (bd=40 N=12 profile, the flagship size)
Within the ~1.46 s matvec: `main_loop` GEMM 283 ms (dedup≈1 ceiling) · `ali_dense_alloc`
220 ms / **1.19 GiB = 77% of allocations** · `permute_B` 122 ms · `permute_back` 56 ms ·
setup 59 ms · `permute_A` ~12 ms (now ~halved + −81% alloc by prepermute). So the
permute budget is ~12% of the matvec; alloc + GEMM dominate.

## Already ruled out (don't re-try without a new idea)
- **Strided write to kill the step-3 permute_back:** NO. It's an *intra-block* keepA
  permutation (dense-tail `l3` pulled forward), which a single GEMM cannot scatter —
  the strided path explicitly rejects it (`:297`).
- **Reordering step-1 output / Lenv to make step-2 permB identity:** NO. Step 1 is a
  dense ITensors `*` whose output order is not controllable; `_reorder_env_for_aliased`
  tried and measured no reduction in step-2 permB. Also permutedims cost is ~pattern-
  invariant (full copy), so a "nicer" permB isn't cheaper — only identity is free.
- **Making Lenv/Renv aliased to gain output-order control (aliased×dense→aliased/dense):**
  net LOSS — env has no channel dedup (`project_aliased_env_no_sparsity`: 1.67× larger,
  saturated channels) and the aliased kernel is ~1.4× slower on dedup≈1 operands
  (`project_aliased_eigsolve_dedensify_slower`). Would slow the efficient dense env
  steps (~528 ms) to save ≤178 ms of permutes.
- **NRED conservation:** the two reduced dense legs that get contracted at step 4
  (`l3(2,p0)` from the vector, `l3(4,p1)` from H) are of different origin and straddle;
  the `[keepA;keepB;c_prefix]` canonical mode can only trail one block, so zeroing the
  step-3 output permute_back reopens a permute at the step-4/permA side. Documented as
  irreducible with the current single canonical mode (`project_nred_output_ordering`).

## Open questions for the next session
1. Can a **two-block / multi-trailing output mode** (not the single `[keepA;keepB;c_prefix]`)
   let BOTH straddling reduced legs trail, breaking the NRED conservation?
2. Can the step-3 `labelsC` keepA-internal order be chosen to keep keepA contiguous
   (enabling strided write, no permute_back) **without** creating a step-4 cost —
   given step 4 is a dense `*` here (not the aliased kernel, unlike the Path-B world
   where NRED's conclusion was drawn)? This may differ in the dense-ψ path.
3. The biggest prize is not permutes at all: `ali_dense_alloc` (1.19 GiB, 77% of
   allocations) and the dedup≈1 GEMM ceiling. Can the densify buffer be pooled/reused
   (cf. `project_matvec_intermediate_pooling`) or the GEMM batched?

## How to regenerate this trace
```
cd edited_packages
./bench_daemon/run_bench.sh --N 8 --bd 10 --model pxp --seeds 0 --profile
# do_trace is at abstractprojmpo.jl:453 (fires once on first aliased bulk bond);
# per-step indices + [permB #N] dumps + perm_profile land in the pxp_*.log.
```
