# Matvec index-order & data-movement analysis (dense-ψ, aliased PHP)

Ground-truth index orders from the N=8 bd=10 `do_trace` dump
(`results/bench_single_N8_bd10_20260709_114503/pxp_N8_bd10_seed0.log`) and the
H-build log. Dims generalized to the flagship: **ψ-bond `D=40`, physical `d=3`,
channel (MPO strand / fused-sparse) `c=4`**. Only ψ-bonds grow with maxdim;
channels stay ≤4 (sparse).

## 0. Legend
`l1,l3` = outer ψ links; `l2` = internal (MPO) link; `s2,s3` = physical sites.
plev: `⁰`=ket(p0), `¹`=MPO-bond bra strand(p1), `³¹/²¹` = ψ-bra output bond.
ψ-bond legs (dim D): `l1⁰, l1³¹, l3⁰, l3²¹`.  channels (dim c): `l1⁴¹,l2⁴¹,l3⁴¹,F1,F2,F3`.

## 1. Ground-truth index orders

### psi (dense MPS) → φ
`φ = psi[b]·psi[b+1]` = **`[l1⁰(D), s2⁰(d), s3⁰(d), l3⁰(D)]`** (trace `v.inds`).
psi site convention: `[left-link, site, right-link]`.

### Dense PHP H[k] (site 3, pre-fuse) — 8 legs, the P″HP′ sandwich
`[s3⁰(d), s3¹(d), l2⁰, l2², l3⁰, l3², l2¹(c), l3¹(c)]`
(each link carries 3 strands: ket p0, MPO-bond p1 dim-c, bra p2.)

### Aliased PHP H[k] (post-fuse + prepermute) — prefix P=4 + dense tail
- H_b (site 2): prefix `[s2⁰, s2¹, F1, F2]` + tail `[l2⁴¹, l1⁴¹]`
- H_{b+1} (site 3): prefix `[s3⁰, s3¹, F2, F3]` + tail `[l3⁴¹, l2⁴¹]`  (tail descending-link = prepermute)
Fuse: `F_k = fuse(ket+bra ψ-projector strands)` stays dim c; the p1 MPO-bond strands `lk¹` are the dense tail (dim c).

## 2. Per-step trace — CURRENT defaults (what the log shows)

| step | op (kind) | contract | Hv AFTER (actual trace order) | permute |
|---|---|---|---|---|
| 1 | ·L (dense `*`) | l1⁰ | `[s2⁰,s3⁰,l3⁰, l1⁴¹,l1³¹,F1]` | born wrong for step2 |
| 2 | ·H_b (aliased) | s2⁰,F1,l1⁴¹ | `[l2⁴¹,s2¹,l3⁰,l1³¹,s3⁰,F2]` | **permB=[4,2,3,5,1,6] FIRES** |
| 3 | ·H_{b+1} (aliased) | s3⁰,F2,l2⁴¹ | `[s2¹,l3⁰,l1³¹,l3⁴¹,s3¹,F3]` | permB=id; **permute_back FIRES** |
| 4 | ·R (dense `*`) | l3⁰,F3,l3⁴¹ | `[s2¹,l1³¹,s3¹,l3²¹]` | internal `*` order |

Sizes (D=40,d=3,c=4): φ = D²d² = **14.4K**; T1=T2=T3 = D²d²c² = **230.4K** (16× φ).
Per-block GEMM step2/3 = M(keepA=c=4)×K(red=c=4)×N(keepB=D·D·d=4800) — **memory-bound**.
Measured churn (N12/bd40): step2 permB 100%(968), step3 permute_back ~82%(792).

**Round-trip mismatch:** input φ = `[l1,s2,s3,l3]`; output(noprime) = `[s2,l1,s3,l3]`
→ KrylovKit vecops auto-permute every iteration (extra churn, off-profile).

## 3. Kernel knobs (aliased×dense→dense) and dense-step knobs

| id | knob | cost model |
|---|---|---|
| K1 | permB (transpose B → [red;keepB;shared]) | full copy of B (230K) + alloc; free iff already that order |
| K2 | **strided-read-B** (NOT BUILT) | 0 pre-pass; needs red & keepB each uniform-stride, one unit-stride |
| K3 | direct/strided-write-C | 0; needs one kept group leading-run + other contiguous-run |
| K4 | permute_back (transpose C → labelsC) | full copy of C (230K) + scratch + zero |
| K5 | labelsC / NRED | sparse-prefix reorder = free key-remap; dense-tail reorder = template copy |
| K6 | prepermute-A (H tail, offline) | once; makes permA≈id (already ON) |
| K7 | φ reorder (once/bond) | ~0/matvec (amortized); OFF for dense-ψ today |
| K8 | L/R construction order (constant) | free |
| K9 | step-1/4 custom dense contraction | choose operand order; strided-write epilogue |
| K10 | segmented strided read/write (NEW idea) | trades 1 transpose for ~(trap-dim) small GEMMs |

## 4. Data-movement per operand: transpose vs strided-read
permute path B-traffic = read 230K (strided) + write 230K + GEMM-read 230K = **690K**.
strided-read = GEMM-read 230K (strided) only = **230K → 3× less**, and −230K alloc.

## 5. The optimization walk (fold is FIXED; minimize transposes)

### Step 1 → 2: KILLABLE (→ 0 transposes)
Set **φ = `[l1⁰, s3⁰, l3⁰, s2⁰]`** (K7, s2⁰ last) and **L = `[l1⁰, l1⁴¹, F1, l1³¹]`** (K8),
do step-1 as GEMM with **L as rows, φ as cols** (K9). Natural GEMM output =
`[L-cols | φ-cols]` = **`[l1⁴¹,F1,l1³¹ , s3⁰,l3⁰,s2⁰]`** = "Layout C". Then at step 2:
```
red   = l1⁴¹  @pos1  stride 1      (K unit-stride ✓)
keepB = {l1³¹,s3⁰,l3⁰} nested stride16 (16·40=640·3=1920 ✓ uniform)
shared= {F1(st4), s2⁰(st76800)}  → per-block offsets only
⇒ B_mat = 4×4800, ldb=16 → stock strided BLAS (K2). NO permB.
```
The trapped shared index s2⁰ is pushed OUT of keepB by the φ reorder — no interleave,
step-1 is a plain 2-block GEMM. Bit-identical (accumulation order unchanged).

### Step 2 → 3: ONE transpose FORCED (derived, not assumed)
step-3 contracts s3⁰, which was a **keepB spectator** at step 2. step-3-keepB =
`{l1³¹,l3⁰,s2¹}` spans step-2's keepB (`l1³¹,l3⁰`) AND c_prefix (`s2¹`), with the
trap s3⁰ (step-2 keepB) and F2 (step-2 c_prefix) interleaved.
- If step 2 **direct-writes** its natural `[keepA;keepB;c_prefix]` → step-3-keepB is
  broken by s3⁰ → step 3 needs permB.
- If step 2 emits step-3's `[red;keepB;shared]` (permB@3 = id) → keepB is broken by the
  c_prefix leg s2¹ → step 2 needs permute_back.
Either way **one ~230K transpose of T2**. This is the real "NRED conservation",
now derived from the actual index split. *Only* escape: **K10** — treat keepB as two
uniform-stride segments across the trap (s3⁰ dim d=3) → ~3 small GEMMs/block instead of
1 transpose. Memory-bound GEMMs vs a 230K copy: plausibly a net win, untested.

### Step 3 → 4: KILLABLE (→ 0 transposes)
Let step 3 **direct-write** its natural `[keepA=l3⁴¹; keepB; c_prefix]` (no permute_back),
and make **step 4 a custom dense contraction (K9)** that **strided-reads** that order.
Step 4 contracts {l3⁰,F3,l3⁴¹} — gather via strides, no pre-transpose.

### Step 4 output: cycle fixed-point
Output must equal φ's order (K7) after noprime so KrylovKit doesn't auto-permute.
Step 4 (mine) emits it via a strided-write epilogue (fused, ≤1 pass) OR pick the
canonical order to be a natural step-4 GEMM output. The once/bond φ reorder (K7)
absorbs any residual for the *next* bond cheaply.

## 6. Net result
Transposes per matvec: **2 forced (permB@2 + permute_back@3) → 1 forced (2→3 handoff)**,
and that last one is a K10 candidate. Everything else becomes strided (3× less B/C
traffic) or once-per-bond amortized. Plus the round-trip auto-permute is removed by
fixing the cycle order.

Orthogonal (not permutes): dedup = **2.697** (measured) ⇒ template reads bounce and
there's ~2.7× GEMM-amplification unclaimed by the one-mul!-per-block loop
(revive legacy template-load-once+axpy, NOT naive batched-GEMM which regressed 2.9×).

## 7. Build order (all layout-only / bit-identical unless noted)
1. K2 strided-read-B in `contract_aliased_dense_to_dense!` (mirror of strided-write) — kills permB@2 once fed Layout C.
2. K7 dense sibling of `reorder_aliased_by_rank` + ungate for dense-ψ; K8 L/R order; K9 step-1 operand order → produce Layout C.
3. K9 step-4 custom + step-3 direct-write → kill permute_back@3.
4. K10 (segmented) for the forced 2→3 transpose — measure vs leaving it.
