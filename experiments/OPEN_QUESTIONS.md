# Open questions — dense vs aliased PHP benchmark

Parked issues found while building and piloting the benchmark. None of them
affect the **timing** or **PHP-memory** columns, which are the deliverable: every
variant runs the same fixed `nsweeps` at the same `maxdim`, and `php_bytes` is a
direct per-variant measurement. They affect how much the **energy** columns can
be trusted as a correctness check.

Pilot data referenced below: `kl_min1_v26_bd40`, seed 0, 15 sweeps, bd40,
BLAS=5, Intel E5-2640v4 (`sh02-01n01`), SLURM 36901529_0.

---

## 1. ψ₀ differs between the baseline and the SparseBackends trees

**Status: open. Highest priority of the three — it is why energies cannot be
compared at fixed sweep count.**

Sweep-1 energies, seed 0:

| variant | sweep-1 energy |
|---|---|
| `orig_dense`  | −15.92073998 |
| `sb_dense`    | −15.94736267 |
| `sb_aliased`  | −15.94736268 |
| `sb_fused`    | −15.94736268 |

The three `sb_*` variants agree to 8 digits — same ψ₀. The baseline starts
somewhere else, so the two arms follow different convergence trajectories and
their energies at sweep 15 are not comparable. Last-sweep deltas confirm neither
is converged: `orig_dense` −1.42e-04, `sb_*` ≈ −6.2e-04.

**Not an RNG problem.** `random_mps` is byte-identical between the forks
(diffed), and `Random.seed!(seed)` is called immediately before it. The
divergence must be in the step after, which contains no randomness:

```julia
psi = replaceprime(ConsOps2[j] * psi, 1 => 0)   # deterministic linear map
normalize!(psi)
```

**Leading hypothesis:** `*` on MPO×MPS resolves to a `contract` whose truncation
defaults differ between `packages/` (min-edits) and `SparseBackends/`
(sparse-edits). Same input, different function, so seeding cannot help.

**Likely fix:** pin an explicit `cutoff`/`maxdim` on the projection in both
trees, exactly as was done for the sandwich (see item 2).

### The probe that will localise it

The two trees load different ITensors forks and **can never share a process**, so
tensors cannot be compared directly. Instead each side independently prints
quantities that are meaningful across processes — basis-independent scalars
(trace, Frobenius norm, bond dimension) or expectation values on a
**deterministic Néel product state**, so no random index IDs enter.

```bash
cd tensornetworks
julia --project=.              experiments/manual_tests/probe_psi0.jl   kl_min1_v26_bd40 0 > /tmp/probe_orig.txt
julia --project=SparseBackends SparseBackends/experiments/probe_psi0.jl kl_min1_v26_bd40 0 > /tmp/probe_sb.txt
diff /tmp/probe_orig.txt /tmp/probe_sb.txt
```

Files: [`probe_psi0.jl`](probe_psi0.jl) (this tree),
`experiments/manual_tests/probe_psi0.jl` (baseline tree), and the shared
sequence `experiments/psi0_probe_body.jl`. The SparseBackends side deliberately
uses the **dense** sandwich, so it is the direct counterpart of the baseline and
any difference is attributable to the packages, not to the aliased backend.

Read the **first** differing line:

| first divergence | meaning |
|---|---|
| `php_norm` / `php_tr` / `probe_neel_php` | the PHP **operators** genuinely differ → item 2 is unresolved and the baseline arm needs rethinking |
| `psiraw_*` | `random_mps` differs after all, contradicting the diff |
| `after_proj_1_*` | the deterministic MPO×MPS projection differs — the current hypothesis |

---

## 2. Is `is_ctn_compression=true` really equivalent to `cutoff=0.0`?

**Status: partially resolved, needs the probe above to confirm.**

`is_ctn_compression=true`
([mpo.jl:1138-1195](../ITensorMPS.jl/src/mpo.jl#L1138-L1195)) skips
`orthogonalize`, skips the `factorize` SVD, skips `truncate!`, and calls
`collapse_all_bonds!` — the **exact, untruncated** product. `packages/` has no
such kwarg, so the baseline uses the plain zip-up path.

The baseline originally truncated at `cutoff=1e-12`, which made it solve in a
smaller variational space (χ=76 vs 80, |ΔE| = 2.9e-02 at maxdim 10). It now uses
`cutoff = 0.0` (`ORIG_*_CUTOFF` in `experiments/manual_tests/models.jl`), after
which χ agrees (80 vs 80) and PHP memory nearly agrees (4.120 vs 4.155 MB).

**What is still unproven.** Both paths should represent the same operator, but in
**different gauges** — canonical (SVD) vs collapsed — so identical tensors are
not expected and per-site comparison is meaningless. Matching χ is weak evidence.
Worse, extrapolating the pilot's convergence deltas geometrically
(`sb_*` ratio ≈0.62 → ≈ −16.3439; `orig_dense` ratio ≈0.76 → ≈ −16.3460) leaves
them ~2e-3 apart *at convergence*, which gauge cannot explain. Either the
extrapolation is too crude (likely — convergence is not purely geometric) or the
operators really do differ. The `probe_neel_php` / `php_norm` / `php_tr` lines
settle it.

---

## 3. `SB_ALIASED_NTHREADS` is pinned to 1, which handicaps the aliased path

**Status: deliberate, but worth one measurement.**

The aliased matvec is **not** thread-hardcoded — the only
`BLAS.set_num_threads(1)` in the whole sparse-edits stack is inside
`_contract_dense_threaded!`
([contract_aliased_dense_shared.jl:328](../SparseBackends/src/tensoralgebra/contract_aliased_dense_shared.jl#L328)),
reached only when `SB_ALIASED_NTHREADS > 1 && Threads.nthreads() > 1`. At the
default the call goes to `_contract_dense_serial_outstat!`, whose `mul!`s use the
ambient BLAS thread count like every other variant. So `BENCH_BLAS_THREADS` is
fully in effect for all four variants — the comparison is fair in that sense.

But there is a real asymmetry. `blksize = 19`, 3312 blocks, 234 templates: the
aliased matvec is thousands of tiny GEMMs, which BLAS threads poorly, while dense
does a few large ones that scale well. The kernel's *own* task-parallelism over
those 3312 blocks is exactly what would help — and that is what
`SB_ALIASED_NTHREADS` enables.

It is pinned to 1 because that path is documented as FP-close but **not
bit-identical** to serial (the Phase-3 reduction reorders sums), which is
unwanted inside a correctness comparison.

**Worth doing:** a one-off `SB_ALIASED_NTHREADS=5` run at one config to measure
what the aliased path can do when it can use the cores. Report it separately from
the bit-identical grid, never merged into it.

---

## 4. `build_seconds` is contaminated by codegen for the first variant in a process

**Status: known, low priority.**

`run_group.jl` runs the three `sb_*` variants in one process to pay
SparseBackends' codegen once. Consequence: the first variant's `build_seconds`
includes JIT. Pilot, seed 0: `sb_aliased` 53.5 s vs `sb_fused` 0.1 s **for the
same operator**.

DMRG timings are unaffected — each variant runs its own discarded warmup sweep
before its timed run. Only `build_seconds` is affected, and only for the first
variant in each process. A clean build-time comparison needs one variant per
process.

---

## 5. Per-variant peak RSS is contaminated by process packing

**Status: accepted; a separate one-off run was agreed for this.**

`Sys.maxrss()` is a process high-water mark and cannot be reset, so in a packed
`run_group.jl` process only the first variant's `maxrss_bytes` is clean.

Use instead: `php.summarysize_bytes` (exact per variant — this is the Hamiltonian
memory number), `allocated_bytes` / `gc_seconds` from `GC_Diff` (exact per run),
and SLURM `MaxRSS` for whole-task peak (pilot seed 0: 2.8 GB).

---

## 6. Timing rows are only comparable at equal thread count

**Status: not yet enforced in the collector.**

`blas_threads` is recorded in every JSON, but
`collect_sparse_bench.py` does not refuse to mix rows with different values. The
pilot ran at **5** (dropped from 10 because the pinned E5-2640v4 class was
fragmented — 104 idle CPUs but only one node with a 10-CPU hole; at 5 CPUs eleven
nodes qualified). Any future run at a different thread count must not be pooled
with these rows. Add a guard before publishing anything.
