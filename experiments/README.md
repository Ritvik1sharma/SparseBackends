# Dense vs aliased PHP benchmark

Head-to-head runtime and Hamiltonian-memory comparison of four ways to build and
apply the projected Hamiltonian `P†HP`, for the KL (Kitaev ladder) and PXP models.

## The four variants

| variant | project | operator | matvec |
|---|---|---|---|
| `orig_dense` | `tensornetworks/` — `packages/` ITensors, branch `min-edits` | dense PHP | stock |
| `sb_dense` | `SparseBackends/` — branch `sparse-edits` | dense PHP | stock |
| `sb_aliased` | `SparseBackends/` | `AliasedBlockSparse` PHP | `run_mode=:standard` |
| `sb_fused` | `SparseBackends/` | `AliasedBlockSparse` PHP | `run_mode=:fused` |

`orig_dense` is the control: it answers "did the modified packages change the
physics?". If `orig_dense` and `sb_dense` disagree on energy, nothing downstream
is trustworthy. `sb_aliased` / `sb_fused` share one operator and differ only in
the matvec kernel.

`orig_dense` lives in `experiments/manual_tests/` rather than here because it
must run under a **different Julia project** — the two ITensors forks cannot be
loaded in one process.

## The grid

Defined once in [`configs.jl`](configs.jl) (mirrored to
`experiments/manual_tests/configs.jl` — keep the two in sync).

| config | model | sites | maxdim | notes |
|---|---|---|---|---|
| `kl_min1_v26_bd40`   | KL  | 26  (nplaq 12) | 40  | PSIGN −1 |
| `kl_min1_v66_bd80`   | KL  | 66  (nplaq 32) | 80  | PSIGN −1 |
| `kl_min1_v130_bd100` | KL  | 130 (nplaq 64) | 100 | PSIGN −1 |
| `pxp_bd20`  | PXP | 100 | 20  | ground + first excited (weight 20) |
| `pxp_bd60`  | PXP | 100 | 60  | ground + first excited |
| `pxp_bd120` | PXP | 100 | 120 | ground + first excited |

15 sweeps, seeds 0/1/2. 6 configs × 3 seeds × 4 variants = **72 runs**
(108 DMRG runs, since PXP does two).

The operators are copied verbatim from the validated drivers
`test_sparse_ham/test_check_working_aliased.jl` (KL) and
`test_sparse_ham/test_pxp_aliased.jl` (PXP), so this measures exactly the
operators those tests compare. The KL constraint is built by **merging** all
plaquette MPOs into one `P` and sandwiching once — matching the SparseBackends
path, not the sequential per-plaquette sandwich in `experiments/kl/*/*/v*.jl`.

## Running

`run_group.jl` is the single entry point. It takes the config id, a
comma-separated variant list and a seed:

```bash
cd tensornetworks

# one variant
julia --project=SparseBackends SparseBackends/experiments/run_group.jl \
      kl_min1_v26_bd40 sb_aliased 0

# all three sb variants in ONE process -- what Sherlock runs
julia --project=SparseBackends SparseBackends/experiments/run_group.jl \
      kl_min1_v26_bd40 sb_aliased,sb_fused,sb_dense 0

# the original-ITensors baseline lives in the other project tree
julia --project=. experiments/manual_tests/run_group.jl kl_min1_v26_bd40 0
```

Grouping variants matters: SparseBackends' codegen (~10 min, per-process because
the kernels specialise on runtime-determined tensor shapes) is then paid once
instead of once per variant.

There are deliberately **no per-(config, variant) wrapper files**. They existed
until commit `87d4987` -- 39 of them, each a single `run_config(id, variant)`
call -- and were removed because nothing referenced them, the Sherlock path
(`sherlock_scripts/run_scripts/run_bench_task.sh`) has always used `run_group.jl`,
and they silently went stale every time a config was added. To add an experiment,
add one entry to `configs.jl`; no new file is needed.

Variant order is deliberate: each writes its JSON the moment it finishes, so an
OOM in `sb_dense` at the large configs cannot lose the aliased results.

Env knobs: `BENCH_SEED`, `BENCH_OUTDIR`, `BENCH_BLAS_THREADS`, and
`BENCH_NSWEEPS` / `BENCH_MAXDIM` (smoke tests only — both are recorded in the
JSON so an overridden run is never mistaken for a production one).

## On Sherlock

```bash
cd /home/groups/sachour/rsharma3/hpc_artifact/tensornetworks
./sherlock_scripts/submit_sparse_bench.sh --configs kl_min1_v26_bd40 --dry-run
./sherlock_scripts/submit_sparse_bench.sh --configs kl_min1_v26_bd40
```

One array task per (config, seed); within a task the variants run **serially on
one node**, so the variant-to-variant ratio is hardware-identical. Seeds scatter
across nodes, so seed spread is the honest error bar. Every task is pinned to the
same CPU class via `--constraint`; see `sherlock_scripts/sparse_bench_config.sh`.

Collect:

```bash
python3 sherlock_scripts/run_scripts/collect_sparse_bench.py \
        $SCRATCH/sparse_bench/results -o summary.csv --sacct-job <JOBID>
```

## Threading

`CPUS_PER_TASK` is the single core budget and is passed to every variant as
`BENCH_BLAS_THREADS`, so all four scale through the same mechanism. It is
recorded in every result JSON — never compare rows with different values.

`SB_ALIASED_NTHREADS` is left at **1** on purpose. Above 1 the aliased matvec
takes a task-parallel path that pins BLAS to 1 internally and is documented in
`contract_aliased_dense_shared.jl` as FP-close but *not* bit-identical to serial
(the reduction reorders sums) — unwanted inside a correctness comparison. At the
default the aliased kernel goes through `_contract_dense_serial_outstat!`, whose
`mul!` calls use the full BLAS thread count like everything else. Note that COO
is a *construction-time* backend only; the DMRG matvec is dense GEMM over
`AliasedBlockSparse` templates.

## What is measured

Timing excludes codegen: each run does one warmup sweep at `maxdim=10` (discarded),
then two GC passes, then the timed run. Per-sweep times come from a `SweepTimer`
observer — `dmrg` calls `checkdone!` once per sweep in both ITensorMPS forks, so
the instrumentation is identical on both sides.

Per run, in `<config>__<variant>__seed<n>.json`:

- **timing** — `total_seconds`, `sweep_seconds[]`, `mean_sweep_seconds`, `build_seconds`
- **energy** — per-sweep energies, plus final `energy_php` and `energy_raw_H`
  (`inner(psi', H_raw, psi)` against the *unprojected* H, which is the
  cross-variant comparable number)
- **memory** — `php.summarysize_bytes` (the Hamiltonian itself, exact per variant),
  `allocated_bytes` and `gc_seconds` from `GC_Diff`, and for the aliased variants
  a `php.aliased_footprint` with block/template counts and compression ratios
- **provenance** — host, CPU model, thread counts, SLURM ids, git commit

Caveat: `maxrss_bytes` is a *process* high-water mark and cannot be reset, so in a
packed `run_group.jl` process only the first variant's value is clean. Use
`php.summarysize_bytes` for per-variant Hamiltonian memory and SLURM `MaxRSS` for
whole-task peak; a dedicated one-variant-per-process run is needed for clean
per-variant peak RSS.
