#!/bin/bash
# ==============================================================================
# run_bench.sh — aliased-PHP / dense-ψ benchmark battery (KL + PXP), driven
# through a persistent DaemonMode Julia server so ITensors + SparseBackends are
# compiled/loaded ONCE, not per run.
#
# Runners exercised (both live in test_sparse_ham/, both run DENSE PHP+dense-ψ
# FIRST, then ALIASED PHP+dense-ψ in the same invocation):
#   KL  : test_check_working_aliased.jl   (arg: N_plaq → 2N+2 sites, S=1)
#   PXP : test_pxp_aliased.jl             (arg: N sites, S=1; ground + excited)
#
# Each runner internally does a discarded 1-sweep JIT warmup, then the timed
# run at outputlevel=1 (which prints per-sweep "After sweep … time=…" lines).
# summarize_bench.py reports the mean per-sweep time EXCLUDING sweep 1, plus
# final energies and dense-vs-aliased |ΔE|.
#
# MATRIX (full):
#   KL  : N ∈ {12,64}   bd ∈ {40,80}    seeds {0,1,2}   10 sweeps   (S=1, +C sector)
#   PXP : N = 100       bd ∈ {40,160}   seeds {0,1,2}   10 sweeps   (ground+excited)
#     (bd sets BOTH mindim and maxdim — the runners use maxdim=mindim=bd.)
#
# --testing : tiny correctness pass — KL and PXP at N=8, bd=10, seed 0 only.
#
# Usage:
#   ./run_bench.sh --testing      # quick energy-correctness smoke test
#   ./run_bench.sh                # full battery (long!)
# Env overrides: NSWEEPS, DAEMON_PORT, OUT
# ==============================================================================
set -u

PROJ="/home/ritvik/temp/temp/edited_packages"
SELFDIR="$PROJ/bench_daemon"
TS="test_sparse_ham"
KL_FILE="$PROJ/$TS/test_check_working_aliased.jl"
PXP_FILE="$PROJ/$TS/test_pxp_aliased.jl"
CLIENT="$SELFDIR/dclient.jl"

TESTING=0; SINGLE=0; ARG_PROFILE=0
ARG_N=""; ARG_BD=""; ARG_SEEDS=""; ARG_MODEL="both"
while [ $# -gt 0 ]; do
  case "$1" in
    --testing)      TESTING=1; shift;;
    --profile)      ARG_PROFILE=1; shift;;            # + SB_PERM_PROFILE + SB_ALIAS_STATS
    --N)            ARG_N="$2"; shift 2;;
    --bd)           ARG_BD="$2"; shift 2;;
    --seeds)        ARG_SEEDS="$2"; shift 2;;         # e.g. --seeds "0 1 2" or --seeds 0
    --model)        ARG_MODEL="$2"; shift 2;;         # kl | pxp | both
    -h|--help)
      echo "Usage: $0 [--testing] [--N <n> --bd <bd>] [--seeds \"0 1 2\"] [--model kl|pxp|both]"
      echo "  (no flags)          full matrix: KL N∈{12,64} bd∈{40,80}, PXP N=100 bd∈{40,160}, seeds 0 1 2"
      echo "  --testing           N=8 bd=10 seed 0, both models"
      echo "  --N 12 --bd 40      single fixed config for both models (seeds default 0 1 2)"
      exit 0;;
    *) echo "unknown arg: $1 (try --help)"; exit 1;;
  esac
done
# --N together with --bd selects a single fixed (N,bd) config.
if [ -n "$ARG_N" ] || [ -n "$ARG_BD" ]; then
  [ -n "$ARG_N" ] && [ -n "$ARG_BD" ] || { echo "ERROR: --N and --bd must be given together."; exit 1; }
  SINGLE=1
fi

NSWEEPS="${NSWEEPS:-10}"
PORT="${DAEMON_PORT:-3999}"
STAMP=$(date +%Y%m%d_%H%M%S)
if   [ "$TESTING" = 1 ]; then MODE=testing
elif [ "$SINGLE"  = 1 ]; then MODE="single_N${ARG_N}_bd${ARG_BD}"
else                          MODE=full; fi
LOGDIR="${OUT:-$PROJ/results/bench_${MODE}_${STAMP}}"
TMPDIR="$LOGDIR/expr"
mkdir -p "$TMPDIR"

echo "=============================================================="
echo " mode      : $MODE"
echo " model     : $ARG_MODEL"
echo " seeds     : ${ARG_SEEDS:-(mode default)}"
echo " project   : $PROJ"
echo " logdir    : $LOGDIR"
echo " daemon    : port $PORT   nsweeps=$NSWEEPS"
echo "=============================================================="

# ── Daemon lifecycle ──────────────────────────────────────────────────────────
DPID=""
cleanup () {
  echo "[cleanup] stopping daemon (port $PORT)…"
  julia --project="$PROJ" -e "using DaemonMode; try sendExitCode($PORT) catch end" >/dev/null 2>&1
  [ -n "$DPID" ] && kill "$DPID" 2>/dev/null
}
trap cleanup EXIT

echo "[daemon] starting…"
DAEMON_PORT="$PORT" julia --project="$PROJ" -e "using DaemonMode; serve($PORT)" \
  > "$LOGDIR/daemon.log" 2>&1 &
DPID=$!
echo "[daemon] pid=$DPID"

# wait until it accepts a trivial expr
ready=0
for i in $(seq 1 60); do
  if julia --project="$PROJ" -e "using DaemonMode; try; runexpr(\"1+1\"; port=$PORT); catch; exit(1); end" >/dev/null 2>&1; then
    ready=1; echo "[daemon] ready (attempt $i)"; break
  fi
  sleep 1
done
[ "$ready" = 1 ] || { echo "[daemon] FAILED to come up — see $LOGDIR/daemon.log"; exit 1; }

# warm the package load ONCE (the expensive precompiled-load happens here, not
# inside a timed test); JIT of the DMRG kernels is handled per-run by the
# runner's internal discarded warmup sweep.
echo "[daemon] warming package load (using SparseBackends, ITensors, ITensorMPS)…"
cat > "$TMPDIR/_warm.jl" <<'EOF'
using SparseBackends, ITensors, ITensorMPS
println("daemon warm: packages loaded")
EOF
DAEMON_PORT="$PORT" julia --project="$PROJ" "$CLIENT" "$TMPDIR/_warm.jl" 2>&1 | tail -1

# Lines echoed LIVE to the console (full output always goes to the per-test log).
# Keeps you updated as each sweep + each dense/aliased comparison lands, without
# flooding the terminal with the full DMRG/timer firehose.
CONSOLE_FILTER='### RUN|After sweep|HEAD-TO-HEAD|Energy comparison|Energy at start|final:|E0 =|E1 =|gap =|ratio aliased|SUMMARY \(|match \(aliased|values differ|^ *\||wall =|ERROR|Usage:|Gap \(|First excited'

# ── One test = one daemon run (does dense then aliased internally) ────────────
run_test () {
  local tag="$1" testfile="$2" N="$3" BD="$4" SEED="$5" model="$6"
  local log="$LOGDIR/${tag}.log"
  local expr="$TMPDIR/expr_${tag}.jl"
  # KL: run in the -1 projector eigenvalue sector; PXP has no such knob.
  local psign="1.0"; [ "$model" = "kl" ] && psign="-1.0"
  local prof="0"; [ "$ARG_PROFILE" = 1 ] && prof="1"
  cat > "$expr" <<EOF
ENV["BENCH_MAXDIM"]="$BD"
ENV["BENCH_NSWEEPS"]="$NSWEEPS"
ENV["BENCH_SEED"]="$SEED"
ENV["BENCH_SPIN"]="3"
ENV["BENCH_PSIGN"]="$psign"
ENV["BENCH_BLAS_THREADS"]="1"
ENV["BENCH_WEIGHT"]="20.0"
ENV["DMRG_DIAG"]="0"
ENV["SB_PERM_PROFILE"]="$prof"
ENV["SB_ALIAS_STATS"]="$prof"
empty!(ARGS); push!(ARGS, "$N")
println("### RUN model=$model N=$N bd=$BD seed=$SEED nsweeps=$NSWEEPS psign=$psign ###")
include("$testfile")
EOF
  echo "[$(date +%H:%M:%S)] START $tag  ->  ${tag}.log"
  local T0; T0=$(date +%s)
  # full output -> log (tee); key comparison lines -> console (grep, live).
  DAEMON_PORT="$PORT" julia --project="$PROJ" "$CLIENT" "$expr" 2>&1 \
    | tee "$log" \
    | grep --line-buffered -E "$CONSOLE_FILTER" \
    | sed -u "s/^/  [$tag] /"
  echo "[$(date +%H:%M:%S)] DONE  $tag  ($(( $(date +%s)-T0 ))s)"
}

# ── Matrix ────────────────────────────────────────────────────────────────────
RUN_KL=1; RUN_PXP=1
[ "$ARG_MODEL" = "pxp" ] && RUN_KL=0
[ "$ARG_MODEL" = "kl"  ] && RUN_PXP=0
if [ "$TESTING" = 1 ]; then
  KL_NS="8";     KL_BDS="10";     SEEDS="0"
  PXP_N="8";     PXP_BDS="10"
elif [ "$SINGLE" = 1 ]; then                     # single fixed (N,bd) for both models
  KL_NS="$ARG_N"; KL_BDS="$ARG_BD"
  PXP_N="$ARG_N"; PXP_BDS="$ARG_BD"
  SEEDS="${ARG_SEEDS:-0 1 2}"
else
  KL_NS="12 64"; KL_BDS="40 80";  SEEDS="0 1 2"
  PXP_N="100";   PXP_BDS="40 160"
fi

if [ "$RUN_KL" = 1 ]; then
  echo; echo "########## KL (aliased PHP vs dense PHP, dense ψ) ##########"
  for N in $KL_NS; do
    for BD in $KL_BDS; do
      for S in $SEEDS; do
        run_test "kl_N${N}_bd${BD}_seed${S}" "$KL_FILE" "$N" "$BD" "$S" kl
      done
    done
  done
fi

if [ "$RUN_PXP" = 1 ]; then
  echo; echo "########## PXP (aliased PHP vs dense PHP, dense ψ; ground+excited) ##########"
  for BD in $PXP_BDS; do
    for S in $SEEDS; do
      run_test "pxp_N${PXP_N}_bd${BD}_seed${S}" "$PXP_FILE" "$PXP_N" "$BD" "$S" pxp
    done
  done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo; echo "########## SUMMARY ##########"
python3 "$SELFDIR/summarize_bench.py" "$LOGDIR" | tee "$LOGDIR/SUMMARY.txt"
echo
echo "Full per-run logs: $LOGDIR/*.log"
echo "Summary saved to : $LOGDIR/SUMMARY.txt"
