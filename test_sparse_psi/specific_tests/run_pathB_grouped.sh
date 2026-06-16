#!/bin/bash
# Path B (M-corrected eigsolve) + grouped SVD (joint SVD per row group).
# This is the configuration that actually works for Path B — produces iso ψ
# within each row group → M ≈ I → eigsolve well-behaved.
set -e
OUT="${OUT:-/home/ritvik/temp/results/pathB_grouped}"
mkdir -p "$OUT"
NSWEEPS="${NSWEEPS:-6}"
MD="${MD:-40}"

run_one () {
  local label="$1"; shift; local log="$1"; shift
  echo "============================================================"
  echo "[$(date)] START $label  ->  $log"
  echo "============================================================"
  local T0=$(date +%s)
  "$@" > "$log" 2>&1 || echo "  (run exited non-zero; continuing)"
  echo "[$(date)] DONE  $label  ($(( $(date +%s) - T0 ))s)"
}

declare -A TAG=( ["true"]="plus1" ["false"]="minus1" )
START=$(date +%s)
for N in 4 12 32; do
  for EIGNV in true false; do
    tag="${TAG[$EIGNV]}"
    log="$OUT/pathB_grouped_N${N}_sign${tag}_md${MD}.log"
    run_one "Path B + grouped SVD N=$N sign=$EIGNV md=$MD" "$log" \
      env BMF_ISO_PATH=0 SB_USE_GROUPED_SVD=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
      julia --project=.. test_profile_bareh.jl \
        --N-plaq $N --eignv $EIGNV --maxdim $MD --n-sweeps $NSWEEPS
  done
done
echo "ALL DONE in $(( $(date +%s) - START ))s -- logs in $OUT"
