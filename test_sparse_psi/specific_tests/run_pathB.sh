#!/bin/bash
# Path B (M^{-1/2} via geneigsolve) sweep at configurable maxdim.
# Same setup as run_md40.sh except BMF_ISO_PATH=0 (Path B).
# No dense re-runs (Path B is a sparse-side variant; dense already in /results).
set -e
OUT="${OUT:-/home/ritvik/temp/results/pathB}"
mkdir -p "$OUT"

NSWEEPS="${NSWEEPS:-6}"
MD="${MD:-40}"
NS_LIST="${NS_LIST:-4,12,32}"
SIGNS_LIST="${SIGNS_LIST:-true,false}"

while [ $# -gt 0 ]; do
  case "$1" in
    --md)      MD="$2";         shift 2;;
    --nsweeps) NSWEEPS="$2";    shift 2;;
    --ns)      NS_LIST="$2";    shift 2;;
    --signs)   SIGNS_LIST="$2"; shift 2;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done
IFS=',' read -ra NS_ARR    <<< "$NS_LIST"
IFS=',' read -ra SIGNS_ARR <<< "$SIGNS_LIST"
echo "Path B sweep: MD=$MD  NSWEEPS=$NSWEEPS  Ns=${NS_ARR[*]}  signs=${SIGNS_ARR[*]}"

declare -A SIGN_TAG=( ["true"]="plus1" ["false"]="minus1" )

run_one () {
  local label="$1"; shift; local log="$1"; shift
  echo "============================================================"
  echo "[$(date)] START $label  ->  $log"
  echo "============================================================"
  local T0=$(date +%s)
  "$@" > "$log" 2>&1 || echo "  (run exited non-zero; continuing)"
  echo "[$(date)] DONE  $label  ($(( $(date +%s) - T0 ))s)"
}

START=$(date +%s)
for N in "${NS_ARR[@]}"; do
  for EIGNV in "${SIGNS_ARR[@]}"; do
    tag="${SIGN_TAG[$EIGNV]}"
    log="$OUT/pathB_N${N}_sign${tag}_md${MD}.log"
    run_one "Path B sparse N=$N sign=$EIGNV md=$MD (SVD, no GS)" "$log" \
      env BMF_ISO_PATH=0 \
      julia --project=.. test_profile_bareh.jl \
        --N-plaq $N --eignv $EIGNV --maxdim $MD --n-sweeps $NSWEEPS
  done
done
END=$(date +%s)
echo "============================================================"
echo "ALL DONE in $((END-START))s -- logs in $OUT"
