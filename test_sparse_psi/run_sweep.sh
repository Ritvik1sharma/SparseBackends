#!/bin/bash
# Consolidated sweep runner for BOTH benchmarks: KL ((I+C)/2 Kitaev-ladder
# projector) and PXP (Rydberg-blockade R1 constraint). Runs dense baseline +
# sparse-psi Path A for each, across a maxdim sweep. Ground state for both;
# excited state additionally for PXP (weight=20, orthogonal to ground).
#
# Usage:
#   ./run_sweep.sh                 # both benchmarks, md ∈ {20,40,80}
#   BENCH=kl  ./run_sweep.sh       # KL only
#   BENCH=pxp ./run_sweep.sh       # PXP only
#   MDS="40"  ./run_sweep.sh       # single maxdim
#   OUT=/path/to/logs ./run_sweep.sh
#
# Env knobs:
#   BENCH   kl | pxp | both        (default both)
#   MDS     space-separated maxdims (default "20 40 80")
#   OUT     output log dir          (default /home/ritvik/temp/results/md_sweep)
#   N_KL    KL chain plaquettes      (default 32)
#   N_PXP   PXP chain length         (default 100)
set -e

BENCH="${BENCH:-both}"
MDS="${MDS:-20 40 80}"
OUT="${OUT:-/home/ritvik/temp/results/md_sweep}"
N_KL="${N_KL:-32}"
N_PXP="${N_PXP:-100}"
mkdir -p "$OUT"

# Production sparse-psi Path A flags (see README "Recommended environment flags")
SPARSE_FLAGS=(BMF_ISO_PATH=1 SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1)

run () {
  local lab="$1"; local log="$2"; shift 2
  echo "============================================================"
  echo "[$(date)] START $lab  ->  $log"
  echo "============================================================"
  local T0; T0=$(date +%s)
  "$@" > "$log" 2>&1 || echo "  (exit non-zero; continuing)"
  echo "[$(date)] DONE  $lab  ($(( $(date +%s) - T0 ))s)"
}

do_kl () {
  for MD in $MDS; do
    run "KL-dense-N$N_KL-md$MD"  "$OUT/kl_dense_N${N_KL}_md$MD.log" \
      julia --project=.. test_dense_kl.jl  --N-plaq "$N_KL" --eignv true --maxdim "$MD" --n-sweeps 6
    run "KL-sparse-N$N_KL-md$MD" "$OUT/kl_sparse_N${N_KL}_md$MD.log" \
      env "${SPARSE_FLAGS[@]}" \
      julia --project=.. test_sparse_kl.jl --N-plaq "$N_KL" --eignv true --maxdim "$MD" --n-sweeps 6
  done
}

do_pxp () {
  for MD in $MDS; do
    run "PXP-dense-N$N_PXP-md$MD"  "$OUT/pxp_dense_md$MD.log" \
      julia --project=.. test_dense_pxp.jl  --N "$N_PXP" --maxdim "$MD" --mindim "$MD" --n-sweeps 10
    run "PXP-sparse-N$N_PXP-md$MD" "$OUT/pxp_sparse_md$MD.log" \
      env "${SPARSE_FLAGS[@]}" \
      julia --project=.. test_sparse_pxp.jl --N "$N_PXP" --maxdim "$MD" --mindim "$MD" --n-sweeps 10
  done
}

case "$BENCH" in
  kl)   do_kl ;;
  pxp)  do_pxp ;;
  both) do_kl; do_pxp ;;
  *) echo "unknown BENCH='$BENCH' (want: kl | pxp | both)" >&2; exit 1 ;;
esac

echo "ALL DONE -- logs in $OUT"
echo "Aggregate/plot with: python3 make_plots.py"
