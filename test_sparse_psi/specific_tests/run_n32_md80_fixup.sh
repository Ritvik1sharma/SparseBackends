#!/bin/bash
# One-off: complete the N=32 md=80 runs that got cut off.
# - dense  N=32 +1  (sparse +1 already done; use its final E as target)
# - sparse N=32 -1
# - dense  N=32 -1  (use that sparse's E as target)
set -e
OUT=/home/ritvik/temp/results
MD=80
NSWEEPS=6

extract_final_E () { grep -E '^final E:' "$1" | tail -1 | awk '{print $3}'; }

run_one () {
  local label="$1"; shift; local log="$1"; shift
  echo "============================================================"
  echo "[$(date)] START $label  ->  $log"
  echo "============================================================"
  local T0=$(date +%s)
  "$@" > "$log" 2>&1 || echo "  (run exited non-zero; continuing)"
  echo "[$(date)] DONE  $label  ($(( $(date +%s) - T0 ))s)"
}

# --- dense N=32 +1 (target from existing sparse +1 log) ---
E_PLUS=$(extract_final_E "$OUT/sparse_N32_signplus1_md${MD}.log")
echo "sparse N=32 +1 md=$MD target = $E_PLUS"
run_one "dense N=32 +1 md=$MD (target=$E_PLUS)" "$OUT/dense_N32_signplus1_md${MD}.log" \
  julia --project=.. test_dense_n16.jl --N-plaq 32 --eignv true --maxdim $MD \
    --n-sweeps $NSWEEPS --target-energy $E_PLUS

# --- sparse N=32 -1 ---
run_one "sparse N=32 -1 md=$MD" "$OUT/sparse_N32_signminus1_md${MD}.log" \
  env SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
  julia --project=.. test_profile_bareh.jl --N-plaq 32 --eignv false --maxdim $MD \
    --n-sweeps $NSWEEPS

# --- dense N=32 -1 ---
E_MINUS=$(extract_final_E "$OUT/sparse_N32_signminus1_md${MD}.log")
echo "sparse N=32 -1 md=$MD target = $E_MINUS"
run_one "dense N=32 -1 md=$MD (target=$E_MINUS)" "$OUT/dense_N32_signminus1_md${MD}.log" \
  julia --project=.. test_dense_n16.jl --N-plaq 32 --eignv false --maxdim $MD \
    --n-sweeps $NSWEEPS --target-energy $E_MINUS

echo "ALL FIXUP DONE"
