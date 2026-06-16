#!/bin/bash
# Sparse (balanced+adaptive) vs dense at configurable maxdim,
# N list, and projector-sign sectors.
#   psign=+1 → P = ∏(I+C)/2     (smaller bond dims, easier)
#   psign=-1 → P = ∏(I-C)/2     (larger bond dims)
#
# Per (N, sign): sparse first; parse sparse final E; run dense with
# --target-energy=<E_sp> so dense reports the cumulative time at which it
# crosses sparse's energy (does not early-exit; finishes all sweeps).
#
# Each runner prints a SUMMARY block with final E, total time incl/excl sweep
# 1 (JIT), avg/sweep excl JIT, and a final-state report (footprint MiB,
# per-site storage, linkdims).
#
# Logs:  /home/ritvik/temp/results/{sparse,dense}_N{N}_sign{plus,minus}1_md{MD}.log

set -e
OUT=/home/ritvik/temp/results
mkdir -p "$OUT"

# Defaults (override via CLI flags or env vars).
#   ./run_md40.sh                       → MD=40, all Ns, both signs
#   ./run_md40.sh --md 80               → maxdim=80
#   ./run_md40.sh --md 160 --ns 4,12    → maxdim=160 at N=4 and N=12
#   ./run_md40.sh --md 80 --signs true  → maxdim=80, only +1 sector
#   ./run_md40.sh --md 40 --nsweeps 8   → 8 total sweeps (1 JIT + 7 prod)
NSWEEPS="${NSWEEPS:-6}"   # sweep 1 = JIT; rest are production
MD="${MD:-40}"
NS_LIST="${NS_LIST:-4,12,32}"
SIGNS_LIST="${SIGNS_LIST:-true,false}"   # true=+1, false=-1

while [ $# -gt 0 ]; do
  case "$1" in
    --md)       MD="$2";         shift 2;;
    --nsweeps)  NSWEEPS="$2";    shift 2;;
    --ns)       NS_LIST="$2";    shift 2;;
    --signs)    SIGNS_LIST="$2"; shift 2;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done
IFS=',' read -ra NS_ARR    <<< "$NS_LIST"
IFS=',' read -ra SIGNS_ARR <<< "$SIGNS_LIST"
echo "Config: MD=$MD  NSWEEPS=$NSWEEPS  Ns=${NS_ARR[*]}  signs=${SIGNS_ARR[*]}"

run_one () {
  local label="$1"; shift
  local log="$1";   shift
  echo "============================================================"
  echo "[$(date)] START $label  ->  $log"
  echo "============================================================"
  local T0=$(date +%s)
  "$@" > "$log" 2>&1 || echo "  (run exited non-zero; continuing)"
  local T1=$(date +%s)
  echo "[$(date)] DONE  $label  ($((T1-T0))s)"
}

extract_final_E () {
  local log="$1"
  grep -E '^final E:' "$log" | tail -1 | awk '{print $3}'
}

# eignv flag: true → +1, false → -1
declare -A SIGN_LABEL=( ["true"]="+1" ["false"]="-1" )
declare -A SIGN_TAG=(   ["true"]="plus1" ["false"]="minus1" )

START=$(date +%s)
for N in "${NS_ARR[@]}"; do
  for EIGNV in "${SIGNS_ARR[@]}"; do
    sign_lbl="${SIGN_LABEL[$EIGNV]}"
    sign_tag="${SIGN_TAG[$EIGNV]}"
    sp_log="$OUT/sparse_N${N}_sign${sign_tag}_md${MD}.log"
    dn_log="$OUT/dense_N${N}_sign${sign_tag}_md${MD}.log"

    # --- sparse ---
    run_one "sparse N=$N sign=$sign_lbl md=$MD" "$sp_log" \
      env SB_USE_QR=1 SB_BALANCED_OWNERSHIP=1 SB_ADAPTIVE_RANK=1 \
      julia --project=.. test_profile_bareh.jl \
        --N-plaq $N --eignv $EIGNV --maxdim $MD --n-sweeps $NSWEEPS

    # --- pull sparse final E to drive dense's early-target reporting ---
    E_SP=$(extract_final_E "$sp_log")
    if [ -z "$E_SP" ] || [ "$E_SP" = "NaN" ]; then
      echo "  WARN: could not extract sparse final E for N=$N sign=$sign_lbl; running dense without --target-energy"
      run_one "dense  N=$N sign=$sign_lbl md=$MD" "$dn_log" \
        julia --project=.. test_dense_n16.jl \
          --N-plaq $N --eignv $EIGNV --maxdim $MD --n-sweeps $NSWEEPS
    else
      echo "  sparse N=$N sign=$sign_lbl final E = $E_SP  →  passing as dense --target-energy"
      run_one "dense  N=$N sign=$sign_lbl md=$MD (target=$E_SP)" "$dn_log" \
        julia --project=.. test_dense_n16.jl \
          --N-plaq $N --eignv $EIGNV --maxdim $MD --n-sweeps $NSWEEPS \
          --target-energy $E_SP
    fi
  done
done
END=$(date +%s)
echo "============================================================"
echo "ALL DONE in $((END-START))s -- logs in $OUT"
