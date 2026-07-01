#!/usr/bin/env bash
# ab_kernels.sh — A/B the AliasedBS×Dense serial kernel variants on the aliased KL workload.
#
# Compares, on the SAME workload, the kernel-schedule / finalize / output-order variants:
#   legacy        reduction-stationary HEAD kernel  (append!-copy finalize) + permA output scheme
#   serial-permA  current reduction-stationary kernel (buffer-swap finalize) + permA output scheme
#   serial-lmap   current reduction-stationary kernel (buffer-swap finalize) + interleaved lmap scheme
#   outstat       output-stationary kernel (production default)               + interleaved lmap scheme
#
# Toggle axes (all gate the SAME output; result is bit-identical up to BLAS-thread FP, ~1e-12):
#   SB_ALIASED_LEGACY=1     -> route to _contract_dense_serial_legacy! (git-HEAD kernel)
#   SB_ALIASED_OUTSTAT=0/1  -> reduction-stationary _contract_dense_serial! (0) vs output-stationary (1)
#   SB_ALIASED_INTERLEAVE=0 -> old scheme: emit canonical output order, reorder via next-step permA
#   SB_ALIASED_INTERLEAVE=1 -> new scheme: emit interleaved order, realize via in-kernel lmap strided write
#
# Usage:   ./ab_kernels.sh [N_PLAQ] [MAXDIM] [N_SWEEPS] [CONFIGS...]
#   N_PLAQ   default 2     (use 12 for the real perf regime)
#   MAXDIM   default 10    (use 40 for the real perf regime)
#   N_SWEEPS default 3     (sweep 1 is JIT warmup)
#   CONFIGS  default "legacy serial-permA serial-lmap"  (add "outstat" for the production kernel)
# Env:
#   ROOFLINE=1   -> set SB_ROOFLINE=1 for the per-phase timing breakdown (show_roofline/TIMER)
#   THREADS=N    -> OPENBLAS_NUM_THREADS (default: leave as-is; set 1 for deterministic GEMM)
#
# Examples:
#   ./ab_kernels.sh                              # N=2 md=10, correctness (energy) check, 3 configs
#   ./ab_kernels.sh 12 40 4                       # N=12 md=40 perf regime
#   ROOFLINE=1 ./ab_kernels.sh 12 40 4 serial-permA serial-lmap   # timing breakdown, 2 configs
#   THREADS=1 ./ab_kernels.sh 12 40 4             # deterministic-GEMM run (bit-identical comparison)
#
# Runs sequentially (no CPU contention), logs FULL output per config to /tmp/ab_<cfg>.log.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/test_aliased_psi" || { echo "cannot find test_aliased_psi under $HERE"; exit 1; }

N=${1:-2}; MD=${2:-10}; SW=${3:-3}
shift $(( $# < 3 ? $# : 3 )) || true
CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(legacy serial-permA serial-lmap densify)

# flags per config: "LEGACY OUTSTAT INTERLEAVE RUNMODE"
#   densify = current serial kernel but run_mode=:bop_densify (env-dressed, DENSIFIED
#   local-solve seed) — the densified-intermediates variant, included ONLY to measure
#   the perf gap vs the sparse kernels (not a target implementation).
flags_for() {
  case "$1" in
    legacy)       echo "1 0 0 bop_aliased" ;;
    serial-permA) echo "0 0 0 bop_aliased" ;;
    serial-lmap)  echo "0 0 1 bop_aliased" ;;
    outstat)      echo "0 1 1 bop_aliased" ;;
    densify)      echo "0 0 1 bop_densify" ;;
    *) echo "BAD" ;;
  esac
}

RF=""; [ "${ROOFLINE:-0}" = "1" ] && RF="SB_ROOFLINE=1"
BL=""; [ -n "${THREADS:-}" ] && BL="OPENBLAS_NUM_THREADS=${THREADS}"

echo "## ab_kernels: N=$N maxdim=$MD sweeps=$SW  configs=[${CONFIGS[*]}]  roofline=${ROOFLINE:-0} threads=${THREADS:-default}"
echo "## load: $(cat /proc/loadavg)"
for cfg in "${CONFIGS[@]}"; do
  read -r LG OS IL RM <<<"$(flags_for "$cfg")"
  [ "$LG" = "BAD" ] && { echo "unknown config: $cfg (valid: legacy serial-permA serial-lmap outstat densify)"; continue; }
  log="/tmp/ab_${cfg}.log"
  echo "### $cfg  (LEGACY=$LG OUTSTAT=$OS INTERLEAVE=$IL run_mode=$RM)  $(date)  -> $log"
  env $RF $BL SB_ALIASED_ENABLE=1 SB_ALIASED_LEGACY=$LG SB_ALIASED_OUTSTAT=$OS SB_ALIASED_INTERLEAVE=$IL \
      julia --project=.. test_aliased_kl.jl --N-plaq "$N" --maxdim "$MD" --n-sweeps "$SW" --run-mode "$RM" > "$log" 2>&1
  echo "    rc=$?  per-sweep E: $(grep -oE 'E=-[0-9.]+' "$log" | tr '\n' ' ')"
  if [ "${ROOFLINE:-0}" = "1" ]; then
    grep -E 'finalize=|scratch_alloc=|SETUP \(|avg per sweep' "$log" | sed 's/^/    /'
  fi
done

echo; echo "## final-E comparison (15-digit) ##"
prev=""; prevcfg=""
for cfg in "${CONFIGS[@]}"; do
  e=$(grep -oE '\-[0-9]+\.[0-9]{13,}' "/tmp/ab_${cfg}.log" 2>/dev/null | tail -1)
  printf "  %-14s %s\n" "$cfg" "${e:-<none>}"
  if [ -n "$prev" ]; then
    [ "$e" = "$prev" ] && echo "    == $prevcfg : SAME (15-dig)" || echo "    != $prevcfg : DIFFER (expected up to ~1e-12 with multithread BLAS; use THREADS=1 to force bit-identical)"
  fi
  prev="$e"; prevcfg="$cfg"
done
