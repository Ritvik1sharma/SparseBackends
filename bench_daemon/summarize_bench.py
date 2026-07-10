#!/usr/bin/env python3
# Summarize per-sweep DMRG timing + energies from the benchmark logs.
#
# Each run_dmrg_ground / run_dmrg_excited call prints, for its (single) timed
# run at outputlevel=1:
#     After sweep <n> energy=<E>  maxlinkdim=<d> maxerr=<e> time=<t>
# ...followed by a marker line:
#     ========== TIMER REPORT: <label>  (wall = <w> s, JIT excluded) ==========
# We attribute the "After sweep" lines that precede a TIMER REPORT to that
# report's <label>, then report the mean per-sweep time EXCLUDING sweep 1 and
# the final-sweep energy.  Filenames encode the config (model/N/bd/seed).
import os, re, sys, glob

SWEEP_RE  = re.compile(r"After sweep\s+(\d+)\s+energy=(.+?)\s+maxlinkdim=(\d+)\s+maxerr=(\S+)\s+time=([\d.eE+-]+)")
REPORT_RE = re.compile(r"TIMER REPORT:\s+(.+?)\s+\(wall\s*=\s*([\d.eE+-]+)\s*s")

def parse_log(path):
    """Return list of (label, wall, [(sweep,energy,time),...]) in file order."""
    blocks, cur = [], []
    for line in open(path, errors="replace"):
        m = SWEEP_RE.search(line)
        if m:
            cur.append((int(m.group(1)), m.group(2).strip(), float(m.group(5))))
            continue
        r = REPORT_RE.search(line)
        if r:
            blocks.append((r.group(1).strip(), float(r.group(2)), cur))
            cur = []
    return blocks

# Canonicalize the runner labels to a short backend tag.
def backend_tag(label):
    up = label.upper()
    kind = "excited" if "EXCITED" in up else ("ground" if "GROUND" in up else "ground")
    be   = "ALIASED" if "ALIAS" in up else "DENSE"
    return be, kind

def fmt_energy(e):
    # strip complex " + 0.0im" noise for display but keep it if imag is nonzero
    e = e.replace(" ", "")
    m = re.match(r"^(-?[\d.eE+-]+)\+(-?[\d.eE+-]+)im$", e)
    if m and abs(float(m.group(2))) < 1e-9:
        return m.group(1)
    return e

def main():
    logdir = sys.argv[1]
    logs = sorted(f for f in glob.glob(os.path.join(logdir, "*.log"))
                  if os.path.basename(f) != "daemon.log")
    print(f"\n{'='*118}")
    print(f"BENCHMARK SUMMARY  ({logdir})")
    print(f"per-sweep mean EXCLUDES sweep 1 | n=#sweeps timed | E=final-sweep energy")
    print(f"{'='*118}")
    hdr = f"{'test':30s} {'backend':8s} {'state':8s} {'n':>3s} {'mean/sw(s)':>11s} {'sweep1(s)':>10s} {'wall(s)':>9s}  {'final E':>22s}"
    for path in logs:
        name = os.path.basename(path)[:-4]
        blocks = parse_log(path)
        if not blocks:
            print(f"\n[{name}]  (no completed sweeps found — check log for errors)")
            continue
        print(f"\n[{name}]")
        print("  " + hdr)
        # collect finals for a per-file dense-vs-aliased delta, keyed by state
        finals = {}  # (state) -> {backend: energy_str}
        for label, wall, sweeps in blocks:
            if not sweeps:
                continue
            be, state = backend_tag(label)
            sw_sorted = sorted(sweeps)
            times = [t for (n, e, t) in sw_sorted]
            t1 = times[0] if times else float("nan")
            rest = times[1:]
            mean_rest = sum(rest) / len(rest) if rest else float("nan")
            fe = fmt_energy(sw_sorted[-1][1])
            finals.setdefault(state, {})[be] = fe
            print(f"  {label[:30]:30s} {be:8s} {state:8s} {len(sw_sorted):3d} "
                  f"{mean_rest:11.3f} {t1:10.3f} {wall:9.3f}  {fe:>22s}")
        # energy deltas dense vs aliased
        for state, d in finals.items():
            if "DENSE" in d and "ALIASED" in d:
                try:
                    de = abs(float(re.match(r'-?[\d.eE+-]+', d['DENSE']).group()) -
                             float(re.match(r'-?[\d.eE+-]+', d['ALIASED']).group()))
                    flag = "  <-- DIFFER" if de > 1e-3 else ""
                    print(f"    |dE| {state:8s} dense-vs-aliased = {de:.3e}{flag}")
                except Exception:
                    pass

if __name__ == "__main__":
    main()
