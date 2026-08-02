#!/usr/bin/env python3
"""Plot per-sweep DMRG truncation error for orig_dense / sb_dense / sb_fused.

Purpose: show (a) how the truncation error evolves over sweeps, (b) that the
SparseBackends variants track the dense reference as they must, and (c) where
orig_dense departs from it.

Data source is the per-sweep `maxerr=` field of the DMRG logs, NOT the JSON
`truncerr` (which stores only the max over sweeps).

    python3 plot_truncerr.py <maxerr.json> [more.json ...] -o out.pdf

Colours are slots 1-3 of the validated categorical palette. Only THREE series are
plotted on purpose: at four slots the all-pairs check hard-fails (orange<->yellow
normal-vision deltaE 13.7 light / 10.6 dark, below the 15 floor), and the rule for
that is to cut series rather than paper over it with extra encoding. sb_aliased is
omitted -- its truncation error is indistinguishable from sb_fused (they share the
operator), so it would add a line without adding information.

Every series is also direct-labelled: the aqua slot sits below 3:1 contrast on the
light surface, which obligates visible labels rather than colour alone.
"""
import argparse, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import matplotlib.patheffects as pe

# validated categorical slots 1-3 (light / dark)
LIGHT = {"orig_dense": "#2a78d6", "sb_dense": "#eb6834", "sb_fused": "#1baf7a"}
DARK  = {"orig_dense": "#3987e5", "sb_dense": "#d95926", "sb_fused": "#199e70"}
ORDER = ["orig_dense", "sb_dense", "sb_fused"]
LABEL = {"orig_dense": "orig_dense", "sb_dense": "sb_dense", "sb_fused": "sb_fused"}

SURF   = {"light": "#fcfcfb", "dark": "#1a1a19"}
INK    = {"light": "#0b0b0b", "dark": "#ffffff"}
INK2   = {"light": "#52514e", "dark": "#c3c2b7"}
GRID   = {"light": "#e6e5e1", "dark": "#333330"}


def load(paths):
    merged = {}
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        tag = "1 thread" if "blas1" in os.path.basename(p) or "maxerr.json" == os.path.basename(p) else "5 threads"
        for cfg, kinds in d.items():
            for kind, variants in kinds.items():
                merged[(cfg, kind, tag)] = variants
    return merged


def plot(merged, out, mode="light"):
    col = LIGHT if mode == "light" else DARK
    panels = sorted(merged.keys(), key=lambda k: (k[0], k[1]))
    panels = [k for k in panels if any(v in merged[k] for v in ORDER)]
    n = len(panels)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow),
                             squeeze=False, facecolor=SURF[mode])
    for ax in axes.flat:
        ax.set_visible(False)

    for i, key in enumerate(panels):
        cfg, kind, tag = key
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        ax.set_facecolor(SURF[mode])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID[mode])
        ax.grid(True, which="major", color=GRID[mode], linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2[mode], labelsize=8, length=3)

        # Stagger the direct-label y-offsets. sb_dense and sb_fused are often
        # numerically IDENTICAL (they share the operator on PXP), so their end
        # points coincide and un-staggered labels overprint into unreadable mush.
        ystag = {"orig_dense": 9.0, "sb_dense": 0.0, "sb_fused": -9.0}
        for v in ORDER:
            rows = merged[key].get(v)
            if not rows:
                continue
            xs = [r["sweep"] for r in rows]
            # a maxerr of exactly 0 (sweep 1 before any truncation) cannot be
            # drawn on a log axis; drop those points rather than clamp them,
            # which would invent a value.
            pts = [(x, r["maxerr"]) for x, r in zip(xs, rows) if r["maxerr"] > 0]
            if not pts:
                continue
            X, Y = zip(*pts)
            # A 2px surface-coloured ring on the markers keeps overlapping series
            # readable where they coincide.
            ax.plot(X, Y, color=col[v], linewidth=2.0, marker="o", markersize=4,
                    markeredgecolor=SURF[mode], markeredgewidth=0.8,
                    label=LABEL[v], zorder=3)
            ax.annotate(LABEL[v], (X[-1], Y[-1]), textcoords="offset points",
                        xytext=(6, ystag[v]), fontsize=7.5, color=col[v],
                        va="center", zorder=4,
                        path_effects=[pe.withStroke(linewidth=2.5,
                                                    foreground=SURF[mode])])

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.set_title(f"{cfg}  ·  {kind}  ·  {tag}", fontsize=9.5,
                     color=INK[mode], loc="left", pad=6)
        ax.set_xlabel("sweep", fontsize=8.5, color=INK2[mode])
        ax.set_ylabel("truncation error (maxerr)", fontsize=8.5, color=INK2[mode])
        ax.margins(x=0.22)

    # one legend for the whole figure — identity is never colour-alone
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   fontsize=9, labelcolor=INK[mode], bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("DMRG truncation error per sweep — SparseBackends variants vs the original-ITensors dense baseline",
                 fontsize=11.5, color=INK[mode], x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out, dpi=200, facecolor=SURF[mode], bbox_inches="tight")
    print(f"wrote {out}  ({n} panels, mode={mode})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("-o", "--out", default="truncerr_per_sweep.pdf")
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    a = ap.parse_args()
    m = load(a.json)
    if not m:
        sys.exit("no data")
    plot(m, a.out, a.mode)
