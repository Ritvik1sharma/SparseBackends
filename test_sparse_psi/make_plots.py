#!/usr/bin/env python3
"""Generate comparison plots for PXP + KL md-sweep results."""
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path("/home/ritvik/temp/results/md_sweep_clean")

# ============ PXP data (CLEAN, no contention) ============
# (md, footprint_MiB, runtime_excl_jit_s, excited_runtime_s, ground_E, excited_E)
pxp_data = {
    "dense": [
        (20, 0.932,  21.273, 30.338, -120.998633861685, -120.000741216186),
        (40, 3.493,  54.202, 70.617, -120.998633747794, -120.018582225361),
        (80, 13.649, 191.088, 205.835, -120.998633872698, -120.018763138557),
    ],
    "sparse": [
        (20, 1.360,  14.203, 24.197, -120.998633893860, -120.020389391580),
        (40, 2.700,  24.898, 42.021, -120.998633879810, -120.014330813605),
        (80, 9.850,  124.407, 178.596, -120.998633873381, -120.011148309935),
    ],
}

# PXP reference: best ground = min across all (lowest); best excited = same.
pxp_gs_ref = min(e for vs in pxp_data.values() for (_,_,_,_, e, _) in vs)
pxp_ex_ref = min(e for vs in pxp_data.values() for (_,_,_,_,_, e) in vs)
print(f"PXP refs: gs={pxp_gs_ref:.12f}  ex={pxp_ex_ref:.12f}")

# ============ KL data (N=32 +1 sector) ============
# (md, footprint_MiB, runtime_excl_jit_s, ground_E)
kl_data = {
    "dense": [
        (20, 1.141,  24.453,  -44.113005830066),
        (40, 4.270,  69.827,  -44.115699912159),
        (80, 16.424, 351.349, -44.116154014441),
    ],
    "sparse": [
        (20, 1.438,  7.437,   -43.679789612866),
        (40, 3.061,  16.970,  -44.092840126148),
        (80, 12.928, 116.196, -44.116185113840),
    ],
}
kl_gs_ref = min(e for vs in kl_data.values() for (_,_,_, e) in vs)
print(f"KL ref: gs={kl_gs_ref:.12f}")

# ============ Write CSV ============
with open(OUT_DIR / "pxp_summary.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["model","method","md","footprint_MiB","gs_time_s","ex_time_s","gs_E","ex_E","gs_err","ex_err"])
    for m,vs in pxp_data.items():
        for (md, fp, gt, et, ge, ee) in vs:
            w.writerow(["PXP", m, md, fp, gt, et, ge, ee, ge - pxp_gs_ref, ee - pxp_ex_ref])
with open(OUT_DIR / "kl_summary.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["model","method","md","footprint_MiB","gs_time_s","gs_E","gs_err"])
    for m,vs in kl_data.items():
        for (md, fp, gt, ge) in vs:
            w.writerow(["KL_N32_plus1", m, md, fp, gt, ge, ge - kl_gs_ref])

# ============ Plots ============
def plot_one(ax, data, ykey, title, ylabel, log=True):
    for method, color in [("dense", "C0"), ("sparse", "C1")]:
        mds = [r[0] for r in data[method]]
        ys  = [r[ykey] for r in data[method]]
        ax.plot(mds, ys, "o-", label=method, color=color, lw=2, ms=8)
    ax.set_xlabel("maxdim")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_xticks([20, 40, 80])
    ax.set_xticklabels(["20", "40", "80"])

# PXP: 4 panels (footprint, runtime, gs_err, ex_err)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle("PXP model (N=100)  sparse Path A vs dense", fontsize=14)

# footprint
plot_one(axes[0,0], pxp_data, 1, "Memory footprint (ground state ψ)", "footprint [MiB]", log=True)

# runtime: ground-only and ground+excited combined
ax = axes[0,1]
for method, color in [("dense", "C0"), ("sparse", "C1")]:
    mds = [r[0] for r in pxp_data[method]]
    gs_t = [r[2] for r in pxp_data[method]]
    total = [r[2] + r[3] for r in pxp_data[method]]
    ax.plot(mds, gs_t, "o-",  label=f"{method} ground",  color=color, lw=2, ms=8)
    ax.plot(mds, total,"s--", label=f"{method} gs+ex",   color=color, lw=2, ms=8, alpha=0.6)
ax.set_xlabel("maxdim"); ax.set_ylabel("time excl JIT [s]")
ax.set_title("Runtime (post-JIT)"); ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.legend()
ax.set_xticks([20,40,80]); ax.set_xticklabels(["20","40","80"])

# ground state error: |E - E_ref|
ax = axes[1,0]
for method, color in [("dense", "C0"), ("sparse", "C1")]:
    mds = [r[0] for r in pxp_data[method]]
    err = [max(r[4] - pxp_gs_ref, 1e-12) for r in pxp_data[method]]
    ax.plot(mds, err, "o-", label=method, color=color, lw=2, ms=8)
ax.set_xlabel("maxdim"); ax.set_ylabel("|E_gs − best E_gs| [a.u.]")
ax.set_title("Ground-state energy error")
ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.legend()
ax.set_xticks([20,40,80]); ax.set_xticklabels(["20","40","80"])

# excited state error
ax = axes[1,1]
for method, color in [("dense", "C0"), ("sparse", "C1")]:
    mds = [r[0] for r in pxp_data[method]]
    err = [max(r[5] - pxp_ex_ref, 1e-12) for r in pxp_data[method]]
    ax.plot(mds, err, "o-", label=method, color=color, lw=2, ms=8)
ax.set_xlabel("maxdim"); ax.set_ylabel("|E_ex − best E_ex| [a.u.]")
ax.set_title("Excited-state energy error")
ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.legend()
ax.set_xticks([20,40,80]); ax.set_xticklabels(["20","40","80"])

plt.tight_layout()
plt.savefig(OUT_DIR / "pxp_md_sweep.png", dpi=120, bbox_inches="tight")
print(f"Saved {OUT_DIR / 'pxp_md_sweep.png'}")

# KL: 3 panels
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("KL (I+C)/2 projector  N=32 +1 sector  sparse Path A vs dense", fontsize=14)

plot_one(axes[0], kl_data, 1, "Memory footprint", "footprint [MiB]", log=True)
plot_one(axes[1], kl_data, 2, "Runtime (post-JIT)", "time excl JIT [s]", log=True)

ax = axes[2]
for method, color in [("dense", "C0"), ("sparse", "C1")]:
    mds = [r[0] for r in kl_data[method]]
    err = [max(r[3] - kl_gs_ref, 1e-12) for r in kl_data[method]]
    ax.plot(mds, err, "o-", label=method, color=color, lw=2, ms=8)
ax.set_xlabel("maxdim"); ax.set_ylabel("|E_gs − best E_gs| [a.u.]")
ax.set_title("Ground-state energy error")
ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.legend()
ax.set_xticks([20,40,80]); ax.set_xticklabels(["20","40","80"])

plt.tight_layout()
plt.savefig(OUT_DIR / "kl_md_sweep.png", dpi=120, bbox_inches="tight")
print(f"Saved {OUT_DIR / 'kl_md_sweep.png'}")
