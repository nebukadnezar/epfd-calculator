#!/usr/bin/env python3
# coding: utf-8
"""Plot true (up-scaled Monte-Carlo) vs analytic (10log10 N CDF-shift) VLBI data
loss against constellation size, for the validated lines. Shows the analytic
method is conservative (true loss >= analytic for the non-saturated lines).

Manuscript figure fig:linval (two-column): 11 pt text, no panel titles (the
caption carries them), 300 dpi.

Usage:  python plot_validation.py [OUTPATH]   (default output/figs/linearity_validation.png)
"""
import csv, os, sys
import settings as cfg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11})

rows = list(csv.DictReader(open(cfg.ref("linearity_validation_full.log.csv"))))
for r in rows:
    for k in ("freq_mhz", "target", "loss_true_pct", "loss_analytic_pct", "delta_db"):
        r[k] = float(r[k])

lines = sorted({r["freq_mhz"] for r in rows})
# same per-detection colours as the breach / E_lim figures (analyze_results)
colors = {7860.0: "#0072B2", 8599.0: "#882255"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for fr in lines:
    sub = sorted([r for r in rows if r["freq_mhz"] == fr], key=lambda r: r["target"])
    Ns = [r["target"] for r in sub]
    c = colors.get(fr, "tab:blue")
    ax1.plot(Ns, [r["loss_true_pct"] for r in sub], "-o", color=c, lw=2, ms=8,
             label=f"{fr:.0f} MHz true (up-scaled MC)")
    ax1.plot(Ns, [r["loss_analytic_pct"] for r in sub], "--s", color=c, lw=2, ms=8,
             mfc="none", mew=1.5, label=f"{fr:.0f} MHz analytic (CDF shift)")
    ax2.plot(Ns, [r["delta_db"] for r in sub], "-o", color=c, lw=2, ms=8,
             label=f"{fr:.0f} MHz")

from matplotlib.ticker import NullFormatter
ticks = sorted({r["target"] for r in rows})
for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.set_xlabel("total LEO satellites")
    ax.set_xticks(ticks)                      # the simulated populations
    ax.set_xticklabels([f"{t/1000:.0f}k" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(color="0.9", which="both")
ax1.set_ylabel("VLBI data loss (%)")
ax1.legend(loc="upper left")
ax1.set_ylim(-2, 103)
ax2.set_ylabel("98th-percentile aggregate, true $-$ analytic (dB)")
ax2.legend(loc="lower left")
ax2.axhline(0, color="0.5", lw=0.8)
p = sys.argv[1] if len(sys.argv) > 1 else cfg.out("figs", "linearity_validation.png")
os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
plt.tight_layout(); plt.savefig(p, dpi=300); plt.close(fig)
print(f"wrote {p}")
