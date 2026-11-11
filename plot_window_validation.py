#!/usr/bin/env python3
# coding: utf-8
"""Plot VLBI breach population N_breach vs integration/scan window (task #5),
from window_validation.csv. Sparse DTD harmonics slope down (2000 s conservative);
the dense UEMR line slopes up (2000 s anti-conservative). Worst station (Hobart).

Manuscript figure fig:window (single column): 13.5 pt text, no title (the
caption carries it), class legend only (frequencies are written on the
curves), 300 dpi.

Usage:  python plot_window_validation.py [OUTPATH]  (default output/figs/window_validation.png)
"""
import csv, sys
import settings as cfg
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

FS = 13.5
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
                     "ytick.labelsize": FS, "legend.fontsize": FS})

STN = "Hb"
rows = [r for r in csv.DictReader(open(cfg.ref("window_validation.csv"))) if r["station"] == STN]
N0 = 11981
PRESENT, S300 = N0, 300_000

# class per line (DTD downlink/LO/harmonics vs platform UEMR)
CLS = {2595: "DTD LO", 2620: "DTD downlink", 5240: "DTD harmonic",
       7860: "DTD harmonic", 10480: "DTD harmonic", 2700: "UEMR (platform)"}
COL = {"DTD downlink": "k", "DTD LO": "tab:purple", "DTD harmonic": "tab:red",
       "UEMR (platform)": "tab:blue"}
HALO = [pe.withStroke(linewidth=3, foreground="white")]

d = defaultdict(dict)
for r in rows:
    d[round(float(r["freq_mhz"]))][int(r["window_s"])] = int(r["n_breach"])

fig, ax = plt.subplots(figsize=(7.2, 6.24))
for f in sorted(d):
    ws = sorted(d[f])
    ax.plot(ws, [d[f][w] for w in ws], "o-", color=COL[CLS[f]], lw=2.2, ms=7)
    # frequency written above the first point; 2700 above its last point
    # (its 10 s point is crowded by the 2595 and 7860 labels)
    w = ws[-1] if f == 2700 else ws[0]
    ax.annotate(f"{f}", (w, d[f][w]), textcoords="offset points",
                xytext=(2, 8) if f == 2700 else (-2, 8),
                ha="right" if f == 2700 else "left", va="bottom", fontsize=FS,
                color=COL[CLS[f]], path_effects=HALO)
for y, lab, x, ha in [(PRESENT, "present fleet (~12k)", 2000, "right"),
                      (S300, "300k scenario", 125, "left")]:
    ax.axhline(y, ls="--", color="0.5", lw=1.2)
    ax.text(x, y * 0.85, lab, ha=ha, va="top", color="0.4", path_effects=HALO)
for x in (100, 2000):
    ax.axvline(x, ls=":", color="0.7", lw=1.0)
ax.axvspan(10, 30, color="0.9", zorder=0)
ax.text(17, 3, "VGOS scan\n10–30 s", ha="center", va="bottom", color="0.4")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([10, 30, 100, 2000]); ax.set_xticklabels(["10", "30", "100", "2000"])
ax.set_xlabel("integration / scan window (s)")
ax.set_ylabel("$N_{\\rm breach}$ (total LEO satellites)")
handles = [Line2D([], [], color=c, marker="o", lw=2.2, ms=7, label=k)
           for k, c in COL.items()]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
           ncol=2, frameon=False, columnspacing=2.0)
ax.grid(True, which="both", color="0.93")
p = sys.argv[1] if len(sys.argv) > 1 else cfg.out("figs", "window_validation.png")
plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
print(f"wrote {p}")
