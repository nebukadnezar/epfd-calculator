"""Figure for Sec. 3.3 (HH's request): the interpolated RA.769 thresholds
S_lim(nu) across 1-14 GHz for both criteria (VLBI 1%-of-noise, single-dish
continuum 10%), the tabulated RA.769 band centres they pass through, the
measured SNIFFLES emission lines they are evaluated at, and the VGOS
sub-bands / legacy X-band for orientation. The interpolation is linear in dB
vs log10(nu), so the segments are straight on a log frequency axis.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import astropy.units as u
from pycraf import protection
import epfd_core as ec, sniffles_inputs as si
import settings as cfg
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
F = np.logspace(np.log10(1.0), np.log10(14.0), 400)          # GHz sweep
MODES = [("vlbi", "VLBI: 1% of receiver noise power", "tab:blue"),
         ("continuum", "single-dish continuum: 10% of the rms noise\nfluctuation after 2000 s", "tab:red")]

# Printed single-column (canvas width kept, 20 % taller
# than the submitted 7.2 x 4.6 in canvas, all text 13.5 pt, legend below the axes).
plt.rcParams.update({"font.size": 13.5, "axes.labelsize": 13.5, "xtick.labelsize": 13.5,
                     "ytick.labelsize": 13.5, "legend.fontsize": 13.5})
fig, ax = plt.subplots(figsize=(7.2, 5.5))

# VGOS sub-bands (regulatory ranges, Table 2) + legacy X-band; labels sit above
# the axes so they never collide with the curves or the legend
BANDS = [("A", 2.90, 3.40), ("B", 4.80, 5.85), ("C", 8.75, 9.80), ("D", 12.75, 14.00)]
for nm, lo, hi in BANDS:
    ax.axvspan(lo, hi, color="moccasin", alpha=0.85, zorder=0)
    ax.text(np.sqrt(lo*hi), 1.01, nm, ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), color="saddlebrown", fontweight="bold")
ax.text(2.6, 1.01, "VGOS sub-bands:", ha="right", va="bottom",
        transform=ax.get_xaxis_transform(), color="saddlebrown", fontweight="bold")
ax.axvspan(8.2, 8.95, color="lightsteelblue", alpha=0.9, zorder=0)
ax.text(np.sqrt(8.2*8.95), 0.985, "legacy X", ha="center", va="top",
        transform=ax.get_xaxis_transform(), color="steelblue", fontsize=12.5)

for mode, lab, c in MODES:
    s = np.array([ec.ra769_limit(f*1e9, mode=mode)["slim_nu_db"] for f in F])
    ax.plot(F, s, color=c, lw=1.8, label=f"{lab}, interpolated")
    t = protection.ra769_limits(mode=mode)
    fm = t["frequency"].to_value(u.GHz)
    sl = t["Slim_nu"].to_value(u.dB(u.W/u.m**2/u.Hz))
    m = (fm >= 0.9) & (fm <= 16)
    ax.plot(fm[m], sl[m], "o", color=c, ms=6, mec="k", mew=0.6, zorder=5,
            label="RA.769 tabulated centres" if mode == "vlbi" else None)

# measured SNIFFLES lines, marked on both curves
lines = sorted(set(round(r["freq_mhz"]) for r in si.load_inputs()))
lines = [f for f in lines if 1000 <= f <= 14000]
for f_mhz in lines:
    f = f_mhz/1e3
    ax.axvline(f, color="0.55", lw=0.5, ls=":", zorder=1)
    for mode, _, c in MODES:
        s = ec.ra769_limit(f_mhz*1e6, mode=mode)["slim_nu_db"]
        ax.plot(f, s, "v", color=c, ms=4.5, zorder=6)
ax.plot([], [], "v", color="0.3", ms=4.5, ls="none",
        label="measured emission/radiation detections")

ax.set_xscale("log")
ax.set_xticks([1, 1.5, 2, 3, 4, 5, 7, 8.4, 10, 14])
ax.set_xticklabels(["1", "1.5", "2", "3", "4", "5", "7", "8.4", "10", "14"])
ax.set_xlim(1, 14); ax.set_ylim(-258, -184)
ax.set_xlabel("frequency (GHz)")
ax.set_ylabel(r"$S_{\rm lim}$  (dB(W m$^{-2}$ Hz$^{-1}$))")
ax.grid(alpha=0.25, which="both")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False,
           labelspacing=0.3, handlelength=2.2)
fig.tight_layout()
# optional output directory; default: output/figs
outdir = sys.argv[1] if len(sys.argv) > 1 else cfg.FIGS
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "fig_slim_interp.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
for f_mhz in lines:
    v = ec.ra769_limit(f_mhz*1e6, mode="vlbi")["slim_nu_db"]
    ct = ec.ra769_limit(f_mhz*1e6, mode="continuum")["slim_nu_db"]
    print(f"  {f_mhz:6d} MHz  VLBI {v:7.1f}  cont {ct:7.1f}")
