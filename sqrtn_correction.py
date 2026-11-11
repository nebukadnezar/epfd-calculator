#!/usr/bin/env python3
# coding: utf-8
"""
sqrtn_correction.py — corrected N-scaling of the aggregate EPFD distribution.

Reviewer request (Winkel): the rigid CDF shift (analytic +10log10 G) ignores
that the aggregate's fluctuations grow only as sqrt(N) while its mean grows as
N, so the true distribution narrows as the sky fills (fig:linval). This script
implements and validates two corrected models against the saved TRUE up-scaled
Monte-Carlo results (data/linearity_validation_full.log.csv):

  analytic : x' = G * x                            (rigid shift, production)
  gauss    : x' = G*mu_c + sqrt(G)*(x - mu_c)      (CLT scaling, per cell c)
  bootsum  : x' = sum of m iid per-cell draws + frac * one more,  G = m + frac
             (compound resampling: an up-scaled fleet of G independent clones
              of the present fleet IS the sum of G iid aggregates, which is
              exactly how the true validation fleets were constructed)

The now-population arrays behind the saved validation rows were not cached, so
the two small now-runs are recomputed here with the IDENTICAL settings used by
validate_linearity.py (station Hb, seed 1, niters 30, 2000 s at 4 s steps);
losses are compared against the RAW RA.769 VLBI threshold exactly as in the
saved CSV (no bandwidth correction), since the comparison tests distribution
shape, not threshold convention.

Output: console table (true vs analytic vs gauss vs bootsum), the result
rows cached to output/sqrtn_correction.csv, the comparison figure
(--fig PATH, default output/figs/fig_sqrtn_correction.png; two-column
manuscript figure, 11 pt), and the production-basis corrected margins at
100k/300k from the cached output/pfd arrays for the main detections.
--replot re-draws the figure from the cached rows without recomputing.

IMPORTANT: run ALONE (no other heavy compute).
"""

import argparse
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import omm_tools as ot
import epfd_core as ec
import sniffles_inputs as si
import settings as cfg
from validate_linearity import build_subpop
from run_auscope import STATIONS, TOTAL_LEO_CONSTELLATIONS

NSYN = 400          # synthetic iterations for the resampled ensembles
SEED = 1


def upscale_gauss(pfd, G, rng):
    """CLT scaling per cell: G*mu + sqrt(G)*(x-mu)."""
    mu = pfd.mean(axis=0, keepdims=True)
    return G * mu + math.sqrt(G) * (pfd - mu)


def upscale_bootsum(pfd, G, rng, nsyn=NSYN):
    """Compound resampling per cell: sum of floor(G) iid iteration draws plus
    frac * one further draw. Returns (nsyn, ncells)."""
    ni, nc = pfd.shape
    m, frac = int(math.floor(G)), G - math.floor(G)
    out = np.zeros((nsyn, nc))
    for k in range(m):
        out += pfd[rng.integers(0, ni, nsyn), :][..., :]
    if frac > 0:
        out += frac * pfd[rng.integers(0, ni, nsyn), :]
    return out


def p98db(x):
    return 10 * math.log10(np.percentile(x, 98.))


def loss_pct(x, slim):
    return 100.0 * np.mean(x.flatten() > slim)


def plot_figure(rows_out, out):
    """Two-column manuscript figure (fig:sqrtncorr): 11 pt text, 300 dpi."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter
    FS = 11
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
                         "ytick.labelsize": FS, "legend.fontsize": FS})
    lines = sorted({r["freq"] for r in rows_out})
    MODELS = [("analytic", "tab:red", "--", "rigid shift ($\\times G$)"),
              ("gauss", "tab:green", ":", "CLT scaling ($\\sqrt{G}$ fluctuations)"),
              ("bootsum", "tab:blue", "-", "compound resampling")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
    for fr, mk in zip(lines, ("o", "s")):
        rs = sorted([r for r in rows_out if r["freq"] == fr], key=lambda r: r["target"])
        Ns = [r["target"] for r in rs]
        for key, col, ls, _ in MODELS:
            axes[0].plot(Ns, [r[f"p98_{key}"] - r["p98_true"] for r in rs],
                         marker=mk, color=col, ls=ls, lw=2.0, ms=8, mec="k", mew=0.6)
            axes[1].plot(Ns, [r[f"loss_{key}"] - r["loss_true"] for r in rs],
                         marker=mk, color=col, ls=ls, lw=2.0, ms=8, mec="k", mew=0.6)
    ticks = sorted({r["target"] for r in rows_out})
    for ax in axes:
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("total LEO satellites")
        ax.set_xticks(ticks)                      # the simulated populations
        ax.set_xticklabels([f"{t/1000:.0f}k" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("98th-percentile aggregate, model $-$ true (dB)")
    axes[1].set_ylabel("VLBI data loss, model $-$ true (percentage points)")
    model_handles = [Line2D([], [], color=c, ls=s, lw=2.0, label=l) for _, c, s, l in MODELS]
    line_handles = [Line2D([], [], color="0.35", marker=m, lw=0, ms=8,
                           mec="k", mew=0.6, label=f"{fr:.0f} MHz")
                    for fr, m in zip(lines, ("o", "s"))]
    axes[0].set_ylim(top=axes[0].get_ylim()[1] + 1.8)   # headroom for the legend
    axes[0].legend(handles=model_handles, loc="upper left")
    axes[1].legend(handles=line_handles, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omm", default=cfg.OMM)
    ap.add_argument("--valcsv", default=cfg.ref("linearity_validation_full.log.csv"))
    ap.add_argument("--niters", type=int, default=30)
    ap.add_argument("--time-step", type=float, default=4.0)
    ap.add_argument("--cache", default=cfg.out("sqrtn_correction.csv"))
    ap.add_argument("--fig", default=os.path.join(cfg.FIGS, "fig_sqrtn_correction.png"))
    ap.add_argument("--replot", action="store_true",
                    help="re-draw the figure from --cache, no recompute")
    args = ap.parse_args()

    if args.replot:
        rows_out = list(csv.DictReader(open(args.cache)))
        for r in rows_out:
            for k in r:
                r[k] = int(r[k]) if k == "target" else float(r[k])
        plot_figure(rows_out, args.fig)
        return

    val = list(csv.DictReader(open(args.valcsv)))
    for r in val:
        for k in r:
            if k not in ("station",):
                r[k] = float(r[k])
    lines = sorted({r["freq_mhz"] for r in val})
    station = val[0]["station"]
    S = STATIONS[station]

    rng = np.random.default_rng(SEED)
    print(f"Loading OMM {args.omm} ...")
    recs_all = ot.load_omm(args.omm)
    current_total = sum(1 for r in recs_all
                        if r["CONSTELLATION"] in TOTAL_LEO_CONSTELLATIONS)
    epoch_mjd = float(np.median([ot._epoch_to_mjd(r["EPOCH"])
                                 for r in recs_all[:: max(1, len(recs_all)//200)]]))
    all_lines = {round(r["freq_mhz"]): r for r in si.load_inputs()}

    rows_out = []
    for fr in lines:
        line = all_lines[round(fr)]
        eirp = line["eirp_atca"] if line["eirp_atca"] is not None else line["eirp_floor"]
        cf = fr * 1e6
        sub = build_subpop(recs_all, line, np.random.default_rng(SEED))
        sub_tles = np.array(ot.to_pytles(sub), dtype=object)
        print(f"\n=== {fr:.0f} MHz : recomputing now-population "
              f"({len(sub)} radiators, validation settings) ===")
        r0 = ec.compute_epfd(
            sub_tles, S["lon"], S["lat"], S["alt"], S["diam"], cf, eirp,
            modes=("vlbi",), min_el=S.get("min_el", 5), grid_size=5.0,
            niters=args.niters, time_range=2000.0, time_step=args.time_step,
            mjd_epoch=epoch_mjd, seed=SEED, make_plots=False, return_arrays=True)
        pfd_now = r0["pfd_nu"]
        slim = ec.ra769_limit(cf, mode="vlbi")["Slim_nu_W_m2_Hz"]
        print(f"  now p98 = {p98db(pfd_now):.2f} dB "
              f"(saved validation run: {val[0]['pfd98_now_db'] if val[0]['freq_mhz']==fr else [v['pfd98_now_db'] for v in val if v['freq_mhz']==fr][0]:.2f})")

        for v in [v for v in val if v["freq_mhz"] == fr]:
            G = v["target"] / current_total
            an = pfd_now * G
            ga = upscale_gauss(pfd_now, G, rng)
            bs = upscale_bootsum(pfd_now, G, rng)
            row = dict(freq=fr, target=int(v["target"]),
                       p98_true=v["pfd98_true_db"],
                       p98_analytic=p98db(an), p98_gauss=p98db(ga),
                       p98_bootsum=p98db(bs),
                       loss_true=v["loss_true_pct"],
                       loss_analytic=loss_pct(an, slim),
                       loss_gauss=loss_pct(ga, slim),
                       loss_bootsum=loss_pct(bs, slim))
            rows_out.append(row)
            print(f"  N={row['target']:>6}: p98 true {row['p98_true']:7.2f} | "
                  f"analytic {row['p98_analytic']:7.2f} "
                  f"({row['p98_analytic']-row['p98_true']:+.2f}) | "
                  f"gauss {row['p98_gauss']:7.2f} "
                  f"({row['p98_gauss']-row['p98_true']:+.2f}) | "
                  f"bootsum {row['p98_bootsum']:7.2f} "
                  f"({row['p98_bootsum']-row['p98_true']:+.2f})")
            print(f"            loss true {row['loss_true']:5.1f}% | "
                  f"analytic {row['loss_analytic']:5.1f} | "
                  f"gauss {row['loss_gauss']:5.1f} | "
                  f"bootsum {row['loss_bootsum']:5.1f}")

    # ---- cache the rows, then the comparison figure (appendix fig:sqrtncorr) ----
    with open(args.cache, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {args.cache}")
    plot_figure(rows_out, args.fig)

    # ---- production-basis corrected margins at the paper scenarios ----
    print("\n--- production caches: corrected p98 shift vs analytic (dB) ---")
    for fr in lines:
        try:
            d = np.load(os.path.join(cfg.PFD, f"{station}_{fr:.0f}_atca.npz"))
        except FileNotFoundError:
            continue
        pfd = d["pfd_nu"]
        for target in (100_000.0, 300_000.0):
            G = target / current_total
            corr = (p98db(pfd * G)
                    - p98db(upscale_bootsum(pfd, G, np.random.default_rng(SEED))))
            print(f"  {fr:6.0f} MHz  N={target:8.0f}:  analytic p98 is "
                  f"{corr:+.2f} dB above corrected (E_lim conservative by that)")


if __name__ == "__main__":
    main()
