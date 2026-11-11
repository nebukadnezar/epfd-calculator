#!/usr/bin/env python3
# coding: utf-8
"""
epfd_angular_decomp.py — where does the aggregate EPFD power come from?

The paper states the aggregate is
"dominated by the satellites far from boresight". Is that true, and does a large
fraction actually come from close to the main beam, especially for high-N
constellations? This script tests it directly with the *production* EPFD engine.

It re-uses the exact code paths of the paper's results:
  * epfd_core._propagate         (cysgp4 propagation, real OMM orbits)
  * epfd_core.sample_on_cell_grid_m1583  (ITU-R M.1583 pointing grid)
  * pycraf antenna.ras_pattern   (ITU-R RA.1631 receiving pattern)
  * pycraf geometry / conversions (angular distance, free-space loss)
  * sniffles_inputs / omm_tools  (per-line EIRP, radiating subpopulation)

The ONLY change from epfd_core._iteration_prx is that we do NOT sum the
per-satellite received power over satellites; we instead retain each visible
satellite's off-boresight angle and bin its power contribution by that angle.
The per-cell total is identical to the production aggregate.

Outputs (per facility x line):
  * cumulative fraction of aggregate power vs off-boresight angle (curve + table)
  * fraction inside the main beam / <1 / <2 / <5 / <10 / <20 / <45 deg
  * the same decomposition restricted to the top-2% cells by aggregate power
    (the realisations that set the 98th-percentile compliance statistic)
  * the median off-boresight angle that carries the aggregate
A PNG (cumulative-fraction curves) and a text table are written to --out.

Note on N-scaling: the paper scales N by cloning the aggregate (linear in the
number of radiators), so the *mean* angular decomposition is N-invariant by
construction. What grows with N is the probability that a satellite lands within
a given angle of boresight in a single realisation, so the high tail (which sets
the 98th percentile) gains near-beam weight as N grows. The mean-vs-tail columns
below quantify that at the present population; the per-realisation main-beam-hit
probability scales linearly with N (reported at the end).

Copyright (C) 2025+ Indermuehle. GPLv3.
"""

import argparse
import os
import re
import time

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy import units as u, constants as const

from pycraf import conversions as cnv
from pycraf import antenna, geometry

import omm_tools as ot
import sniffles_inputs as si
import epfd_core as ec
import settings as cfg
from run_auscope import STATIONS, SOURCE_REGEX


# angle bin edges (deg): 0, then log-spaced 0.05..90, to resolve the near-in region
BIN_EDGES = np.concatenate([[0.0], np.logspace(np.log10(0.05), np.log10(90.0), 90)])
BIN_MIDS = 0.5 * (BIN_EDGES[1:] + BIN_EDGES[:-1])


def hpbw_deg(diam_m, cf_hz):
    """Half-power beamwidth (deg), ~1.22 lambda/D (first-null ~ same order)."""
    lam = const.c.value / cf_hz
    return np.degrees(1.22 * lam / diam_m)


def _iter_decomp(topo_el_i, topo_az_i, sat_dist_i, sat_el_i, sat_az_i,
                 tel_az_i, tel_el_i, n_tsteps, freq, p_tx, d_rx, eta_a_rx,
                 thresholds_deg):
    """One Monte-Carlo iteration, mirroring epfd_core._iteration_prx but
    binning per-satellite power by off-boresight angle instead of summing.

    Two histograms are accumulated, because "where the power comes from" has two
    legitimate readings that differ sharply:
      hist_pooled  : power-weighted, pooled over cells -> the MEAN aggregate,
                     dominated by the rare bright near-beam transits.
      hist_typical : each cell's contributions normalised by that cell's own
                     total before accumulation -> the mean of the PER-REALISATION
                     split, i.e. the typical (persistent) pointing.
    Returns (hist_pooled, hist_typical, n_nonempty, cell_total, cell_within).
      cell_total  : (ncells,)            aggregate power per cell (== production)
      cell_within : (ncells, nthresh)    power within each threshold angle per cell
    """
    ncells = tel_az_i.shape[0]
    nthr = len(thresholds_deg)
    hist_pooled = np.zeros(BIN_MIDS.size, dtype=np.float64)
    hist_typical = np.zeros(BIN_MIDS.size, dtype=np.float64)
    cell_total = np.zeros(ncells, dtype=np.float64)
    cell_within = np.zeros((ncells, nthr), dtype=np.float64)
    n_nonempty = 0

    vis = topo_el_i > 0.
    if np.count_nonzero(vis) == 0:
        return hist_pooled, hist_typical, n_nonempty, cell_total, cell_within

    FSPL = cnv.free_space_loss(sat_dist_i[vis, np.newaxis] * u.km, freq).to(cnv.dB)
    G_tx = np.zeros((np.count_nonzero(vis), 1)) * cnv.dBi   # 0 dBi isotropic Tx
    thr = np.asarray(thresholds_deg)

    chunk = 100
    nchunks = ncells // chunk + 1
    for c in range(nchunks):
        sl = slice(c * chunk, min((c + 1) * chunk, ncells))
        if sl.start >= ncells:
            break
        ang = geometry.true_angular_distance(
            tel_az_i[np.newaxis, sl] * u.deg, tel_el_i[np.newaxis, sl] * u.deg,
            topo_az_i[vis, np.newaxis] * u.deg, topo_el_i[vis, np.newaxis] * u.deg)
        G_rx = antenna.ras_pattern(ang, d_rx, const.c / freq, eta_a_rx)
        # per-(vis-sat, cell) received power [W], time-averaged like production
        p_lin = (p_tx + G_tx + FSPL + G_rx).to_value(u.W) / n_tsteps
        ang_deg = ang.to_value(u.deg)

        col_tot = p_lin.sum(axis=0)                       # aggregate per cell
        cell_total[sl] = col_tot
        for k, t in enumerate(thr):
            cell_within[sl, k] = np.where(ang_deg < t, p_lin, 0.0).sum(axis=0)
        # pooled (mean-aggregate) histogram
        hist_pooled += np.histogram(ang_deg.ravel(), bins=BIN_EDGES,
                                    weights=p_lin.ravel())[0]
        # per-realisation-normalised histogram (each cell weighted to unit total)
        good = col_tot > 0
        n_nonempty += int(np.count_nonzero(good))
        p_norm = np.zeros_like(p_lin)
        p_norm[:, good] = p_lin[:, good] / col_tot[good][np.newaxis, :]
        hist_typical += np.histogram(ang_deg.ravel(), bins=BIN_EDGES,
                                     weights=p_norm.ravel())[0]
    return hist_pooled, hist_typical, n_nonempty, cell_total, cell_within


def decompose(station, freq_mhz, omm_recs, epoch_mjd, rng, scenarios,
              niters=8, time_range=2000.0, time_step=2.0,
              grid_size=5.0, seed=1):
    """Run the angular decomposition for one facility x line.

    scenarios : dict {label: growth_factor G} for the breach-set analysis
                (G=1.0 is the present population; G=300000/N0 the 300k scenario).
    """
    S = STATIONS[station]
    line = next(r for r in si.load_inputs() if round(r["freq_mhz"]) == round(freq_mhz))
    cf = freq_mhz * 1e6
    freq = cf * u.Hz

    # radiating subpopulation, identical construction to run_auscope
    rgx = SOURCE_REGEX.get(line["source"])
    sub = [r for r in omm_recs if rgx and re.search(rgx, r.get("OBJECT_NAME", ""), re.I)]
    af = line["active_fraction"] if line["active_fraction"] is not None else 1.0
    n_emit = max(1, int(round(len(sub) * af)))
    if len(sub) > n_emit:
        idx = np.sort(rng.choice(len(sub), n_emit, replace=False))
        sub = [sub[i] for i in idx]
    tles = np.array(ot.to_pytles(sub), dtype=object)

    # EIRP bracket is irrelevant to the fractions (aggregate is linear in it); use atca
    eirp = line["eirp_atca"] if line["eirp_atca"] is not None else line["eirp_floor"]
    p_tx = eirp * cnv.dB_W
    d_rx = S["diam"] * u.m
    eta_q = 1.0 * 100 * u.percent
    min_el = S.get("min_el", 5)

    hpbw = hpbw_deg(S["diam"], cf)
    thresholds = [hpbw, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0]
    thr_labels = ["mainbeam(HPBW)", "<1deg", "<2deg", "<5deg",
                  "<10deg", "<20deg", "<45deg"]

    # time grid + pointing grid (production sampler)
    start_mjd = epoch_mjd + np.random.default_rng(seed).uniform(0, 1, niters)
    td = np.arange(0, time_range, time_step) / 86400.
    mjds = start_mjd[:, np.newaxis] + td[np.newaxis]
    n_tsteps = mjds.shape[1]
    tel_az, tel_el, grid_info = ec.sample_on_cell_grid_m1583(
        niters, step_size=grid_size * u.deg,
        lat_range=(min_el * u.deg, 90 * u.deg), rnd_seed=seed)
    ncells = len(grid_info)

    res = ec._propagate(mjds, tles, S["lon"], S["lat"], S["alt"])
    topo = res["topo"]; sazel = res["sat_azel"]
    topo_az, topo_el = topo[..., 0], topo[..., 1]
    sat_az, sat_el, sat_dist = sazel[..., 0], sazel[..., 1], sazel[..., 2]

    hist_pooled = np.zeros(BIN_MIDS.size)
    hist_typical = np.zeros(BIN_MIDS.size)
    n_nonempty = 0
    cell_tot_all, cell_within_all = [], []
    for i in range(niters):
        hp, ht, nne, ct, cw = _iter_decomp(
            topo_el[i], topo_az[i], sat_dist[i], sat_el[i], sat_az[i],
            tel_az[i], tel_el[i], n_tsteps, freq, p_tx, d_rx, eta_q, thresholds)
        hist_pooled += hp
        hist_typical += ht
        n_nonempty += nne
        cell_tot_all.append(ct)
        cell_within_all.append(cw)
    cell_tot = np.concatenate(cell_tot_all)                # (niters*ncells,)
    cell_within = np.concatenate(cell_within_all, axis=0)  # (niters*ncells, nthr)

    # per-realisation fractional split (guard empty cells)
    good = cell_tot > 0
    frac_per = np.zeros_like(cell_within)
    frac_per[good] = cell_within[good] / cell_tot[good][:, None]

    # (1) POOLED / mean-aggregate split: power-weighted over all realisations
    frac_pooled = cell_within.sum(axis=0) / cell_tot.sum()
    # (2) TYPICAL realisation: mean over realisations of the per-cell split
    frac_typical = frac_per[good].mean(axis=0)
    # (3) BREACHING realisations: those exceeding the RA.769 S_lim, at the present
    #     population and the 300k scenario (linear scaling shifts the level, not the
    #     split, so we just move the threshold crossing). Convert received power ->
    #     incident spectral PFD exactly as production.
    pfd_now = (cnv.powerflux_from_prx(cell_tot * 1e60 * u.W, freq, 0 * cnv.dBi)
               * 1e-60).to_value(u.W / u.m**2)
    C = pfd_now.sum() / cell_tot.sum()            # scalar W->PFD/Hz conversion
    # count-based attribution: a breach is "near-beam driven" at cutoff angle k if
    # removing the power within that angle would drop it back below the limit.
    cut_idx = {"HPBW": 0, "<2deg": 2, "<5deg": 3}
    breach = {}
    for scen, G in scenarios.items():
        pfd = pfd_now * G
        b = {}
        for mode in ("vlbi", "continuum"):
            slim = ec.ra769_limit(cf, mode=mode)["Slim_nu_W_m2_Hz"]
            sel = pfd > slim
            nb = int(np.count_nonzero(sel))
            entry = dict(loss_pct=100.0 * nb / cell_tot.size, n_breach=nb,
                         frac=None, nearbeam_driven={})
            if nb:
                entry["frac"] = cell_within[sel].sum(axis=0) / cell_tot[sel].sum()
                for lab, k in cut_idx.items():
                    pfd_excl = (cell_tot - cell_within[:, k]) * C * G
                    driven = sel & (pfd_excl <= slim)
                    entry["nearbeam_driven"][lab] = 100.0 * np.count_nonzero(driven) / nb
            b[mode] = entry
        breach[scen] = b

    # cumulative fraction vs angle (pooled and typical), median-carrying angle
    cum_pooled = np.cumsum(hist_pooled) / hist_pooled.sum()
    cum_typical = np.cumsum(hist_typical) / hist_typical.sum()
    med_pooled = np.interp(0.5, cum_pooled, BIN_MIDS)
    med_typical = np.interp(0.5, cum_typical, BIN_MIDS)

    return dict(
        station=station, name=S["name"], diam=S["diam"], min_el=min_el,
        freq_mhz=freq_mhz, n_emit=len(sub), niters=niters, ncells=ncells,
        hpbw_deg=hpbw, thresholds=thresholds, thr_labels=thr_labels,
        frac_pooled=frac_pooled, frac_typical=frac_typical, breach=breach,
        cum_pooled=cum_pooled, cum_typical=cum_typical,
        median_pooled_deg=med_pooled, median_typical_deg=med_typical)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omm", default=cfg.OMM)
    ap.add_argument("--cases", nargs="*", default=["Yg:7860", "Pks:7860", "Yg:2656"],
                    help="facility:freqMHz pairs (facility keys from run_auscope.STATIONS)")
    ap.add_argument("--niters", type=int, default=8)
    ap.add_argument("--time-range", type=float, default=2000.0)
    ap.add_argument("--time-step", type=float, default=2.0)
    ap.add_argument("--grid-size", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(cfg.FIGS, "angular_decomp"))
    ap.add_argument("--fig", default=None,
                    help="PNG path for the paper figure (default: <out>.png)")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure from the cached curves (<out>.npz), no recompute")
    ap.add_argument("--n-now", type=float, default=None,
                    help="present total LEO N0 for the N-scaling hit-probability note")
    ap.add_argument("--n-future", type=float, default=300000.,
                    help="future total N for the hit-probability note")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cache = args.out + ".npz"
    if args.replot:
        d = np.load(cache, allow_pickle=True)
        plot_curves(list(d["results"]), args.fig or args.out + ".png")
        return
    print(f"Loading OMM {args.omm} ...")
    recs = ot.load_omm(args.omm)
    sample = recs[:: max(1, len(recs) // 200)]
    epoch_mjd = float(np.median([ot._epoch_to_mjd(r["EPOCH"]) for r in sample]))
    rng = np.random.default_rng(args.seed)

    total_leo = ["Starlink", "OneWeb", "Kuiper", "GuoWang", "Qianfan"]
    n0 = sum(1 for r in recs if r["CONSTELLATION"] in total_leo)
    if args.n_now is None:
        args.n_now = float(n0)
    scenarios = {"now": 1.0, "300k": args.n_future / n0}
    print(f"present total LEO N0 = {n0}; 300k growth x{scenarios['300k']:.1f}")

    results = []
    for case in args.cases:
        stn, fr = case.split(":")
        fr = float(fr)
        t0 = time.time()
        print(f"decomposing {stn} @ {fr:.0f} MHz ...")
        r = decompose(stn, fr, recs, epoch_mjd, rng, scenarios,
                      niters=args.niters, time_range=args.time_range,
                      time_step=args.time_step, grid_size=args.grid_size,
                      seed=args.seed)
        r["case"] = case
        results.append(r)
        print(f"   {r['n_emit']} radiators, {r['ncells']} cells, "
              f"HPBW={r['hpbw_deg']:.3f} deg, median-carrying angle "
              f"pooled={r['median_pooled_deg']:.2f} / typical="
              f"{r['median_typical_deg']:.2f} deg  [{time.time()-t0:.0f}s]")

    # ---- text table ----
    txt = os.path.join(os.path.dirname(args.out) or ".",
                       os.path.basename(args.out) + ".txt")
    with open(txt, "w") as fh:
        def w(s=""):
            print(s); fh.write(s + "\n")
        w("EPFD angular decomposition — fraction of aggregate power vs "
          "off-boresight angle")
        w(f"(niters={args.niters}, time {args.time_range:.0f}s/{args.time_step:.0f}s, "
          f"grid {args.grid_size:.0f}deg, RA.1631 pattern, real OMM orbits, "
          f"N0={n0:.0f})")
        w("")
        w("Two readings of 'where the power comes from':")
        w("  MEAN     = power-weighted over all pointings; set by the rare bright "
          "main-beam transits.")
        w("  TYPICAL  = mean of the per-pointing split; the persistent pointing "
          "that drives the data-loss statistic.")
        w("  BREACH   = mean split over only the pointings that exceed the RA.769 "
          "VLBI limit at each scenario.")
        w("")
        for r in results:
            w(f"== {r['name']} ({r['diam']:.0f} m, min_el {r['min_el']}deg) "
              f"@ {r['freq_mhz']:.0f} MHz ; {r['n_emit']} radiators ==")
            w(f"   HPBW ~ {r['hpbw_deg']:.3f} deg ; median-carrying off-boresight "
              f"angle: MEAN={r['median_pooled_deg']:.2f} deg, "
              f"TYPICAL={r['median_typical_deg']:.2f} deg")
            bd = r["breach"]["300k"]["vlbi"]
            fb = bd["frac"]
            w(f"   {'region':<16}{'MEAN %':>10}{'TYPICAL %':>12}"
              f"{'BREACH% (300k,VLBI)':>22}")
            for k, lab in enumerate(r["thr_labels"]):
                bcol = f"{100*fb[k]:>22.2f}" if fb is not None else f"{'--':>22}"
                w(f"   {lab:<16}{100*r['frac_pooled'][k]:>10.2f}"
                  f"{100*r['frac_typical'][k]:>12.2f}{bcol}")
            brest = f"{100*(1-fb[-1]):>22.2f}" if fb is not None else f"{'--':>22}"
            w(f"   {'>=45deg (rest)':<16}{100*(1-r['frac_pooled'][-1]):>10.2f}"
              f"{100*(1-r['frac_typical'][-1]):>12.2f}{brest}")
            w(f"   (VLBI data loss: now={r['breach']['now']['vlbi']['loss_pct']:.1f}%, "
              f"300k={r['breach']['300k']['vlbi']['loss_pct']:.1f}%; "
              f"continuum 300k={r['breach']['300k']['continuum']['loss_pct']:.1f}%)")
            w("   (MEAN/TYPICAL/BREACH rows are cumulative: power within that "
              "angle of boresight)")
            # count-based attribution: what fraction of breaching pointings STOP
            # breaching if the near-beam power (within a cutoff) is removed
            for mode in ("vlbi", "continuum"):
                for scen in ("now", "300k"):
                    e = r["breach"][scen][mode]
                    if e["n_breach"] and e["nearbeam_driven"]:
                        nd = e["nearbeam_driven"]
                        w(f"   near-beam-DRIVEN breaches [{mode},{scen}] "
                          f"(un-breach if power within cutoff removed): "
                          f"HPBW {nd['HPBW']:.0f}%, <2deg {nd['<2deg']:.0f}%, "
                          f"<5deg {nd['<5deg']:.0f}%  (of {e['n_breach']} breaching, "
                          f"loss {e['loss_pct']:.1f}%)")
            w("")
    print(f"\nwrote table -> {txt}")

    keep = ("name", "diam", "freq_mhz", "hpbw_deg", "cum_typical", "cum_pooled")
    np.savez(cache, results=np.array([{k: r[k] for k in keep} for r in results],
                                     dtype=object))
    print(f"wrote cache -> {cache}")
    plot_curves(results, args.fig or args.out + ".png")


def plot_curves(results, png):
    """Cumulative-fraction plot: MEAN (dashed) vs TYPICAL (solid).

    Styled to match the paper's analysis figures (cf. plot_slim_interp.py):
    same canvas, all text at 13.5 pt, light "both" grid, no title (the
    caption carries it)."""
    plt.rcParams.update({"font.size": 13.5, "axes.labelsize": 13.5,
                         "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
                         "legend.fontsize": 13.5, "legend.title_fontsize": 13.5})
    CASE_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    case_handles = []
    for r, c in zip(results, CASE_COLORS):
        lab = f"{r['name']} {r['diam']:.0f} m @ {r['freq_mhz']:.0f} MHz"
        ln, = ax.semilogx(BIN_MIDS, 100 * r["cum_typical"], lw=1.8, color=c,
                          label=lab)
        ax.semilogx(BIN_MIDS, 100 * r["cum_pooled"], lw=1.4, ls="--", color=c)
        ax.axvline(r["hpbw_deg"], ls=":", lw=1.4, color=c, alpha=0.9, zorder=2)
        case_handles.append(ln)
    ax.axhline(50, color="0.55", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel("off-boresight angle of contributing satellites (deg)")
    ax.set_ylabel("cumulative fraction of aggregate EPFD power (%)")
    ax.grid(alpha=0.25, which="both")
    ax.set_ylim(0, 100); ax.set_xlim(BIN_MIDS[0], 90)
    ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(["0.1", "1", "10"])
    # two legends: cases (colour) and the typical/mean linestyle key
    style_handles = [
        Line2D([], [], color="0.35", lw=1.8, ls="-",
                   label="typical pointing (persistent loss)"),
        Line2D([], [], color="0.35", lw=1.4, ls="--",
                   label="mean, power-weighted (bright transits)")]
    # one legend below the axes (cases in the left column, line styles in the
    # right); the larger text no longer fits inside the plot without covering curves
    fig.tight_layout()
    fig.legend(handles=case_handles + style_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.savefig(png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"wrote plot  -> {png}")


if __name__ == "__main__":
    main()
