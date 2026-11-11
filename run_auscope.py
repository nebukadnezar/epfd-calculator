#!/usr/bin/env python3
# coding: utf-8
"""
run_auscope.py — multi-station EPFD driver for the geoVLBI coexistence study.

For each AuScope station x SNIFFLES emission line x level-bracket, run the EPFD
engine at the CURRENT radiating population (real OMM orbits), then analytically
scale the aggregate to the 100k / 300k total-LEO scenarios (the aggregate spectral
PFD is linear in the number of radiators, same spatial distribution -> shift the
CDF by 10*log10(N_ratio)). Compare against BOTH RA.769 criteria (single-dish RAS
continuum, and VLBI). Emit sky-coverage + CDF plots and a results CSV.

Linearity of the analytic scaling is validated separately with a few true
up-scaled EPFD runs (see --validate).

"""

import argparse
import csv
import os
import re
import time

# Single-thread the BLAS libs BEFORE numpy is imported: we parallelise the
# data-loss step over processes (joblib/loky), so per-worker BLAS threads only
# oversubscribe. NOTE: do NOT pin OMP_NUM_THREADS here -- cysgp4.propagate_many
# parallelises the (parent-side) propagation via OpenMP and has no thread arg, so
# OMP_NUM_THREADS must stay >1 for fast propagation. Set it via the launcher
# (e.g. export OMP_NUM_THREADS=48); it defaults to all cores if unset.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import omm_tools as ot
import epfd_core as ec
import sniffles_inputs as si
import settings as cfg

# --- Observer stations (edit coords/diam/min_el as confirmed) --------------
# AuScope VGOS (12 m broadband); plus major Australian RAS facilities. Coords
# from ARSP footnotes AUS87/AUS103. min_el per station (Parkes has a ~30 deg
# elevation limit). diam in m, alt in km.
# fmin/fmax (MHz) = the facility's observing-band coverage; lines outside are
# skipped ("facilities per their band"). VGOS hardware is 2-14 GHz broadband.
STATIONS = cfg.stations()      # stations.json (see config.json)

# --- map SNIFFLES 'source' -> OMM OBJECT_NAME regex (the radiating subpop) ---
# Editable. DTC = Starlink 11xxx (SNIFFLES); V2-mini/other approximated by all
# Starlink (v2-mini dominate the fleet). Refine as version tagging improves.
SOURCE_REGEX = {
    "Starlink V2-mini DTD":     r"STARLINK-11\d{3}",
    # Ku FSS downlink (10.7-12.7 GHz) is carried by the standard Starlink fleet
    # (exclude the 11xxx DTC-only series via negative lookahead) AND by OneWeb,
    # whose real OMM orbits (~1200 km polar) are propagated with the same
    # per-satellite EIRP density.
    "Starlink/OneWeb Ku":       r"STARLINK-(?!11\d{3})\d|ONEWEB",
    "Starlink V2-mini Ku":      r"STARLINK-(?!11\d{3})\d",
    "Starlink V2-mini Ku/std":  r"STARLINK",
    "Starlink V2-mini":         r"STARLINK",
    "Starlink/other":           r"STARLINK",
    "OneWeb":                   r"ONEWEB",
    "GuoWang":                  r"HULIANWANG|GUOWANG|^GW-",
}

# total-LEO growth scenarios (the aggregate scales by target/current_total).
# Finer ladder so the breach-vs-N figure resolves the present->2M sweep; the
# aggregate is linear in N so the extra scenarios are free (analytic CDF shift).
SCENARIOS = {
    "now": None, "30k": 30_000, "50k": 50_000, "100k": 100_000,
    "200k": 200_000, "300k": 300_000, "500k": 500_000,
    "1m": 1_000_000, "2m": 2_000_000,
}

# constellations counted as "current total LEO" for the growth factor
TOTAL_LEO_CONSTELLATIONS = ["Starlink", "OneWeb", "Kuiper", "GuoWang", "Qianfan"]

# memory guard: bound the propagate_many array (niters*ntsteps*sat_chunk
# elements, 7 float64 fields = 56 bytes each). compute_epfd now propagates the
# constellation in SATELLITE CHUNKS sized to keep each chunk's array under
# PROP_BUDGET elements, accumulating the (additive) per-cell power across chunks.
# So niters and total N are decoupled from memory: a million-satellite line runs
# at full niters, bounded by chunk size; only wall-clock grows. Raise the budget
# on big-memory hosts via --prop-budget for fewer chunks.
PROP_BUDGET = cfg.PROP_BUDGET


def _epoch_mjd(records):
    """Median TLE epoch (MJD) of the records — propagation must use this."""
    sample = records[:: max(1, len(records) // 200)]
    return float(np.median([ot._epoch_to_mjd(r["EPOCH"]) for r in sample]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omm", default=cfg.OMM)
    ap.add_argument("--stations", nargs="*", default=list(STATIONS))
    ap.add_argument("--lines", nargs="*", type=float,
                    help="restrict to these line freqs (MHz); default all")
    ap.add_argument("--brackets", nargs="*", default=["floor", "atca"])
    ap.add_argument("--niters", type=int, default=20,
                    help="iterations for the data-loss statistics (converged ~20)")
    ap.add_argument("--plot-niters", type=int, default=200,
                    help="iterations for the plotted (ATCA-bracket) cases, so the "
                         "per-cell sky maps are smooth (~200)")
    ap.add_argument("--time-step", type=float, default=2.0)
    ap.add_argument("--time-range", type=float, default=2000.0)
    ap.add_argument("--grid-size", type=float, default=5.0)
    ap.add_argument("--min-el", type=float, default=5.0)
    ap.add_argument("--n-jobs", type=int, default=cfg.N_JOBS,
                    help="parallel workers (-1 = all cores)")
    ap.add_argument("--prop-budget", type=float, default=PROP_BUDGET,
                    help="max niters*ntsteps*sat_chunk elements per propagation "
                         "chunk (~56 bytes each); raise on big-memory hosts")
    ap.add_argument("--figpath", default=cfg.FIGS)
    ap.add_argument("--outcsv", default=os.path.join(cfg.OUT, "auscope_results.csv"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    modes = ("continuum", "vlbi")
    rng = np.random.default_rng(args.seed)

    print(f"Loading OMM {args.omm} ...")
    recs_all = ot.load_omm(args.omm)
    current_total = sum(1 for r in recs_all
                        if r["CONSTELLATION"] in TOTAL_LEO_CONSTELLATIONS)
    epoch_mjd = _epoch_mjd(recs_all)
    print(f"current total LEO (modeled) = {current_total}; epoch MJD = {epoch_mjd:.3f}")
    print(f"growth factors: 100k x{100000/current_total:.2f}, 300k x{300000/current_total:.2f}")

    rows = si.load_inputs()
    if args.lines:
        want = {round(f) for f in args.lines}
        rows = [r for r in rows if round(r["freq_mhz"]) in want]

    # build radiating PyTles per line ONCE (observer-independent)
    line_tles = {}
    for ln in rows:
        rgx = SOURCE_REGEX.get(ln["source"])
        sub = [r for r in recs_all if rgx and re.search(rgx, r.get("OBJECT_NAME", ""), re.I)]
        af = ln["active_fraction"] if ln["active_fraction"] is not None else 1.0
        n_emit = max(1, int(round(len(sub) * af)))
        if len(sub) > n_emit:
            idx = np.sort(rng.choice(len(sub), n_emit, replace=False))
            sub = [sub[i] for i in idx]
        line_tles[ln["freq_mhz"]] = (np.array(ot.to_pytles(sub), dtype=object),
                                     len(sub))
        print(f"  line {ln['freq_mhz']:.0f} MHz [{ln['source']}]: "
              f"{len(sub)} radiators (af={af})")

    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    results = []
    t_start = time.time()
    for stn in args.stations:
        S = STATIONS[stn]
        for ln in rows:
            # skip lines outside this facility's observing band
            if not (S.get("fmin", 0) <= ln["freq_mhz"] <= S.get("fmax", 1e9)):
                continue
            tles, n_emit_now = line_tles[ln["freq_mhz"]]
            cf = ln["freq_mhz"] * 1e6
            for bracket in args.brackets:
                eirp = ln["eirp_floor"] if bracket == "floor" else ln["eirp_atca"]
                if eirp is None:
                    continue
                lbl = f"{stn}_{ln['freq_mhz']:.0f}MHz"
                # plot the measured (ATCA) bracket only; floor is a uniform dB
                # shift of the same distribution and is kept in the CSV.
                make_plots = (bracket == "atca")
                # data-loss statistics converge by ~20 iterations; the per-cell sky
                # maps need more, so plotted (ATCA) cases run at --plot-niters.
                req_niters = args.plot_niters if make_plots else args.niters
                t0 = time.time()
                r = ec.compute_epfd(
                    tles, S["lon"], S["lat"], S["alt"], S["diam"], cf, eirp,
                    modes=modes, min_el=S.get("min_el", args.min_el),
                    grid_size=args.grid_size,
                    niters=req_niters, time_range=args.time_range,
                    time_step=args.time_step, n_jobs=args.n_jobs,
                    mjd_epoch=epoch_mjd, seed=args.seed,
                    figpath=args.figpath, label=f"{lbl}_now",
                    obslabel=S["name"], make_plots=make_plots, return_arrays=True,
                    prop_budget=args.prop_budget)
                pfd_nu = r["pfd_nu"]; grid = r["grid_info"]
                # cache the now-population aggregate so any future constellation
                # size can be derived analytically (pfd_nu * N/N0) without
                # re-propagating — see derive_from_pfd in analyze_results.py.
                pfddir = os.path.join(os.path.dirname(args.outcsv), "pfd")
                os.makedirs(pfddir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(pfddir, f"{stn}_{ln['freq_mhz']:.0f}_{bracket}.npz"),
                    pfd_nu=pfd_nu, cf_hz=cf, eirp_dbw_hz=eirp,
                    n_emit_now=n_emit_now, current_total=current_total,
                    station=stn, eclass=ln["eclass"], source=ln["source"])
                for scen, target in SCENARIOS.items():
                    G = 1.0 if target is None else target / current_total
                    pfd_s = pfd_nu * G
                    m, per_cell = ec.metrics_from_pfd(pfd_s, cf, modes=modes,
                                                      n_cells=len(grid))
                    if scen != "now" and make_plots:   # scaled CDF (ATCA bracket)
                        ec._plot_cdf(pfd_s, m,
                            title=f"{S['name']}  {ln['freq_mhz']:.0f} MHz  "
                                  f"{scen} (xG={G:.1f}, {int(n_emit_now*G)} emitters)",
                            path=os.path.join(args.figpath, f"{lbl}_{scen}_cdf.png"))
                    for mode in modes:
                        results.append(dict(
                            station=stn, freq_mhz=ln["freq_mhz"],
                            eclass=ln["eclass"], source=ln["source"],
                            bracket=bracket, eirp_dbw_hz=round(eirp, 1),
                            scenario=scen, growth=round(G, 2),
                            n_emit=int(round(n_emit_now * G)), mode=mode,
                            data_loss_pct=round(m[mode]["data_loss_pct"], 3),
                            slim_db=m[mode]["slim_nu_db"],
                            margin_db=round(m[mode]["ras_margin_db"], 2),
                            interp=m[mode]["interpolated"]))
                print(f"  {lbl}: now-loss(cont/vlbi)="
                      f"{r['modes']['continuum']['data_loss_pct']:.1f}/"
                      f"{r['modes']['vlbi']['data_loss_pct']:.1f}%  "
                      f"[{time.time()-t0:.0f}s]")

    keys = list(results[0].keys())
    with open(args.outcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(results)
    print(f"\nWrote {len(results)} rows to {args.outcsv} in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
