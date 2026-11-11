#!/usr/bin/env python3
# coding: utf-8
"""
window_validation.py -- VGOS integration-window sensitivity test (task #5).

The production EPFD averages the aggregate over a time_range=2000 s window, the
RA.769 single-dish reference integration. But a geodetic VGOS scan is 10-30 s,
and the VLBI threshold is tau-INDEPENDENT (1% of receiver noise power; confirmed
in RA.769 and Report ITU-R RA.2507), so for VLBI the 2000 s averaging has no
offsetting threshold conservatism and may understate the peak a short scan sees.

Model: compute_epfd holds the telescope pointing fixed for the whole window while
the satellites move, so a run with time_range=W IS a W-second scan at that
pointing. Sweeping W over {10, 30, 100, 2000} s with random start times, the 98th
percentile of the (start-time x cell) aggregate is the "2% worst scan" at each
scan length -- bad transits included. W=10/30 s = realistic VGOS scan bracket;
W=100 s = RA.2507's VGOS-continuum anchor; W=2000 s reproduces the production
tab:key (consistency check). Only the averaging window varies; the tau-independent
VLBI threshold is fixed.

Deliverable: N_breach and 98th-pct VLBI data loss vs window length per line, where
N_breach = current_total * 10**(margin_now/10), margin_now = slim_db - pfd98_db.

"""
import argparse, csv, os, re, time
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import omm_tools as ot
import epfd_core as ec
import sniffles_inputs as si
import settings as cfg
from run_auscope import (STATIONS, SOURCE_REGEX, TOTAL_LEO_CONSTELLATIONS,
                         _epoch_mjd)

# binding (present-fleet) + marginal (N_breach near the scenarios) VLBI lines
DEFAULT_LINES = [2595, 2620, 2700, 5240, 7860, 10480]
# window_s : niters  (more iters for shorter windows -> enough independent
# orbital configs to resolve the 98th percentile; short windows are cheap)
DEFAULT_WINDOWS = {10: 300, 30: 200, 100: 100, 2000: 50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omm", default=cfg.OMM)
    ap.add_argument("--stations", nargs="*", default=["Hb", "Ke", "Yg"])
    ap.add_argument("--lines", nargs="*", type=float, default=DEFAULT_LINES)
    ap.add_argument("--windows", default="10:300,30:200,100:100,2000:50",
                    help="comma list of window_s:niters")
    ap.add_argument("--grid-size", type=float, default=5.0)
    ap.add_argument("--n-jobs", type=int, default=cfg.N_JOBS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outcsv", default=os.path.join(cfg.OUT, "window_validation.csv"))
    args = ap.parse_args()

    windows = {}
    for tok in args.windows.split(","):
        w, n = tok.split(":")
        windows[int(w)] = int(n)

    rng = np.random.default_rng(args.seed)
    print(f"Loading OMM {args.omm} ...")
    recs_all = ot.load_omm(args.omm)
    current_total = sum(1 for r in recs_all
                        if r["CONSTELLATION"] in TOTAL_LEO_CONSTELLATIONS)
    epoch_mjd = _epoch_mjd(recs_all)
    print(f"current total LEO (modelled) N0 = {current_total}; epoch MJD = {epoch_mjd:.3f}")

    rows = si.load_inputs()
    want = {round(f) for f in args.lines}
    rows = [r for r in rows if round(r["freq_mhz"]) in want]

    # radiating subpop per line (observer-independent), built once
    line_tles = {}
    for ln in rows:
        rgx = SOURCE_REGEX.get(ln["source"])
        sub = [r for r in recs_all if rgx and re.search(rgx, r.get("OBJECT_NAME", ""), re.I)]
        af = ln["active_fraction"] if ln["active_fraction"] is not None else 1.0
        n_emit = max(1, int(round(len(sub) * af)))
        if len(sub) > n_emit:
            idx = np.sort(rng.choice(len(sub), n_emit, replace=False))
            sub = [sub[i] for i in idx]
        line_tles[ln["freq_mhz"]] = (np.array(ot.to_pytles(sub), dtype=object), len(sub))
        print(f"  line {ln['freq_mhz']:.0f} MHz [{ln['source']}]: {len(sub)} radiators (af={af})")

    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    results, t_start = [], time.time()
    for stn in args.stations:
        S = STATIONS[stn]
        for ln in rows:
            if not (S.get("fmin", 0) <= ln["freq_mhz"] <= S.get("fmax", 1e9)):
                continue
            tles, n_emit_now = line_tles[ln["freq_mhz"]]
            cf = ln["freq_mhz"] * 1e6
            eirp = ln["eirp_atca"]                      # measured (ATCA) bracket
            if eirp is None:
                continue
            # emission-bandwidth correction (Sec. 4.4): a line of width bw_hz is
            # diluted in the 32 MHz VGOS channel, raising the effective threshold by
            # 10log10(max(1, 32e6/bw_hz)). Equivalent to lowering the aggregate pfd
            # by the same factor before comparison. Must match recompute_with_bw.py.
            corr = max(1.0, 32e6 / ln["bw_hz"])
            corr_db = 10 * np.log10(corr)
            for W, niters in sorted(windows.items()):
                ts = 2.0 if W >= 2000 else 1.0          # match production at 2000 s
                t0 = time.time()
                r = ec.compute_epfd(
                    tles, S["lon"], S["lat"], S["alt"], S["diam"], cf, eirp,
                    modes=("vlbi",), min_el=S.get("min_el", 5.0),
                    grid_size=args.grid_size, niters=niters, time_range=float(W),
                    time_step=ts, n_jobs=args.n_jobs, mjd_epoch=epoch_mjd,
                    seed=args.seed, make_plots=False, return_arrays=True)
                m, _ = ec.metrics_from_pfd(r["pfd_nu"] / corr, cf, modes=("vlbi",),
                                           n_cells=len(r["grid_info"]))
                m = m["vlbi"]
                margin = m["ras_margin_db"]             # slim_eff_db - pfd98_db
                n_breach = current_total * 10 ** (margin / 10.0)
                results.append(dict(
                    station=stn, freq_mhz=ln["freq_mhz"], source=ln["source"],
                    window_s=W, niters=niters, n_emit_now=n_emit_now,
                    current_total=current_total,
                    bw_hz=ln["bw_hz"], bw_corr_db=round(corr_db, 1),
                    slim_db=round(m["slim_nu_db"], 2),
                    pfd98_db=round(m["pfd98_db"], 2),
                    margin_db=round(margin, 2),
                    data_loss_now_pct=round(m["data_loss_pct"], 3),
                    n_breach=int(round(n_breach))))
                print(f"  {stn} {ln['freq_mhz']:.0f}MHz W={W:>4}s n={niters:>3}: "
                      f"margin={margin:+6.2f} dB  N_breach={n_breach:,.0f}  "
                      f"[{time.time()-t0:.0f}s]")

    keys = list(results[0].keys())
    with open(args.outcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(results)
    print(f"\nWrote {len(results)} rows to {args.outcsv} in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
