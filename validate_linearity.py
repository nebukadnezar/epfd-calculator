#!/usr/bin/env python3
# coding: utf-8
"""
validate_linearity.py — check the central assumption behind the breach-vs-N and
E_lim-vs-N extrapolations: that the aggregate spectral PFD (hence pfd98) is LINEAR
in the number of radiators, so scaling the now-population CDF by 10log10(N/N0) is
equivalent to actually simulating N satellites.

Method: for a few small-subpopulation lines and one station, simulate the present
radiating subpop (pfd98_now), then build a genuinely up-scaled fleet with
constellation.proportional_upscale_tles() (real OMM orbits cloned with random
RAAN / mean anomaly, preserving the altitude+inclination distribution) at
n_emit_now * G and simulate it directly (pfd98_true). Compare:

  analytic:  pfd98_now_db + 10*log10(G)
  true:      pfd98_true_db
  delta = true - analytic   (should be ~0 dB if the aggregate is linear in N)

Also compares VLBI data-loss% from the analytic CDF shift vs the true up-scaled run.

IMPORTANT: run ALONE (no other heavy compute).
"""

import argparse
import csv
import math
import re
import time

import numpy as np

import omm_tools as ot
import epfd_core as ec
import sniffles_inputs as si
import constellation as con
import settings as cfg
from run_auscope import STATIONS, SOURCE_REGEX, TOTAL_LEO_CONSTELLATIONS


def build_subpop(recs_all, line, rng):
    """Same radiating-subpop selection as run_auscope.main()."""
    rgx = SOURCE_REGEX.get(line["source"])
    sub = [r for r in recs_all if rgx and re.search(rgx, r.get("OBJECT_NAME", ""), re.I)]
    af = line["active_fraction"] if line["active_fraction"] is not None else 1.0
    n_emit = max(1, int(round(len(sub) * af)))
    if len(sub) > n_emit:
        idx = np.sort(rng.choice(len(sub), n_emit, replace=False))
        sub = [sub[i] for i in idx]
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omm", default=cfg.OMM)
    ap.add_argument("--station", default="Hb")
    ap.add_argument("--lines", nargs="*", type=float, default=[5240.0, 8599.0],
                    help="small-subpop lines to validate (default DTC 5240, OneWeb 8599)")
    ap.add_argument("--targets", nargs="*", type=int, default=[100_000, 300_000])
    ap.add_argument("--niters", type=int, default=30)
    ap.add_argument("--time-step", type=float, default=4.0)   # 500 tsteps -> modest memory
    ap.add_argument("--time-range", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outcsv", default=cfg.out("linearity_validation.csv"))
    args = ap.parse_args()

    S = STATIONS[args.station]
    rng = np.random.default_rng(args.seed)
    print(f"Loading OMM {args.omm} ...")
    recs_all = ot.load_omm(args.omm)
    current_total = sum(1 for r in recs_all
                        if r["CONSTELLATION"] in TOTAL_LEO_CONSTELLATIONS)
    epoch_mjd = float(np.median([ot._epoch_to_mjd(r["EPOCH"])
                                 for r in recs_all[:: max(1, len(recs_all)//200)]]))
    print(f"current total LEO = {current_total}; station = {S['name']}; "
          f"epoch MJD = {epoch_mjd:.3f}")

    all_lines = {round(r["freq_mhz"]): r for r in si.load_inputs()}
    results = []
    for fr in args.lines:
        line = all_lines[round(fr)]
        eirp = line["eirp_atca"] if line["eirp_atca"] is not None else line["eirp_floor"]
        cf = fr * 1e6
        sub = build_subpop(recs_all, line, rng)
        n_now = len(sub)
        sub_tles = np.array(ot.to_pytles(sub), dtype=object)
        print(f"\n=== {fr:.0f} MHz [{line['source']}] : {n_now} now-radiators, "
              f"eirp={eirp:.1f} dB(W/Hz) ===")

        # now-population reference (return the pfd array for analytic scaling)
        r0 = ec.compute_epfd(
            sub_tles, S["lon"], S["lat"], S["alt"], S["diam"], cf, eirp,
            modes=("vlbi",), min_el=S.get("min_el", 5), grid_size=5.0,
            niters=args.niters, time_range=args.time_range, time_step=args.time_step,
            mjd_epoch=epoch_mjd, seed=args.seed, make_plots=False, return_arrays=True)
        pfd_now = r0["pfd_nu"]
        pfd98_now_db = 10*math.log10(np.percentile(pfd_now, 98.))
        loss_now = r0["modes"]["vlbi"]["data_loss_pct"]
        print(f"  now: pfd98={pfd98_now_db:.2f} dB, vlbi loss={loss_now:.1f}%")

        for target in args.targets:
            G = target / current_total
            n_up = int(round(n_now * G))
            t0 = time.time()
            # TRUE up-scaled fleet (real orbits cloned, RAAN/MA randomised)
            up_tles = con.proportional_upscale_tles(sub, n_up, seed=args.seed)
            rU = ec.compute_epfd(
                up_tles, S["lon"], S["lat"], S["alt"], S["diam"], cf, eirp,
                modes=("vlbi",), min_el=S.get("min_el", 5), grid_size=5.0,
                niters=args.niters, time_range=args.time_range, time_step=args.time_step,
                mjd_epoch=epoch_mjd, seed=args.seed, make_plots=False, return_arrays=True)
            pfd98_true_db = 10*math.log10(np.percentile(rU["pfd_nu"], 98.))
            loss_true = rU["modes"]["vlbi"]["data_loss_pct"]
            # ANALYTIC prediction (run_auscope method): shift now-CDF by 10log10(G)
            pfd98_analytic_db = pfd98_now_db + 10*math.log10(G)
            m_an, _ = ec.metrics_from_pfd(pfd_now * G, cf, modes=("vlbi",))
            loss_analytic = m_an["vlbi"]["data_loss_pct"]
            delta = pfd98_true_db - pfd98_analytic_db
            print(f"  {target} (G={G:.1f}, {len(up_tles)} sats): "
                  f"pfd98 true={pfd98_true_db:.2f} vs analytic={pfd98_analytic_db:.2f} "
                  f"-> delta={delta:+.2f} dB | loss true={loss_true:.1f}% vs "
                  f"analytic={loss_analytic:.1f}% [{time.time()-t0:.0f}s]")
            results.append(dict(
                freq_mhz=fr, station=args.station, n_now=n_now, target=target,
                G=round(G, 2), n_up=len(up_tles),
                pfd98_now_db=round(pfd98_now_db, 2),
                pfd98_true_db=round(pfd98_true_db, 2),
                pfd98_analytic_db=round(pfd98_analytic_db, 2),
                delta_db=round(delta, 2),
                loss_true_pct=round(loss_true, 1),
                loss_analytic_pct=round(loss_analytic, 1)))

    with open(args.outcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    deltas = [r["delta_db"] for r in results]
    print(f"\nWrote {len(results)} rows to {args.outcsv}")
    print(f"linearity: |delta| max={max(abs(d) for d in deltas):.2f} dB, "
          f"mean={np.mean(deltas):+.2f} dB (≈0 confirms linear-in-N scaling)")


if __name__ == "__main__":
    main()
