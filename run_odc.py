#!/usr/bin/env python3
# coding: utf-8
"""
run_odc.py — EPFD from the SpaceX orbital-data-centre (AI1) dawn-dusk SSO
constellation, for the two presets (realistic ~210k, stress 1M).

Method (mirrors run_auscope's now->scenario scaling, but on the synthetic ODC
geometry): build a representative SUBSAMPLE of each preset's dawn-dusk multi-shell
distribution (constellation.odc_shell_tles with a reduced total), propagate it
once per facility x frequency with the satellite-chunking compute_epfd, cache the
aggregate spectral pfd_nu, then scale it analytically to the target fleet sizes
(aggregate spectral PFD is linear in N at fixed spatial distribution).

EMISSION = UEMR ONLY. An ODC is a compute platform, NOT a comms satellite: it
carries no intended DTD/DTC downlink, and therefore none of the downlink spurious
harmonics either. The only emission modelled is the unintended platform radiation
(UEMR) that dominates on dense digital hardware, anchored to the V2-mini UEMR
floor EIRP spectral density (a parameter; the ODC's own UEMR is unmeasured, and
150 kW of compute could well exceed it). Compared against both RA.769 criteria
(single-dish continuum + VLBI).

Note on bandwidth: UEMR is narrowband, so (as for the platform UEMR in the main
pipeline) the emission-bandwidth correction applies downstream -- it dilutes the
VGOS/VLBI impact heavily and leaves single-dish continuum as the binding case.

A true full-N spot check (no scaling) is available via --validate-n for
validating the linear extrapolation at the ODC's actual target N.

"""
import argparse
import csv
import os
import time

# single-thread BLAS before numpy import (see note in run_auscope.py). Do NOT pin
# OMP_NUM_THREADS -- cysgp4 propagation needs it >1 (set via the launcher).
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import epfd_core as ec
import constellation as C
import settings as cfg
from run_auscope import STATIONS

# representative evaluation frequencies (MHz): VGOS A/B/C sub-band reps (leaner;
# was 5 incl L-band 1400 and VGOS-D 13000 -- 3 covers the geodetic case).
DEFAULT_FREQS = [3300.0, 5200.0, 8600.0]

# V2-mini UEMR floor anchor [dB(W/Hz)] (SNIFFLES detection floor; ODC emission is
# a parameter -- see sniffles_inputs.csv eirp_floor_dbw_hz, ~-118 to -120.5).
DEFAULT_FLOOR_DBW_HZ = -119.0

# target fleet sizes per preset geometry (analytic scaling from the subsample).
# realistic is capped near its clumped capacity (~210k); stress runs to 1M.
TARGETS = {
    "realistic": [50_000, 100_000, 209_877],
    "stress":    [50_000, 100_000, 250_000, 500_000, 1_000_000],
}
SUBSAMPLE_N = 20_000     # propagate this many; scale to the targets above (leaner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="*", default=["realistic", "stress"])
    ap.add_argument("--stations", nargs="*", default=list(STATIONS))
    ap.add_argument("--freqs", nargs="*", type=float, default=DEFAULT_FREQS)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR_DBW_HZ,
                    help="per-satellite EIRP spectral density [dB(W/Hz)]")
    ap.add_argument("--subsample", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--niters", type=int, default=30)
    ap.add_argument("--time-range", type=float, default=2000.0)
    ap.add_argument("--time-step", type=float, default=2.0)
    ap.add_argument("--grid-size", type=float, default=5.0)
    ap.add_argument("--n-jobs", type=int, default=cfg.N_JOBS)
    ap.add_argument("--prop-budget", type=float, default=cfg.PROP_BUDGET)
    ap.add_argument("--epoch-mjd", type=float, default=60860.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--validate-n", type=int, default=0,
                    help="if >0, ALSO propagate the full N directly (no scaling) "
                         "at each facility/freq to validate the linear extrapolation")
    ap.add_argument("--outcsv", default=os.path.join(cfg.OUT, "odc_results.csv"))
    args = ap.parse_args()

    modes = ("continuum", "vlbi")
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    FIELDS = ["preset", "geometry", "method", "n_sats", "station", "freq_mhz",
              "mode", "floor_dbw_hz", "data_loss_pct", "slim_db", "margin_db", "interp"]
    fh = open(args.outcsv, "w", newline="")          # write incrementally per call
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader(); fh.flush()
    n_written = 0
    t_start = time.time()

    for preset in args.presets:
        # representative subsample of this preset's dawn-dusk multi-shell geometry
        t0 = time.time()
        sub = C.odc_shell_tles(preset, epoch_mjd=args.epoch_mjd, seed=args.seed,
                               total=args.subsample, verbose=True)
        n_sub = len(sub)
        print(f"[{preset}] subsample {n_sub} sats built in {time.time()-t0:.0f}s", flush=True)

        for stn in args.stations:
            S = STATIONS[stn]
            for fr in args.freqs:
                if not (S.get("fmin", 0) <= fr <= S.get("fmax", 1e9)):
                    continue
                cf = fr * 1e6
                t1 = time.time()
                r = ec.compute_epfd(
                    sub, S["lon"], S["lat"], S["alt"], S["diam"], cf, args.floor,
                    modes=modes, min_el=S.get("min_el", 5.0),
                    grid_size=args.grid_size, niters=args.niters,
                    time_range=args.time_range, time_step=args.time_step,
                    n_jobs=args.n_jobs, mjd_epoch=args.epoch_mjd, seed=args.seed,
                    make_plots=False, return_arrays=True,
                    prop_budget=args.prop_budget)
                pfd_nu = r["pfd_nu"]; grid = r["grid_info"]
                call_rows = []
                # scale the subsample aggregate to each target fleet size
                for target in TARGETS[preset]:
                    G = target / n_sub
                    m, _ = ec.metrics_from_pfd(pfd_nu * G, cf, modes=modes,
                                               n_cells=len(grid))
                    for mode in modes:
                        call_rows.append(dict(
                            preset=preset, geometry=preset, method="scaled",
                            n_sats=int(target), station=stn, freq_mhz=fr,
                            mode=mode, floor_dbw_hz=args.floor,
                            data_loss_pct=round(m[mode]["data_loss_pct"], 3),
                            slim_db=m[mode]["slim_nu_db"],
                            margin_db=round(m[mode]["ras_margin_db"], 2),
                            interp=m[mode]["interpolated"]))
                print(f"  [{preset}] {stn} {fr:.0f}MHz subN={n_sub} "
                      f"loss(cont/vlbi @1x)="
                      f"{r['modes']['continuum']['data_loss_pct']:.1f}/"
                      f"{r['modes']['vlbi']['data_loss_pct']:.1f}%  "
                      f"[{time.time()-t1:.0f}s]", flush=True)

                # optional: true full-N propagation (no scaling) to validate
                if args.validate_n > 0:
                    t2 = time.time()
                    full = C.odc_shell_tles(preset, epoch_mjd=args.epoch_mjd,
                                            seed=args.seed + 1, total=args.validate_n)
                    rv = ec.compute_epfd(
                        full, S["lon"], S["lat"], S["alt"], S["diam"], cf, args.floor,
                        modes=modes, min_el=S.get("min_el", 5.0),
                        grid_size=args.grid_size, niters=args.niters,
                        time_range=args.time_range, time_step=args.time_step,
                        n_jobs=args.n_jobs, mjd_epoch=args.epoch_mjd,
                        seed=args.seed + 1, make_plots=False,
                        prop_budget=args.prop_budget)
                    for mode in modes:
                        mv = rv["modes"][mode]
                        call_rows.append(dict(
                            preset=preset, geometry=preset, method="true",
                            n_sats=int(len(full)), station=stn, freq_mhz=fr,
                            mode=mode, floor_dbw_hz=args.floor,
                            data_loss_pct=round(mv["data_loss_pct"], 3),
                            slim_db=mv["slim_nu_db"],
                            margin_db=round(mv["ras_margin_db"], 2),
                            interp=mv["interpolated"]))
                    print(f"    [validate] {stn} {fr:.0f}MHz true N={len(full)} "
                          f"[{time.time()-t2:.0f}s]", flush=True)

                writer.writerows(call_rows); fh.flush()      # <-- incremental flush
                n_written += len(call_rows)
    fh.close()
    print(f"\nWrote {n_written} rows to {args.outcsv} in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
