#!/usr/bin/env python3
# coding: utf-8
"""
run_example.py - out-of-the-box example of the per-satellite limit calculation.

For one station and a few frequencies (see the "example" block of config.json):

  1. load the orbital-element snapshot and select the radiating satellites,
  2. run the Monte-Carlo EPFD at the present fleet (compute_epfd),
  3. compare the 98th percentile of the aggregate with the RA.769 thresholds
     (single-dish continuum and VLBI), including the emission-bandwidth
     correction,
  4. derive the per-satellite limit for larger fleets analytically (the
     aggregate is linear in the number of radiators, no re-propagation).

Outputs, all below <output_dir>/example/:
  example_results.csv   margins, data loss and limits per frequency and criterion
  example_limits.txt    the same as a readable table, with the run times
  figs/                 sky maps and cumulative distributions per frequency

The margins are also printed next to the shipped reference results
(data/auscope_results.csv, the values behind the paper). The two agree to
within the Monte-Carlo scatter, a few tenths of a dB at the default settings.

Usage:  ./run.sh run_example.py        (or: python run_example.py)
"""

import csv
import math
import multiprocessing as mp
import os
import platform
import re
import time

# single-thread the BLAS libraries before numpy is imported (see run_auscope.py)
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import settings as cfg
import omm_tools as ot
import epfd_core as ec
import sniffles_inputs as si
from run_auscope import (STATIONS, SOURCE_REGEX, TOTAL_LEO_CONSTELLATIONS,
                         _epoch_mjd)
from recompute_with_bw import CHAN_BW_HZ
from analyze_results import eirp_to_dbuv_m, eirp_lim_at, B_REF

MODES = ("continuum", "vlbi")
SEED = 1


def reference_margins(station):
    """Margins of the shipped reference results: {(freq, mode): margin_db}."""
    ref = {}
    with open(cfg.ref("auscope_results.csv")) as fh:
        for r in csv.DictReader(fh):
            if (r["station"], r["bracket"], r["scenario"]) == (station, "atca", "now"):
                ref[(round(float(r["freq_mhz"])), r["mode"])] = float(r["margin_db"])
    return ref


def main():
    ex = cfg.CFG["example"]
    stn = ex["station"]
    S = STATIONS[stn]
    sizes = [float(n) for n in ex["fleet_sizes"]]
    figdir = cfg.out("example", "figs", "x")[:-2]
    rng = np.random.default_rng(SEED)
    t_start = time.time()

    print(f"machine: {platform.machine()}, {mp.cpu_count()} cores, "
          f"n_jobs={cfg.N_JOBS}")
    print(f"loading {cfg.OMM} ...")
    recs = ot.load_omm(cfg.OMM)
    n0 = sum(1 for r in recs if r["CONSTELLATION"] in TOTAL_LEO_CONSTELLATIONS)
    epoch_mjd = _epoch_mjd(recs)
    print(f"{len(recs)} objects, {n0} in the modelled LEO constellations, "
          f"epoch MJD {epoch_mjd:.2f}  [{time.time() - t_start:.0f} s]")

    inputs = {round(r["freq_mhz"]): r for r in si.load_inputs()}
    ref = reference_margins(stn)
    rows, timing = [], []
    for f_mhz in ex["frequencies_mhz"]:
        line = inputs[round(f_mhz)]
        cf = line["freq_mhz"] * 1e6
        eirp = line["eirp_atca"]          # measured per-satellite level, dB(W/Hz)

        # radiating sub-population: name match, thinned to the active fraction
        rgx = SOURCE_REGEX[line["source"]]
        sub = [r for r in recs if re.search(rgx, r.get("OBJECT_NAME", ""), re.I)]
        af = line["active_fraction"] if line["active_fraction"] is not None else 1.0
        n_emit = max(1, int(round(len(sub) * af)))
        if len(sub) > n_emit:
            keep = np.sort(rng.choice(len(sub), n_emit, replace=False))
            sub = [sub[i] for i in keep]
        tles = np.array(ot.to_pytles(sub), dtype=object)

        t0 = time.time()
        r = ec.compute_epfd(
            tles, S["lon"], S["lat"], S["alt"], S["diam"], cf, eirp,
            modes=MODES, min_el=S["min_el"], grid_size=ex["grid_size_deg"],
            niters=ex["niters"], time_range=ex["time_range_s"],
            time_step=ex["time_step_s"], n_jobs=cfg.N_JOBS, mjd_epoch=epoch_mjd,
            seed=SEED, figpath=figdir, label=f"{stn}_{f_mhz:.0f}MHz",
            obslabel=S["name"], make_plots=True, return_arrays=True,
            prop_budget=cfg.PROP_BUDGET)
        dt = time.time() - t0
        timing.append((f_mhz, len(tles), dt))

        # emission-bandwidth correction of the threshold (see recompute_with_bw.py)
        corr_db = 10 * math.log10(max(1.0, CHAN_BW_HZ / line["bw_hz"]))
        pfd = r["pfd_nu"]
        pfd98_db = 10 * math.log10(np.percentile(pfd, 98.0))
        b_ref = min(line["bw_hz"], B_REF)     # field-strength reference bandwidth
        for mode in MODES:
            slim_db = r["modes"][mode]["slim_nu_db"] + corr_db
            margin = slim_db - pfd98_db
            loss = 100.0 * np.mean(pfd >= 10 ** (slim_db / 10))
            row = dict(station=stn, freq_mhz=f_mhz, source=line["source"],
                       n_radiators=len(tles), mode=mode,
                       eirp_used_dbw_hz=round(eirp, 1),
                       slim_db=round(slim_db, 2), bw_corr_db=round(corr_db, 1),
                       pfd98_db=round(pfd98_db, 2), margin_db=round(margin, 2),
                       reference_margin_db=ref.get((round(f_mhz), mode), ""),
                       data_loss_pct=round(loss, 2))
            for n in sizes:
                lim = eirp_lim_at(eirp, margin, n, n0)
                row[f"eirp_lim_{n:.0f}_dbw_hz"] = round(lim, 1)
                row[f"e_lim_{n:.0f}_dbuv_m"] = round(eirp_to_dbuv_m(lim, b_ref=b_ref), 1)
            rows.append(row)
        print(f"  {f_mhz:.0f} MHz: {len(tles)} radiators, {dt:.0f} s")

    total = time.time() - t_start
    out_csv = cfg.out("example", "example_results.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    lines = [f"Station {S['name']} ({S['diam']:.0f} m), {ex['niters']} iterations of "
             f"{ex['time_range_s']} s at {ex['time_step_s']} s, present fleet N0 = {n0}",
             "",
             "Margin of the present fleet to the RA.769 threshold (negative = exceeded):",
             f"{'MHz':>6} {'criterion':>10} {'radiators':>9} {'margin dB':>10} "
             f"{'reference':>10} {'diff':>6} {'loss %':>7}"]
    for r in rows:
        refm = r["reference_margin_db"]
        d = f"{r['margin_db'] - refm:+6.2f}" if refm != "" else "     -"
        refs = f"{refm:10.2f}" if refm != "" else f"{'-':>10}"
        lines.append(f"{r['freq_mhz']:6.0f} {r['mode']:>10} {r['n_radiators']:9d} "
                     f"{r['margin_db']:10.2f} {refs} {d} {r['data_loss_pct']:7.2f}")
    lines += ["",
              "Per-satellite limit E_lim in dB(uV/m) at 10 m, in min(emission "
              "bandwidth, 1 MHz),",
              "for a total LEO fleet of N satellites:",
              f"{'MHz':>6} {'criterion':>10} " + " ".join(f"{'N=%.0f' % n:>11}" for n in sizes)]
    for r in rows:
        lines.append(f"{r['freq_mhz']:6.0f} {r['mode']:>10} " + " ".join(
            f"{r['e_lim_%.0f_dbuv_m' % n]:11.1f}" for n in sizes))
    lines += ["",
              f"Run time on {platform.machine()} with {mp.cpu_count()} cores "
              f"(n_jobs={cfg.N_JOBS}):"]
    lines += [f"  {f:6.0f} MHz, {n:5d} radiators: {dt:6.1f} s" for f, n, dt in timing]
    lines.append(f"  total, including loading the snapshot: {total:.1f} s")
    text = "\n".join(lines)
    out_txt = cfg.out("example", "example_limits.txt")
    with open(out_txt, "w") as fh:
        fh.write(text + "\n")
    print("\n" + text)
    print(f"\nwrote {out_csv}\nwrote {out_txt}\nfigures in {figdir}")


if __name__ == "__main__":
    main()
