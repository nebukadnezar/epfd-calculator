#!/usr/bin/env python3
# coding: utf-8
"""
recompute_with_bw.py — apply the emission-bandwidth correction to the RA.769
data-loss comparison, from the cached present-population aggregates (no
re-propagation).

Physics: a radiometer/correlator integrates interference POWER over its detection
channel of width df_ch. For an emission line of width dnu_e radiating at aggregate
per-Hz spectral pfd S_agg (what the EPFD computes, summed over satellites and
linear in N), the in-channel interference power is S_agg * dnu_e (Doppler drift
over the 2000 s redistributes this within the channel but conserves the total).
The RA.769 detrimental condition is

    S_agg * dnu_e  >=  Slim_nu * df_ch      (Slim_nu = published per-Hz threshold)

i.e. compare S_agg against an EFFECTIVE threshold

    Slim_eff = Slim_nu * max(1, df_ch / dnu_e).

Broadband emission (dnu_e >= df_ch, e.g. the 20-40 MHz downlink harmonics) -> factor 1,
unchanged. Narrowband lines (dnu_e ~ 1 kHz: the UEMR, and the 2595/5190 frequency
conversion products) -> raised by ~10log10(df_ch/dnu_e) ~ 45 dB
(df_ch = 32 MHz VGOS sub-channel). The aggregate still scales linearly with N; only
the threshold it must cross changes.

Inputs : output/pfd/*.npz (cached pfd_nu per station x line x bracket from run_auscope),
         emission bandwidths bw_hz from data/sniffles_inputs.csv.
Output : output/auscope_results.csv, identical schema to run_auscope.py, so
         analyze_results.py consumes it unchanged.

Extensible: edit bw_hz in the CSV (per line) and CHAN_BW_HZ below, then re-run (seconds).
"""

import csv
import glob
import os

import numpy as np
from scipy.stats import percentileofscore

import epfd_core as ec
import sniffles_inputs as si
import settings as cfg

CHAN_BW_HZ = 32e6          # VGOS sub-channel detection bandwidth (configurable)
MODES = ("continuum", "vlbi")
SCENARIOS = {"now": None, "30k": 30_000, "50k": 50_000, "100k": 100_000,
             "200k": 200_000, "300k": 300_000, "500k": 500_000,
             "1m": 1_000_000, "2m": 2_000_000}
PFDDIR = cfg.PFD
OUT = os.path.join(cfg.OUT, "auscope_results.csv")


def main():
    inp = {round(r["freq_mhz"]): r for r in si.load_inputs()}
    rows, corr_log = [], {}
    # Every (station,line) has a floor .npz holding the propagation GEOMETRY; the
    # atca .npz (when present) is the SAME geometry at a different reference EIRP.
    # So we read the floor cache only and emit BOTH brackets from it, rescaling the
    # aggregate (linear in per-satellite EIRP) to each bracket's level. This makes
    # the atca bracket free for lines calibrated AFTER the run (no re-propagation):
    # a flux added to the CSV produces its atca result by the EIRP ratio alone.
    for path in sorted(glob.glob(os.path.join(PFDDIR, "*_floor.npz"))):
        d = np.load(path, allow_pickle=True)
        pfd_geom = d["pfd_nu"]
        cf = float(d["cf_hz"]); eirp_stored = float(d["eirp_dbw_hz"])
        ctot = float(d["current_total"]); n_now = int(d["n_emit_now"])
        stn = str(d["station"]); source = str(d["source"])
        fr = cf / 1e6
        line = inp[round(fr)]
        # eclass from the input table, not the npz: the npz was written by the
        # EPFD run and freezes whatever class was current then, so reading it
        # back would silently regress any later reclassification.
        eclass = line["eclass"]

        emit_bw = line["bw_hz"]                     # emission bandwidth [Hz]
        corr_db = 10 * np.log10(max(1.0, CHAN_BW_HZ / emit_bw))
        corr_log[round(fr)] = (eclass, emit_bw, corr_db)

        brackets = [("floor", line["eirp_floor"])]
        if line["eirp_atca"] is not None:           # calibrated -> derive atca too
            brackets.append(("atca", line["eirp_atca"]))
        for bracket, eirp_cur in brackets:
            if eirp_cur is None:
                eirp_cur = eirp_stored
            pfd_nu = pfd_geom * 10 ** ((eirp_cur - eirp_stored) / 10.0)
            for mode in MODES:
                lim = ec.ra769_limit(cf, mode=mode)
                slim_db = lim["slim_nu_db"] + corr_db   # bandwidth-corrected threshold
                slim = 10 ** (slim_db / 10)
                for scen, target in SCENARIOS.items():
                    G = 1.0 if target is None else target / ctot
                    pfd_s = pfd_nu * G
                    pfd98 = np.percentile(pfd_s, 98.)
                    dloss = 100 - percentileofscore(pfd_s.flatten(), slim, kind="strict")
                    margin = slim_db - 10 * np.log10(pfd98)
                    rows.append(dict(
                        station=stn, freq_mhz=fr, eclass=eclass, source=source,
                        bracket=bracket, eirp_dbw_hz=round(eirp_cur, 1), scenario=scen,
                        growth=round(G, 2), n_emit=int(round(n_now * G)), mode=mode,
                        data_loss_pct=round(float(dloss), 3), slim_db=round(slim_db, 2),
                        margin_db=round(float(margin), 2), interp=lim["interpolated"]))

    keys = list(rows[0].keys())
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}  (detection channel = {CHAN_BW_HZ/1e6:.0f} MHz)")
    print(f"\n{'freq':>7} {'class':>9} {'emit_bw':>10} {'thresh_corr_dB':>14}")
    for fr in sorted(corr_log):
        ec_, bw, c = corr_log[fr]
        bw_s = f"{bw/1e3:.0f} kHz" if bw < 1e6 else f"{bw/1e6:.0f} MHz"
        print(f"{fr:7.0f} {ec_:>9} {bw_s:>10} {c:>+14.1f}")


if __name__ == "__main__":
    main()
