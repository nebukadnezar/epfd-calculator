"""Time-step convergence of the Monte-Carlo EPFD (Sec 4.3):
is the 2 s sampling of the 2000 s window fine enough?

Runs the binding VLBI line (5240 MHz DTD 2nd harmonic, Hobart 12 m, present
fleet) with identical seeds at time_step = 2.0, 1.0 and 0.5 s. Same start
epochs and pointings (seed-controlled), so any difference is purely sampling
density. Reports data loss (both criteria) and the 98th-pct aggregate.
"""
import os, re, glob, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import omm_tools as ot, epfd_core as ec
import settings as cfg

HERE = os.path.dirname(__file__)
OUT = cfg.out("timestep_validation%s.csv" % ("_fine" if len(sys.argv) > 1 else ""))

# 5240 MHz DTD line (sniffles_inputs.csv): ATCA bracket
SFD_JY, DIST_KM, ACT = 246705.0, 890.0, 0.869
CF = 5240e6
EIRP = 10*np.log10(SFD_JY) + 10*np.log10(1e-26*4*np.pi*(DIST_KM*1e3)**2)

recs = ot.load_omm(cfg.OMM)
sub = [r for r in recs if re.match(r"STARLINK-11\d{3}", r.get("OBJECT_NAME", ""))]
rng = np.random.default_rng(1)
n = int(round(len(sub)*ACT))
sub = [sub[i] for i in np.sort(rng.choice(len(sub), n, replace=False))]
tles = np.array(ot.to_pytles(sub), dtype=object)
epoch = float(np.median([ot._epoch_to_mjd(r["EPOCH"]) for r in recs[::150]]))
print(f"{len(tles)} DTD emitters, EIRP {EIRP:.1f} dB(W/Hz)", flush=True)

STEPS = tuple(float(s) for s in sys.argv[1:]) or (2.0, 1.0, 0.5)
rows = []
for ts in STEPS:
    r = ec.compute_epfd(tles, 147.44, -42.80, 0.04, 12.0, CF, EIRP,
                        modes=("continuum", "vlbi"), min_el=5, grid_size=5.0,
                        niters=50, time_range=2000, time_step=ts, n_jobs=cfg.N_JOBS,
                        mjd_epoch=epoch, seed=3, make_plots=False,
                        return_arrays=True)
    pfd = r["pfd_nu"]
    p98 = 10*np.log10(np.percentile(pfd, 98.))
    line = dict(time_step=ts, p98_db=round(p98, 3))
    for mode in ("continuum", "vlbi"):
        slim = ec.ra769_limit(CF, mode=mode)["slim_nu_db"]
        loss = 100*np.mean(pfd > 10**(slim/10))
        line[f"loss_{mode}"] = round(float(loss), 3)
    rows.append(line)
    print(f"ts={ts:3.1f}s  p98={p98:8.3f} dB  "
          f"loss C={line['loss_continuum']:6.2f}%  V={line['loss_vlbi']:6.2f}%", flush=True)

import csv
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print("wrote", OUT)
