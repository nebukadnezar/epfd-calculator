"""The 150.05-153 MHz RAS band under Starlink UEMR: what LOFAR should see,
anchored in the Bassa et al. (2024, A&A 689, L10) per-satellite measurements,
swept over three epochs. Fills the gap Bassa+24 name explicitly: "EPFD
simulations of the Generation 2 Starlink constellation would be required...
beyond the scope of this paper."

INPUTS (measured, Bassa+24 Tables B.2/B.3, 150.05-153 MHz column)
------------------------------------------------------------------
Per-satellite broadband power flux densities S_nu (Jy) at the observed
distance d (km), LOFAR HBA, 2024-07-19. We scale each to 1000 km
(S1000 = S*(d/1000)^2, their Fig. 2 convention) and use the class MEAN for
aggregate work (power sums; the median is also reported). The isotropic
per-satellite EIRP density follows their implicit convention:
  EIRP_nu = S1000 * 1e-26 * 4 pi (1e6 m)^2   [W/Hz]
Field strengths quoted at 10 m in 120 kHz (CISPR < 1 GHz), as in Bassa+24
Fig. 2 and the cross-check (validate_crosscheck.py).

CLASSES AND SCENARIOS
---------------------
Classes by launch designator and name (v2-Mini launches began 2023-028;
DTC = STARLINK-11xxx):
  v1.x    : Gen1 (v1.0 + v1.5), ~550 km
  v2-Mini : Gen2 standard, 448-482 km shells
  DTC     : Gen2 direct-to-cell, ~360 km
Scenarios:
  A "Bassa epoch"  : fleet with designator <= 2024-117 (the newest launch
                     group observed in their tables), real orbits
  B "present day"  : full Starlink from the 2026-07-03 OMM snapshot
  C "full buildout": Gen1 4408 (authorised) + Gen2 29988 (as filed)
                     at v2-Mini levels (conservative: DTC are fainter here)
Counts within a class scale linearly (linear-in-N validated in the paper,
conservative). Class aggregates are computed sample-aligned (same seed,
epoch, grid) so they sum correctly before taking percentiles.

MACHINERY
---------
Identical to validate_crosscheck.py: epfd_core at 151.525 MHz
(S_lim = -259 dB(W/m2/Hz), RA.769 Table 1), LOFAR core (52.915 N, 6.869 E),
70 m dish eta=1 (the equivalent-dish convention used in the cross-check and, per their
Sec. 4, by Di Vruno+23; its ~4 dB anti-conservatism for a real station is
documented in lofar_pattern_compare.py), el > 5 deg, 3 deg grid, 50 x 2000 s
windows @ 2 s, p98 anchor (identified as the community convention in
validate_crosscheck.py).

BENCHMARK: Di Vruno+23 (quoted by Bassa+24): E_lim = 11.7 dB(uV/m) for the
Gen1 4408-satellite constellation. We reproduce it by scaling our v1.x run
to 4408 satellites.

OUTPUTS: per scenario, (i) the aggregate p98 vs S_lim with the measured
levels ("what LOFAR should see"), the exceedance in dB and the RA.769 data
loss; (ii) the uniform per-satellite E_lim that would hold the aggregate at
the threshold.
"""
import os, re, sys, glob, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import epfd_core as ec, omm_tools as ot
import settings as cfg

HERE = os.path.dirname(__file__)
CF = 151.525e6; Z0 = 376.730
LON, LAT, ALT = 6.869, 52.915, 0.015          # LOFAR core
EIRP_UNIT = -130.0                            # dB(W/Hz) trial, scaled out
RUN = dict(modes=("continuum",), min_el=5, grid_size=3.0, niters=50,
           time_range=2000, time_step=2.0, duty=1.0, eta_a_rx=1.0,
           n_jobs=cfg.N_JOBS, seed=5, make_plots=False, return_arrays=True)

# ---- Bassa+24 measured levels: (S_nu Jy, d km), 150.05-153 MHz ----
V2MINI = [(10.8,489.3),(9.6,488.2),(12.4,489.1),(59.4,488.0),(60.7,489.2),
          (23.8,558.7),(37.9,490.2),(49.3,489.4),(52.3,489.1),(37.1,481.4),
          (41.3,482.3),(40.6,482.4),(62.6,488.1),(37.5,486.9),(4.0,481.9),
          (39.0,453.2),(45.7,454.3),(50.8,453.7),(50.7,454.4),(59.5,453.8),
          (51.7,454.5),(17.1,453.2),(49.6,453.7),(43.5,454.3)]
DTC =    [(10.3,348.3),(14.8,346.9),(19.5,346.6),(84.9,345.4),(5.3,347.0),
          (8.3,348.5)]
# v1.x from Table B.2 (values 0.5-2.9 Jy at ~550 km; transcription of the
# legible entries; class contributes <5% of any aggregate here)
V1X = [(0.9,553.2),(2.9,553.9),(1.3,553.8),(1.0,554.2),(0.5,553.6),
       (1.3,553.7),(0.8,553.3),(1.2,554.0),(0.9,554.5),(1.3,419.9),
       (2.0,554.2),(0.7,552.8),(1.1,554.0),(0.7,554.1),(1.2,552.9),
       (1.1,554.5),(1.5,554.5),(1.4,553.4),(1.9,553.9),(1.1,553.2),
       (1.4,553.2),(1.2,555.0),(1.0,553.4),(1.0,554.7),(2.9,552.9),
       (1.0,554.6),(2.0,553.3),(1.9,553.4),(0.8,545.5),(1.0,547.6),
       (1.0,546.4),(0.6,547.4),(1.0,547.3),(0.8,547.3),(1.4,547.9),
       (1.0,547.6),(0.5,546.2),(0.8,546.4),(0.5,547.2),(1.4,546.3),
       (0.8,546.7)]

def s1000(pairs):
    return np.array([s*(d/1000.0)**2 for s, d in pairs])
def eirp_nu(s1000_jy):                        # W/Hz, isotropic convention
    return s1000_jy*1e-26*4*np.pi*1e12
def e_dbuvm(eirp):                            # dB(uV/m) @10m /120kHz
    pfd = eirp*120e3/(4*np.pi*100.0)
    return 20*np.log10(np.sqrt(pfd*Z0)) + 120

LEVELS = {}
for name, pairs in [("v1x", V1X), ("v2mini", V2MINI), ("dtc", DTC)]:
    s = s1000(pairs)
    LEVELS[name] = dict(mean=s.mean(), median=np.median(s), n=len(s))
    print(f"{name:7}: S@1000km mean {s.mean():6.2f} Jy, median {np.median(s):6.2f} Jy "
          f"(n={len(s)});  E(mean) = {e_dbuvm(eirp_nu(s.mean())):.1f} dB(uV/m)@10m/120kHz",
          flush=True)

# ---- classify the current catalogue ----
recs = ot.load_omm(cfg.OMM)
def desig(r):
    m = re.match(r"(\d{4})-(\d{3})", r.get("OBJECT_ID", "") or "")
    return (int(m.group(1)), int(m.group(2))) if m else (9999, 999)
star = [r for r in recs if re.match(r"STARLINK", r.get("OBJECT_NAME", ""))]
def classify(r):
    y, n = desig(r)
    if re.match(r"STARLINK-11\d{3}", r["OBJECT_NAME"]): return "dtc"
    if y < 2023 or (y == 2023 and n < 28): return "v1x"
    return "v2mini"
CLS = {"v1x": [], "v2mini": [], "dtc": []}
for r in star: CLS[classify(r)].append(r)
epoch = float(np.median([ot._epoch_to_mjd(r["EPOCH"]) for r in recs[::150]]))
BASSA_CUT = (2024, 117)
counts_now = {k: len(v) for k, v in CLS.items()}
counts_bassa = {k: sum(1 for r in v if desig(r) <= BASSA_CUT) for k, v in CLS.items()}
counts_build = {"v1x": 4408, "v2mini": 29988, "dtc": 0}
print(f"\ncounts: present {counts_now} (total {sum(counts_now.values())})")
print(f"        Bassa epoch (<=2024-117) {counts_bassa} (total {sum(counts_bassa.values())})")
print(f"        full buildout {counts_build} (total {sum(counts_build.values())})", flush=True)

# ---- one sample-aligned unit-EIRP pattern run per class ----
AGG = {}
for name, subset in CLS.items():
    tles = np.array(ot.to_pytles(subset), dtype=object)
    print(f"\nEPFD pattern run: {name} ({len(tles)} sats)", flush=True)
    r = ec.compute_epfd(tles, LON, LAT, ALT, 70.0, CF, EIRP_UNIT,
                        mjd_epoch=epoch, **RUN)
    AGG[name] = r["pfd_nu"]                    # [niters, ncells] at unit EIRP

slim_db = ec.ra769_limit(CF, mode="continuum")["slim_nu_db"]; slim = 10**(slim_db/10)
unit = 10**(EIRP_UNIT/10)

def scenario(counts, levels=LEVELS, stat="mean"):
    """Sample-aligned class sum, scaled to counts and measured levels."""
    tot_meas = sum(AGG[k]*(counts[k]/max(counts_now[k], 1)) *
                   (eirp_nu(levels[k][stat])/unit) for k in AGG if counts[k])
    tot_unit = sum(AGG[k]*(counts[k]/max(counts_now[k], 1)) for k in AGG if counts[k])
    p98_m = np.percentile(tot_meas, 98)
    p98_u = np.percentile(tot_unit, 98)
    excess = 10*np.log10(p98_m/slim)
    loss = 100*np.mean(tot_meas > slim)
    elim = e_dbuvm(unit*slim/p98_u)
    return excess, loss, elim

print(f"\nthreshold S_lim = {slim_db:.1f} dB(W/m2/Hz); anchor = pooled p98; "
      f"levels = class means of Bassa+24 measurements")
print(f"{'scenario':<28}{'p98 vs S_lim':>13}{'data loss':>11}{'E_lim':>18}")
for lab, c in [("A Bassa epoch (2024-07-19)", counts_bassa),
               ("B present day (2026-07-03)", counts_now),
               ("C full buildout (34 396)", counts_build)]:
    ex, lo, el = scenario(c)
    print(f"{lab:<28}{ex:+10.1f} dB{lo:9.1f} %{el:12.1f} dB(uV/m)")

# ---- benchmark: Di Vruno+23 Gen1 4408 -> expect E_lim ~ 11.7 dB(uV/m) ----
ex, lo, el = scenario({"v1x": 4408, "v2mini": 0, "dtc": 0})
print(f"\nbenchmark Gen1 4408 (Di Vruno+23 quote 11.7): our E_lim = {el:.1f} dB(uV/m)")
np.savez(cfg.out("bassa_150mhz.npz"),
         **{f"agg_{k}": v for k, v in AGG.items()},
         counts_now=str(counts_now), counts_bassa=str(counts_bassa),
         slim_db=slim_db, eirp_unit=EIRP_UNIT)
