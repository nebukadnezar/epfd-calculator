"""Quantify the error of modelling a LOFAR-like station as an equivalent dish.

Compares the aggregate satellite pickup of two receive-pattern models at
150 MHz (HBA band) for an international-LOFAR-class station:

  DISH   : ITU-R RA.1631 pattern for a 70 m parabola (the equivalent-dish
           assumption), whose far-sidelobe floor is -12..-7 dBi.
  HYBRID : aperture-array physics. Station gain = array factor x element
           pattern; the mean far sidelobe of an N-element array is the
           embedded element pattern, NOT a dish floor. Modelled as
              G = max( RA1631(theta)          # array-factor near-in envelope
                       G_tile(theta),          # analogue tile beam, ~19 dBi,
                                               #  FWHM ~ 0.886 lam/5 m (~20 deg)
                       G_elem(za_sat) )        # embedded-element floor,
                                               #  ~6 dBi x cos^1.5(zenith angle)
           (For an LBA-like station drop G_tile; the floor argument is the
           same with the LBA dipole pattern.)

The per-satellite EIRP cancels in the dish/hybrid ratio, so the input level
is set to unity and results are reported as aggregate shifts in dB.
Population: all Starlink + OneWeb from the 2026-07-03 snapshot (UEMR carriers
at 150 MHz per Bassa et al. 2024). Site: LOFAR core, Exloo NL.
"""
import os, re, glob, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import astropy.units as u
from pycraf import antenna, geometry, conversions as cnv
import cysgp4, omm_tools as ot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
import settings as cfg
CF = 150e6; lam = 3e8/CF                      # 2 m
D_EQ = 70.0                                    # equivalent dish (cross-check convention)
LON, LAT, ALT = 6.869, 52.915, 0.015           # LOFAR core
NREAL, STEP, DUR = 8, 4.0, 2000.0
CELLS = [(az, el) for el in (15, 30, 45, 60, 75) for az in (0, 90, 180, 270)]

# --- the two receive patterns (linear gain) ---
G_T0 = 4*np.pi*25.0/lam**2                     # 5x5 m tile aperture -> ~19 dBi
FWHM_T = np.degrees(0.886*lam/5.0)             # tile beam ~20 deg at 150 MHz
G_E0, N_E = 4.0, 1.5                           # embedded element: 6 dBi, cos^1.5

def g_dish(theta_deg):
    return antenna.ras_pattern(theta_deg*u.deg, D_EQ*u.m, lam*u.m,
                               50*u.percent).to_value(cnv.dimless)

def g_hybrid(theta_deg, el_sat_deg):
    g_af = g_dish(theta_deg)                                   # near-in envelope
    g_tile = G_T0*np.exp(-4*np.log(2)*(theta_deg/FWHM_T)**2)   # tile beam
    za = np.radians(90.0 - el_sat_deg)
    g_elem = G_E0*np.cos(za)**N_E                              # element floor
    return np.maximum(np.maximum(g_af, g_tile), g_elem)

recs = ot.load_omm(cfg.OMM)
sub = [r for r in recs if re.search(r"STARLINK|ONEWEB", r.get("OBJECT_NAME", ""), re.I)]
tles = np.array(ot.to_pytles(sub), dtype=object)
epoch = float(np.median([ot._epoch_to_mjd(r["EPOCH"]) for r in recs[::150]]))
rng = np.random.default_rng(11)
starts = epoch + rng.uniform(0, 1, NREAL)
ts = np.arange(0, DUR, STEP); nt = len(ts)
print(f"{len(tles)} satellites; {NREAL} windows x {nt} steps; {len(CELLS)} pointings", flush=True)
obs = cysgp4.PyObserver(LON, LAT, ALT)

agg_d = np.zeros((NREAL, len(CELLS)))          # window-mean aggregates
agg_h = np.zeros((NREAL, len(CELLS)))
inst_d, inst_h = [], []
for r_i, s0 in enumerate(starts):
    mjds = s0 + ts/86400.0
    res = cysgp4.propagate_many(mjds[:, None], tles[None, :], obs, do_eci_pos=False,
            do_eci_vel=False, do_geo=False, do_topo=True, do_sat_azel=True,
            sat_frame="zxy", on_error="coerce_to_nan")
    taz, tel = res["topo"][..., 0], res["topo"][..., 1]
    sd = res["sat_azel"][..., 2]
    pfd = 1.0/(4*np.pi*(sd*1e3)**2)            # unit EIRP density
    for c_i, (paz, pel) in enumerate(CELLS):
        Sd = np.zeros(nt); Sh = np.zeros(nt)
        for k in range(nt):
            vis = np.isfinite(tel[k]) & (tel[k] > 0)
            if not vis.any(): continue
            ang = geometry.true_angular_distance(paz*u.deg, pel*u.deg,
                    taz[k, vis]*u.deg, tel[k, vis]*u.deg).to_value(u.deg)
            Sd[k] = np.nansum(pfd[k, vis]*g_dish(ang))
            Sh[k] = np.nansum(pfd[k, vis]*g_hybrid(ang, tel[k, vis]))
        agg_d[r_i, c_i] = Sd.mean(); agg_h[r_i, c_i] = Sh.mean()
        inst_d.append(Sd); inst_h.append(Sh)
    print(f"  window {r_i+1}/{NREAL} done", flush=True)

inst_d = np.concatenate(inst_d); inst_h = np.concatenate(inst_h)
dB = lambda x: 10*np.log10(x)
delta_win = dB(agg_h) - dB(agg_d)              # per (window, cell)
print("\n===== HYBRID minus DISH, 2000 s window-mean aggregate (dB) =====")
print(f"  all cells: median {np.median(delta_win):+.1f}  "
      f"p10 {np.percentile(delta_win,10):+.1f}  p90 {np.percentile(delta_win,90):+.1f}")
els = sorted(set(e for _, e in CELLS))
for el in els:
    m = [i for i, (a, e) in enumerate(CELLS) if e == el]
    print(f"  el {el:2d}: median {np.median(delta_win[:, m]):+.1f} dB")
fpop = 10**(np.median(delta_win)/10)
print(f"  -> equivalent to under-counting the fleet by a factor {fpop:.0f}")
d98 = dB(np.percentile(inst_h, 98)) - dB(np.percentile(inst_d, 98))
print(f"  instantaneous p98 shift: {d98:+.1f} dB")

np.savez(cfg.out("lofar_pattern_compare.npz"),
         cells=np.array(CELLS), agg_d=agg_d, agg_h=agg_h)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
for d, lab, c in [(inst_d, f"RA.1631 dish, D={D_EQ:.0f} m", "tab:blue"),
                  (inst_h, "hybrid station pattern (tile + element floor)", "tab:red")]:
    s = np.sort(d[d > 0]); ax.plot(dB(s), 1-np.arange(len(s))/len(s), color=c, lw=1.6, label=lab)
ax.set_yscale("log"); ax.grid(alpha=0.3)
ax.set_xlabel("instantaneous aggregate (dB, unit per-satellite EIRP)")
ax.set_ylabel("fraction exceeding"); ax.legend(fontsize=8)
ax.set_title("aggregate under the two patterns", fontsize=10)
med_el = [np.median(delta_win[:, [i for i, (a, e) in enumerate(CELLS) if e == el]]) for el in els]
ax2.plot(els, med_el, "o-", color="tab:red")
ax2.set_xlabel("pointing elevation (deg)"); ax2.set_ylabel("hybrid - dish (dB)")
ax2.set_title("window-mean aggregate shift vs pointing elevation", fontsize=10)
ax2.grid(alpha=0.3)
fig.suptitle(f"LOFAR-like station at {CF/1e6:.0f} MHz: equivalent-dish vs aperture-array pattern "
             f"(LOFAR core, {len(tles)} satellites)", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = cfg.out("figs", "lofar_pattern_compare.png")
fig.savefig(out, dpi=125); print("wrote", out)
