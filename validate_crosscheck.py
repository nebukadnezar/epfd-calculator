"""Cross-validation of our EPFD machinery against an independent pycraf-based
pipeline (E. van der Wateren), on a synthetic low-frequency test case.

PURPOSE
-------
The reference pipeline derives per-satellite radiation limits independently.
This script reproduces its test case with OUR pipeline
(epfd_core, the code behind every EPFD result in the paper) so the two
machineries can be compared number-to-number. It validates code, not physics
assumptions: both pipelines model the LOFAR station as a 70 m RA.1631 dish
(the merits of that assumption are examined separately in
lofar_pattern_compare.py).

REFERENCE CONFIGURATION                             -> OUR MAPPING
---------------------------------------------------------------------------
freq          = 151.525 MHz                          cf_hz = 151.525e6
ras_bandwidth = 2.95 MHz                             (RA.769 Table 1 row)
p_lim         = -199 dB(W)                           = RA.769 Delta-P_H
pfd_lim       = -194 dB(W/m2)                        = RA.769 S_H*Delta-f
                                                     spectral: -194 - 10log10(2.95e6)
                                                     = -258.7 dB(W/m2/Hz)
                                                     [ec.ra769_limit gives -259]
d_rx          = 70 m, eta = 100%                     d_rx_m=70, eta_a_rx=1.0
G_tx          = 0 dBi isotropic                      our standard convention
duty          = 100%                                 duty=1
observer      = lon 0, lat 50, alt 0                 same
eq_lat_range  = (-30 deg, +90 deg)                   we mask pointings with
                                                     declination < -30 deg
                                                     (also report unmasked)
min_elevation = 5 deg                                min_el=5
grid_size     = 3 deg                                grid_size=3.0
niters        = 100                                  50 first pass (our
                                                     convergence-validated
                                                     setting; escalate if
                                                     marginal)
time          = 2000 s @ 1 s                         2000 s @ 2 s (validated:
                                                     data-loss identical at
                                                     2/1/0.5/0.25 s; p98 ~1 dB
                                                     conservative at 2 s, see
                                                     validate_timestep.py)
constellation: Walker-like, inc 55 deg, alt 750 km,  built here with the same
               75 planes x 100 sats = 7500,          cysgp4 TLE generator and
               ecc 0.001, argp 0                     the same MA/RAAN gridding
                                                     as the reference pipeline    

REFERENCE RESULT: E_lim = 4.3 +/- 0.2 dB(uV/m) at 10 m in 120 kHz (CISPR sub-GHz
reference bandwidth), i.e. the per-satellite field strength at which the
7500-satellite aggregate just meets the RA.769 continuum threshold.

ANCHOR-STATISTIC AMBIGUITY
--------------------------
E_lim depends on WHICH statistic of the aggregate is set equal to the
threshold. We compute E_lim under several anchors and identify hers by
which one reproduces 4.3:
  mean    : pooled linear mean of all (cell x iteration) 2000 s aggregates
  median  : pooled median
  p90/p98 : pooled percentiles (p98 is our paper's convention, RA.1513 2%)
  maxcell : max over cells of the per-cell mean (worst pointing)

CONVERSION CHAIN (EIRP density -> field strength)
-------------------------------------------------
The aggregate is linear in the per-satellite EIRP density, so one run at a
trial EIRP gives every anchor:  EIRP_lim = EIRP_trial * S_lim / stat.
Then, at reference distance r = 10 m in reference bandwidth b = 120 kHz:
  pfd  = EIRP_lim,nu * b / (4 pi r^2)      [W/m2]
  E    = sqrt(pfd * Z0),  Z0 = 376.73 ohm  [V/m]
  E_dBuV/m = 20 log10(E) + 120
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import cysgp4
import epfd_core as ec
import settings as cfg

NITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
TSTEP = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

MJD_EPOCH = 60860.0                      # arbitrary fixed epoch (2026-07-04)
EIRP_TRIAL = -160.0                      # dB(W/Hz), scaled out at the end
CF = 151.525e6
Z0 = 376.730

# ---- constellation: mirror the reference pipeline ----
def walker(mjd_epoch, alt_km, inc_deg, nplanes, sats_per_plane, seed=42):
    rng = np.random.default_rng(seed)
    mas = np.linspace(0.0, 360.0, sats_per_plane, endpoint=False)
    mas = mas + rng.uniform(0, 360, 1)               # common random phase
    raans = np.linspace(0.0, 360.0, nplanes, endpoint=False)
    mas_g, raans_g = np.meshgrid(mas, raans)
    mm = cysgp4.satellite_mean_motion(alt_km)
    tles = []
    for i, (ma, raan) in enumerate(zip(mas_g.flatten(), raans_g.flatten()), 1):
        tles.append(cysgp4.PyTle(*cysgp4.tle_linestrings_from_orbital_parameters(
            f"TEST {i}", i, mjd_epoch, inc_deg, raan, 0.001, 0.0, ma % 360.0, mm)))
    return np.array(tles, dtype=object)

tles = walker(MJD_EPOCH, 750.0, 55.0, 75, 100)
print(f"constellation: {len(tles)} satellites (75 planes x 100)", flush=True)

r = ec.compute_epfd(tles, 0.0, 50.0, 0.0, 70.0, CF, EIRP_TRIAL,
                    modes=("continuum",), min_el=5, grid_size=3.0,
                    niters=NITERS, time_range=2000, time_step=TSTEP, duty=1.0,
                    eta_a_rx=1.0, n_jobs=cfg.N_JOBS, mjd_epoch=MJD_EPOCH, seed=5,
                    make_plots=False, return_arrays=True)
pfd = r["pfd_nu"]                                    # [niters, ncells], W/m2/Hz
gi = r["grid_info"]; az, el = gi["cell_lon"], gi["cell_lat"]
slim_db = ec.ra769_limit(CF, mode="continuum")["slim_nu_db"]
slim = 10**(slim_db/10)
print(f"threshold S_lim = {slim_db:.1f} dB(W/m2/Hz) "
      f"(reference -194 dBW/m2 / 2.95 MHz = {-194-10*np.log10(2.95e6):.1f})", flush=True)

# reference eq_lat_range: mask pointings below declination -30 deg
dec = np.degrees(np.arcsin(np.sin(np.radians(50))*np.sin(np.radians(el)) +
                           np.cos(np.radians(50))*np.cos(np.radians(el)) *
                           np.cos(np.radians(az))))
mask = dec >= -30.0
print(f"pointings: {pfd.shape[1]} total, {mask.sum()} with dec >= -30", flush=True)

def e_lim_dbuvm(stat_w):
    """EIRP that puts `stat_w` at S_lim -> field strength dB(uV/m) @10m/120kHz."""
    eirp_lim = 10**(EIRP_TRIAL/10) * slim/stat_w        # W/Hz
    pfd_ref = eirp_lim * 120e3 / (4*np.pi*10.0**2)      # W/m2 in 120 kHz at 10 m
    return 20*np.log10(np.sqrt(pfd_ref*Z0)) + 120

for label, sel in [("dec >= -30 (reference range)", pfd[:, mask]),
                   ("all visible sky", pfd)]:
    s = sel.flatten()
    anchors = dict(mean=s.mean(), median=np.median(s),
                   p90=np.percentile(s, 90), p98=np.percentile(s, 98),
                   maxcell=sel.mean(axis=0).max())
    print(f"\n===== {label} =====")
    for k, v in anchors.items():
        print(f"  anchor {k:8}: aggregate {10*np.log10(v):7.1f} dB(W/m2/Hz) "
              f"-> E_lim = {e_lim_dbuvm(v):5.1f} dB(uV/m) @10m/120kHz")
print(f"\nReference result: 4.3 +/- 0.2 dB(uV/m) @10m/120kHz")
np.savez(cfg.out(f"crosscheck_{NITERS}_{TSTEP}.npz"),
         pfd_nu=pfd, az=az, el=el, dec=dec, eirp_trial=EIRP_TRIAL, slim_db=slim_db)
