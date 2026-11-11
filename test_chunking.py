#!/usr/bin/env python3
"""Regression test: satellite-chunked compute_epfd must equal the single-pass
result (the per-cell aggregate is additive over satellites, so chunking is exact)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import cysgp4
import epfd_core as ec

mjd = 60860.0
N = 400
rng = np.random.default_rng(0)
tles = np.array([cysgp4.PyTle(*cysgp4.tle_linestrings_from_orbital_parameters(
    f"S{i}", i + 1, mjd, 53.0, (i * 1.7) % 360, 1e-4, 0.0, (i * 0.9) % 360,
    cysgp4.satellite_mean_motion(550.0))) for i in range(N)], dtype=object)

common = dict(obs_lon=147.44, obs_lat=-42.80, obs_alt=0.04, d_rx_m=22.0,
              cf_hz=8640e6, eirp_spec_dbw_hz=-120.0, modes=("continuum", "vlbi"),
              min_el=5.0, grid_size=5.0, niters=8, time_range=2000, time_step=2.0,
              n_jobs=1, mjd_epoch=mjd, seed=3, make_plots=False, return_arrays=True)

# single pass (one chunk covers all sats) vs forced many small chunks
r_full = ec.compute_epfd(tles, sat_chunk=10_000, **common)   # 1 chunk
r_chunk = ec.compute_epfd(tles, sat_chunk=37, **common)      # 11 chunks

a, b = r_full["pfd_nu"], r_chunk["pfd_nu"]
rel = np.nanmax(np.abs(a - b) / (np.abs(a) + 1e-300))
print(f"pfd_nu shape {a.shape}; max relative diff (1 chunk vs 11 chunks) = {rel:.2e}")
for mode in ("continuum", "vlbi"):
    la = r_full["modes"][mode]["data_loss_pct"]
    lb = r_chunk["modes"][mode]["data_loss_pct"]
    print(f"  {mode:9s} data_loss: full={la:.4f}%  chunked={lb:.4f}%  d={abs(la-lb):.2e}")
assert rel < 1e-9, "chunked aggregate diverges from single-pass!"
print("CHUNKING REGRESSION OK")
