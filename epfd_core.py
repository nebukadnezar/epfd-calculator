#!/usr/bin/env python3
# coding: utf-8
"""
epfd_core.py — importable EPFD engine for the geoVLBI study.

A focused re-packaging of the validated core of `epfd_calculator_parallel.py`
(Winkel/Di Vruno/Józsa/Indermuehle), updated for the installed pycraf API and
worked entirely in *spectral* (per-Hz) units, which match both RA.769 Slim_nu
[dB(W/m^2/Hz)] and the SNIFFLES emission characterisation (EIRP spectral density
in dB(W/Hz), SFD in Jy = 1e-26 W/m^2/Hz).

  - ra769_limit       : RA.769 Slim_nu (per-Hz) with non-RA.769 interpolation
  - eirp_spec_from_sfd: EIRP spectral density [dB(W/Hz)] from SFD(Jy), distance
  - eirp_total_from_sfd: total EIRP [dBm] over a bandwidth (for saturation)
  - compute_epfd      : aggregate spectral EPFD -> data loss vs RA.769  (metric a)
  - saturation_peak   : single-sat peak in-beam Rx power [dBm]          (metric b)

Copyright (C) 2020+ Winkel, Di Vruno, Józsa; 2025+ Indermuehle. GPLv3.
"""

import os
import warnings
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy import units as u, constants as const
from astropy.utils.misc import NumpyRNGContext
from scipy.stats import percentileofscore

from pycraf import conversions as cnv
from pycraf import protection, antenna, geometry
import cysgp4
from joblib import Parallel, delayed, parallel_backend

Z0 = 376.730313668          # ohm, free-space impedance (dBuV/m conversions)
_MODE = {"cont": "continuum", "continuum": "continuum",
         "spec": "spectroscopy", "spectroscopy": "spectroscopy",
         "vlbi": "vlbi"}


# --------------------------------------------------------------------------
# RA.769 limits  (Slim_nu, per-Hz spectral PFD)
# --------------------------------------------------------------------------
def ra769_limit(cf_hz, mode="vlbi", interp_non_ra769=True):
    """
    Return dict(freq_ghz, Slim_nu_W_m2_Hz, interpolated) for the RA.769 band
    containing cf_hz. If cf_hz is outside all tabulated bands and
    interp_non_ra769, interpolate Slim_nu in dB vs log10(freq) between the two
    nearest band centres (RA.769 tabulates discrete band centres only).
    """
    t = protection.ra769_limits(mode=_MODE[mode])
    f_mhz = t["frequency"].to_value(u.MHz)
    slim_db = t["Slim_nu"].to_value(u.dB(u.W / u.m**2 / u.Hz))   # dB(W/m^2/Hz)
    cf_mhz = cf_hz / 1e6

    # exact-band: RA.769 gives discrete centres; treat nearest centre within a
    # tolerance as "in band". For our purposes we always evaluate at the centre
    # or interpolate; pick the closest centre if extremely close.
    idx_exact = np.where(np.abs(f_mhz - cf_mhz) < 1.0)[0]
    if idx_exact.size:
        i = idx_exact[0]
        return dict(freq_ghz=f_mhz[i] / 1e3,
                    Slim_nu_W_m2_Hz=10 ** (slim_db[i] / 10),
                    slim_nu_db=float(slim_db[i]), interpolated=False)

    if not interp_non_ra769:
        raise ValueError(f"{cf_mhz:.3f} MHz not an RA.769 {mode} band centre")

    if cf_mhz <= f_mhz.min():
        i = 0
        return dict(freq_ghz=cf_hz / 1e9, Slim_nu_W_m2_Hz=10 ** (slim_db[i] / 10),
                    slim_nu_db=float(slim_db[i]), interpolated=True)
    if cf_mhz >= f_mhz.max():
        i = -1
        return dict(freq_ghz=cf_hz / 1e9, Slim_nu_W_m2_Hz=10 ** (slim_db[i] / 10),
                    slim_nu_db=float(slim_db[i]), interpolated=True)

    j = np.searchsorted(f_mhz, cf_mhz)
    f1, f2 = f_mhz[j - 1], f_mhz[j]
    s1, s2 = slim_db[j - 1], slim_db[j]
    w = (np.log10(cf_mhz) - np.log10(f1)) / (np.log10(f2) - np.log10(f1))
    s_db = (1 - w) * s1 + w * s2
    return dict(freq_ghz=cf_hz / 1e9, Slim_nu_W_m2_Hz=10 ** (s_db / 10),
                slim_nu_db=float(s_db), interpolated=True)


# --------------------------------------------------------------------------
# EIRP from SFD
# --------------------------------------------------------------------------
def eirp_spec_from_sfd(sfd_jy, dist_m):
    """EIRP spectral density [dB(W/Hz)] from SFD (Jy) at slant range dist_m."""
    pfd_nu = sfd_jy * 1e-26                       # W/m^2/Hz
    eirp_nu_w_hz = pfd_nu * 4 * np.pi * dist_m**2  # W/Hz
    return 10 * np.log10(eirp_nu_w_hz)


def eirp_total_from_sfd(sfd_jy, bw_hz, dist_m):
    """Total EIRP [dBm] over bandwidth bw_hz (for saturation metric)."""
    pfd = sfd_jy * 1e-26 * bw_hz                  # W/m^2
    eirp_w = pfd * 4 * np.pi * dist_m**2          # W
    return 10 * np.log10(eirp_w * 1000.0)


# --------------------------------------------------------------------------
# Sky-cell grid (ITU-R M.1583)
# --------------------------------------------------------------------------
def sample_on_cell_grid_m1583(niters, step_size=5 * u.deg,
                              lat_range=(0 * u.deg, 90 * u.deg), rnd_seed=None):
    """Sample telescope pointings on an equal-area sky-cell grid (M.1583)."""
    def sample(niters, low_lon, high_lon, low_lat, high_lat):
        z_low = np.cos(np.radians(90 - low_lat))
        z_high = np.cos(np.radians(90 - high_lat))
        az = np.random.uniform(low_lon, high_lon, size=niters)
        el = 90 - np.degrees(np.arccos(np.random.uniform(z_low, z_high, size=niters)))
        return az, el

    cell_edges, cell_mids, solid_angles, tel_az, tel_el = [], [], [], [], []
    lat_range = (lat_range[0].to_value(u.deg), lat_range[1].to_value(u.deg))
    ncells_lat = int((lat_range[1] - lat_range[0]) / step_size.to_value(u.deg) + 0.5)
    edge_lats = np.linspace(lat_range[0], lat_range[1], ncells_lat + 1, endpoint=True)
    mid_lats = 0.5 * (edge_lats[1:] + edge_lats[:-1])
    with NumpyRNGContext(rnd_seed):
        for low_lat, mid_lat, high_lat in zip(edge_lats[:-1], mid_lats, edge_lats[1:]):
            ncells_lon = int(360 * np.cos(np.radians(mid_lat)) / step_size.to_value(u.deg) + 0.5)
            edge_lons = np.linspace(0, 360, ncells_lon + 1, endpoint=True)
            mid_lons = 0.5 * (edge_lons[1:] + edge_lons[:-1])
            solid_angle = (edge_lons[1] - edge_lons[0]) * np.degrees(
                np.sin(np.radians(high_lat)) - np.sin(np.radians(low_lat)))
            for low_lon, mid_lon, high_lon in zip(edge_lons[:-1], mid_lons, edge_lons[1:]):
                cell_edges.append((low_lon, high_lon, low_lat, high_lat))
                cell_mids.append((mid_lon, mid_lat))
                solid_angles.append(solid_angle)
                a, e = sample(niters, low_lon, high_lon, low_lat, high_lat)
                tel_az.append(a)
                tel_el.append(e)
    tel_az = np.array(tel_az).T
    tel_el = np.array(tel_el).T
    grid_info = np.column_stack([cell_mids, cell_edges, solid_angles])
    grid_info.dtype = np.dtype([
        ('cell_lon', np.float64), ('cell_lat', np.float64),
        ('cell_lon_low', np.float64), ('cell_lon_high', np.float64),
        ('cell_lat_low', np.float64), ('cell_lat_high', np.float64),
        ('solid_angle', np.float64)])
    return tel_az, tel_el, grid_info[:, 0]


def _sat_gain_func(az, el):
    az, el = np.broadcast_arrays(az, el)
    return np.zeros(az.shape, dtype=np.float64) * cnv.dBi   # 0 dBi isotropic Tx


def _propagate(mjds, sats_tles, obs_lon, obs_lat, obs_alt, sat_frame="zxy"):
    observer = cysgp4.PyObserver(obs_lon, obs_lat, obs_alt)
    # Request only the fields we use: topo (az/el/dist) for the link geometry and
    # sat_azel (slant range) for the FSPL. Suppressing eci_pos/eci_vel/geo cuts the
    # propagation array from 16 to 7 float64 per (iter,tstep,sat) element
    # (128 -> 56 bytes), a 2.3x memory saving with no change to the result.
    return cysgp4.propagate_many(
        mjds[:, :, np.newaxis], sats_tles[np.newaxis, np.newaxis, :], observer,
        do_eci_pos=False, do_eci_vel=False, do_geo=False,
        do_topo=True, do_sat_azel=True,
        sat_frame=sat_frame, on_error="coerce_to_nan")


def _iteration_prx(topo_el_i, topo_az_i, sat_dist_i, sat_el_i, sat_az_i,
                   tel_az_i, tel_el_i, n_tsteps, freq, p_tx, d_rx, eta_a_rx):
    """One Monte-Carlo iteration. Inputs are this iteration's slices only
    (so the parallel workers don't each receive the full propagation arrays)."""
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
    ncells = tel_az_i.shape[0]
    p_rx_iter = np.zeros(ncells, dtype=np.float64)
    vis = topo_el_i > 0.
    if np.count_nonzero(vis) == 0:
        return p_rx_iter
    FSPL = cnv.free_space_loss(sat_dist_i[vis, np.newaxis] * u.km, freq).to(cnv.dB)
    G_tx = _sat_gain_func(sat_az_i[vis, np.newaxis] * u.deg,
                          sat_el_i[vis, np.newaxis] * u.deg)
    chunk = 100
    nchunks = ncells // chunk + 1
    for c in range(nchunks):
        sl = slice(c * chunk, min((c + 1) * chunk, ncells))
        ang = geometry.true_angular_distance(
            tel_az_i[np.newaxis, sl] * u.deg, tel_el_i[np.newaxis, sl] * u.deg,
            topo_az_i[vis, np.newaxis] * u.deg, topo_el_i[vis, np.newaxis] * u.deg)
        G_rx = antenna.ras_pattern(ang, d_rx, const.c / freq, eta_a_rx)
        _p = np.sum((p_tx.to(cnv.dB_W) + G_tx + FSPL + G_rx).to_value(u.W), axis=0) / n_tsteps
        p_rx_iter[sl] = _p
    return p_rx_iter


def _plot_skygrid_rect(grid_info, values, cnv_label, title, path,
                       cmap="viridis", vmin=None, vmax=None):
    """Rectangular az/el sky-cell map coloured by `values` (one per cell)."""
    values = np.asarray(values)
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax
    denom = (vmax - vmin) or 1e-20
    norm = (values - vmin) / denom
    fig = plt.figure(figsize=(12, 4))
    plt.bar(grid_info['cell_lon_low'],
            height=grid_info['cell_lat_high'] - grid_info['cell_lat_low'],
            width=grid_info['cell_lon_high'] - grid_info['cell_lon_low'],
            bottom=grid_info['cell_lat_low'],
            color=plt.get_cmap(cmap)(norm), align='edge')
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap),
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, ax=plt.gca()).set_label(cnv_label)
    plt.title(title)
    plt.xlabel("Azimuth [deg]"); plt.ylabel("Elevation [deg]")
    plt.xlim(0, 360); plt.ylim(0, 90)
    plt.savefig(path, bbox_inches="tight", dpi=110); plt.close(fig)
    return path


def _plot_skygrid_polar(grid_info, values, cnv_label, title, path,
                        cmap="viridis", vmin=None, vmax=None, font_scale=1.0,
                        ring_font=None, dpi=110, ring_halo=False):
    """Polar all-sky map (zenith at centre, horizon at rim) coloured by `values`.
    Azimuth N at top, increasing clockwise (N->E->S->W). ring_font overrides
    the elevation-ring label size (default 7 * font_scale); ring_halo draws a
    white outline behind the ring labels (for dark colour maps)."""
    if ring_font is None:
        ring_font = 7 * font_scale
    values = np.asarray(values)
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax
    denom = (vmax - vmin) or 1e-20
    norm = (values - vmin) / denom
    theta_l = np.radians(grid_info['cell_lon_low'])
    theta_w = np.radians(grid_info['cell_lon_high'] - grid_info['cell_lon_low'])
    r_out = 1 - grid_info['cell_lat_low'] / 90.0     # low elevation -> rim
    r_in = 1 - grid_info['cell_lat_high'] / 90.0     # high elevation -> centre
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(theta_l, height=(r_out - r_in), width=theta_w, bottom=r_in,
           color=plt.get_cmap(cmap)(norm), edgecolor="none", align="edge")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 7))
    ax.set_yticklabels([str(int(x)) for x in np.linspace(90, 0, 7)], fontsize=ring_font)
    if ring_halo:
        import matplotlib.patheffects as pe
        for t in ax.get_yticklabels():
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=10 * font_scale)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap),
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax, pad=0.10, shrink=0.85)
    cb.set_label(cnv_label, fontsize=10 * font_scale)
    cb.ax.tick_params(labelsize=10 * font_scale)
    ax.set_title(title, pad=18, fontsize=12 * font_scale)
    plt.savefig(path, bbox_inches="tight", dpi=dpi); plt.close(fig)
    return path


_MODE_LABEL = {"continuum": "Continuum", "spectroscopy": "Spectroscopy",
               "vlbi": "VLBI"}


def _plot_cdf(pfd_nu, modes_metrics, title, path, paper=False):
    """CDF of aggregate PFD with each mode's RA.769 limit line + data loss.

    paper=True renders the manuscript version (single-column figure): same
    canvas width, 20 percent taller, all text 13.5 pt, no title (the caption
    carries it), short legend labels, 300 dpi."""
    flat = np.sort(pfd_nu.flatten())
    flat_db = 10 * np.log10(flat)
    y = 100 * np.linspace(0, 1, flat_db.size, endpoint=True)
    fs = 13.5 if paper else plt.rcParams["font.size"]
    fig = plt.figure(figsize=(9, 7.2) if paper else (9, 6))
    plt.plot(flat_db[::max(1, flat_db.size // 4000)],
             y[::max(1, flat_db.size // 4000)], "k-",
             label="aggregate EPFD" if paper else "CDF (all iters)")
    plt.axhline(98., color="0.6", ls=":", lw=1)
    colors = {"continuum": "tab:red", "spectroscopy": "tab:green", "vlbi": "tab:blue"}
    for mode, m in modes_metrics.items():
        c = colors.get(mode, "tab:purple")
        if paper:
            nm = {"continuum": "continuum", "vlbi": "VLBI"}.get(mode, mode)
            # no loss percentage in the legend: the caption and the tables carry
            # it, from the production (floor-cache) pipeline, and the plotted
            # array is a different Monte-Carlo realisation (point-level scatter)
            lab = f"{nm} limit: {m['slim_nu_db']:.1f} dB"
        else:
            lab = (f"RA.769 {_MODE_LABEL.get(mode, mode)}: {m['slim_nu_db']:.1f} dB  "
                   f"(loss {m['data_loss_pct']:.2f}%, margin {m['ras_margin_db']:+.1f} dB)")
        plt.axvline(m["slim_nu_db"], color=c, ls="--", label=lab)
    plt.grid(color="0.85")
    plt.xlabel("Aggregate PFD [dB(W m$^{-2}$ Hz$^{-1}$)]", fontsize=fs)
    plt.ylabel("Cumulative probability [%]", fontsize=fs)
    plt.tick_params(labelsize=fs)
    if paper:
        plt.legend(loc="upper left", fontsize=fs)
        plt.savefig(path, bbox_inches="tight", dpi=300)
    else:
        plt.title(title); plt.legend(loc="upper left", fontsize=8)
        plt.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return path


def metrics_from_pfd(pfd_nu, cf_hz, modes=("continuum", "vlbi"),
                     interp_non_ra769=True, n_cells=None):
    """
    Per-mode data-loss / margin metrics from an aggregate spectral-PFD array
    pfd_nu [W/m^2/Hz] of shape (niters, ncells). Reused for analytic N-scaling
    (shift pfd_nu by the linear factor, recompute) without re-propagating.
    """
    to_db = lambda x: 10 * np.log10(np.asarray(x))
    pfd_avg = np.mean(pfd_nu, axis=0)
    pfd98 = np.percentile(pfd_nu, 98.)
    ncells = n_cells if n_cells is not None else pfd_nu.shape[1]
    modes_out, per_cell_loss = {}, {}
    for mode in modes:
        lim = ra769_limit(cf_hz, mode=mode, interp_non_ra769=interp_non_ra769)
        slim = lim["Slim_nu_W_m2_Hz"]
        data_loss = 100 - percentileofscore(pfd_nu.flatten(), slim, kind="strict")
        bad = np.count_nonzero(pfd_avg > slim)
        per_cell_loss[mode] = np.array([
            100 - percentileofscore(pfd_nu[:, c], slim, kind="strict")
            for c in range(pfd_nu.shape[1])])
        modes_out[mode] = dict(
            data_loss_pct=float(data_loss),
            slim_nu_db=float(lim["slim_nu_db"]),
            pfd98_db=float(to_db(pfd98)),
            ras_margin_db=float(lim["slim_nu_db"] - to_db(pfd98)),
            bad_cell_frac=float(bad / ncells),
            interpolated=bool(lim["interpolated"]))
    return modes_out, per_cell_loss


def compute_epfd(sats_tles, obs_lon, obs_lat, obs_alt, d_rx_m, cf_hz,
                 eirp_spec_dbw_hz, modes=("continuum", "vlbi"), min_el=5.0,
                 grid_size=5.0, niters=10, time_range=2000, time_step=1.0,
                 duty=1.0, eta_a_rx=1.0, n_jobs=-1, mjd_epoch=None, seed=0,
                 interp_non_ra769=True, figpath=None, label=None,
                 make_plots=True, obslabel=None, return_arrays=False,
                 sat_chunk=None, prop_budget=2.0e8):
    """
    Aggregate spectral EPFD vs RA.769 limits (metric a), for one or more modes.

    The aggregate spectral PFD is a single physical quantity; each mode in
    `modes` ('continuum' = single-dish RAS, 'spectroscopy', 'vlbi') only changes
    the threshold line it is compared against. Returns metrics per mode and, if
    make_plots and figpath, writes a sky-coverage map and a CDF/data-loss plot.

    eirp_spec_dbw_hz : per-satellite EIRP spectral density in dB(W/Hz) (0 dBi Tx).
    """
    freq = cf_hz * u.Hz
    p_tx = (eirp_spec_dbw_hz + 10 * np.log10(duty)) * cnv.dB_W  # carries /Hz implicitly
    d_rx = d_rx_m * u.m

    if mjd_epoch is None:
        raise ValueError(
            "mjd_epoch is required: pass the TLE/OMM epoch so satellites are not "
            "propagated far from epoch (use the snapshot epoch, e.g. via "
            "omm_tools._epoch_to_mjd / the median OMM EPOCH).")
    rng = np.random.default_rng(seed)
    start_mjd = mjd_epoch + rng.uniform(0, 1, niters)          # spread over a day
    td = np.arange(0, time_range, time_step) / 86400.
    mjds = start_mjd[:, np.newaxis] + td[np.newaxis]
    n_tsteps = mjds.shape[1]

    tel_az, tel_el, grid_info = sample_on_cell_grid_m1583(
        niters, step_size=grid_size * u.deg,
        lat_range=(min_el * u.deg, 90 * u.deg), rnd_seed=seed)
    ncells = len(grid_info)

    njobs = mp.cpu_count() if n_jobs == -1 else n_jobs
    eta_q = eta_a_rx * 100 * u.percent          # ras_pattern wants a percent Quantity

    # Satellite chunking: the aggregate per-cell received power is a SUM over
    # satellites (see _iteration_prx, axis=0), so we propagate the constellation
    # in chunks and accumulate. This bounds the peak propagation array to
    # niters*ntsteps*sat_chunk elements regardless of the total population, so a
    # million-satellite run fits in memory at full niters; only wall-clock grows
    # with N. The accumulation is exact (no approximation): summing chunk
    # contributions then flooring equals flooring the full sum.
    nsats_tot = len(sats_tles)
    if sat_chunk is None:
        sat_chunk = max(1, int(prop_budget / max(1, niters * n_tsteps)))
    sat_chunk = min(sat_chunk, max(1, nsats_tot))
    nchunks = (nsats_tot + sat_chunk - 1) // sat_chunk

    p_rx = np.zeros((niters, ncells), dtype=np.float64)
    for ci in range(nchunks):
        chunk_tles = sats_tles[ci * sat_chunk:(ci + 1) * sat_chunk]
        res = _propagate(mjds, chunk_tles, obs_lon, obs_lat, obs_alt)
        topo = res["topo"]; sazel = res["sat_azel"]
        topo_az, topo_el = topo[..., 0], topo[..., 1]
        sat_az, sat_el, sat_dist = sazel[..., 0], sazel[..., 1], sazel[..., 2]

        def _slice(i):     # pass only iteration i's data to the worker (memory)
            return (topo_el[i], topo_az[i], sat_dist[i], sat_el[i], sat_az[i],
                    tel_az[i], tel_el[i], n_tsteps, freq, p_tx, d_rx, eta_q)
        if njobs == 1:
            p_rx_chunk = np.array([_iteration_prx(*_slice(i)) for i in range(niters)])
        else:
            # loky backend (fork-safe). inner_max_num_threads=1 pins each worker's
            # OMP/BLAS pools to a single thread, so the njobs workers don't each
            # spawn OpenMP threads and oversubscribe (load >> ncores). The parent-
            # side cysgp4 propagation, run outside this block, still uses the full
            # OMP_NUM_THREADS for its own parallelism.
            with parallel_backend("loky", inner_max_num_threads=1):
                p_rx_chunk = np.array(Parallel(n_jobs=njobs)(
                    delayed(_iteration_prx)(*_slice(i)) for i in range(niters)))
        p_rx += p_rx_chunk
        del res, topo, sazel, p_rx_chunk

    p_rx[p_rx < 1e-89] = 1e-89
    # received spectral power -> incident spectral PFD [W/m^2/Hz]
    pfd_nu = (cnv.powerflux_from_prx(p_rx * 1e60 * u.W, freq, 0 * cnv.dBi) * 1e-60
              ).to_value(u.W / u.m**2)
    to_db = lambda x: 10 * np.log10(np.asarray(x))

    pfd_avg = np.mean(pfd_nu, axis=0)                 # per cell, mode-independent

    modes_out, per_cell_loss = metrics_from_pfd(
        pfd_nu, cf_hz, modes=modes, interp_non_ra769=interp_non_ra769,
        n_cells=len(grid_info))
    out = dict(freq_ghz=cf_hz / 1e9, n_sats=int(len(sats_tles)),
               n_cells=int(len(grid_info)), modes=modes_out, plots=[])
    if return_arrays:
        out["pfd_nu"] = pfd_nu
        out["pfd_avg"] = pfd_avg
        out["grid_info"] = grid_info

    if make_plots and figpath:
        os.makedirs(figpath, exist_ok=True)
        lab = label or f"epfd_{cf_hz/1e6:.0f}MHz"
        site = obslabel or ""
        # 1) sky-coverage map: average PFD per cell (polar)
        p = _plot_skygrid_polar(
            grid_info, to_db(pfd_avg), cnv_label="Average PFD / cell [dB(W m$^{-2}$ Hz$^{-1}$)]",
            title=f"{site}  {cf_hz/1e6:.0f} MHz  ({len(sats_tles)} sats)\nsky coverage of aggregate EPFD",
            path=os.path.join(figpath, f"{lab}_skycoverage_avgpfd.png"))
        out["plots"].append(p)
        # 2) data-loss-per-cell map, one per criterion (polar)
        for mode in modes:
            p = _plot_skygrid_polar(
                grid_info, per_cell_loss[mode],
                cnv_label=f"Data loss / cell [%] ({_MODE_LABEL.get(mode, mode)})",
                title=f"{site}  {cf_hz/1e6:.0f} MHz  ({len(sats_tles)} sats)\n"
                      f"data loss vs RA.769 {_MODE_LABEL.get(mode, mode)}",
                path=os.path.join(figpath, f"{lab}_dataloss_{mode}.png"),
                cmap="inferno", vmin=0)
            out["plots"].append(p)
        # 3) CDF with all mode limit lines
        p = _plot_cdf(pfd_nu, out["modes"],
                      title=f"{site}  {cf_hz/1e6:.0f} MHz  ({len(sats_tles)} sats)",
                      path=os.path.join(figpath, f"{lab}_cdf.png"))
        out["plots"].append(p)
    return out


def saturation_peak(eirp_dbm_total, dist_m, d_rx_m, eta_a_rx=0.6,
                    p_lin_dbm=-40.0, p_dmg_dbm=12.0):
    """
    Metric (b): peak received power [dBm] for a single satellite of total EIRP
    eirp_dbm_total sitting in the main beam at slant range dist_m, into a dish of
    diameter d_rx_m. Compared with receiver linear/damage limits.
    """
    a_eff = eta_a_rx * np.pi * (d_rx_m / 2.0) ** 2
    eirp_w = 10 ** (eirp_dbm_total / 10) / 1000.0
    pfd = eirp_w / (4 * np.pi * dist_m ** 2)
    p_rx_dbm = 10 * np.log10(pfd * a_eff * 1000.0)
    return dict(p_rx_dbm=float(p_rx_dbm), pfd_w_m2=float(pfd),
                over_linear_db=float(p_rx_dbm - p_lin_dbm),
                over_damage_db=float(p_rx_dbm - p_dmg_dbm))
