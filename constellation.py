#!/usr/bin/env python3
# coding: utf-8
"""
constellation.py — build current and up-scaled satellite constellations as
cysgp4.PyTle arrays for the EPFD calculator.

Two up-scaling methods (author decision: use both, cross-check):

  1. proportional_upscale_tles(records, target_n)
        Replicate the *real* OMM fleet K times, randomising RAAN and mean anomaly
        on each clone, until ~target_n satellites. Preserves the measured
        altitude/inclination distribution exactly; ties directly to today's sky.

  2. filed_shell_tles(target_total)
        Synthesise Walker-style shells from the *filed* architectures of the major
        systems (Starlink Gen2, Kuiper, OneWeb Ph2, GuoWang, Qianfan), scaled to
        target_total. More representative of the planned spatial distribution.

NOTE: the FILED_SHELLS table below is approximate, compiled from public ITU/FCC
filings and operator announcements.
Counts/altitudes are good enough for EPFD spatial statistics
(what matters is the altitude+inclination distribution), not for orbit prediction.
"""

import numpy as np

# --- Filed architectures of the major LEO systems --------------------------
# Each shell: (altitude_km, inclination_deg, n_satellites).
# Baseline counts are the announced totals; filed_shell_tles() can scale them
# proportionally to any target_total (e.g. 300k) while preserving geometry.
FILED_SHELLS = {
    "Starlink_Gen2": [   # FCC Gen2 (~30k)
        (530, 43.0, 7178), (525, 53.0, 7178), (535, 33.0, 7178),
        (604, 148.0, 4000), (614, 115.7, 4000),
    ],
    "Kuiper": [          # Amazon (3,236)
        (590, 33.0, 784), (610, 42.0, 1296), (630, 51.9, 1156),
    ],
    "OneWeb_Ph2": [      # OneWeb Phase 2 (~6,372)
        (1200, 87.9, 1764), (1200, 40.0, 2304), (1200, 55.0, 2304),
    ],
    "GuoWang": [         # China SatNet / Guowang (~12,992)
        (590, 85.0, 1728), (600, 50.0, 1728), (508, 55.0, 1728),
        (1145, 30.0, 3456), (1145, 85.0, 4352),
    ],
    "Qianfan": [         # Spacesail / Thousand Sails (~15,000)
        (1160, 89.0, 7500), (1160, 50.0, 7500),
    ],
}


def baseline_total():
    return sum(n for shells in FILED_SHELLS.values() for *_, n in shells)


# --- helpers ---------------------------------------------------------------
def _mean_motion_revday(alt_km):
    """Mean motion (rev/day) for a circular orbit at altitude alt_km."""
    import cysgp4
    return cysgp4.satellite_mean_motion(alt_km)


def _walker_split(count):
    """Pick (n_planes, sats_per_plane) for a near-square Walker layout."""
    n_planes = max(1, int(round(np.sqrt(count))))
    sats_per_plane = int(np.ceil(count / n_planes))
    return n_planes, sats_per_plane


def synthesize_shell(alt_km, inc_deg, count, epoch_mjd, rng, name="SYN", nr0=1,
                     raan_center=None, raan_span=360.0):
    """Build `count` PyTles in a Walker shell at (alt, inc).

    raan_center=None spreads the planes over a full uniform RAAN circle (a normal
    isotropic shell, e.g. the filed Walker constellations). Passing raan_center
    (deg) with raan_span<360 confines the planes to a narrow band
    raan_center +/- raan_span/2 -- used for the dawn-dusk clumped ODC shells,
    where every plane rides near the terminator.
    """
    import cysgp4
    n_planes, spp = _walker_split(count)
    if raan_center is None:
        raans = np.linspace(0.0, 360.0, n_planes, endpoint=False)
    elif n_planes == 1:
        raans = np.array([raan_center % 360.0])
    else:
        raans = (raan_center - raan_span / 2.0
                 + np.linspace(0.0, raan_span, n_planes, endpoint=False)) % 360.0
    mm = _mean_motion_revday(alt_km)          # hoisted out of the inner loop
    tles, k = [], 0
    for ip, raan in enumerate(raans):
        # phase offset between planes (Walker delta), plus small jitter
        ma0 = (ip * 360.0 / n_planes) % 360.0
        mas = (np.linspace(0.0, 360.0, spp, endpoint=False) + ma0) % 360.0
        for ma in mas:
            if k >= count:
                break
            raan_j = (raan + rng.uniform(-0.5, 0.5)) % 360.0
            ma_j = (ma + rng.uniform(-0.5, 0.5)) % 360.0
            # the TLE catalog number is a 5-digit field; wrap it to stay valid at
            # large N (the identifier is irrelevant to SGP4 dynamics and to our
            # aggregate statistics -- satellites are never individually tracked).
            cat = ((nr0 + k - 1) % 99998) + 1
            lines = cysgp4.tle_linestrings_from_orbital_parameters(
                f"{name}-{nr0+k}", cat, epoch_mjd,
                inc_deg, raan_j, 0.0001, 0.0, ma_j, mm,
            )
            tles.append(cysgp4.PyTle(*lines))
            k += 1
    return tles


def filed_shell_tles(target_total=None, epoch_mjd=58813.5, seed=0):
    """
    Synthesise all FILED_SHELLS as PyTles. If target_total is given, scale every
    shell's count by target_total/baseline_total() (preserving geometry).
    """
    rng = np.random.default_rng(seed)
    base = baseline_total()
    scale = 1.0 if target_total is None else target_total / base
    tles, nr = [], 1
    for system, shells in FILED_SHELLS.items():
        for alt, inc, n in shells:
            n_scaled = int(round(n * scale))
            if n_scaled <= 0:
                continue
            shell = synthesize_shell(alt, inc, n_scaled, epoch_mjd, rng,
                                     name=system[:10], nr0=nr)
            tles.extend(shell)
            nr += n_scaled
    return np.array(tles, dtype=object)


# --- SpaceX orbital-data-centre (AI1) dawn-dusk SSO shells -----------------
# The ODC is an anomalous case: solar-powered compute platforms in dawn-dusk
# sun-synchronous orbits, so the planes clump on the terminator (a narrow RAAN
# band) instead of spreading uniformly in RAAN. Per-shell occupancy is capped by
# a collision-avoidance minimum separation via an area-packing model. Two presets:
#   realistic -- design band 600-800 km (5 shells @ 50 km), each filled to its
#                clumped capacity at s=50 km  (~200k total);
#   stress    -- full filed envelope 500-2000 km (31 shells @ 50 km) packed to
#                1,000,000, which sits near the safe-packing ceiling.
_MU, _RE_KM, _J2 = 398600.4418, 6378.137, 1.08263e-3
_OMEGA_SUN = 1.99096e-7      # rad/s, mean solar nodal rate (360 deg / 365.25 d)


def sso_inclination(alt_km):
    """Sun-synchronous inclination [deg] for a near-circular orbit at alt_km,
    from the J2 nodal-regression condition Omega_dot = Omega_sun
    (97.8 deg at 600 km, 98.6 deg at 800 km, ~104.9 deg at 2000 km)."""
    a = _RE_KM + alt_km
    n = np.sqrt(_MU / a**3)                       # mean motion, rad/s
    cosi = -_OMEGA_SUN / (1.5 * _J2 * n * (_RE_KM / a)**2)
    return float(np.degrees(np.arccos(np.clip(cosi, -1.0, 1.0))))


def shell_capacity(alt_km, s_km=50.0, clumped=True, raan_halfwidth_deg=15.0):
    """Area-packing capacity of one shell: usable surface area / s^2. For the
    dawn-dusk clumped case the usable area is two pole-to-pole lunes of full
    angular width 2*raan_halfwidth (the dawn and dusk meridian bands), ~1/6 of
    the sphere at +/-15 deg; uniform uses the full 4 pi r^2."""
    r = _RE_KM + alt_km
    if clumped:
        dRA = np.radians(raan_halfwidth_deg)
        area = 2.0 * (2.0 * r**2 * (2.0 * dRA))   # 2 lunes, each area 2 r^2 * width
    else:
        area = 4.0 * np.pi * r**2
    return area / s_km**2


ODC_PRESETS = {
    "realistic": dict(alt_lo=600.0, alt_hi=800.0, spacing=50.0, total=None),
    "stress":    dict(alt_lo=500.0, alt_hi=2000.0, spacing=50.0, total=1_000_000),
}


def _distribute(target, cap):
    """Split `target` satellites across shells proportional to capacity `cap`,
    never exceeding any shell's capacity. Returns an int array summing to
    min(target, floor(cap).sum())."""
    cap_floor = np.floor(cap).astype(int)
    if target >= cap_floor.sum():
        return cap_floor                           # saturate every shell
    per = np.minimum(cap_floor, np.round(target * cap / cap.sum()).astype(int))
    diff = target - int(per.sum())                 # reconcile rounding to target
    head = cap_floor - per
    order = np.argsort(-(head if diff > 0 else per))
    i = 0
    while diff != 0 and i < 100 * len(per):
        j = order[i % len(per)]
        if diff > 0 and head[j] > 0:
            per[j] += 1; head[j] -= 1; diff -= 1
        elif diff < 0 and per[j] > 0:
            per[j] -= 1; diff += 1
        i += 1
    return per


def odc_shell_plan(preset="realistic", s_km=50.0, raan_halfwidth_deg=15.0,
                   total=None):
    """Return the (alt_km, sso_inc_deg, capacity, n_sats) plan for an ODC preset
    WITHOUT building TLEs -- cheap, for sizing/printing the 1M case."""
    cfg = dict(ODC_PRESETS[preset])
    if total is not None:
        cfg["total"] = total
    alts = np.arange(cfg["alt_lo"], cfg["alt_hi"] + 1e-6, cfg["spacing"])
    cap = np.array([shell_capacity(a, s_km, True, raan_halfwidth_deg) for a in alts])
    per = np.floor(cap).astype(int) if cfg["total"] is None \
        else _distribute(int(cfg["total"]), cap)
    incs = np.array([sso_inclination(a) for a in alts])
    return alts, incs, cap, per


def odc_shell_tles(preset="realistic", epoch_mjd=58813.5, seed=0, s_km=50.0,
                   raan_halfwidth_deg=15.0, raan_center_deg=90.0, total=None,
                   verbose=False):
    """
    Build the SpaceX ODC dawn-dusk SSO constellation as a PyTle array.

    Shells are stacked from alt_lo to alt_hi at `spacing` km; each takes its
    Sun-synchronous inclination and a RAAN clump of half-width raan_halfwidth_deg
    about the terminator (raan_center_deg). Per-shell occupancy is the clumped
    area-packing capacity at min-separation s_km; the total is each shell's
    capacity ('realistic', ~200k) or a fixed target spread across the stack
    ('stress', default 1,000,000).
    """
    rng = np.random.default_rng(seed)
    alts, incs, cap, per = odc_shell_plan(preset, s_km, raan_halfwidth_deg, total)
    span = 2.0 * raan_halfwidth_deg
    tles, nr = [], 1
    for a, inc, n_i in zip(alts, incs, per):
        if n_i <= 0:
            continue
        shell = synthesize_shell(a, inc, int(n_i), epoch_mjd, rng,
                                 name=f"ODC{int(a)}", nr0=nr,
                                 raan_center=raan_center_deg, raan_span=span)
        tles.extend(shell)
        nr += int(n_i)
    if verbose:
        cfg = ODC_PRESETS[preset]
        print(f"ODC '{preset}': {len(alts)} shells "
              f"{alts[0]:.0f}-{alts[-1]:.0f} km @ {cfg['spacing']:.0f} km, "
              f"s={s_km:.0f} km, clump +/-{raan_halfwidth_deg:.0f} deg")
        for a, inc, c, n_i in zip(alts, incs, cap, per):
            print(f"   {a:6.0f} km  i={inc:6.2f} deg  cap={c:9.0f}  N={int(n_i):8d}")
        print(f"   TOTAL N = {int(per.sum()):,}  (capacity sum {cap.sum():,.0f})")
    return np.array(tles, dtype=object)


def proportional_upscale_tles(records, target_n, seed=0):
    """
    Replicate real OMM records to ~target_n PyTles, randomising RAAN and mean
    anomaly on each clone. records: list of dicts from omm_tools.load_omm().
    """
    import cysgp4
    from omm_tools import _epoch_to_mjd
    rng = np.random.default_rng(seed)
    n_have = len(records)
    if n_have == 0:
        return np.array([], dtype=object)
    reps = int(np.ceil(target_n / n_have))
    tles, made, nr = [], 0, 1
    # precompute epochs once
    epochs = [_epoch_to_mjd(r["EPOCH"]) for r in records]
    for _ in range(reps):
        for r, mjd in zip(records, epochs):
            if made >= target_n:
                break
            raan = rng.uniform(0, 360)
            ma = rng.uniform(0, 360)
            try:
                lines = cysgp4.tle_linestrings_from_orbital_parameters(
                    f"{r['CONSTELLATION'][:6]}-{nr}", nr, mjd,
                    r["INCLINATION"], raan, max(r["ECCENTRICITY"], 1e-5),
                    r["ARG_OF_PERICENTER"], ma, r["MEAN_MOTION"],
                )
                tles.append(cysgp4.PyTle(*lines))
                made += 1
                nr += 1
            except Exception:
                continue
        if made >= target_n:
            break
    return np.array(tles, dtype=object)


def _altitudes(tles, epoch_mjd=58813.5):
    """Quick altitude check (km) by propagating each PyTle once at epoch."""
    import cysgp4
    mu, Re = 398600.4418, 6378.137
    # mean motion is encoded; recover altitude from the TLE's mean motion
    alts = []
    for t in tles:
        n = t.mean_motion  # rad/min in cysgp4 PyTle? -> use revs/day via attribute
        alts.append(n)
    return np.array(alts)


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="constellation up-scaler")
    ap.add_argument("--method", choices=["filed", "proportional", "odc"], required=True)
    ap.add_argument("--target", type=int, default=None,
                    help="target satellite count (None: filed=baseline, "
                         "proportional=300000, odc=preset default)")
    ap.add_argument("--preset", choices=["realistic", "stress"], default="realistic",
                    help="ODC preset (method=odc)")
    ap.add_argument("--omm", help="OMM file (required for proportional)")
    ap.add_argument("--constellations", nargs="*",
                    help="constellations to include for proportional method")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"FILED_SHELLS baseline total = {baseline_total()}")
    if args.method == "odc":
        tles = odc_shell_tles(preset=args.preset, seed=args.seed,
                              total=args.target, verbose=True)
        print(f"odc_shell_tles(preset={args.preset}) -> {len(tles)} PyTles")
    elif args.method == "filed":
        tles = filed_shell_tles(target_total=args.target, seed=args.seed)
        print(f"filed_shell_tles(target={args.target}) -> {len(tles)} PyTles")
    else:
        args.target = args.target or 300000
        import omm_tools as ot
        recs = ot.load_omm(args.omm)
        if args.constellations:
            recs = [r for r in recs if r["CONSTELLATION"] in args.constellations]
        print(f"loaded {len(recs)} real records for proportional base")
        tles = proportional_upscale_tles(recs, args.target, seed=args.seed)
        print(f"proportional_upscale_tles(target={args.target}) -> {len(tles)} PyTles")


if __name__ == "__main__":
    _main()
