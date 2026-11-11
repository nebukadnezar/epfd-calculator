#!/usr/bin/env python3
"""Validate odc_shell_tles: SSO inclinations, per-shell capacity, the 1M stress
plan (without building it), and an end-to-end build + SGP4 propagation of the
realistic preset (recovered altitude + RAAN clump width)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import cysgp4
import constellation as C

print("=== SSO inclination vs altitude ===")
for h in [600, 700, 800, 1000, 1500, 2000]:
    print(f"  {h:5.0f} km -> i = {C.sso_inclination(h):6.2f} deg")
assert abs(C.sso_inclination(600) - 97.79) < 0.1
assert abs(C.sso_inclination(800) - 98.60) < 0.1

print("\n=== clumped per-shell capacity (s=50 km, +/-15 deg) ===")
for h in [600, 800, 2000]:
    print(f"  {h:5.0f} km -> {C.shell_capacity(h):,.0f}")

print("\n=== STRESS plan (1,000,000 across 500-2000 km, plan only) ===")
alts, incs, cap, per = C.odc_shell_plan("stress")
print(f"  shells={len(alts)}  capacity_sum={cap.sum():,.0f}  "
      f"allocated={int(per.sum()):,}  per-shell N {int(per.min())}-{int(per.max())}")
assert int(per.sum()) == 1_000_000, int(per.sum())
assert (per <= np.floor(cap)).all(), "a shell exceeds its capacity!"

print("\n=== REALISTIC build + propagation check ===")
mjd = 60860.0
tl = C.odc_shell_tles("realistic", seed=1, epoch_mjd=mjd, verbose=True)
print(f"  built {len(tl):,} PyTles")

# recover altitude from SGP4 (geo lon/lat/alt) for a random sample
rng = np.random.default_rng(0)
idx = rng.choice(len(tl), 800, replace=False)
obs = cysgp4.PyObserver(0.0, 0.0, 0.0)
res = cysgp4.propagate_many(np.array([[mjd]]), tl[idx][None, None, :], obs,
                            do_eci_pos=False, do_eci_vel=False, do_geo=True,
                            do_topo=False, do_sat_azel=False, on_error="coerce_to_nan")
alt = res["geo"][0, 0, :, 2]
print(f"  recovered altitude: mean {np.nanmean(alt):.0f} km, "
      f"range {np.nanmin(alt):.0f}-{np.nanmax(alt):.0f} km "
      f"(expect 600-800 +/- ~21 km geodetic oblateness swing)")
# allow the +/-21 km equatorial-vs-polar geodetic offset (+ ~7 km from e=1e-4)
assert 575 < np.nanmin(alt) and np.nanmax(alt) < 832

# RAAN clump: parse RAAN from TLE line 2 (cols 18-25) for a clumped vs uniform shell
def raans_of(tles):
    out = []
    for t in tles:
        l2 = t.tle_strings()[2]
        out.append(float(l2[17:25]))
    return np.array(out)

clump = C.synthesize_shell(700, 98.0, 2000, mjd, np.random.default_rng(2),
                           name="CL", raan_center=90.0, raan_span=30.0)
uni = C.synthesize_shell(700, 98.0, 2000, mjd, np.random.default_rng(2), name="UN")
rc, ru = raans_of(clump), raans_of(uni)
print(f"  clumped RAAN: {rc.min():.1f}-{rc.max():.1f} deg (expect ~75-105)")
print(f"  uniform RAAN: {ru.min():.1f}-{ru.max():.1f} deg (expect ~0-360)")
assert rc.min() > 73 and rc.max() < 107, (rc.min(), rc.max())
assert ru.max() - ru.min() > 300

print("\nODC GENERATOR OK")
