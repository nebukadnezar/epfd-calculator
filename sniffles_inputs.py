#!/usr/bin/env python3
# coding: utf-8
"""
sniffles_inputs.py — per-satellite emission/radiation inputs for the EPFD driver,
drawn from SNIFFLES-I (Indermuehle & Lourenço 2026, MNRAS) and follow-up ATCA
measurements.

The EDITABLE SOURCE OF TRUTH is `data/sniffles_inputs.csv`. Update a
value there and simply re-run the driver — nothing here needs changing. As new
data are analysed, edit the CSV (add rows, refine SFD/active fractions, etc.).

Levels are per-satellite EIRP spectral density in dB(W/Hz) (the quantity the EPFD
engine takes; 0 dBi Tx, per-Hz throughout). Two brackets per line:
  - FLOOR : SNIFFLES detection-floor EIRP density [dB(W/Hz)] (Table 2) -> lower bound
  - ATCA  : derived from the ATCA-calibrated apparent SFD [Jy] at slant range dist:
              EIRP_nu[dB(W/Hz)] = 10*log10(SFD_Jy) + 10*log10(1e-26*4*pi*dist_m^2)

Class terminology is strict: 'spurious' = unwanted EMISSION (ITU-R SM track),
'uemr' = unintended RADIATION (CISPR track), 'intended' = licensed downlink.

'spurious' follows RR No. 1.145 / ITU-R SM.329 and covers two subcategories:
  - harmonic emissions (SM.329 1.1.1): whole multiples of the CENTRE frequency,
    i.e. 5240/7860/10480 = 2/3/4 x 2620. Broadband (they inherit the modulated
    carrier's 10 MHz).
  - frequency conversion products (SM.329 1.1.4): the frequencies, "or whole
    multiples thereof, of any oscillations generated to produce the carrier",
    i.e. the local oscillator at 2595 and its 2nd multiple at 5190. Narrowband
    (they inherit the oscillator's ~5-10 kHz). SM.329 1.1.4 excludes harmonic
    emissions, so 5190 is NOT a harmonic emission: it is not a multiple of 2620.
The LO belongs to the transmitter chain, so 2595/5190 are spurious EMISSION,
not UEMR. Note class does not predict bandwidth: bw_hz is measured, and the
narrowband/broadband split cuts across 'spurious'.

Blank CSV cells -> None (e.g. lines with a detection but no calibrated ATCA flux
are FLOOR-only; blank active_fraction means "not quantified", driver defaults 1.0).
"""

import csv
import os
import numpy as np

import settings as cfg
CSV_PATH = cfg.INPUTS_CSV


def eirp_nu_dbw_hz(sfd_jy, dist_km):
    """EIRP spectral density [dB(W/Hz)] from apparent SFD (Jy) at slant range."""
    if sfd_jy is None or dist_km is None:
        return None
    return 10 * np.log10(sfd_jy) + 10 * np.log10(1e-26 * 4 * np.pi * (dist_km * 1e3) ** 2)


def _num(s):
    s = (s or "").strip()
    return None if s == "" else float(s)


def load_inputs(path=CSV_PATH):
    """Load the editable CSV and return a list of enriched input dicts."""
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            r = dict(
                freq_mhz=_num(raw["freq_mhz"]),
                vgos_subband=raw["vgos_subband"].strip(),
                geodetic_band=raw["geodetic_band"].strip(),
                eclass=raw["eclass"].strip(),
                source=raw["source"].strip(),
                eirp_floor=_num(raw["eirp_floor_dbw_hz"]),
                sfd_atca_jy=_num(raw["sfd_atca_jy"]),
                dist_km=_num(raw["dist_km"]),
                bw_hz=_num(raw["bw_hz"]),
                duty=_num(raw["duty"]),
                active_fraction=_num(raw["active_fraction"]),
                ref=raw["ref"].strip(),
                notes=raw["notes"].strip(),
            )
            r["eirp_atca"] = eirp_nu_dbw_hz(r["sfd_atca_jy"], r["dist_km"])
            rows.append(r)
    return rows


# convenience module-level load (re-import to refresh after editing the CSV)
ROWS = load_inputs()


def _fmt(v, f="{:.1f}"):
    return "  -  " if v is None else f.format(v)


def _main():
    rows = load_inputs()
    hdr = (f"{'freq[MHz]':>9} {'VGOS':>6} {'geod':>4} {'class':>8} {'source':>22} "
           f"{'EIRPfloor':>9} {'SFD[Jy]':>9} {'EIRPatca':>8} {'dist':>5} "
           f"{'BW[MHz]':>7} {'duty':>4} {'actfr':>5}")
    print(f"# SNIFFLES per-satellite inputs (EIRP spectral density in dB(W/Hz))")
    print(f"# source CSV: {CSV_PATH}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['freq_mhz']:9.0f} {r['vgos_subband']:>6} {r['geodetic_band']:>4} "
              f"{r['eclass']:>8} {r['source']:>22} {_fmt(r['eirp_floor']):>9} "
              f"{_fmt(r['sfd_atca_jy'],'{:.3g}'):>9} {_fmt(r['eirp_atca']):>8} "
              f"{_fmt(r['dist_km'],'{:.0f}'):>5} {r['bw_hz']/1e6:7.0f} "
              f"{_fmt(r['duty']):>4} {_fmt(r['active_fraction'],'{:.2f}'):>5}")


if __name__ == "__main__":
    _main()
