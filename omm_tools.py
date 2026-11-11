#!/usr/bin/env python3
# coding: utf-8
"""
omm_tools.py — load CCSDS OMM (XML) snapshots, classify satellites by
constellation, and convert to cysgp4 PyTle objects for the EPFD calculator.

Snapshots are CCSDS OMM XML files (optionally bz2-compressed), as distributed
by Space-Track or CelesTrak.


CLI:
    python omm_tools.py --omm data/full_omm_2026-06-21_03.xml.bz2            # summary
    python omm_tools.py --omm <file> --constellation Starlink --tle out.tle
"""

import argparse
import bz2
import re
import sys
from xml.etree import ElementTree as ET

# Leaf OMM tags we need from each <omm> record.
_FIELDS = (
    "OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "EPOCH", "MEAN_MOTION",
    "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
)

# Constellation classification by OBJECT_NAME. Order matters (first match wins).
# Patterns are matched case-insensitively against the object name.
_CONSTELLATIONS = [
    ("Starlink",  re.compile(r"^STARLINK", re.I)),
    ("OneWeb",    re.compile(r"^ONEWEB", re.I)),
    ("Kuiper",    re.compile(r"KUIPER", re.I)),                # Amazon LEO
    ("GuoWang",   re.compile(r"HULIANWANG|GUOWANG|^GW-", re.I)),  # SNIFFLES "Hulianwang-Digui"
    ("Qianfan",   re.compile(r"QIANFAN|SPACESAIL|^G60", re.I)),   # Thousand Sails / Spacesail
    ("Iridium",   re.compile(r"^IRIDIUM", re.I)),
]


def classify(name):
    """Return the constellation label for an OBJECT_NAME, or 'Other'."""
    if not name:
        return "Other"
    for label, pat in _CONSTELLATIONS:
        if pat.search(name):
            return label
    return "Other"


def _strip_ns(tag):
    """Drop any XML namespace prefix from a tag."""
    return tag.rsplit("}", 1)[-1]


def _open(path):
    return bz2.open(path, "rb") if path.endswith(".bz2") else open(path, "rb")


def load_omm(path, constellation=None, name_regex=None):
    """
    Parse an OMM XML (optionally .bz2) snapshot.

    Parameters
    ----------
    path : str
        Path to the OMM XML(.bz2) file.
    constellation : str or None
        If given, keep only records whose classify() matches this label.
    name_regex : str or None
        If given, keep only records whose OBJECT_NAME matches this regex.

    Returns
    -------
    list of dict
        One dict per satellite with keys from _FIELDS plus 'CONSTELLATION'.
        Numeric orbital fields are returned as float / int where sensible.
    """
    name_pat = re.compile(name_regex, re.I) if name_regex else None
    records = []
    with _open(path) as fh:
        # iterparse over end events; collect leaf text within each <omm>, clear after.
        for event, elem in ET.iterparse(fh, events=("end",)):
            if _strip_ns(elem.tag) != "omm":
                continue
            rec = {}
            for leaf in elem.iter():
                t = _strip_ns(leaf.tag)
                if t in _FIELDS and leaf.text is not None:
                    rec[t] = leaf.text.strip()
            elem.clear()  # free memory — 54 MB files otherwise blow up RAM

            name = rec.get("OBJECT_NAME", "")
            con = classify(name)
            rec["CONSTELLATION"] = con
            if constellation and con != constellation:
                continue
            if name_pat and not name_pat.search(name):
                continue
            # cast numerics
            for k in ("MEAN_MOTION", "ECCENTRICITY", "INCLINATION",
                      "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY"):
                if k in rec:
                    rec[k] = float(rec[k])
            if "NORAD_CAT_ID" in rec:
                rec["NORAD_CAT_ID"] = int(rec["NORAD_CAT_ID"])
            records.append(rec)
    return records


def _epoch_to_mjd(epoch_iso):
    """Convert an OMM ISO-UTC EPOCH string to MJD (UTC)."""
    from astropy.time import Time
    return Time(epoch_iso, format="isot", scale="utc").mjd


def to_pytles(records):
    """
    Convert OMM records to cysgp4.PyTle objects.

    Drag (BSTAR) and mean-motion derivatives are omitted: over the RA.769 2000 s
    window they are negligible for visibility/PFD purposes.
    """
    import cysgp4
    tles = []
    for i, r in enumerate(records, start=1):
        try:
            mjd = _epoch_to_mjd(r["EPOCH"])
            lines = cysgp4.tle_linestrings_from_orbital_parameters(
                r.get("OBJECT_NAME", f"OMM-{i}")[:24],
                r.get("NORAD_CAT_ID", i),
                mjd,
                r["INCLINATION"], r["RA_OF_ASC_NODE"], r["ECCENTRICITY"],
                r["ARG_OF_PERICENTER"], r["MEAN_ANOMALY"], r["MEAN_MOTION"],
            )
            tles.append(cysgp4.PyTle(*lines))
        except Exception as e:  # skip malformed records, report at the end
            print(f"  warn: skipped {r.get('OBJECT_NAME','?')}: {e}", file=sys.stderr)
    return tles


def write_tle_file(records, path):
    """Write records as a 3-line TLE text file (name/line1/line2) for --tlefile."""
    import cysgp4
    n = 0
    with open(path, "w") as fh:
        for i, r in enumerate(records, start=1):
            try:
                mjd = _epoch_to_mjd(r["EPOCH"])
                l0, l1, l2 = cysgp4.tle_linestrings_from_orbital_parameters(
                    r.get("OBJECT_NAME", f"OMM-{i}")[:24],
                    r.get("NORAD_CAT_ID", i), mjd,
                    r["INCLINATION"], r["RA_OF_ASC_NODE"], r["ECCENTRICITY"],
                    r["ARG_OF_PERICENTER"], r["MEAN_ANOMALY"], r["MEAN_MOTION"],
                )
                fh.write(f"{l0}\n{l1}\n{l2}\n")
                n += 1
            except Exception as e:
                print(f"  warn: skipped {r.get('OBJECT_NAME','?')}: {e}", file=sys.stderr)
    return n


def summary(records):
    """Return a dict {constellation: count} plus 'TOTAL'."""
    counts = {}
    for r in records:
        counts[r["CONSTELLATION"]] = counts.get(r["CONSTELLATION"], 0) + 1
    counts["TOTAL"] = len(records)
    return counts


def _main():
    ap = argparse.ArgumentParser(description="OMM loader / classifier")
    ap.add_argument("--omm", required=True, help="OMM XML(.bz2) file")
    ap.add_argument("--constellation", help="filter to one constellation label")
    ap.add_argument("--name-regex", help="filter OBJECT_NAME by regex")
    ap.add_argument("--tle", help="write filtered records to this TLE file")
    ap.add_argument("--check-propagate", action="store_true",
                    help="build PyTles and report how many propagate cleanly")
    args = ap.parse_args()

    recs = load_omm(args.omm, constellation=args.constellation,
                    name_regex=args.name_regex)
    print(f"Loaded {len(recs)} records from {args.omm}")
    # full-catalogue summary regardless of filter (re-load unfiltered if filtered)
    if args.constellation or args.name_regex:
        allrecs = load_omm(args.omm)
    else:
        allrecs = recs
    print("Constellation breakdown (full catalogue):")
    for k, v in sorted(summary(allrecs).items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {v:7d}")

    if args.tle:
        n = write_tle_file(recs, args.tle)
        print(f"Wrote {n} TLEs to {args.tle}")
    if args.check_propagate:
        tles = to_pytles(recs[:2000])
        print(f"Built {len(tles)} PyTle objects from first {min(2000,len(recs))} records")


if __name__ == "__main__":
    _main()
