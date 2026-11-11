#!/usr/bin/env python3
# coding: utf-8
"""
settings.py - one place for everything that depends on the local environment.

All scripts take their input files, output directory and compute settings from
config.json in this directory (or from the file named by the EPFD_CONFIG
environment variable). Relative paths in the config are resolved against this
directory, so the scripts can be started from anywhere.

  python         interpreter used by ./run.sh (e.g. a virtual environment)
  omm_file       orbital-element snapshot, CCSDS OMM XML (optionally .bz2)
  inputs_csv     per-satellite emission/radiation inputs
  stations_file  observer stations (JSON)
  reference_dir  shipped reference results (read only)
  output_dir     everything the scripts write goes below this directory
  n_jobs         parallel workers, -1 = all cores
  prop_budget    max elements per propagation chunk (~56 bytes each)
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.environ.get("EPFD_CONFIG", os.path.join(HERE, "config.json"))

with open(CONFIG_FILE) as _fh:
    CFG = json.load(_fh)


def _path(key):
    p = os.path.expanduser(CFG[key])
    return p if os.path.isabs(p) else os.path.join(HERE, p)


OMM = _path("omm_file")
INPUTS_CSV = _path("inputs_csv")
STATIONS_FILE = _path("stations_file")
REF = _path("reference_dir")
OUT = _path("output_dir")
FIGS = os.path.join(OUT, "figs")
PFD = os.path.join(OUT, "pfd")
TABLES = os.path.join(OUT, "tables")
N_JOBS = int(CFG.get("n_jobs", -1))
PROP_BUDGET = float(CFG.get("prop_budget", 5e7))


def out(*parts):
    """Path below the output directory; its parent directory is created."""
    path = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def ref(name):
    """Path of a shipped reference-result file."""
    return os.path.join(REF, name)


def stations():
    """Observer stations: {key: dict(name, lon, lat, alt, diam, min_el, ...)}."""
    with open(STATIONS_FILE) as fh:
        return json.load(fh)
