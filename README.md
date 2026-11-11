# EPFD calculator: per-satellite emission and radiation limits

Monte-Carlo equivalent power flux density (EPFD) calculator for satellite
constellations. It computes the aggregate interference of a satellite fleet at a
radio telescope, compares it with the ITU-R RA.769 protection thresholds
(single-dish continuum and VLBI), and derives the per-satellite emission or
radiation limit that keeps a fleet of any size below the threshold.

This is the code behind

> B. Indermuehle, L. McCallum, E. van der Wateren, H. Hase, B. Winkel,
> L. Lourenço, F. Di Vruno, M. Lindqvist, G. Hellbourg and G. I. G. Józsa (2026),
> "Towards Genuine Coexistence: Per-Satellite Emission and Radiation Limits to
> Protect Radio Astronomy and Geodetic VLBI at 1-14 GHz from Satellite
> Constellations", PASA. [arXiv:2608.11659](https://arxiv.org/abs/2608.11659)

It builds on [pycraf](https://github.com/bwinkel/pycraf) and
[cysgp4](https://github.com/bwinkel/cysgp4).

## Quick start

Tested with Python 3.13, pycraf 2.1 and cysgp4 0.4. A laptop is sufficient.

```sh
python3 -m venv ~/venv
~/venv/bin/pip install -r requirements.txt
./run.sh run_example.py
```

`run.sh` starts the Python interpreter named in `config.json` (default
`~/venv/bin/python`). If your environment is already active, `python
run_example.py` does the same.

The example runs Hobart (12 m) at 2700, 5240 and 8599 MHz against the shipped
orbital snapshot (29,496 objects, epoch 2026-06-21). It takes about one minute
on an 18-core Apple M5 (64 s measured; 51 s of that is the 2700 MHz case with
8177 radiating satellites). Run time scales with the number of radiating
satellites and the number of iterations, and inversely with the core count.

It writes to `output/example/`:

| file | content |
|---|---|
| `example_limits.txt` | margins, per-satellite limits and run times as a table |
| `example_results.csv` | the same numbers, machine readable |
| `figs/*_skycoverage_avgpfd.png` | sky map of the aggregate power flux density |
| `figs/*_dataloss_continuum.png`, `*_dataloss_vlbi.png` | data loss per sky cell |
| `figs/*_cdf.png` | cumulative distribution of the aggregate with the thresholds |

Expected output (20 iterations, seed 1):

```
Margin of the present fleet to the RA.769 threshold (negative = exceeded):
   MHz  criterion radiators  margin dB  reference   diff  loss %
  2700  continuum      8177     -24.18     -24.25  +0.07  100.00
  2700       vlbi      8177      17.33      17.25  +0.08    0.00
  5240  continuum       558     -60.67     -61.03  +0.36  100.00
  5240       vlbi       558     -18.85     -19.21  +0.36   67.97
  8599  continuum       653      -6.35      -6.57  +0.22   11.87
  8599       vlbi       653      38.75      38.52  +0.23    0.00

Per-satellite limit E_lim in dB(uV/m) at 10 m, in min(emission bandwidth, 1 MHz),
for a total LEO fleet of N satellites:
   MHz  criterion     N=50000    N=100000    N=300000   N=1000000
  2700  continuum        34.9        31.9        27.1        21.9
  2700       vlbi        76.4        73.4        68.6        63.4
  5240  continuum        32.9        29.9        25.2        19.9
  5240       vlbi        74.8        71.8        67.0        61.8
  8599  continuum        50.9        47.9        43.2        37.9
  8599       vlbi        96.0        93.0        88.3        83.0
```

The `reference` column is the value behind the paper (200 iterations), read
from `data/auscope_results.csv`. Small differences in the last digit between
machines are normal. The figures show the RA.769 thresholds without the
emission-bandwidth correction; the tables include it.

## Adapting it to your environment

Everything that depends on the local setup is in `config.json`. Relative paths
are resolved against this directory. To keep several setups, point the
`EPFD_CONFIG` environment variable at another config file.

| key | meaning |
|---|---|
| `python` | interpreter used by `run.sh` |
| `omm_file` | orbital-element snapshot, CCSDS OMM XML, optionally `.bz2` |
| `inputs_csv` | per-satellite emission/radiation inputs |
| `stations_file` | observer stations |
| `reference_dir` | shipped reference results (read only) |
| `output_dir` | everything the scripts write goes below this directory |
| `n_jobs` | parallel workers, `-1` uses all cores |
| `prop_budget` | size of a propagation chunk in elements of about 56 bytes; the default `5e7` needs about 3 GB, lower it on small machines |
| `example` | station, frequencies, iterations and fleet sizes of `run_example.py` |

To run your own case:

- **Station**: add an entry to `stations.json` (longitude and latitude in deg,
  altitude in km, dish diameter in m, minimum elevation in deg, observing band
  `fmin`/`fmax` in MHz).
- **Emitter**: add a row to `data/sniffles_inputs.csv`. The columns that matter
  are `freq_mhz`, `source` (selects the satellites, see `SOURCE_REGEX` in
  `run_auscope.py`), `sfd_atca_jy` and `dist_km` (spectral flux density in Jy at
  that range, converted to an isotropic e.i.r.p. density), `bw_hz` (emission
  bandwidth) and `active_fraction`.
- **Orbits**: any OMM XML file from Space-Track or CelesTrak. Synthetic fleets
  that need no orbital data are in `constellation.py`.

The per-satellite limit does not depend on the level entered: the margin is
linear in it, so `limit = level + margin - 10 log10(N / N0)` is the same for any
input level. Only the reported exceedance of a measured satellite depends on it.

## Method in brief

1. Propagate the radiating satellites over many 2000 s windows (the RA.769
   integration time) with random start epochs.
2. For each window and each cell of an equal-area sky grid (ITU-R M.1583), sum
   the power received through the RA.1631 antenna pattern from all satellites
   above the horizon.
3. Take the 98th percentile of the aggregate over all cells and windows (the 2%
   data-loss criterion of RA.1513) and compare it with the RA.769 threshold.
   Outside the tabulated bands the threshold is interpolated in log frequency.
4. Scale to larger fleets analytically. The aggregate is linear in the number
   of radiators, so no further propagation is needed.
5. Convert the limiting e.i.r.p. density to a field strength at 10 m.

## Files

| file | purpose |
|---|---|
| `run_example.py` | start here |
| `epfd_core.py` | the engine: RA.769 thresholds, sky grid, aggregate EPFD, metrics |
| `omm_tools.py` | read OMM snapshots, classify constellations, build TLE objects |
| `constellation.py` | synthetic and up-scaled fleets, orbital data-centre shells |
| `sniffles_inputs.py` | loader for the per-satellite input table |
| `settings.py`, `config.json`, `stations.json` | configuration |

Reproducing the paper:

| file | purpose |
|---|---|
| `run_auscope.py` | all stations and frequencies; writes `output/auscope_results.csv` and the aggregates in `output/pfd/` |
| `recompute_with_bw.py` | emission-bandwidth correction and fleet scaling from the cached aggregates |
| `analyze_results.py` | limit tables and figures; by default from the shipped `data/auscope_results.csv` (a few seconds), or `--csv output/auscope_results.csv` for your own run |
| `run_odc.py` | orbital data-centre constellations |
| `epfd_angular_decomp.py` | contribution to the aggregate by angle from boresight |

The full grid (`run_auscope.py`, 9 stations, 20 frequencies) is far larger than
the example and has not been timed on a laptop. Restrict it with `--stations`
and `--lines`.

Validation:

| file | purpose |
|---|---|
| `test_chunking.py`, `test_odc.py` | quick regression tests, no orbital data needed |
| `validate_crosscheck.py` | synthetic 7500-satellite Walker case at 151.5 MHz against an independent pipeline |
| `validate_linearity.py`, `sqrtn_correction.py` | linear fleet scaling against true up-scaled runs |
| `validate_timestep.py`, `window_validation.py` | time step and integration window |
| `plot_validation.py`, `plot_window_validation.py`, `plot_slim_interp.py` | figures from the shipped validation results |

Low frequencies (150 MHz): `bassa_150mhz.py`, `lofar_pattern_compare.py`. Note
that the dish pattern under-estimates the far-sidelobe pickup of an aperture
array (about 4 dB in the aggregate at 150 MHz), and that CISPR uses a 120 kHz
reference bandwidth below 1 GHz.

## Data

- `data/full_omm_2026-06-21_03.xml.bz2`: the orbital-element snapshot used in
  the paper (source: Space-Track).
- `data/sniffles_inputs.csv`: per-satellite levels from the SNIFFLES-I survey.
- `data/auscope_results.csv`, `data/odc_results.csv` and the validation CSV
  files: the results behind the paper's tables and figures.

## Licence and citation

GNU General Public License v3, see `LICENSE`. The engine derives from the EPFD
calculator by B. Winkel, F. Di Vruno and G. Józsa.

If you use this code, please cite the paper above, and pycraf (Winkel & Jessner
2018) and cysgp4.
