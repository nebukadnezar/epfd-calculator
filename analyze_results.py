#!/usr/bin/env python3
# coding: utf-8
"""
analyze_results.py — turn the EPFD matrix output (auscope_results.csv) into the
paper's derived products:

  - output/figs/breach_vs_N_vlbi.png : VLBI data loss vs total-LEO count (per line)
  - output/figs/elim_vs_N.png        : per-satellite RA.769-protective field-strength
                                limit E_lim vs total-LEO count (per line)
  - output/tables/key_results.tex  : key data-loss table (inline)
  - output/tables/cispr_limits.tex : per-satellite emission/radiation limit + exceedance
  - output/tables/appendix_results.tex : full station x line x scenario x criterion grid
  - console summary + a few derived numbers

Per-satellite limit logic (workflow step 7). The aggregate spectral PFD is linear
in the number of radiators; at population scaled by G the 98th-pct aggregate is
pfd98(G) = pfd98_now + 10log10(G). The max tolerable per-satellite EIRP spectral
density that keeps pfd98 at the RA.769 limit (Slim_nu) for a target population is
  eirp_lim(N) = eirp_used + margin_now - 10log10(N/N0)
where margin_now = Slim_nu - pfd98_now (dB). This limit is INDEPENDENT of the
measured/detected per-satellite level (eirp_used cancels: eirp_lim = Slim - K -
10log10(N/N0), K = pfd98_now - eirp_used the aggregate transfer factor): it is the
RA.769-protective ceiling, NOT the SNIFFLES detection floor. The measured level
exceeds it by
  exceedance(N) = eirp_used - eirp_lim(N) = 10log10(N/N0) - margin_now.

Field strength (CISPR form) of a per-Hz EIRP density, in a reference bandwidth
B_ref at distance d_ref:
  EIRP_tot[dBW] = eirp[dB(W/Hz)] + 10log10(B_ref)
  E[dB(uV/m)]  = EIRP_tot + 10log10(Z0/(4 pi d_ref^2)) + 90 + 120 conversion
We compute E numerically. d_ref = 10 m (locked); B_ref = 1 MHz (CISPR RBW >1 GHz).
For the fixed B_ref/d_ref here E[dB(uV/m)] = eirp[dB(W/Hz)] + 174.8 (a constant
offset), so the ITU-R SM e.i.r.p.-density quantity is recoverable from E directly.
"""

import argparse
import csv
import math
import os
import numpy as np
import sniffles_inputs as si
import settings as cfg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = cfg.ref("auscope_results.csv")   # shipped reference results
TBL = cfg.TABLES
FIG = cfg.FIGS

CT = 11981.0          # current total modeled LEO (fallback; derived from CSV below)
Z0 = 376.730313668
D_REF = 10.0          # m  (CISPR bench distance, locked)
B_REF = 1e6           # Hz (CISPR resolution bandwidth above 1 GHz)

# constellation sizes (total LEO) at which the CISPR table reports E_lim
TABLE_SIZES = [50_000.0, 200_000.0, 300_000.0]
TABLE_EXCEED_N = 300_000.0     # size used for the measured-exceedance column

STATIONS = {"Hb": "Hobart", "Ke": "Katherine", "Yg": "Yarragadee"}

# Fixed per-line plot style (colour, marker): categorical hues assigned by hand
# so that curves which overlap in the figures (the 100%-pinned pair 2620/5190,
# the 5240/7860 risers, the near-zero 2656/8599/8640/10480 bundle) get maximally
# different colours; the marker is the secondary encoding for CVD/print.
SERIES_STYLE = {
    2595: ("#009E73", "o"), 2620: ("#000000", "*"), 2656: ("#56B4E9", "s"),
    2700: ("#CC79A7", "^"), 4995: ("#E69F00", "v"), 5190: ("#999933", "D"),
    5240: ("#D55E00", "P"), 7860: ("#0072B2", "X"), 8599: ("#882255", "<"),
    8640: ("#009988", ">"), 10480: ("#666666", "d"),
    # L-band set (left panel of the two-panel continuum figure; palette
    # validated 2026-07-29, worst CVD pair dE 11.0 with markers as the
    # secondary encoding)
    1320: ("#994455", "o"), 1350: ("#0055AA", "s"), 1440: ("#BB8800", "^"),
    1454: ("#117733", "D"), 1474: ("#7733AA", "v"), 1620: ("#CC5511", "P"),
    1680: ("#00A0B0", "X"), 1777: ("#CC2266", "d"),
}
_FALLBACK = ["#117733", "#332288", "#88CCEE", "#DDCC77", "#AA4499"]

def style_of(fr):
    """(colour, marker) for a line; deterministic fallback for unstyled freqs."""
    key = round(fr)
    if key in SERIES_STYLE:
        return SERIES_STYLE[key]
    return _FALLBACK[key % len(_FALLBACK)], "o"

CLASS_LABEL = {"spurious": "Spurious", "uemr": "UEMR", "intended": "Intended", "int": "Intended"}
def CL(ec):
    return CLASS_LABEL.get(ec, ec.upper())

def DAG(ec):
    """Marker for intended, licensed in-band emission.

    Within its authorised band an operator may transmit at its licensed power,
    so figures tabulated against an intended line (a per-satellite limit, an
    exceedance, a data loss) size the band-sharing problem and are never a
    compliance verdict. The captions define the dagger.
    """
    return "$^\\dagger$" if ec in ("intended", "int") else ""


def eirp_to_dbuv_m(eirp_dbw_hz, b_ref=B_REF, d_ref=D_REF):
    """Per-Hz EIRP density -> field strength dB(uV/m) over b_ref at d_ref."""
    eirp_tot_w = 10 ** (eirp_dbw_hz / 10.0) * b_ref          # W (over b_ref)
    pfd = eirp_tot_w / (4 * math.pi * d_ref ** 2)            # W/m^2
    e = math.sqrt(Z0 * pfd)                                  # V/m
    return 20 * math.log10(e) + 120.0                        # dB(uV/m)


def eirp_lim_at(eirp_used, margin_now, N, n0):
    """RA.769-protective per-satellite EIRP spectral density [dB(W/Hz)] at total
    LEO population N (reference population n0)."""
    return eirp_used + margin_now - 10 * math.log10(N / n0)


def load():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        for k in ("freq_mhz", "eirp_dbw_hz", "growth", "data_loss_pct",
                  "slim_db", "margin_db"):
            r[k] = float(r[k])
        r["n_emit"] = int(r["n_emit"])
    return rows


def idx(rows):
    d = {}
    for r in rows:
        d[(r["station"], r["freq_mhz"], r["bracket"], r["mode"], r["scenario"])] = r
    return d


def _label_to_target(label):
    """Scenario label -> total-LEO target N ('50k'->5e4, '2m'->2e6, 'now'->None)."""
    if label == "now":
        return None
    mult = {"k": 1e3, "m": 1e6}[label[-1].lower()]
    return float(label[:-1]) * mult


def scenario_sizes(rows):
    """Map scenario label -> total-LEO N. The labels encode the target N directly;
    the 'now' population (current total LEO) is recovered as target/growth from the
    largest-target scenario (where the 2-dp growth rounding is least significant)."""
    cur = None
    for r in rows:
        t = _label_to_target(r["scenario"])
        if t and r["growth"] > 0 and (cur is None or t > cur[0]):
            cur = (t, t / r["growth"])
    current_total = cur[1] if cur else CT
    s2n = {r["scenario"]: (current_total if _label_to_target(r["scenario"]) is None
                           else _label_to_target(r["scenario"])) for r in rows}
    return s2n, current_total


def main():
    global CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=FIG,
                    help="directory for the figure PNGs (default output/figs/)")
    ap.add_argument("--figs-only", action="store_true",
                    help="write the figures only, leave the LaTeX tables alone")
    ap.add_argument("--csv", default=CSV,
                    help="EPFD results CSV (default: the shipped reference "
                         "results; pass output/auscope_results.csv for your own run)")
    args = ap.parse_args()
    CSV = args.csv
    figdir = args.figdir
    os.makedirs(TBL, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    rows = load()
    D = idx(rows)
    lines = sorted({r["freq_mhz"] for r in rows})
    scen_N, N0 = scenario_sizes(rows)
    scen_order = sorted(scen_N, key=lambda s: scen_N[s])
    print(f"N0 (current total LEO) = {N0:.0f}")
    print("size ladder: " + ", ".join(f"{s}={scen_N[s]:.0f}" for s in scen_order))

    # CISPR field-strength reference bandwidth per line: the emission occupies
    # min(emission bandwidth, CISPR RBW). Above 1 GHz CISPR uses a 1 MHz RBW, so a
    # narrowband (kHz) line deposits all its power in one window and its field
    # strength must be computed over its own bandwidth, not the full 1 MHz.
    bw_of = {round(r["freq_mhz"]): min(r["bw_hz"], B_REF) for r in si.load_inputs()}
    def E_dbuv(eirp_dbw_hz, fr):
        return eirp_to_dbuv_m(eirp_dbw_hz, b_ref=bw_of.get(round(fr), B_REF))

    # ---- per-line worst-station summary (ATCA), parameterised by criterion ----
    def build_summary(mode):
        summ = {}
        for fr in lines:
            best = None
            for st in STATIONS:
                key = (st, fr, "atca", mode, "now")
                if key in D and (best is None or D[key]["data_loss_pct"] > best[1]):
                    best = (st, D[key]["data_loss_pct"])
            if best is None:
                continue
            st = best[0]
            now = D[(st, fr, "atca", mode, "now")]
            mnow = now["margin_db"]
            Nb = N0 * 10 ** (mnow / 10.0)         # size where pfd98 reaches Slim
            loss = {s: D[(st, fr, "atca", mode, s)]["data_loss_pct"]
                    for s in scen_order if (st, fr, "atca", mode, s) in D}
            summ[fr] = dict(st=st, eclass=now["eclass"], eirp_used=now["eirp_dbw_hz"],
                            margin_now=mnow, slim_db=now["slim_db"], Nb=Nb, loss=loss)
        return summ

    def build_cispr(summ):
        c = {}
        for fr in lines:
            if fr not in summ:
                continue
            s = summ[fr]
            E_lim = {n: E_dbuv(eirp_lim_at(s["eirp_used"], s["margin_now"], n, N0), fr)
                     for n in TABLE_SIZES}
            eirp_lim_exc = eirp_lim_at(s["eirp_used"], s["margin_now"], TABLE_EXCEED_N, N0)
            c[fr] = dict(E_meas=E_dbuv(s["eirp_used"], fr), E_lim=E_lim,
                         exceed=s["eirp_used"] - eirp_lim_exc,
                         eclass=s["eclass"], st=s["st"])
        return c

    summary = build_summary("vlbi")           # VLBI: key_results, breach/elim figs, cispr
    summary_c = build_summary("continuum")    # single-dish: continuum cispr table

    # L-band lines lie below the AuScope band (fmin 2 GHz) and have no VLBI
    # tier; for the continuum per-satellite TABLE (not the figures) extend the
    # summary with the worst-affected facility (most negative margin, either
    # bracket -- E_lim is independent of the input level, which cancels).
    FAC_STATIONS = ["ASK", "ATCA", "Mop", "Pks", "Ced", "Hb26"]
    summary_c_tbl = dict(summary_c)
    for fr in lines:
        if fr in summary_c_tbl:
            continue
        best = None
        for st in FAC_STATIONS:
            for br in ("atca", "floor"):
                k = (st, fr, br, "continuum", "now")
                if k in D and (best is None or D[k]["margin_db"] < best["margin_db"]):
                    best = D[k]
        if best:
            summary_c_tbl[fr] = dict(
                st=best["station"], eclass=best["eclass"],
                eirp_used=best["eirp_dbw_hz"], margin_now=best["margin_db"],
                slim_db=best["slim_db"],
                Nb=N0 * 10 ** (best["margin_db"] / 10.0), loss={})

    cispr = build_cispr(summary)
    cispr_c = build_cispr(summary_c_tbl)

    print("\n=== VLBI data loss (ATCA), worst of 3 stations, full ladder + breach N ===")
    for fr in lines:
        if fr not in summary:
            continue
        s = summary[fr]
        loss_str = " ".join(f"{sc}:{s['loss'][sc]:.0f}" for sc in scen_order if sc in s["loss"])
        print(f"{fr:7.0f} {CL(s['eclass']):>8} Nb={s['Nb']:8.0f} {s['st']:>3}  {loss_str}")
    print("\n=== Per-satellite limit (dB uV/m): VLBI exc@300k | continuum exc@300k ===")
    for fr in lines:
        if fr in cispr:
            print(f"{fr:7.0f} {CL(cispr[fr]['eclass']):>8}  VLBI {cispr[fr]['exceed']:+6.1f}"
                  f"   cont {cispr_c.get(fr, {}).get('exceed', float('nan')):+6.1f}")

    # provisional marker: all result values red except the calibrated lines
    # a line is a solid (black) measurement iff it has an ATCA flux in the CSV;
    # floor-only lines (no ATCA value) remain provisional (red placeholders).
    CALIBRATED = {round(r["freq_mhz"]) for r in si.load_inputs()
                  if r["sfd_atca_jy"] is not None}
    def is_cal(fr):
        return round(fr) in CALIBRATED
    def R(x, fr):
        return x if is_cal(fr) else f"\\textcolor{{red}}{{{x}}}"

    # ---- breach-vs-N figure (VLBI, ATCA, worst station per line, full ladder) ----
    # Single-column figure in the paper (fig:breach). Text 17 pt on the 9 in
    # canvas prints at the same size as 13.5 pt on a 7.2 in canvas.
    with plt.rc_context({"font.size": 17, "axes.labelsize": 17,
                         "xtick.labelsize": 17, "ytick.labelsize": 17,
                         "legend.fontsize": 17}):
        fig, ax = plt.subplots(figsize=(9, 7.2))
        for i, fr in enumerate(fr for fr in lines if fr in summary):
            s = summary[fr]
            Ns = [scen_N[sc] for sc in scen_order if sc in s["loss"]]
            dl = [s["loss"][sc] for sc in scen_order if sc in s["loss"]]
            col, mk = style_of(fr)
            ls = "-" if is_cal(fr) else "--"
            # staggered markevery: coincident curves (100%-pinned, near-zero bundle)
            # mark alternating scenario points, so every series stays identifiable
            ax.plot(Ns, dl, ls, marker=mk, color=col, lw=2.4,
                    ms=18 if mk == "*" else 13, mec="white", mew=1.0,
                    markevery=(i % 3, 3),
                    label=f"{fr:.0f} MHz ({CL(s['eclass'])}, {s['st']})")
        ax.axvline(N0, color="0.5", ls=":", lw=1.4)
        ax.text(N0 * 1.06, 3, "today", rotation=90, color="0.4")
        ax.set_xscale("log")
        ax.set_xlabel("total NGSO satellites")
        ax.set_ylabel("VLBI data loss (%)")
        ax.grid(color="0.9", which="both")
        ax.set_ylim(-2, 103)
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
                   frameon=False, columnspacing=1.5, handlelength=2.5)
        p = os.path.join(figdir, "breach_vs_N_vlbi.png")
        fig.savefig(p, bbox_inches="tight", dpi=300)
        plt.close(fig)
    print(f"\nwrote {p}")

    # ---- E_lim-vs-N figures: RA.769-protective per-sat field strength vs size,
    #      one for the VLBI criterion and one for the single-dish continuum ----
    Nfine = np.logspace(math.log10(N0), math.log10(2_000_000.0), 200)

    def plot_elim(summ, path, crit_label):
        # Single-column figure in the paper (fig:elim): 17 pt on the 9 in
        # canvas prints like 13.5 pt on a 7.2 in canvas; no title (the caption
        # explains the dotted measured levels); legend below the axes.
        with plt.rc_context({"font.size": 17, "axes.labelsize": 17,
                             "xtick.labelsize": 17, "ytick.labelsize": 17,
                             "legend.fontsize": 17}):
            fig, ax = plt.subplots(figsize=(9, 7.2))
            for i, fr in enumerate(lines):
                if fr not in summ:
                    continue
                s = summ[fr]
                col, mk = style_of(fr)
                E_N = [E_dbuv(eirp_lim_at(s["eirp_used"], s["margin_now"], n, N0), fr)
                       for n in Nfine]
                ls = "-" if is_cal(fr) else "--"
                ax.plot(Nfine, E_N, ls, marker=mk, color=col, lw=2.2,
                        ms=15 if mk == "*" else 11, mec="white", mew=0.9,
                        markevery=(4 * i + 8, 40),
                        label=f"{fr:.0f} MHz ({CL(s['eclass'])}, {s['st']})")
                # measured per-satellite level (N-independent): dotted horizontal marker
                E_meas = E_dbuv(s["eirp_used"], fr)
                ax.plot([N0, 2_000_000.0], [E_meas, E_meas], ":", color=col, lw=1.2, alpha=0.8)
            ax.axvline(N0, color="0.5", ls=":", lw=1.4)
            ax.text(N0 * 1.06, ax.get_ylim()[0] + 2, "today", rotation=90, color="0.4")
            ax.set_xscale("log")
            ax.set_xlabel("total NGSO satellites")
            ax.set_ylabel(r"per-satellite limit $E_{\rm lim}$ (dB($\mu$V/m))")
            ax.grid(color="0.9", which="both")
            # ITU-R twin axis: e.i.r.p. producing the same field at 10 m is a
            # constant offset, EIRP[dBW] = E[dB(uV/m)] - 114.8, whatever the
            # per-line reference bandwidth (both axes express the same quantity).
            E2EIRP = 10 * math.log10(Z0 / (4 * math.pi * D_REF**2)) + 120.0  # 114.77
            ax2 = ax.secondary_yaxis(
                "right", functions=(lambda e: e - E2EIRP, lambda p: p + E2EIRP))
            ax2.set_ylabel("equivalent e.i.r.p. (dBW)")
            fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
                       frameon=False, columnspacing=1.5, handlelength=2.5)
            fig.savefig(path, bbox_inches="tight", dpi=300)
            plt.close(fig)
        print(f"wrote {path}")

    plot_elim(summary, os.path.join(figdir, "elim_vs_N.png"), "VLBI")

    def plot_elim_cont_2panel(summ, path):
        """Two-panel continuum E_lim figure: left = L-band lines (worst
        facility; no VLBI tier exists there), right = the >=2.5 GHz lines
        (worst AuScope station). Two-column figure in the paper (fig:elim_cont):
        11 pt text, no title (the caption explains solid / dashed / dotted),
        legends below the panels, 300 dpi."""
        with plt.rc_context({"font.size": 11, "axes.labelsize": 11,
                             "xtick.labelsize": 11, "ytick.labelsize": 11,
                             "legend.fontsize": 11}):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            panels = [(axes[0], [fr for fr in sorted(summ) if fr < 2000], 2),
                      (axes[1], [fr for fr in sorted(summ) if fr >= 2000], 3)]
            for ax, frs, ncol in panels:
                for i, fr in enumerate(frs):
                    s = summ[fr]
                    col, mk = style_of(fr)
                    E_N = [E_dbuv(eirp_lim_at(s["eirp_used"], s["margin_now"], n, N0), fr)
                           for n in Nfine]
                    # floor-only detections: long dash + OPEN markers (fill survives
                    # curve overlap where a dash pattern alone does not) + legend tag
                    if is_cal(fr):
                        ax.plot(Nfine, E_N, "-", marker=mk, color=col, lw=1.6,
                                ms=8, mec="white", mew=0.7, markevery=(4 * i + 8, 40),
                                label=f"{fr:.0f} MHz ({s['st']})")
                    else:
                        ax.plot(Nfine, E_N, ls=(0, (5, 2.5)), marker=mk, color=col,
                                lw=1.8, ms=9, mfc="none", mec=col, mew=1.4,
                                markevery=(4 * i + 8, 40),
                                label=f"{fr:.0f} MHz ({s['st']}, floor)")
                    E_meas = E_dbuv(s["eirp_used"], fr)
                    ax.plot([N0, 2_000_000.0], [E_meas, E_meas], ":", color=col,
                            lw=1.0, alpha=0.8)
                ax.axvline(N0, color="0.5", ls=":", lw=1.0)
                ax.set_xscale("log")
                ax.set_xlabel("total NGSO satellites")
                ax.grid(alpha=0.25, which="both")
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=ncol,
                          frameon=False, columnspacing=1.5)
            axes[0].set_ylabel(r"per-satellite limit $E_{\rm lim}$ (dB($\mu$V/m))")
            axes[0].text(N0 * 1.06, axes[0].get_ylim()[0] + 2, "today",
                         rotation=90, color="0.4")
            # ITU-R twin axis on the right panel (constant offset, see plot_elim)
            _e2 = 10 * math.log10(Z0 / (4 * math.pi * D_REF**2)) + 120.0
            ax2 = axes[1].secondary_yaxis(
                "right", functions=(lambda e: e - _e2, lambda p: p + _e2))
            ax2.set_ylabel("equivalent e.i.r.p. (dBW)")
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        print(f"wrote {path}")

    plot_elim_cont_2panel(summary_c_tbl, os.path.join(figdir, "elim_vs_N_cont.png"))

    if args.figs_only:
        print("figures only: LaTeX tables not rewritten")
        return

    # ---- LaTeX: key results (inline) ----
    with open(os.path.join(TBL, "key_results.tex"), "w") as fh:
        fh.write("% auto-generated by analyze_results.py\n")
        fh.write("\\begin{tabular}{@{\\extracolsep{\\fill}}rlrrrr}\n\\toprule\n")
        fh.write("Freq & Class & \\multicolumn{3}{c}{VLBI data loss (\\%)} & $N_{\\rm breach}$\\\\\n")
        fh.write("(MHz) & & now & 100k & 300k & (sats)\\\\\n\\hline\n")
        for fr in lines:
            if fr not in summary:
                continue
            s = summary[fr]
            now = f"{s['loss'].get('now', float('nan')):.1f}"
            k100 = f"{s['loss'].get('100k', float('nan')):.1f}"
            k300 = f"{s['loss'].get('300k', float('nan')):.1f}"
            # N_breach to ~3 significant figures (reviewer request): grouped
            # integer below 1e5, scientific \num form above
            if s["Nb"] < 1e5:
                nb = "\\num{%.0f}" % float("%.3g" % s["Nb"])
            else:
                _e = int(math.floor(math.log10(s["Nb"])))
                nb = "\\num{%.1fe%d}" % (s["Nb"] / 10**_e, _e)
            fh.write(f"{fr:.0f}{DAG(s['eclass'])} & {CL(s['eclass'])} & "
                     f"{R(now, fr)} & {R(k100, fr)} & {R(k300, fr)} & {R(nb, fr)}\\\\\n")
        fh.write("\\botrule\n\\end{tabular}\n")

    # ---- LaTeX: per-satellite limit tables (VLBI and single-dish continuum) ----
    def write_cispr(cisprd, path):
        with open(os.path.join(TBL, path), "w") as fh:
            fh.write("% auto-generated by analyze_results.py\n")
            fh.write("\\begin{tabular}{@{\\extracolsep{\\fill}}rl"
                     + "r" * (1 + len(TABLE_SIZES) + 1) + "}\n\\toprule\n")
            fh.write("Freq & Class & $E_{\\rm meas}$ & "
                     + "".join(f"$E_{{\\rm lim}}^{{{int(n/1000)}\\rm k}}$ & " for n in TABLE_SIZES)
                     + "Exceed\\\\\n")
            fh.write("(MHz) & & \\multicolumn{" + str(1 + len(TABLE_SIZES))
                     + "}{c}{dB($\\mu$V/m)} & (dB)\\\\\n\\hline\n")
            for fr in lines:
                if fr not in cisprd:
                    continue
                c = cisprd[fr]
                cells = ([f"{c['E_meas']:.1f}"]
                         + [f"{c['E_lim'][n]:.1f}" for n in TABLE_SIZES]
                         + [f"{c['exceed']:+.1f}"])
                fh.write(f"{fr:.0f}{DAG(c['eclass'])} & {CL(c['eclass'])} & "
                         + " & ".join(R(x, fr) for x in cells) + "\\\\\n")
            fh.write("\\botrule\n\\end{tabular}\n")
    write_cispr(cispr, "cispr_limits.tex")           # VLBI-protective
    write_cispr(cispr_c, "cispr_limits_cont.tex")    # single-dish (continuum)-protective

    # ---- LaTeX: full appendix grid, pivoted (C=continuum, V=vlbi) ----
    # Data loss saturates (0/100%) and hides the depth of exceedance, so the grid
    # also carries the present-fleet 98th-percentile margin per criterion,
    # M = Slim - pfd98 (the M0 of the per-satellite-limit derivation): negative
    # values give the dB of aggregate attenuation required to meet the criterion.
    # Margins at the scaled scenarios follow analytically (M - 10log10 G), so only
    # the present-fleet value is tabulated.
    def dl(st, fr, br, md, sc):
        r = D.get((st, fr, br, md, sc))
        return f"{r['data_loss_pct']:.1f}" if r else "---"
    def mg(st, fr, br, md):
        r = D.get((st, fr, br, md, "now"))
        if not r:
            return "---"
        # uncalibrated (floor-only) lines: margins are upper bounds.
        # typography: math minus, no explicit + on positives
        v = r["margin_db"]
        txt = (f"$-${abs(v):.1f}" if v < 0 else f"{v:.1f}")
        return ("" if is_cal(fr) else "$<$") + txt
    combos = sorted({(r["station"], r["freq_mhz"], r["bracket"]) for r in rows},
                    key=lambda c: (c[0], c[1], c[2]))
    # display names for the facility station codes (AuScope Hb/Ke/Yg stay as
    # defined in the caption; Hb26 is the Hobart 26-m single dish)
    STN_DISPLAY = {"ASK": "ASKAP", "Mop": "Mopra", "Pks": "Murriyang",
                   "Ced": "Ceduna"}
    hdr = ("Stn & Freq & Brkt & $C_{\\rm now}$ & $V_{\\rm now}$ & "
           "$C_{100k}$ & $V_{100k}$ & $C_{300k}$ & $V_{300k}$ & "
           "$M^{C}$ & $M^{V}$\\\\")
    with open(os.path.join(TBL, "appendix_results.tex"), "w") as fh:
        fh.write("% auto-generated by analyze_results.py\n")
        fh.write("{\\footnotesize\n\\begin{longtable}{@{}lll rrrrrr rr@{}}\n")
        fh.write("\\caption{Complete EPFD grid for every station, line and level "
                 "bracket: data loss (\\%) under the single-dish continuum ($C$) "
                 "and VLBI ($V$) criteria at the present fleet, \\num{100000} and "
                 "\\num{300000} satellites, and the present-fleet 98th-percentile "
                 "margin per criterion, $M = S_{\\rm lim}-\\mathrm{pfd}_{98}$ "
                 "(dB). Negative $M$ gives directly the aggregate attenuation "
                 "required to meet the criterion; margins at \\num{100000} and "
                 "\\num{300000} satellites follow by subtracting \\SI{9.2}{} and "
                 "\\SI{14.0}{dB}. Floor-bracket rows bound the level from "
                 "below: their losses are lower bounds and their margins upper "
                 "bounds; for the floor-only (red) lines the margin entries "
                 "carry an explicit $<$. The Monte-Carlo sampling uncertainty on the margins "
                 "(resampling standard error of the pooled 98th percentile, "
                 "resampled over iterations) has median \\SI{0.1}{dB} and lies "
                 "below \\SI{0.3}{dB} for \\SI{90}{\\percent} of entries, "
                 "reaching \\SI{1.4}{dB} only for the sparsest radiating "
                 "population (the 21-satellite GuoWang line at \\SI{1777}{MHz}); "
                 "the uncertainty budget is instead dominated by the input "
                 "levels, bracketed by the floor and ATCA rows and by the "
                 "compression-limited $\\geq$ bounds. Freq in MHz.}"
                 "\\label{tab:appendix}\\\\\n")
        fh.write("\\toprule\n" + hdr + "\n\\hline\n\\endfirsthead\n")
        fh.write("\\toprule\n" + hdr + "\n\\hline\n\\endhead\n")
        for st, fr, br in combos:
            cells = ([dl(st, fr, br, m, sc) for sc in ("now", "100k", "300k")
                      for m in ("continuum", "vlbi")]
                     + [mg(st, fr, br, m) for m in ("continuum", "vlbi")])
            cells = [R(c, fr) for c in cells]
            fh.write(f"{STN_DISPLAY.get(st, st)} & {fr:.0f} & {br} & "
                     + " & ".join(cells) + "\\\\\n")
        fh.write("\\botrule\n\\end{longtable}}\n")

    # ---- LaTeX: facility-by-facility single-dish continuum data loss (present fleet) ----
    # one EPFD propagation per facility caches the aggregate geometry; the loss here
    # uses the measured (ATCA) level where available, else the detection floor (a
    # lower bound). Populated once the facility caches are folded in (recompute_with_bw).
    FAC = [("ATCA", "ATCA", 22), ("Mop", "Mopra", 22), ("Pks", "Murriyang", 64),
           ("Ced", "Ceduna", 30), ("Hb26", "Hobart", 26), ("ASK", "ASKAP", 12)]
    present_fac = [(k, n, d) for k, n, d in FAC if any(r["station"] == k for r in rows)]
    eclass_of = {round(r["freq_mhz"]): r["eclass"] for r in rows}
    def cont_now(stn, fr):
        for br in ("atca", "floor"):
            r = D.get((stn, fr, br, "continuum", "now"))
            if r:
                return r["data_loss_pct"]
        return None
    with open(os.path.join(TBL, "facility_continuum.tex"), "w") as fh:
        fh.write("% auto-generated by analyze_results.py\n")
        if not present_fac:
            fh.write("\\emph{(facility EPFD run pending)}\n")
        else:
            fh.write("\\begin{tabular}{@{\\extracolsep{\\fill}}rl"
                     + "r" * len(present_fac) + "}\n\\toprule\n")
            fh.write("Freq & Class & " + " & ".join(n for _, n, _ in present_fac) + "\\\\\n")
            fh.write("(MHz) & & " + " & ".join(f"{d}\\,m" for _, _, d in present_fac)
                     + "\\\\\n\\hline\n")
            for fr in lines:
                cells, any_val = [], False
                for k, _, _ in present_fac:
                    v = cont_now(k, fr)
                    if v is None:
                        cells.append("---")
                    else:
                        cells.append(R(f"{v:.0f}", fr)); any_val = True
                if any_val:
                    ec = eclass_of.get(round(fr), "")
                    fh.write(f"{fr:.0f}{DAG(ec)} & {CL(ec)} & "
                             + " & ".join(cells) + "\\\\\n")
            fh.write("\\botrule\n\\end{tabular}\n")

    # ---- LaTeX: facility-by-facility continuum 98th-percentile MARGIN table ----
    # Companion to facility_continuum.tex: same grid, but the present-fleet margin
    # M = Slim - pfd98 (dB) instead of the data loss. Data loss saturates at the
    # bright lines (everything reads 100%); the margin stays linear in per-satellite
    # level and N, and its negative reads directly as the aggregate attenuation
    # required to restore RA.769 compliance.
    def cont_margin_now(stn, fr):
        for br in ("atca", "floor"):
            r = D.get((stn, fr, br, "continuum", "now"))
            if r:
                return r["margin_db"]
        return None
    with open(os.path.join(TBL, "facility_continuum_margin.tex"), "w") as fh:
        fh.write("% auto-generated by analyze_results.py\n")
        if not present_fac:
            fh.write("\\emph{(facility EPFD run pending)}\n")
        else:
            fh.write("\\begin{tabular}{@{\\extracolsep{\\fill}}rl"
                     + "r" * len(present_fac) + "}\n\\toprule\n")
            fh.write("Freq & Class & " + " & ".join(n for _, n, _ in present_fac) + "\\\\\n")
            fh.write("(MHz) & & " + " & ".join(f"{d}\\,m" for _, _, d in present_fac)
                     + "\\\\\n\\hline\n")
            for fr in lines:
                cells, any_val = [], False
                for k, _, _ in present_fac:
                    v = cont_margin_now(k, fr)
                    if v is None:
                        cells.append("---")
                    else:
                        # floor-only (uncalibrated) lines: the margin is an
                        # upper bound on the true margin -> "<" prefix.
                        # typography: math minus, no explicit + on positives
                        txt = (f"$-${abs(v):.1f}" if v < 0 else f"{v:.1f}")
                        if not is_cal(fr):
                            txt = f"$<$" + txt
                        cells.append(R(txt, fr)); any_val = True
                if any_val:
                    ec = eclass_of.get(round(fr), "")
                    fh.write(f"{fr:.0f}{DAG(ec)} & {CL(ec)} & "
                             + " & ".join(cells) + "\\\\\n")
            fh.write("\\botrule\n\\end{tabular}\n")

    print(f"wrote LaTeX tables to {TBL}/")


if __name__ == "__main__":
    main()
