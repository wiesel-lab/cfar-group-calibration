"""Reproduces Figure 1: detection vs balance for the three candidate scores.

For each score (GLRT, AMF 7x7, ACE 7x7): filled marker = uncalibrated; open
markers = group calibration over superpixels with spacing P from 128 down to
4; ring = calibration over the labeled audited parts. Means over the 141
configuration-planting pairs at alpha = 0.1.

Usage:
    python experiments/figure1.py

Requires results/eval_seed*.csv (experiments/evaluate.py).
Writes results/figure1.pdf and results/figure1.png.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402

SERIES = [("glrt", "GLRT, whole cube", "#2a78d6", "o"),
          ("amf_7", "AMF, 7x7", "#eb6834", "s"),
          ("ace_7", "ACE, 7x7", "#1baf7a", "^")]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": "#e2e2df", "grid.linewidth": 0.5,
                     "pdf.fonttype": 42})   # embedded TrueType fonts (no Type 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="folder with eval_seed*.csv (default: results/)")
    a = ap.parse_args()
    ev = C.load_results(a.results)

    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    for base, label, color, mk in SERIES:
        pts = []
        for p in sorted([p for p in C.LADDER if p >= 4], reverse=True):
            j = C.paired(ev, base, f"sp_{p}", base, "none")
            pts.append((p, j.delta_a.mean(), j.pd_a.mean()))
        uncal = (j.delta_b.mean(), j.pd_b.mean())
        o = C.paired(ev, base, "oracle", base, "none")
        _, dx, py = zip(*pts)
        ax.plot(dx, py, "-", color=color, lw=1.4, zorder=3)
        ax.plot(dx, py, mk, color=color, ms=3.6, mew=0.8, mfc="white", zorder=4)
        ax.plot([uncal[0]], [uncal[1]], mk, color=color, ms=6, mfc=color, mew=0.8,
                zorder=5, label=label)
        ax.plot([o.delta_a.mean()], [o.pd_a.mean()], mk, color=color, ms=6, mfc="none",
                mew=1.3, ls="none", zorder=5)
        if base == "glrt":
            for p, x, y in pts:
                if p in (4, 6, 8, 16, 128):
                    ax.annotate(f"{p}", (x, y), textcoords="offset points", xytext=(0, 4),
                                fontsize=6, color="#52514e", ha="center")
    ax.invert_xaxis()
    ax.set_xlabel(r"balance $\Delta$")
    ax.set_ylabel(r"detection $P_D$")
    ax.set_ylim(0.72, 0.92)
    ax.grid(True, zorder=0)
    ax.legend(loc="lower left", frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.3)
    C.RESULTS.mkdir(exist_ok=True)
    fig.savefig(C.RESULTS / "figure1.pdf")
    fig.savefig(C.RESULTS / "figure1.png", dpi=300)
    print("Saved results/figure1.pdf and results/figure1.png")


if __name__ == "__main__":
    main()
