"""Reproduces Table 1: detection and balance per image at alpha = 0.1.

Rows: the three candidate scores; the GLRT calibrated over blind partitions;
the GLRT calibrated over the labeled audited parts. Per image, values are the
mean over target configurations and plantings; "all" is the mean over all 47
configurations. "worst P_D loss" is the largest drop in P_D from the
uncalibrated GLRT on a single configuration and planting (out of 141).

Usage:
    python experiments/table1.py

Requires results/eval_seed*.csv (experiments/evaluate.py).
Writes results/table1.csv and results/table1.tex.
"""

from __future__ import annotations

import argparse

import pandas as pd

import common as C

DATASETS = [("muufl", "MUUFL"), ("pavia", "Pavia"), ("salinas", "Salinas"), ("indian", "Indian Pines")]
ROWS = [
    ("GLRT in $A$, whole cube", "glrt", "none"),
    ("AMF, $7\\times7$ window", "amf_7", "none"),
    ("ACE, $7\\times7$ window", "ace_7", "none"),
    ("superpixels, $P=16$", "glrt", "sp_16"),
    ("superpixels, $P=8$", "glrt", "sp_8"),
    ("tiles, $P=6$", "glrt", "tile_6"),
    ("tiles, $P=4$", "glrt", "tile_4"),
    ("superpixels, $P=4$", "glrt", "sp_4"),
    ("audited parts (labeled)", "glrt", "oracle"),
]
BLIND = {"sp_16", "sp_8", "tile_6", "tile_4", "sp_4"}


def build(ev):
    sel = ev[(ev.alpha == C.ALPHA) & (ev.min_part_cells == 0)]
    out = []
    for label, score, part in ROWS:
        d = sel[(sel.score == score) & (sel.partition == part)]
        row = {"row": label, "score": score, "partition": part}
        for ds, _ in DATASETS:
            m = C.mean_over_configs(d[d.dataset == ds])
            row[f"{ds}_pd"], row[f"{ds}_delta"] = m.pd, m.delta
        m = C.mean_over_configs(d)
        row["all_pd"], row["all_delta"] = m.pd, m.delta
        if part != "none":
            j = C.paired(ev, score, part, "glrt", "none")
            row["worst_pd_loss"] = -j.d_pd.min()
            row["n_pairs"] = len(j)
        out.append(row)
    return pd.DataFrame(out)


def to_latex(t):
    cols = [f"{ds}_delta" for ds, _ in DATASETS] + ["all_delta"]
    best = {c: t[t.partition.isin(BLIND)][c].min() for c in cols}
    lines = []
    for i, r in t.iterrows():
        cells = []
        for ds in [d for d, _ in DATASETS] + ["all"]:
            dl = f"{r[f'{ds}_delta']:.2f}"
            if r.partition in BLIND and round(r[f"{ds}_delta"], 2) == round(best[f"{ds}_delta"], 2):
                dl = f"\\textbf{{{dl}}}"
            cells += [f"{r[f'{ds}_pd']:.3f}", dl]
        worst = "--" if r.partition == "none" else f"{r.worst_pd_loss:.2f}"
        lines.append(f"{r.row} & " + " & ".join(cells) + f" & {worst} \\\\")
        if i in (2, 7):
            lines.append("\\midrule")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="folder with eval_seed*.csv (default: results/)")
    a = ap.parse_args()
    t = build(C.load_results(a.results))
    C.RESULTS.mkdir(exist_ok=True)
    t.to_csv(C.RESULTS / "table1.csv", index=False)
    (C.RESULTS / "table1.tex").write_text(to_latex(t) + "\n")
    show = t[["row"] + [c for c in t.columns if c.endswith(("_pd", "_delta"))] + ["worst_pd_loss"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSaved results/table1.csv and results/table1.tex")


if __name__ == "__main__":
    main()
