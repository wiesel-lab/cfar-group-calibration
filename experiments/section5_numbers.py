"""Reproduces every number quoted in the text of Section V (Experiments).

Each block prints the sentence of the paper and the values behind it.

Usage:
    python experiments/section5_numbers.py

Requires results/eval_seed*.csv (experiments/evaluate.py).
"""

from __future__ import annotations

import argparse

import numpy as np

import common as C

DATASETS = ["muufl", "pavia", "salinas", "indian"]


def cfg(ev, score, part, alpha=C.ALPHA, mc=0, datasets=None):
    d = ev[(ev.alpha == alpha) & (ev.min_part_cells == mc) & (ev.score == score) & (ev.partition == part)]
    if datasets is not None:
        d = d[d.dataset.isin(datasets)]
    return C.mean_over_configs(d)


def per_image(ev, score, part):
    return {ds: cfg(ev, score, part, datasets=[ds]) for ds in DATASETS}


def header(text):
    print("\n" + "=" * 78 + "\n" + text + "\n" + "-" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="folder with eval_seed*.csv (default: results/)")
    ev = C.load_results(ap.parse_args().results)

    header("Protocol: AMF/ACE window sweep (5x5 .. 31x31), the best-balanced window is reported")
    for kind in ("amf", "ace"):
        for w in C.WINDOWS:
            m = cfg(ev, f"{kind}_{w}", "none")
            print(f"  {kind.upper()} {w:2d}x{w:<2d}  P_D = {m.pd:.3f}   Delta = {m.delta:.3f}")

    header("Protocol: across plantings, P_D varies by < 0.01 and Delta by < 0.04 (Table 1 arms)")
    arms = [("glrt", "none"), ("amf_7", "none"), ("ace_7", "none"), ("glrt", "sp_16"), ("glrt", "sp_8"),
            ("glrt", "tile_6"), ("glrt", "tile_4"), ("glrt", "sp_4"), ("glrt", "oracle")]
    rng_pd, rng_dl = 0.0, 0.0
    for s, p in arms:
        d = ev[(ev.alpha == C.ALPHA) & (ev.min_part_cells == 0) & (ev.score == s) & (ev.partition == p)]
        per_seed = d.groupby("seed")[["pd", "delta"]].mean()
        rng_pd = max(rng_pd, per_seed.pd.max() - per_seed.pd.min())
        rng_dl = max(rng_dl, per_seed.delta.max() - per_seed.delta.min())
    print(f"  largest spread across the three plantings: P_D {rng_pd:.4f}, Delta {rng_dl:.4f}")

    header("The 7x7 ACE reduces this disparity by about one-third, with a detection loss of\n"
           "0.09-0.10 on MUUFL and Pavia and 0.07 on Indian Pines")
    g, a = per_image(ev, "glrt", "none"), per_image(ev, "ace_7", "none")
    for ds in DATASETS:
        print(f"  {ds:8s} P_D loss vs GLRT = {g[ds].pd - a[ds].pd:.3f}")
    ga, aa = cfg(ev, "glrt", "none"), cfg(ev, "ace_7", "none")
    print(f"  Delta: GLRT {ga.delta:.3f} -> ACE {aa.delta:.3f}  (reduction {1 - aa.delta / ga.delta:.1%})")

    header("For each of the three scores, calibration with P = 4 superpixels improves balance\n"
           "in all 141 configuration-planting pairs; calibration over the audited parts\n"
           "reduces mean detection by no more than 0.004 for any base score")
    for b in C.BASES:
        j = C.paired(ev, b, "sp_4", b, "none")
        o = C.paired(ev, b, "oracle", b, "none")
        print(f"  {b:6s} Delta improves in {(j.d_delta < 0).sum()}/{len(j)} pairs;"
              f"  mean P_D change with audited parts = {o.d_pd.mean():+.4f}")

    header("Every blind partition reduces Delta in every image; P = 4 superpixels are the best\n"
           "balanced overall (0.25 vs 0.76) with mean P_D within 0.035 of the GLRT, but lose up\n"
           "to 0.23 on one MUUFL target; tiles P = 6 cut the spread by more than half and never\n"
           "lose more than 0.04")
    blind = ["sp_16", "sp_8", "tile_6", "tile_4", "sp_4"]
    ok = all(per_image(ev, "glrt", p)[ds].delta < g[ds].delta for p in blind for ds in DATASETS)
    print(f"  every blind partition below the GLRT's Delta in every image: {ok}")
    for p in blind:
        m = cfg(ev, "glrt", p)
        j = C.paired(ev, "glrt", p, "glrt", "none")
        w = j.d_pd.idxmin()
        print(f"  {p:7s} Delta {m.delta:.3f} (GLRT {ga.delta:.3f}, ratio {m.delta / ga.delta:.2f})"
              f"  mean P_D change {m.pd - ga.pd:+.3f}  worst {j.d_pd.min():+.3f} at {w[0]} target {w[1]}")

    header("Splitting each audited part into pure groups keeps the within-group term at zero,\n"
           "while moving only 1% of the cells between parts creates more imbalance than an\n"
           "eight-fold refinement")
    sel = ev[(ev.alpha == C.ALPHA) & (ev.min_part_cells == 0) & (ev.score == "glrt")]
    for p in ["oracle"] + [f"split_{k}" for k in C.SPLITS] + [f"mix_{f * 100:g}" for f in C.MIX_FRACS]:
        m = C.mean_over_configs(sel[sel.partition == p], ("pd", "delta", "within", "n_groups"))
        print(f"  {p:9s} groups {m.n_groups:6.1f}  Delta {m.delta:.3f}  within {m.within:.3f}  P_D {m.pd:.3f}")

    header("Permuting the P = 4 superpixel groups removes 86-91% of the balance gain")
    j0 = C.paired(ev, "glrt", "sp_4", "glrt", "none")
    gain = -j0.d_delta.mean()
    lost = [(gain + C.paired(ev, "glrt", f"perm_sp_4_{r}", "glrt", "none").d_delta.mean()) / gain
            for r in range(C.PERM_SEEDS)]
    print("  fraction of the gain lost, per permutation: " + ", ".join(f"{v:.1%}" for v in lost))

    header("Grouping by the 128 mixture components gives Delta = 0.57 at P_D = 0.63")
    m = cfg(ev, "glrt", "modes_128")
    print(f"  Delta {m.delta:.3f}  P_D {m.pd:.3f}")

    header("At lower false-alarm levels on Pavia and Salinas, the smallest spacing that keeps\n"
           "detection within 0.02 of the global score goes from P = 6 (0.05) to P = 12 (0.02)\n"
           "and P = 16 (0.01); the groups stay far better balanced than the global score, but\n"
           "at alpha = 0.01 no better than the best ACE window")
    for alpha, mc in C.LOW_ALPHA.items():
        base = cfg(ev, "glrt", "none", alpha, mc, C.LOW_ALPHA_SETS)
        aces = {s: cfg(ev, s, "none", alpha, mc, C.LOW_ALPHA_SETS)
                for s in C.LOW_ALPHA_BASELINES if s.startswith("ace")}
        best = min(aces, key=lambda s: aces[s].delta)
        print(f"  alpha = {alpha}: GLRT P_D {base.pd:.3f}, Delta {base.delta:.3f};"
              f" best-balanced ACE {best}: Delta {aces[best].delta:.3f}")
        for fam in ("tile", "sp"):
            for p in C.LADDER:
                m = cfg(ev, "glrt", f"{fam}_{p}", alpha, mc, C.LOW_ALPHA_SETS)
                if m.pd >= base.pd - 0.02:
                    print(f"      {fam:4s}: smallest P = {p:2d}  (P_D {m.pd:.3f}, Delta {m.delta:.3f})")
                    break


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
