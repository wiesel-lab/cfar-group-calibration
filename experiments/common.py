"""Settings and helpers shared by the experiment scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import calibrate  # noqa: E402

CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
SEEDS = (0, 1, 2)                       # three independent target plantings

# Section V-A
PCA_DIM = 12
N_COMPONENTS = 128                      # Gaussian-mixture background density
PLANT_FRAC = 0.01
AMPLITUDE_SIGMAS = 2.0
AMAX_FACTOR = 2.0                       # A_max = 2 x planted amplitude
GLRT_GRID = 65                          # amplitudes evaluated for the GLRT
WINDOWS = [5, 7, 9, 11, 15, 21, 31]     # AMF/ACE window sweep
GUARD = 3
LADDER = [2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128]   # tile / superpixel spacing P
SPLITS = [2, 4, 8, 16]                  # pure refinements of the audited parts
MIX_FRACS = [0.005, 0.01, 0.02, 0.05, 0.10]
PERM_SEEDS = 5

ALPHA = 0.10
BASES = ["glrt", "amf_7", "ace_7"]      # the three candidate scores (7x7 windows)
# Lower false-alarm levels, Pavia and Salinas only: alpha -> smallest audited part
LOW_ALPHA = {0.05: 0, 0.02: 1000, 0.01: 2000}
LOW_ALPHA_SETS = ("pavia", "salinas")
LOW_ALPHA_BASELINES = ["glrt", "amf_7", "amf_9", "ace_7", "ace_9", "ace_11", "ace_15"]


def scene_file(seed: int, dataset: str, target: int) -> Path:
    return CACHE / f"seed{seed}" / f"{dataset}_t{target:02d}.npz"


def evaluate(score, groups, sc, alpha, min_part_cells=0):
    """One reading at a pooled background false-alarm rate of exactly alpha.

    `groups=None` is the uncalibrated score (one global threshold). Returns
    P_D, the balance Delta = max_c r_c - min_c r_c over the audited parts, and
    the between/within spreads of the Section IV identity.
    """
    tie, planted, parts = sc["tie"], sc["planted"], sc["parts"]
    g = np.zeros(len(score), dtype=np.int64) if groups is None else groups
    rank, _, counts = calibrate.group_rank(score, g, tie)
    fire = calibrate.fire_at_pooled_rate(rank, tie, ~planted, alpha)
    if min_part_cells:
        sizes = np.bincount(parts[parts >= 0])
        parts = np.where(np.isin(parts, np.flatnonzero(sizes >= min_part_cells)), parts, -1)
    _, r = calibrate.part_rates(fire, parts, planted, alpha)
    out = dict(pd=float(fire[planted].mean()),
               pooled_pfa=float(fire[~planted].mean()),
               delta=float(r.max() - r.min()),
               n_parts=int(len(r)), n_groups=int(len(counts)))
    if groups is not None:
        dec = calibrate.decompose(fire, parts, g, planted, alpha)
        out["between"], out["within"] = dec["between_spread"], dec["within_spread"]
    return out


def load_results(results_dir=None):
    import pandas as pd
    files = sorted(Path(results_dir or RESULTS).glob("eval_seed*.csv"))
    if not files:
        raise SystemExit("No results/eval_seed*.csv found; run experiments/evaluate.py first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def paired(ev, a_score, a_part, b_score, b_part, alpha=ALPHA, mc=0):
    """Per-configuration (dataset, target, seed) pairs of two arms."""
    sel = (ev.alpha == alpha) & (ev.min_part_cells == mc)
    key = ["dataset", "target", "seed"]
    a = ev[sel & (ev.score == a_score) & (ev.partition == a_part)].set_index(key)
    b = ev[sel & (ev.score == b_score) & (ev.partition == b_part)].set_index(key)
    j = a[["pd", "delta"]].join(b[["pd", "delta"]], lsuffix="_a", rsuffix="_b", how="inner")
    j["d_pd"] = j.pd_a - j.pd_b
    j["d_delta"] = j.delta_a - j.delta_b
    return j


def mean_over_configs(d, cols=("pd", "delta")):
    """Mean over target configurations within each planting, then over plantings."""
    return d.groupby("seed")[list(cols)].mean().mean()
