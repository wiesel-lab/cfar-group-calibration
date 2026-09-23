"""Group calibration (Section III) and the balance audit (Sections II and IV).

The method itself is `group_rank` + `group_calibrated_detector`: rank the
detection score inside each group of the designer's partition, then apply a
single threshold. `fire_at_pooled_rate`, `audit` and `decompose` are the
evaluation used in the experiments.
"""

from __future__ import annotations

import numpy as np


def group_rank(score: np.ndarray, groups: np.ndarray, tie: np.ndarray):
    """Normalized within-group rank (Section III): Lambda_i = rank_i / n_g(i).

    Cells are sorted from the largest score to the smallest inside their group,
    so a small Lambda_i means cell i is among the highest-scoring cells of its
    group. Ties are broken by the uniform draw `tie`.

    Returns (Lambda, rank 1..n_g, group sizes n_g).
    """
    n_groups = int(groups.max()) + 1
    counts = np.bincount(groups, minlength=n_groups)
    order = np.lexsort((tie, -score, groups))
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    pos = np.empty(len(score), dtype=np.int64)
    pos[order] = np.arange(len(score)) - starts[groups[order]] + 1
    return pos / counts[groups], pos, counts


def group_calibrated_detector(score, groups, alpha, rng=None, tie=None):
    """Detections of group calibration at level alpha (Section III).

    In every group of n_g cells the floor(alpha n_g) highest-scoring cells are
    declared, and the next-ranked cell with probability equal to the fractional
    part of alpha n_g, so every group receives an expected alarm fraction of
    exactly alpha. The ranks do not depend on alpha.
    """
    rng = np.random.default_rng(rng)
    if tie is None:
        tie = rng.random(len(score))
    _, pos, counts = group_rank(score, groups, tie)
    quota = alpha * counts[groups]
    k = np.floor(quota + 1e-12)
    return (pos <= k) | ((pos == k + 1) & (rng.random(len(score)) < quota - k))


def fire_at_pooled_rate(rank, tie, background, alpha):
    """Single global threshold at which the pooled background rate equals alpha.

    Used in the experiments so that every detector, calibrated or not, is
    compared at the same pooled background false-alarm rate.
    """
    wanted = int(round(alpha * background.sum()))
    order = np.lexsort((tie, rank))
    cum = np.cumsum(background[order])
    cut = np.searchsorted(cum, wanted, side="left")
    fire = np.zeros(len(rank), dtype=bool)
    fire[order[: cut + 1]] = True
    return fire


def part_rates(fire, parts, planted, alpha):
    """r_c: false-alarm rate among the background cells of audited part c,
    normalized by alpha (Section II). Parts with id -1 are not audited."""
    bg = ~planted
    ids = np.unique(parts[bg & (parts >= 0)])
    r = np.array([fire[bg & (parts == c)].mean() / alpha for c in ids])
    return ids, r


def decompose(fire, parts, groups, planted, alpha):
    """Section IV identity: r_c - 1 = between_c + within_c over the designer's groups,

        between_c = sum_g P(g|c) (r_g - 1),  within_c = sum_g P(g|c) (r_{c,g} - r_g),

    on background cells. Returns the spread (max - min over c) of each term.
    """
    bg = ~planted
    g = groups[bg]
    c = parts[bg]
    f = fire[bg].astype(np.float64)
    ng = int(g.max()) + 1
    n_g = np.bincount(g, minlength=ng)
    r_g = np.bincount(g, weights=f, minlength=ng) / np.maximum(n_g, 1) / alpha

    between, within = [], []
    for cc in np.unique(c[c >= 0]):
        m = c == cc
        nz = np.bincount(g[m], minlength=ng)
        p = nz / m.sum()
        r_cg = np.zeros(ng)
        r_cg[nz > 0] = (np.bincount(g[m], weights=f[m], minlength=ng)[nz > 0]
                        / nz[nz > 0] / alpha)
        between.append(float((p * (r_g - 1.0)).sum()))
        within.append(float((p * (r_cg - r_g)).sum()))
    between, within = np.array(between), np.array(within)
    return {"between_spread": float(between.max() - between.min()),
            "within_spread": float(within.max() - within.min())}
