"""Designer partitions (Section IV) and the control partitions of Section V.

Tiles and superpixels are built without labels. `labeled_parts`, `split_parts`
and `mix_parts` use the labels on purpose: they are the labeled reference and
the controlled tests of Section V, not detectors.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt


def compact(labels: np.ndarray) -> np.ndarray:
    """Relabel to 0..G-1 so empty groups never appear."""
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(np.int64)


def tiles(active_idx: np.ndarray, width: int, spacing: int) -> np.ndarray:
    """Square P x P tiles, P = spacing."""
    row, col = np.divmod(active_idx, width)
    ncol = math.ceil(width / spacing)
    return compact((row // spacing) * ncol + (col // spacing))


def superpixels(feat_full, active_full, height, width, spacing,
                compactness=1.0, iters=10) -> np.ndarray:
    """SLIC: seed a grid at `spacing`, then k-means in the joint space of
    position and spectrum, each cell competing only for the nine seeds nearest
    it. `compactness` weighs a one-spacing move against a one-standard-deviation
    average spectral move (the features are whitened, so that is the unit).
    """
    active_idx = np.flatnonzero(active_full)
    rows, cols = np.divmod(active_idx, width)
    nr, nc = math.ceil(height / spacing), math.ceil(width / spacing)
    nseed = nr * nc
    seed_r = np.repeat(np.minimum((np.arange(nr) + 0.5) * spacing, height - 1), nc)
    seed_c = np.tile(np.minimum((np.arange(nc) + 0.5) * spacing, width - 1), nr)
    rr, cc = np.rint(seed_r).astype(int), np.rint(seed_c).astype(int)
    if not active_full.all():
        # move a seed that landed on a removed cell to the nearest kept cell
        miss = ~active_full.reshape(height, width)
        near = distance_transform_edt(miss, return_distances=False, return_indices=True)
        rr, cc = near[0, rr, cc], near[1, rr, cc]
    sfeat = feat_full[rr * width + cc].copy()
    cr, cc_ = seed_r.copy(), seed_c.copy()
    feat = feat_full[active_idx]
    base_r = np.clip(rows // spacing, 0, nr - 1)
    base_c = np.clip(cols // spacing, 0, nc - 1)
    lab = np.zeros(len(active_idx), dtype=np.int64)

    for _ in range(iters):
        best = np.full(len(active_idx), np.inf)
        for dr in (-1, 0, 1):
            r_ = base_r + dr
            rok = (r_ >= 0) & (r_ < nr)
            r_ = np.clip(r_, 0, nr - 1)
            for dc in (-1, 0, 1):
                c_ = base_c + dc
                ok = rok & (c_ >= 0) & (c_ < nc)
                c_ = np.clip(c_, 0, nc - 1)
                cand = r_ * nc + c_
                spec = np.mean((feat - sfeat[cand]) ** 2, axis=1)
                spat = ((rows - cr[cand]) ** 2 + (cols - cc_[cand]) ** 2) / spacing ** 2
                dist = spec + compactness ** 2 * spat
                upd = ok & (dist < best)
                best[upd] = dist[upd]
                lab[upd] = cand[upd]
        cnt = np.bincount(lab, minlength=nseed)
        occ = cnt > 0
        cr[occ] = np.bincount(lab, weights=rows, minlength=nseed)[occ] / cnt[occ]
        cc_[occ] = np.bincount(lab, weights=cols, minlength=nseed)[occ] / cnt[occ]
        for j in range(feat.shape[1]):
            sfeat[occ, j] = np.bincount(lab, weights=feat[:, j], minlength=nseed)[occ] / cnt[occ]
    return compact(lab)


def labeled_parts(parts: np.ndarray) -> np.ndarray:
    """Calibration over the audited parts themselves (the labeled reference)."""
    return compact(parts)


def permuted(groups: np.ndarray, rng) -> np.ndarray:
    """Locality control: keep every group size, destroy the spatial structure."""
    perm = rng.permutation(len(groups))
    out = np.empty_like(groups)
    out[perm] = groups
    return out


def split_parts(parts: np.ndarray, k: int, rng) -> np.ndarray:
    """Pure refinement: cut every audited part into k random sub-groups."""
    return compact(parts.astype(np.int64) * k + rng.integers(0, k, size=len(parts)))


def mix_parts(parts: np.ndarray, frac: float, rng) -> np.ndarray:
    """Controlled mixing: move a fraction `frac` of the audited cells to a
    uniformly random other audited part."""
    out = parts.astype(np.int64).copy()
    aud = np.flatnonzero(parts >= 0)
    ids = np.unique(parts[aud])
    hit = aud[rng.random(len(aud)) < frac]
    if len(hit):
        pos = np.searchsorted(ids, out[hit])
        shift = rng.integers(1, len(ids), size=len(hit))
        out[hit] = ids[(pos + shift) % len(ids)]
    return compact(out)
