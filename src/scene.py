"""One semi-synthetic target configuration (Section V-A).

Each labeled class serves in turn as the target: its cells are removed, its
unit-norm mean spectrum is the signature s, and the signature is planted in
raw radiance at twice the cube's standard deviation on a seeded random 1% of
the remaining cells. The cube is then reduced to d = 12 whitened principal
components, and the background density m (a Gaussian mixture) is fitted to
the reduced cube, planted targets included, without any labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from zlib import crc32

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class Scene:
    dataset: str
    target: int
    height: int
    width: int
    active_full: np.ndarray    # (H*W,) cells kept (the target class is removed)
    x: np.ndarray              # (n, d) PCA coordinates of the kept cells
    s: np.ndarray              # (d,) target signature in PCA coordinates
    amplitude: float           # planted amplitude A
    planted: np.ndarray        # (n,) cells carrying a target (H1)
    parts: np.ndarray          # (n,) audited part id, -1 = not audited
    labels: np.ndarray         # (n,) raw annotation
    gmm: GaussianMixture       # background density m


def _pca(cube_flat, fit_mask, dim):
    fit = cube_flat[fit_mask]
    mu = fit.mean(axis=0)
    cen = fit - mu
    cov = cen.T @ cen / (len(fit) - 1)
    val, vec = np.linalg.eigh(cov)
    order = np.argsort(val)[::-1][:dim]
    val = np.maximum(val[order], 1e-12)
    vec = vec[:, order]
    scale = np.sqrt(val)
    return (cube_flat - mu) @ vec / scale, vec, scale


def build(ds, target, admitted, plant_frac=0.01, pca_dim=12, n_components=128,
          seed=0, amplitude_sigmas=2.0, gmm_max_iter=200):
    H, W, B = ds.cube.shape
    flat = ds.cube.reshape(-1, B)
    lab_full = ds.labels.reshape(-1)
    active_full = lab_full != target
    idx = np.flatnonzero(active_full)
    labels = lab_full[idx]
    n = len(idx)

    # crc32 (not hash()) keeps the planting identical across processes.
    rng = np.random.default_rng(seed * 1000003 + crc32(ds.name.encode()) % 100003 + target)
    planted = np.zeros(n, dtype=bool)
    planted[rng.choice(n, size=int(round(plant_frac * n)), replace=False)] = True

    sig_raw = flat[lab_full == target].mean(axis=0)
    direction = sig_raw / np.linalg.norm(sig_raw)
    amplitude = amplitude_sigmas * float(flat[idx].std())

    planted_flat = flat.copy()
    planted_flat[idx[planted]] += amplitude * direction

    x_full, vec, scale = _pca(planted_flat, active_full, pca_dim)
    x = x_full[active_full]
    s = direction @ vec / scale

    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          reg_covar=1e-6, n_init=1, max_iter=gmm_max_iter,
                          random_state=seed * 7 + target).fit(x)

    # Audited parts: every admitted class except the target, plus one part for
    # the unlabeled cells. Smaller classes stay in the image but are not audited.
    parts = np.full(n, -1, dtype=np.int64)
    parts[labels == 0] = 0
    for c in admitted:
        if c != target:
            parts[labels == c] = c

    return Scene(ds.name, target, H, W, active_full, x, s, amplitude,
                 planted, parts, labels, gmm)
