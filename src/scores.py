"""The three candidate detection scores of Section III-A, computed in the
d = 12 principal-component space; every score is "large means target".

- `glrt_mixture`: the GLRT in the target amplitude against a Gaussian-mixture
  background density fitted to the whole cube.
- `WindowMachine.score(kind="amf" | "ace")`: AMF and ACE with the background
  mean and covariance estimated from a guarded square window.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp


# ---------------------------------------------------------------- mixture GLRT

def _mixture_terms(x: np.ndarray, gmm, s: np.ndarray):
    """log w_k N(x; mu_k, S_k) and the two coefficients of the A-expansion.

    log N(x - A s; mu_k, S_k) = base_k(x) + A * lin_k(x) - 0.5 A^2 quad_k
    """
    n, d = x.shape
    K = gmm.n_components
    base = np.empty((n, K))
    lin = np.empty((n, K))
    quad = np.empty(K)
    const = d * math.log(2 * math.pi)
    for k in range(K):
        cov = gmm.covariances_[k]
        prec = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise RuntimeError("non-positive definite GMM covariance")
        diff = x - gmm.means_[k]
        pdiff = diff @ prec
        base[:, k] = (math.log(gmm.weights_[k])
                      - 0.5 * (const + logdet + np.einsum("ij,ij->i", pdiff, diff)))
        lin[:, k] = pdiff @ s
        quad[k] = s @ prec @ s
    return base, lin, quad


# A 128-component mixture over 200k cells makes the (n, K) work arrays large,
# so both scores below run in cell blocks.
_BLOCK = 20000


def glrt_mixture(x, gmm, s, amax, grid=65):
    """log max_{0<=A<=amax} m(x - A s) / m(x), maximized over a grid of A.

    A = 0 is always admissible, so the score is >= 0 and has an atom at 0;
    ties are broken by a uniform draw in the calibration.
    """
    out = np.empty(len(x))
    grid_a = np.linspace(0.0, amax, grid)[1:]
    for a0 in range(0, len(x), _BLOCK):
        b = min(a0 + _BLOCK, len(x))
        base, lin, quad = _mixture_terms(x[a0:b], gmm, s)
        log_null = logsumexp(base, axis=1)
        best = np.zeros(b - a0)
        for a in grid_a:
            alt = logsumexp(base + a * lin - 0.5 * a * a * quad, axis=1)
            np.maximum(best, alt - log_null, out=best)
        out[a0:b] = best
    return out


# ------------------------------------------------------------ windowed (AMF/ACE)

class WindowMachine:
    """Guarded-window local statistics for many window sizes at one cost.

    The integral images of x and of x x^T are built once per scene; each
    (window, guard) pair is then two rectangle lookups. Cells outside the image
    and cells removed with the target material never contribute.
    """

    def __init__(self, x_img: np.ndarray, valid: np.ndarray):
        H, W, d = x_img.shape
        self.H, self.W, self.d = H, W, d
        self.x_img = x_img
        prod = np.einsum("hwi,hwj->hwij", x_img, x_img).reshape(H, W, d * d)
        self._ix = self._integral(x_img * valid[:, :, None])
        self._ip = self._integral(prod * valid[:, :, None])
        self._ic = self._integral(valid.astype(np.float64)[:, :, None])
        del prod
        xg = x_img.reshape(-1, d)[valid.reshape(-1)]
        self.gmu = xg.mean(axis=0)
        self.gcov = np.cov(xg, rowvar=False)

    @staticmethod
    def _integral(field):
        H, W, C = field.shape
        acc = np.zeros((H + 1, W + 1, C))
        np.cumsum(np.cumsum(field, axis=0), axis=1, out=acc[1:, 1:, :])
        return acc

    def _rect(self, acc, win):
        """Sum over the win x win box centred on each cell, clipped to the image."""
        H, W = self.H, self.W
        r = win // 2
        rows = np.arange(H); cols = np.arange(W)
        r0 = np.clip(rows - r, 0, H); r1 = np.clip(rows + r + 1, 0, H)
        c0 = np.clip(cols - r, 0, W); c1 = np.clip(cols + r + 1, 0, W)
        return (acc[np.ix_(r1, c1)] - acc[np.ix_(r0, c1)]
                - acc[np.ix_(r1, c0)] + acc[np.ix_(r0, c0)])

    def stats(self, win, guard):
        d = self.d
        sx = self._rect(self._ix, win) - self._rect(self._ix, guard)
        sxx = self._rect(self._ip, win) - self._rect(self._ip, guard)
        n = (self._rect(self._ic, win) - self._rect(self._ic, guard))[:, :, 0]
        return sx, sxx.reshape(self.H, self.W, d, d), n

    def score(self, s, win, guard, kind, reg=1e-6):
        H, W, d = self.H, self.W, self.d
        sx, sxx, n = self.stats(win, guard)
        flat_n = n.reshape(-1)
        enough = flat_n >= d + 2
        safe = np.maximum(flat_n, 1.0)

        mu = sx.reshape(-1, d) / safe[:, None]
        cov = sxx.reshape(-1, d, d) / safe[:, None, None]
        cov -= mu[:, :, None] * mu[:, None, :]
        cov *= (safe / np.maximum(flat_n - 1, 1.0))[:, None, None]
        cov[~enough] = self.gcov
        mu[~enough] = self.gmu
        cov += reg * np.trace(self.gcov) / d * np.eye(d)

        z = self.x_img.reshape(-1, d) - mu
        out = np.empty(H * W)
        step = 40000
        for a in range(0, H * W, step):
            b = min(a + step, H * W)
            rhs = np.empty((b - a, d, 2))
            rhs[:, :, 0] = z[a:b]
            rhs[:, :, 1] = s
            sol = np.linalg.solve(cov[a:b], rhs)
            cs = sol[:, :, 1]
            sz = np.einsum("ij,ij->i", z[a:b], cs)
            ss = np.einsum("j,ij->i", s, cs)
            if kind == "amf":
                out[a:b] = sz ** 2 / np.maximum(ss, 1e-30)
            elif kind == "ace":
                zz = np.einsum("ij,ij->i", z[a:b], sol[:, :, 0])
                out[a:b] = sz ** 2 / np.maximum(ss * zz, 1e-30)
            else:
                raise ValueError(kind)
        return out
