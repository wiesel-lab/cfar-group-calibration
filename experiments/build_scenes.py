"""Stage 1: build every target configuration once and cache it.

For each image and each labeled class used as the target: the semi-synthetic
scene, the three candidate scores (GLRT; AMF and ACE for every window in the
sweep), the 128 mixture-component assignment, and every partition used in the
experiments (tiles and superpixels over the spacing ladder, the labeled
parts, their refinements and mixtures, and the permuted superpixels).

Usage:
    python experiments/build_scenes.py --seed 0 --workers 4

Writes cache/seed{seed}/{dataset}_t{target}.npz. Existing files are skipped,
so the script can be interrupted and restarted.
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

import common as C
from src import data, partitions, scene, scores


def build_one(job):
    name, target, seed = job
    out = C.scene_file(seed, name, target)
    if out.exists():
        return f"skip {out.name}"
    t0 = time.time()
    ds = data.load(name)
    sc = scene.build(ds, target, ds.admitted_classes(), plant_frac=C.PLANT_FRAC,
                     pca_dim=C.PCA_DIM, n_components=C.N_COMPONENTS, seed=seed,
                     amplitude_sigmas=C.AMPLITUDE_SIGMAS)
    H, W = sc.height, sc.width
    idx = np.flatnonzero(sc.active_full)
    amax = C.AMAX_FACTOR * sc.amplitude

    S = {"glrt": scores.glrt_mixture(sc.x, sc.gmm, sc.s, amax, C.GLRT_GRID)}
    modes = sc.gmm.predict(sc.x)

    # The windows and the partitions work on the coordinates rounded to float32.
    x = sc.x.astype(np.float32).astype(np.float64)
    x_full = np.zeros((H * W, x.shape[1]))
    x_full[idx] = x
    wm = scores.WindowMachine(x_full.reshape(H, W, -1), sc.active_full.reshape(H, W))
    for w in C.WINDOWS:
        for kind in ("amf", "ace"):
            S[f"{kind}_{w}"] = wm.score(sc.s, w, C.GUARD, kind)[idx]
    del wm

    P = {}
    for p in C.LADDER:
        P[f"tile_{p}"] = partitions.tiles(idx, W, p)
        P[f"sp_{p}"] = partitions.superpixels(x_full, sc.active_full, H, W, p, 1.0, 10)
    P["modes_128"] = partitions.compact(modes)
    P["oracle"] = partitions.labeled_parts(sc.parts)
    base = seed * 7919 + target
    for k in C.SPLITS:
        P[f"split_{k}"] = partitions.split_parts(sc.parts, k, np.random.default_rng(base + 100 + k))
    for f in C.MIX_FRACS:
        P[f"mix_{f * 100:g}"] = partitions.mix_parts(sc.parts, f, np.random.default_rng(base + 200 + int(f * 1000)))
    for r in range(C.PERM_SEEDS):
        P[f"perm_sp_4_{r}"] = partitions.permuted(P["sp_4"], np.random.default_rng(base + 400 + 10 * r))

    tie = np.random.default_rng(seed * 99991 + target).random(len(sc.x))   # shared tie-break draw

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, dataset=name, target=target, seed=seed, height=H, width=W,
        planted=sc.planted, parts=sc.parts, tie=tie,
        **{f"score__{k}": v.astype(np.float32) for k, v in S.items()},
        **{f"part__{k}": v.astype(np.int32) for k, v in P.items()})
    return f"{name:8s} target {target:2d}  n={len(sc.x):6d}  {time.time() - t0:5.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", nargs="+", default=list(data.ALL_DATASETS))
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    jobs = []
    for name in a.datasets:
        ds = data.load(name)
        jobs += [(name, t, a.seed) for t in ds.admitted_classes()]
    # largest images first, for load balance
    order = {"pavia": 0, "salinas": 1, "muufl": 2, "indian": 3}
    jobs.sort(key=lambda j: order[j[0]])
    print(f"{len(jobs)} target configurations, seed {a.seed}", flush=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(build_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
