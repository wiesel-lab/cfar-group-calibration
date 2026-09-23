"""Stage 2: evaluate every arm on the cached scenes of one planting.

Every arm is read at the same pooled background false-alarm rate. At
alpha = 0.1 (all four images):
  - uncalibrated: GLRT, and AMF/ACE for every window of the sweep;
  - GLRT, AMF 7x7 and ACE 7x7 calibrated over tiles and superpixels (every
    spacing P) and over the labeled audited parts;
  - GLRT calibrated over the 128 mixture components, the refined and the
    mixed audited parts, and the permuted P = 4 superpixels.
At alpha = 0.05, 0.02, 0.01 (Pavia and Salinas, audited parts of at least
0, 1000 and 2000 cells): the uncalibrated scores and the calibrated GLRT.

Usage:
    python experiments/evaluate.py --seed 0 --workers 4

Writes results/eval_seed{seed}.csv (one row per configuration and arm).
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

import common as C


def rows_for(path):
    z = np.load(path)
    sc = {k: z[k] for k in ("planted", "parts", "tie")}
    S = {k[7:]: z[k].astype(np.float64) for k in z.files if k.startswith("score__")}
    P = {k[6:]: z[k].astype(np.int64) for k in z.files if k.startswith("part__")}
    ds, target, seed = str(z["dataset"]), int(z["target"]), int(z["seed"])
    ladder = [f"{t}_{p}" for p in C.LADDER for t in ("tile", "sp")]
    controls = ["modes_128"] + [k for k in P if k.startswith(("split_", "mix_", "perm_"))]

    plan = [(s, None, C.ALPHA, 0) for s in S]
    plan += [(b, p, C.ALPHA, 0) for b in C.BASES for p in ladder + ["oracle"]]
    plan += [("glrt", p, C.ALPHA, 0) for p in controls]
    if ds in C.LOW_ALPHA_SETS:
        for alpha, mc in C.LOW_ALPHA.items():
            plan += [(s, None, alpha, mc) for s in C.LOW_ALPHA_BASELINES]
            plan += [("glrt", p, alpha, mc) for p in ladder + ["oracle"]]

    rows = []
    for s, p, alpha, mc in plan:
        r = C.evaluate(S[s], None if p is None else P[p], sc, alpha, mc)
        family, spacing = "none", None
        if p is not None:
            head = p.split("_")
            family = head[0]
            if head[0] in ("tile", "sp"):
                spacing = int(head[1])
        rows.append(dict(dataset=ds, target=target, seed=seed, alpha=alpha,
                         min_part_cells=mc, score=s, partition=p or "none",
                         family=family, P=spacing, **r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    files = sorted((C.CACHE / f"seed{a.seed}").glob("*.npz"), key=lambda f: -f.stat().st_size)
    if not files:
        raise SystemExit(f"No cached scenes for seed {a.seed}; run experiments/build_scenes.py first.")
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.map(rows_for, files, chunksize=1) for r in rs]
    C.RESULTS.mkdir(exist_ok=True)
    out = C.RESULTS / f"eval_seed{a.seed}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(files)} configurations, {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
