# Constant False Alarm Rate (CFAR) Detection by Blind Group Calibration

Daniel Busbib, Shir Schneorson, Danilo Orlando, [Ami Wiesel](https://www.cs.huji.ac.il/~amiw/)

The Hebrew University of Jerusalem · University of Pisa

Official implementation of *Constant False Alarm Rate (CFAR) Detection by Blind
Group Calibration* (submitted to ICASSP 2027).

<p align="center">
  <img src="docs/static/teaser.png" width="560" alt="Detection vs balance for three detection scores">
</p>

## Abstract

CFAR detection requires the false-alarm rate to remain constant as the
background nuisance changes. We consider the case where this nuisance is
discrete, representing different background regions. In this paper, we ask
whether a detector can be calibrated toward this property without knowing the
regions or their number. Our approach is simple: we partition the image into
small spatial groups and rank the detection scores within each group before
applying a single threshold. The underlying scoring function remains
unchanged. Only the threshold varies locally, requiring neither a parametric
background model nor prior knowledge of the regions or their number. We also
derive a simple identity that decomposes regional imbalance into two terms: a
between-group term directly controlled by calibration and a within-group term
arising from groups containing different backgrounds. Experiments on four
public hyperspectral images show that small spatial groups reduce the spread
of false-alarm rates across labeled regions by more than half on average while
preserving most detection power. Disrupting the groups' spatial structure
eliminates most of this improvement.

## The method in a few lines

Group calibration works with any detection score and any partition of the
image into groups:

```python
import numpy as np
from src.calibrate import group_rank, group_calibrated_detector

# score:  (n,) detection score per cell, larger = more target-like
# groups: (n,) group index per cell, e.g. square P x P tiles
lam, _, _ = group_rank(score, groups, tie=np.random.default_rng(0).random(len(score)))
detections = lam <= alpha          # the top alpha fraction of every group, up to rounding

# or with the randomized selection of the next-ranked cell (Section III),
# which gives every group an expected alarm fraction of exactly alpha:
detections = group_calibrated_detector(score, groups, alpha, rng=0)
```

The ranks do not depend on `alpha`, so one calibration serves every
false-alarm level.

## Installation

```bash
git clone https://github.com/wiesel-lab/cfar-group-calibration.git
cd cfar-group-calibration
pip install -r requirements.txt
bash data/download_data.sh
```

Tested with Python 3.12 on macOS.

## Reproducing the results

The experiments run in two stages, once for each of the three target
plantings (seeds 0, 1, 2), followed by one script per result:

```bash
python experiments/build_scenes.py --seed 0 --workers 4   # scenes, scores, partitions -> cache/
python experiments/evaluate.py     --seed 0 --workers 4   # every arm -> results/eval_seed0.csv
# ... the same for --seed 1 and --seed 2, then:
```

| Result in paper | Command |
|---|---|
| Table 1 (detection and balance per image) | `python experiments/table1.py` |
| Figure 1 (detection vs balance for GLRT, AMF, ACE) | `python experiments/figure1.py` |
| Every number quoted in Section V (window sweep, 141/141 pairs, labeled-parts cost, refinement and mixing tests, permutation control, mixture-component grouping, lower false-alarm levels) | `python experiments/section5_numbers.py` |

Or run everything, including the download, with `bash experiments/run_all.sh`.
Outputs are written to `results/`. All randomness is seeded.

Stage 1 is the slow part: it fits a 128-component Gaussian mixture for each of
the 47 target configurations, about 50 CPU-minutes per planting (Pavia is the
largest image). Stage 2 takes a few minutes. Both stages skip configurations
that are already cached, so they can be interrupted and restarted. To try the
pipeline quickly, run stage 1 on one image, e.g.
`python experiments/build_scenes.py --seed 0 --datasets indian` (under a
minute), then stage 2.

## Data

MUUFL Gulfport, Pavia University, Salinas and Indian Pines. The download
script fetches them and checks their MD5 sums; see
[data/README.md](data/README.md) for the sources.

## Repository structure

```
src/
  calibrate.py    group calibration (within-group ranks), the pooled-rate
                  operating point, per-region false-alarm rates and the
                  between/within decomposition of Section IV
  scores.py       GLRT in the target amplitude with a Gaussian-mixture
                  background; AMF and ACE with guarded windows
  partitions.py   tiles, SLIC superpixels, labeled parts and the control
                  partitions (permuted, refined, mixed)
  scene.py        one semi-synthetic target configuration (Section V-A)
  data.py         loaders for the four images
experiments/      one script per stage and per result in the paper
data/             download script
docs/             project page (served via GitHub Pages)
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{busbib2027cfar,
  title  = {Constant False Alarm Rate ({CFAR}) Detection by Blind Group Calibration},
  author = {Busbib, Daniel and Schneorson, Shir and Orlando, Danilo and Wiesel, Ami},
  note   = {Submitted to ICASSP 2027},
  year   = {2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
