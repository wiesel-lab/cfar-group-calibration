"""Loaders for the four hyperspectral images (download: data/download_data.sh).

Each loader returns the raw cube (H, W, B) and an integer label map (H, W)
with 0 = unlabeled. The labels are used only to define the audited parts and
the target signatures, never by the detector or the calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

DATA = Path(__file__).resolve().parents[1] / "data"

# A labeled region is audited (and used as a target) only if it has at least
# 200 cells, so that it sees about twenty expected false alarms at alpha = 0.1.
MIN_PART_CELLS = 200

ALL_DATASETS = ("muufl", "pavia", "salinas", "indian")


@dataclass
class Dataset:
    name: str
    cube: np.ndarray          # (H, W, B) float64
    labels: np.ndarray        # (H, W) int, 0 = unlabeled

    def admitted_classes(self, min_cells: int = MIN_PART_CELLS) -> list[int]:
        """Labeled classes large enough to be an audited part or a target."""
        flat = self.labels.reshape(-1)
        return [int(k) for k in sorted(set(flat.tolist()) - {0})
                if int((flat == k).sum()) >= min_cells]


def load_pavia() -> Dataset:
    cube = loadmat(DATA / "PaviaU.mat")["paviaU"]
    gt = loadmat(DATA / "PaviaU_gt.mat")["paviaU_gt"]
    return Dataset("pavia", cube.astype(np.float64), gt.astype(np.int32))


def load_salinas() -> Dataset:
    cube = loadmat(DATA / "Salinas_corrected.mat")["salinas_corrected"]
    gt = loadmat(DATA / "Salinas_gt.mat")["salinas_gt"]
    return Dataset("salinas", cube.astype(np.float64), gt.astype(np.int32))


def load_indian() -> Dataset:
    cube = loadmat(DATA / "Indian_pines_corrected.mat")["indian_pines_corrected"]
    gt = loadmat(DATA / "Indian_pines_gt.mat")["indian_pines_gt"]
    return Dataset("indian", cube.astype(np.float64), gt.astype(np.int32))


def load_muufl() -> Dataset:
    """MUUFL Gulfport campus 1 with the Du & Zare scene labels (325 x 220 x 64).
    Unlabeled cells are marked -1 in the file and remapped to 0."""
    h = loadmat(DATA / "muufl_gulfport_campus_1_hsi_220_label.mat",
                struct_as_record=False, squeeze_me=True)["hsi"]
    cube = np.asarray(h.Data, dtype=np.float64)
    labels = np.asarray(h.sceneLabels.labels, dtype=np.int32)
    labels[labels < 0] = 0
    return Dataset("muufl", cube, labels)


LOADERS = {"muufl": load_muufl, "pavia": load_pavia,
           "salinas": load_salinas, "indian": load_indian}


def load(name: str) -> Dataset:
    return LOADERS[name]()
