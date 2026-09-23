# Data

The paper uses four public hyperspectral images. Download them with

```bash
bash data/download_data.sh
```

This fetches seven `.mat` files (~96 MB) into this directory and checks their
MD5 sums against the files used for the paper. The files are git-ignored.

| Image | Files | Source |
|---|---|---|
| MUUFL Gulfport, campus 1, with scene labels [Gader et al. 2013; Du & Zare 2017] | `muufl_gulfport_campus_1_hsi_220_label.mat` | [GatorSense/MUUFLGulfport](https://github.com/GatorSense/MUUFLGulfport) |
| Pavia University | `PaviaU.mat`, `PaviaU_gt.mat` | [UPV/EHU](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Salinas (corrected) | `Salinas_corrected.mat`, `Salinas_gt.mat` | [UPV/EHU](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| Indian Pines (corrected) [Baumgardner et al. 2015] | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` | [UPV/EHU](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |

`www.ehu.eus` blocks scripted downloads, so the script fetches the three
UPV/EHU images from a public GitHub mirror
([gokriznastic/HybridSN](https://github.com/gokriznastic/HybridSN)) whose
files are identical to the standard ones (the checksums are verified). If you
prefer, download them by hand from the UPV/EHU page into this directory.

The labels are used only to define the audited regions and the target
signatures. They are never used to fit the background model, to form the
calibration groups, or to set the threshold.
