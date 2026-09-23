#!/usr/bin/env bash
# Downloads the four hyperspectral images used in the paper into data/ (~96 MB)
# and checks them against the MD5 sums of the files used for the paper.
#
# Pavia University, Salinas and Indian Pines are the standard files of the
# Grupo de Inteligencia Computacional (UPV/EHU); www.ehu.eus blocks scripted
# downloads, so they are fetched from a public GitHub mirror with identical
# content. MUUFL Gulfport (campus 1, with scene labels) is fetched from the
# official GatorSense repository.
set -euo pipefail
cd "$(dirname "$0")"

MIRROR=https://raw.githubusercontent.com/gokriznastic/HybridSN/master/data
MUUFL=https://raw.githubusercontent.com/GatorSense/MUUFLGulfport/master/MUUFLGulfportSceneLabels

fetch() {  # url file md5
  if [ ! -f "$2" ]; then
    echo "downloading $2"
    curl -sSfL -o "$2" "$1"
  fi
  local sum
  if command -v md5sum >/dev/null; then sum=$(md5sum "$2" | cut -d' ' -f1); else sum=$(md5 -q "$2"); fi
  if [ "$sum" != "$3" ]; then
    echo "checksum mismatch for $2 (got $sum, expected $3)" >&2
    exit 1
  fi
}

fetch "$MIRROR/PaviaU.mat"                 PaviaU.mat                 165a3c7488995f54a19add47c7eed4cd
fetch "$MIRROR/PaviaU_gt.mat"              PaviaU_gt.mat              b8c3ba44b077c26e24220463aa855bd3
fetch "$MIRROR/Salinas_corrected.mat"      Salinas_corrected.mat      485d8802f4a6b4ebc0767d48dd0da06b
fetch "$MIRROR/Salinas_gt.mat"             Salinas_gt.mat             7b8da653a61bb0271b27b37fb926390f
fetch "$MIRROR/Indian_pines_corrected.mat" Indian_pines_corrected.mat 66dbc9f4a9b7c9b1445f60a87b505101
fetch "$MIRROR/Indian_pines_gt.mat"        Indian_pines_gt.mat        9414943dac1d80faaa9165c8b460510c
fetch "$MUUFL/muufl_gulfport_campus_1_hsi_220_label.mat" muufl_gulfport_campus_1_hsi_220_label.mat ce8b037a16d8fc252a1d5bab1ca38ed4
echo "Done: all files downloaded to data/ and verified."
