# Data

No dataset is stored in this repository. Everything below is fetched from its
original source with `python scripts/download_data.py` (use `--only` for a
subset, `--root` to change the output folder, default `data/`).

| Name | Source | Used for | Access |
|---|---|---|---|
| Indian license plates with labels | Kaggle `kedarsai/indian-license-plates-with-labels` | Plate detector / OCR training data | Kaggle API key (`~/.kaggle/kaggle.json`) |
| Indian number plates (sample) | Hugging Face `Dataclusterlabspvtltd/indian-number-plates-dataset` | Detection evaluation | Public |
| RoadText-1K | IIIT Hyderabad CVIT, `RoadText1k.zip` | Real-plate OCR validation/test crops built from video frames | Public zip; read its `License.txt` |
| COCO val2017 | cocodataset.org | Background images for composite small-plate training data | Public |

## Licences

Each dataset has its own terms. They are not redistributed here; check the
source before using them beyond evaluation, and before publishing anything
derived from them (for example trained weights).

## What is not reproducible from the repo yet

The derived data (RoadText validation/test crops, composite small-plate images,
final training splits) was built in the development notebook
[`notebooks/development_log.ipynb`](../notebooks/development_log.ipynb) and in
scripts that were kept on Google Drive only. The notebook shows how each was
produced (Part 7 rebuilds the detector datasets from the downloads); standalone
scripts for the composite generator and detector training are not written yet
(see `docs/TRAINING.md`).
