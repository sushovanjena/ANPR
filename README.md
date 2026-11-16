# Indian ANPR cascade

Open-source automatic number plate recognition for Indian multi-lane traffic. It
reads plates per vehicle, not per frame: vehicles are detected and tracked first,
each plate is tied to its vehicle, and the final plate is a vote across the
vehicle's whole track. A live-stream runner turns camera frames into plate events.

```
frame -> vehicle detector -> tracker -> plate detector (per vehicle crop)
      -> plate-to-vehicle association -> quality gate -> OCR -> per-track consensus
      -> event (JSONL / SQLite / callback) + latency metrics
```

Design, references and limitations: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Weights and how to publish them: [docs/WEIGHTS.md](docs/WEIGHTS.md). Training
pipeline and what is not in the repo yet: [docs/TRAINING.md](docs/TRAINING.md).
Datasets and how to download them: [docs/DATA.md](docs/DATA.md).
GPU test on Colab: [notebooks/colab_gpu_test.ipynb](notebooks/colab_gpu_test.ipynb).

## Status

| Part | State |
|---|---|
| Vehicle detection, tracking, live-stream runner, events, metrics | Working. Tested on a Colab T4 with a 1080p, 30 fps traffic clip (see below). |
| Orchestration, grammar, consensus, association, config, weights fetching | Unit-tested (81 tests, CPU only). |
| OCR training pipeline (crops, teacher labels, synthetic data, fine-tuning) | In the repo (`src/anpr/ocr/`); unit-tested where it runs on CPU. |
| Plate detector training and composite-data scripts | **Not in the repo** (lost with a Colab runtime); re-added with the retrained weights. |
| Plate detector and OCR weights | Trained earlier but **not yet in the repo**; being retrained and published after submission. The code downloads them automatically once published. |
| End-to-end plate accuracy on video | **Not measured yet.** No claim is made. |

Measured on a Colab T4 (vehicle detector and tracker only, clip played at real
time, two runs on one clip, not a benchmark; drops vary run to run):

| Detector size | p50 / p95 frame time | Frames dropped | Tracks |
|---|---|---|---|
| 640 | 19 / 48 ms | 46 of 1172 (4%) | 28 |
| 1280 | 23 / 63 ms | 96 of 1172 (8%) | 34 |
| 640, second run | 16 / 37 ms | 5 of 1172 | 26 |
| 1280, second run | 21 / 38 ms | 30 of 1172 | 34 |

The tracker finds far fewer vehicles than are visible in this dense scene
(few two-wheelers; COCO has no auto-rickshaw class), so vehicle recall is the main
known weakness of the current vehicle stage.

## Install

```bash
git clone https://github.com/sushovanjena/ANPR.git && cd ANPR
pip install -e . ultralytics rfdetr pytest ruff
pytest -q && ruff check .
```

Python 3.10 or newer. A CUDA GPU is recommended for the models; the tests run on CPU.

## Run

```bash
# Vehicles + tracking only (no plate weights needed)
python -m anpr.run --source traffic.mp4 --no-plates --jsonl out/events.jsonl

# Full pipeline on a recorded file, paced at its frame rate like a camera
python -m anpr.run --config configs/default.yaml --source traffic.mp4

# Live camera with automatic reconnect
python -m anpr.run --config configs/default.yaml --source rtsp://cam/stream --live \
    --camera-id gate-1 --sqlite out/events.db
```

Press Ctrl-C to stop; open tracks are flushed and the final metrics are printed as
JSON (`frames_in`, `frames_processed`, `frames_dropped`, `reconnects`, per-stage
p50/p95 latency).

## Weights

Weights are not stored in git. They are published as GitHub Release assets and listed
with checksums in `weights/manifest.json`.

```bash
python -m anpr.fetch_weights --openocr     # download published weights, clone OpenOCR
```

With `auto_fetch_weights: true` (the default) `python -m anpr.run` does this on first
use. Until the weights are published the full pipeline stops with a clear message;
use `--no-plates` to run the parts that need no trained weights. To use your own
weights, copy them into `weights/` under the names in the manifest, or change the
paths in the config. See [docs/WEIGHTS.md](docs/WEIGHTS.md).

## Events

Each vehicle produces one `confirmed` event when its plate becomes reliable and a
`final` event when it leaves the scene (only if the plate changed or none was read).

```json
{"camera_id": "gate-1", "track_id": 8, "event": "final", "vehicle_class": "car",
 "plate": "DL1CAB1234", "confidence": 0.93, "valid": true, "n_reads": 7,
 "first_seen": 16.9, "last_seen": 20.8, "state": "DL", "plate_box": [412, 233, 498, 262]}
```

`valid` means the string fits the Indian plate grammar (state code, district, series,
number, or Bharat series). Invalid reads are kept, marked, and meant for human review.

## Configuration

`configs/default.yaml` holds every setting (detector size and thresholds, tracker,
quality gate, consensus, queue size, sinks, weight paths). Unknown keys raise an
error so a typo in a deployment config cannot be silently ignored. Any value can be
overridden from the command line for the common ones (`--source`, `--live`,
`--no-plates`, `--camera-id`, `--jsonl`, `--sqlite`, `--max-frames`).

## Repository layout

```
src/anpr/
  pipeline/   vehicle.py, association.py, consensus.py, cascade.py
  track/      bytetrack.py            ByteTrack-style tracker (numpy + scipy)
  models/     plate_detector.py       RF-DETR / YOLO plate detector wrapper
  ocr/        recognizer.py, grammar.py        runtime: RepSVTR adapter, plate grammar
              crops.py, teacher.py, consensus.py, synth.py, finetune.py   training
  stream/     source.py, runner.py, sinks.py, events.py, metrics.py
  eval/       pipeline_eval.py        recall, precision, CER on plate events
  config.py, factory.py, run.py, weights.py, fetch_weights.py
  viz.py      annotated-video renderer (boxes, track ids, plate text) for demos
configs/      default.yaml
weights/      manifest.json           (weight files are downloaded here, not committed)
docs/         ARCHITECTURE.md, WEIGHTS.md, TRAINING.md, DATA.md
scripts/      download_data.py (public datasets)
notebooks/    colab_gpu_test.ipynb, development_log.ipynb
tests/        one test module per component
```

## Code quality

PEP 8 enforced with ruff (`E, F, W, D, I`, Google docstrings); every public function
has a docstring with arguments and returns; models are injected as plain callables,
so each stage is testable without weights or a GPU.

## Credits and licences

- Vehicle detector: Ultralytics YOLO26 (AGPL-3.0). Plate detector: RF-DETR (Apache-2.0).
- OCR: RepSVTR from [OpenOCR](https://github.com/Topdu/OpenOCR), cloned at run time.
- Tracker: algorithm after ByteTrack (Zhang et al., ECCV 2022), reimplemented.
- Test clip: *Dense traffic near Charminar in Hyderabad*, Alexander Bear, Florida Center
  for Instructional Technology, University of South Florida (ETC ClipVideo). Not
  redistributed here; see the [terms](https://etc.usf.edu/clipvideo/license).

Full reference list with arXiv numbers is in `docs/ARCHITECTURE.md`.
