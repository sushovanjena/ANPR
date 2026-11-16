# Indian ANPR: architecture and design decisions

Target: open-source ANPR for Indian multi-lane traffic. Evaluated on a Colab GPU
(no ONNX/TensorRT export in scope). Detector weights are trained last and plugged
in by path.

## 1. Pipeline

```
camera / file ──► FrameGrabber thread ──► LatestQueue (drop-oldest, size 2)
                                              │
                                              ▼
 vehicle detector (COCO) ─► ByteTrack ─► per vehicle crop ─► plate detector
                                                                   │
                       plate↔vehicle association (Hungarian) ◄─────┘
                                   │
                       quality gate (size, blur) ─► OCR (RepSVTR, CTC)
                                   │                 + two-line reader
                                   ▼
        per-track consensus + Indian plate grammar ─► PlateEvent ─► sinks (JSONL, SQLite, callbacks)
```

A cascade rather than one end-to-end model: plates in a multi-lane scene cover
about 2% of the frame, so reading them from vehicle crops gives the plate
detector far more resolution, and the vehicle is the primary object (a plate with
no vehicle is dropped; a vehicle with no readable plate is still reported).

## 2. Plate-to-vehicle association

**What we do.** For each tracked vehicle the plate detector runs on its crop, and
candidate plates are matched to the vehicle by an *affinity* that multiplies
(a) containment of the plate box in the vehicle box, (b) a smooth size prior
(plate area as a fraction of the vehicle box) and (c) a position prior (lower half,
near the horizontal centre), weighted by detector confidence. Matching is solved
one-to-one with the Hungarian algorithm, so a plate that fits two overlapping
vehicles goes to the one it explains best and no vehicle gets two plates. Plates that
fit no vehicle are discarded. Association is then made temporal: every track pools
reads over its lifetime (consensus), so one wrong frame cannot reassign a plate.

**Why this and not a learned association head.** It needs no extra labels, adds
microseconds, and each term is explainable and testable. The cost is that the priors
are hand-set. The state of the art learns the association instead, and is the natural
upgrade if the rule-based version underperforms on a labelled vehicle-plate set.

**State-of-the-art references (part-to-body association).**
Vehicle-plate pairing is the same problem as hand/face-to-person association, which
has strong one-stage solutions:

- BPJDet: Zhou et al., *Body-Part Joint Detection and Association via Extended Object
  Representation*, arXiv:2212.07652; extended in *BPJDet: Extended Object
  Representation for Generic Body-Part Joint Detection*, IEEE TPAMI 2023
  (arXiv:2304.10765). Each body box also regresses the centre of its part, and parts
  are assigned to the body whose predicted centre is nearest.
- PBADet: Gao et al., *A One-Stage Anchor-Free Approach for Part-Body Association*,
  ICLR 2024, arXiv:2402.07814. A single part-to-body centre offset per part; scales
  to many parts and does not depend on a fixed number of parts per body.

Applied to vehicles: add a head that predicts, for each plate, the offset to its
vehicle's centre; association becomes nearest-predicted-centre matching, which is
trained jointly with detection and needs no hand priors.

**Vehicle-plate specific work.**
- Laroca et al., *An efficient and layout-independent automatic license plate
  recognition system based on the YOLO detector*, IET Intelligent Transport Systems,
  2021 (arXiv:1909.01754): vehicle detection first, plate detection inside the
  vehicle, the same cascade used here.
- *End-to-end trainable network for degraded license plate detection via
  vehicle-plate relation mining* (see Papers with Code) and *Simultaneous End-to-End
  Vehicle and License Plate Detection With Multi-Branch Attention Neural Network*
  (IEEE T-ITS): learn the vehicle-plate relation to help detect degraded or small
  plates.

**Matching.** Kuhn, *The Hungarian method for the assignment problem*, 1955
(`scipy.optimize.linear_sum_assignment`).

## 3. Components and sources

| Stage | Choice | Reference |
|---|---|---|
| Vehicle detector | COCO-pretrained YOLO26-n (cars, motorcycles, buses, trucks); no fine-tuning | Ultralytics YOLO26 docs (AGPL-3.0) |
| Tracker | ByteTrack-style two-stage matching with a Kalman filter | Zhang et al., *ByteTrack*, ECCV 2022, arXiv:2110.06864 |
| Plate detector | RF-DETR Nano (Apache-2.0) and YOLO26-n compared | Robinson et al., *RF-DETR*, arXiv:2511.09554 |
| Plate detector data | Kaggle plates plus composites (plates pasted onto real vehicles at realistic scale) | Ghiasi et al., *Simple Copy-Paste*, CVPR 2021, arXiv:2012.07177 |
| OCR | RepSVTR (mobile SVTRv2 family), CTC head, fine-tuned by sequence-level distillation | Du et al., *SVTR*, IJCAI 2022, arXiv:2205.00159; *SVTRv2*, ICCV 2025, arXiv:2411.15858; Graves et al., *CTC*, ICML 2006; Kim and Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016, arXiv:1606.07947 |
| Consensus | Confidence-weighted voting across a track plus plate-grammar repair | this work |

Measured (confidence 0.25, held-out), published RF-DETR Nano weights: AP50 0.960 /
0.958 on Saisirishan val/test and 0.967 on Kaggle test, but only 0.514 on DataCluster
(52 plates; recall 0.15 for plates under 0.5% of the image). An earlier run gave
YOLO26-n 0.906/0.904 on Saisirishan val/test; that model was not retrained. Fine-tuned OCR beat the pretrained model on
Indian plates but lost accuracy on RoadText scene text (71% to about 24%).

## 4. Live-stream design

- **Bounded latency.** The grabber thread writes to a size-2 queue that drops the
  oldest frame, so a slow GPU skips frames instead of lagging behind the camera.
  Dropped frames are counted.
- **Resilience.** RTSP read failures trigger reconnect with exponential backoff;
  sink failures are isolated and counted; SIGINT/SIGTERM flush open tracks.
- **Low latency and no duplicates.** A track emits a `confirmed` event the moment its
  consensus is confident, then stops being re-read (saves compute); a `final` event
  is sent on exit only if the plate changed or none was found.
- **Observability.** Per-stage p50/p95 latency (`vehicle_ms`, `plate_ms`, `ocr_ms`,
  `frame_ms`), FPS in/processed, drops and reconnects.
- **Replay as live.** `realtime: true` paces a recording at its source FPS, so a file
  benchmarks like a camera.

## 5. Live test on Colab

Setup: Colab T4, vehicle detector (YOLO26-n, COCO) and tracker only, a 39 s 1080p
30 fps clip of dense traffic near Charminar played at real time through the live
runner. Two runs on one clip; frame drops vary run to run, so treat the numbers as indicative, not a benchmark.

| Detector size | p50 / p95 frame time | Frames dropped | Tracks |
|---|---|---|---|
| 640 | 19 / 48 ms | 46 of 1172 | 28 |
| 1280 | 23 / 63 ms | 96 of 1172 | 34 (3 motorcycles) |
| 640, second run | 16 / 37 ms | 5 of 1172 | 26 |
| 1280, second run | 21 / 38 ms | 30 of 1172 | 34 (4 motorcycles) |

Findings:
- **Throughput holds.** The runner keeps up with a 30 fps stream at both sizes, with
  drops from the bounded queue rather than growing lag.
- **Cold start matters.** The first call of the detector took about 16 s (weights and
  CUDA initialisation) and dropped about 480 frames. The runner now warms every
  stage before the stream opens, which cut drops from 506 to 46.
- **Vehicle recall is the weak point.** The scene shows many more vehicles than were
  tracked, almost no two-wheelers, and COCO has no auto-rickshaw class (they appear
  as car or truck). A detector fine-tuned for Indian vehicles is the next step.
- **Full cascade does not keep up with 30 fps on a T4.** With the published plate
  and OCR weights, one real-time run with the plain RF-DETR model processed 5.8 of
  29.7 fps (943 of 1172 frames dropped); per stage p50 / p95: vehicle 35 / 131 ms,
  plate detector 50 / 184 ms, OCR 46 / 204 ms, whole frame 74 / 608 ms.
- **Compiled inference predictor helps a lot.** Using RF-DETR's `inference()`
  predictor (now the default, with fallback) one run gave vehicle 20 / 41 ms,
  plate detector 14 / 20 ms, OCR 27 / 118 ms, whole frame 40 / 73 ms, and 11.5 fps
  processed (724 of 1172 dropped). Still below real time; OCR tail latency and
  per-vehicle plate calls are the next costs. Latency stays bounded by the
  drop-oldest queue.
- **Not measured:** end-to-end plate accuracy. The clip has no ground-truth plate
  list; 25 events were written and their correctness is unchecked.

## 6. Limitations (state these up front)

- End-to-end accuracy on video is not measured, and the full cascade is below real
  time on a T4 (section 5). Stages are unit-tested with fakes; the OCR adapter was written against
  the OpenOCR API and now runs on the real checkpoint in the full-pipeline test (section 5),
  but its accuracy on video is unmeasured.
- Grammar repair is a prior: a truncated read can be repaired into a wrong but valid
  string. One-digit districts are accepted only for Delhi.
- Association priors are hand-set and untested on real overlapping two-wheelers.
- OCR forgets non-Indian scene text; a lower learning rate with replay is untested.
- Night, rain and motion blur are covered only by synthetic augmentation.
