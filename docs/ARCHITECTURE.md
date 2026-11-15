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

Measured so far (confidence 0.25, held-out): RF-DETR Nano AP50 0.958/0.965 on
Saisirishan val/test against 0.906/0.904 for YOLO26-n; both weak on tiny plates in
8MP photos (DataCluster, 52 plates). Fine-tuned OCR beat the pretrained model on
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

## 5. Limitations (state these up front)

- Plate detector weights are not in the repo yet; end-to-end accuracy on video is not
  measured. Stages are unit-tested with fakes; the OCR adapter needs its Colab check.
- Grammar repair is a prior: a truncated read can be repaired into a wrong but valid
  string. One-digit districts are accepted only for Delhi.
- Association priors are hand-set and untested on real overlapping two-wheelers.
- OCR forgets non-Indian scene text; a lower learning rate with replay is untested.
- Night, rain and motion blur are covered only by synthetic augmentation.
