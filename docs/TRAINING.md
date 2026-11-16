# Training pipeline

What is in the repository, what is not, and how the pieces fit. Weights produced by
this pipeline are published separately ([WEIGHTS.md](WEIGHTS.md)).

## OCR (in the repo: `src/anpr/ocr/`)

The OCR model is OpenOCR's RepSVTR (a mobile SVTRv2 variant with a CTC head),
fine-tuned for Indian plates with **sequence-level distillation**: a larger
vision-language model reads unlabelled plate crops, its outputs are cleaned and
voted into labels, and the small model learns to reproduce them.

| Step | Module | What it does |
|---|---|---|
| 1. Crops | `crops.py` | Cuts plate crops from YOLO-labelled images (`yolo_to_xyxy`, `extract_crops`). |
| 2. Teacher reads | `teacher.py` | Qwen3-VL in 4-bit transcribes each crop (`load_teacher`, `transcribe`). |
| 3. Label cleaning | `consensus.py` | Normalises replies, strips country marks, repairs look-alike characters, checks the Indian plate format and takes a vote across teacher reads (`clean_label`, `vote`, `cer`). |
| 4. Synthetic plates | `synth.py` | Renders Indian plates (private, commercial, rental, EV, Bharat series, one and two lines) with occlusion, perspective, blur, resolution loss, noise and JPEG artefacts (`generate_dataset`). |
| 5. Fine-tune | `finetune.py` | AdamW with cosine schedule, EMA weights and early stopping on validation CER (`train`). |

Training needs a GPU, `torch`, `transformers`, `bitsandbytes` and a clone of OpenOCR;
these are not in the default install because the runtime pipeline does not need them.
The unit tests cover steps 1, 3 and 4 on CPU.

Known limit: after fine-tuning the model loses accuracy on non-plate scene text
(RoadText 71% to about 24%); a lower learning rate with replay data is untested.

## Plate detector

The detector was trained with RF-DETR Nano (and YOLO26-n for comparison) on Indian
plate datasets plus **composite images**: plates pasted onto real vehicle photos at
realistic scale to create small-plate examples (Simple Copy-Paste, arXiv:2012.07177).

Recipe used for the published weights (`weights-v1`): RF-DETR Nano from the
COCO-pretrained `rf-detr-nano.pth`, trained on 4,743 images (Kaggle train plus
composite small-plate images), validated on 678 Saisirishan images, 20 epochs
maximum, batch 16, EMA weights, early stopping with patience 5 (stopped at epoch 15,
best epoch 9), a checkpoint every epoch. Held-out results are in
[WEIGHTS.md](WEIGHTS.md); the detection metrics code is
`src/anpr/eval/detection.py`.

The composite generator and the training launch are still notebook cells, not
standalone scripts: see Part 7 of
[`notebooks/development_log.ipynb`](../notebooks/development_log.ipynb) (dataset
rebuild, training, evaluation, release upload) and the earlier cells for the composite
data. Turning them into scripts under `scripts/` is open work.

## Dataset inventory

`src/anpr/data/inventory.py` scans a dataset folder and summarises images, labels and
box sizes, to check data before training.
