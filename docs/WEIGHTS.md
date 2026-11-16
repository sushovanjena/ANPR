# Model weights

Weights are too large for git and are not committed. The code fetches them from a
GitHub Release by name, and checks each download against a SHA-256 recorded in
`weights/manifest.json`.

## Files

| File | Role |
|---|---|
| `plate_rfdetr_nano.pth` | Plate detector, RF-DETR Nano (default) |
| `plate_yolo26n.pt` | Plate detector, YOLO26-n (alternative; set `plate.kind: yolo`) |
| `ocr_repsvtr_v2.pt` | RepSVTR OCR state dict, 36 characters plus blank (37-class head) |
| `plate_dict.txt` | OCR dictionary, one character per line |

The vehicle detector (`yolo26n.pt`, COCO) is downloaded by Ultralytics automatically.

## How users get weights

```bash
python -m anpr.fetch_weights --openocr
```

or simply run the pipeline; `auto_fetch_weights: true` downloads on first use. A file
with no URL in the manifest is reported as "not published yet" and skipped.

## How to publish after training

1. Copy the trained files into `weights/` under the names above.
2. Print each checksum: `python -m anpr.fetch_weights --hash weights/ocr_repsvtr_v2.pt`.
3. Create a GitHub Release with tag `weights-v1` and upload the files as assets.
4. In `weights/manifest.json` set each file's `sha256` to the printed value.
   Leave `url` as `null`: once a checksum is present, the URL is built as
   `https://github.com/<repo>/releases/download/<tag>/<file>`. Set `url` explicitly
   to host elsewhere (Hugging Face, S3).
5. Commit the manifest. No code change is needed.

If a release has to be replaced, bump `release_tag` (for example `weights-v2`) so old
checkouts keep working with their recorded checksums.

## Checks done on download

The file is written to a `.part` file, hashed, and renamed only if the hash matches;
a mismatch deletes it and raises an error.

## Training recipes (for reproducing the weights)

- **OCR:** RepSVTR (mobile SVTRv2) fine-tuned on Indian plate crops with synthetic
  plates and sequence-level distillation from a larger recogniser's outputs. Known
  limit: it loses accuracy on non-plate scene text (RoadText 71% to about 24%).
- **Plate detector:** RF-DETR Nano on Indian plate datasets plus composites (plates
  pasted onto real vehicle photos at realistic scale). In earlier held-out tests it
  beat YOLO26-n on the Saisirishan set (AP50 0.958 / 0.965 against 0.906 / 0.904,
  val / test) and was weak for both on tiny plates in 8MP photos.

## Published release (`weights-v1`)

Published: `plate_rfdetr_nano.pth`, `ocr_repsvtr_v2.pt`, `plate_dict.txt`. The YOLO26-n
plate detector (`plate_yolo26n.pt`) was not retrained and is not published.

The detector was retrained from the COCO-pretrained RF-DETR Nano on 4,743 images
(Kaggle train plus composite small-plate images), validated on 678 Saisirishan
images, stopped early at epoch 15 with the best epoch at 9. Held-out results at a
0.25 confidence threshold:

| Set | Plates | AP50 | AP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| Saisirishan val | 678 | 0.960 | 0.689 | 0.927 | 0.932 |
| Saisirishan test | 1,006 | 0.958 | 0.684 | 0.921 | 0.932 |
| Kaggle test | 319 | 0.967 | 0.797 | 0.949 | 0.940 |
| DataCluster | 52 | 0.514 | 0.391 | 0.743 | 0.500 |

DataCluster is a small, harder set (mixed plate sizes, unseen in training); recall
there is 0.15 for plates under 0.5% of the image area.
