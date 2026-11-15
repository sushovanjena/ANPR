# Indian ANPR cascade

Vehicle detector, ByteTrack, plate detector, plate-to-vehicle association, quality
gate, RepSVTR OCR and per-track consensus, with a live-stream runner.
Design and references: `docs/ARCHITECTURE.md`. Colab GPU test:
`notebooks/colab_gpu_test.ipynb`.

```
pip install -e . pytest ruff && pytest && ruff check .
python -m anpr.run --config configs/default.yaml --source rtsp://cam/stream --live
python -m anpr.run --source video.mp4 --no-plates      # vehicles + tracking only
```

Plug in trained weights through `configs/default.yaml` (`plate.weights`,
`ocr.weights`, `ocr.dict_path`). Events go to JSONL/SQLite; metrics print on exit.
