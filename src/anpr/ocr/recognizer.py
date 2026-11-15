"""Plate OCR adapter: fine-tuned RepSVTR (CTC) with two-line plate handling.

The recogniser is built from the OpenOCR code base (Apache-2.0) and the
fine-tuned weights, so it uses exactly the architecture, dictionary and
preprocessing that the fine-tune was trained and validated with.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from anpr.ocr import grammar

OcrFn = Callable[[np.ndarray], tuple[str, float]]

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def ctc_greedy(logits: np.ndarray, alphabet: str = ALPHABET) -> tuple[str, float]:
    """Greedy CTC decode with blank at index 0.

    Args:
        logits: ``(T, C)`` scores, with class 0 the blank and class ``i + 1``
            mapping to ``alphabet[i]``. Probabilities and raw scores both work
            because only the arg-max and a softmax-normalised score are used.
        alphabet: Characters excluding the blank.

    Returns:
        ``(text, confidence)`` where confidence is the mean probability of the
        emitted characters (0 for an empty read).
    """
    x = np.asarray(logits, dtype=float)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    idx = p.argmax(axis=1)
    out, probs, prev = [], [], 0
    for t, k in enumerate(idx):
        if k != 0 and k != prev:
            out.append(alphabet[k - 1])
            probs.append(p[t, k])
        prev = k
    return "".join(out), float(np.mean(probs)) if probs else 0.0


class RepSvtrRecognizer:
    """Fine-tuned RepSVTR plate recogniser.

    Args:
        openocr_dir: Path to a clone of ``Topdu/OpenOCR``.
        weights: Fine-tuned ``best_ema.pt`` (a state dict with a 37-class head).
        dict_path: ``plate_dict.txt`` saved next to the weights.
        device: ``"cuda"``, ``"cpu"`` or None for automatic.
    """

    def __init__(
        self,
        openocr_dir: str | Path,
        weights: str | Path,
        dict_path: str | Path,
        device: str | None = None,
    ) -> None:
        self.openocr_dir = Path(openocr_dir)
        self.weights = Path(weights)
        self.dict_path = Path(dict_path)
        for p in (self.openocr_dir, self.weights, self.dict_path):
            if not p.exists():
                raise FileNotFoundError(f"OCR resource not found: {p}")
        self.device = device
        self._ready = False

    def _build(self) -> None:
        import torch

        sys.path.insert(0, str(self.openocr_dir))
        from openrec.modeling import build_model
        from openrec.postprocess import build_post_process
        from openrec.preprocess import create_operators, transform
        from tools.engine.config import Config
        from tools.infer_det import replace_batchnorm

        cfg = Config(str(self.openocr_dir / "configs/rec/svtrv2/repsvtr_ch.yml")).cfg
        cfg["Global"].update(
            character_dict_path=str(self.dict_path),
            use_space_char=False,
            max_text_length=16,
        )
        cfg["PostProcess"]["character_dict_path"] = str(self.dict_path)
        self.post = build_post_process(cfg["PostProcess"], cfg["Global"])
        cfg["Architecture"]["Decoder"]["out_channels"] = self.post.get_character_num()
        self.model = build_model(cfg["Architecture"])
        state = torch.load(self.weights, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        self.model.load_state_dict(state, strict=True)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch, self.dev = torch, torch.device(dev)
        self.model.to(self.dev).eval()
        replace_batchnorm(self.model.encoder)
        ops = [
            op
            for op in cfg["Eval"]["dataset"]["transforms"]
            if "Label" not in next(iter(op))
        ]
        for op in ops:
            name = next(iter(op))
            if name == "KeepKeys":
                op[name]["keep_keys"] = ["image"]
        self.ops = create_operators(ops, cfg["Global"])
        self.transform = transform
        self._ready = True

    def __call__(self, crop: np.ndarray) -> tuple[str, float]:
        """Read a BGR plate crop and return ``(text, confidence)``."""
        if not self._ready:
            self._build()
        batch = self.transform({"image": crop}, self.ops[1:])
        img = batch[0] if isinstance(batch[0], np.ndarray) else batch[0].numpy()
        x = self.torch.from_numpy(img[None].astype(np.float32)).to(self.dev)
        with self.torch.no_grad():
            preds = self.model(x)
        text, score = self.post(preds, torch_tensor=True)[0]
        return grammar.normalise(text), float(score)


def split_two_lines(crop: np.ndarray, overlap: float = 0.05) -> list[np.ndarray]:
    """Split a roughly square plate crop into its top and bottom lines."""
    h = crop.shape[0]
    mid = h // 2
    pad = max(1, int(h * overlap))
    return [crop[: mid + pad], crop[mid - pad :]]


class PlateReader:
    """Single- or two-line plate reading on top of any line recogniser.

    Standard car plates are wide (about 4:1) and read as one line. Two-wheeler
    plates are often square and carry the state and district on the top line
    and the series and number below. For crops that are not clearly wide, both
    hypotheses are read (whole crop, and top plus bottom joined) and the one
    that fits the Indian plate grammar wins; if both or neither fit, the more
    confident read wins.

    Args:
        ocr_fn: Line recogniser mapping a crop to ``(text, confidence)``.
        two_line_aspect: Crops with width/height below this also try the
            two-line hypothesis.
    """

    def __init__(self, ocr_fn: OcrFn, two_line_aspect: float = 2.6) -> None:
        self.ocr_fn = ocr_fn
        self.two_line_aspect = two_line_aspect

    def __call__(self, crop: np.ndarray) -> tuple[str, float]:
        """Return the best ``(text, confidence)`` for a plate crop."""
        h, w = crop.shape[:2]
        single = self.ocr_fn(crop)
        if h < 8 or w / max(h, 1) >= self.two_line_aspect:
            return single
        top, bottom = (self.ocr_fn(c) for c in split_two_lines(crop))
        joined = (top[0] + bottom[0], min(top[1], bottom[1]))
        cands = [single, joined]
        valid = [c for c in cands if grammar.correct(c[0])[1]]
        pool = valid or cands
        return max(pool, key=lambda c: c[1])
