"""Fine-tune OpenOCR's RepSVTR recogniser on plate text (EMA, early stopping)."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from anpr.ocr.consensus import cer, normalize_plate

OPENOCR = Path("/content/OpenOCR")
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def write_gt(pairs: list[tuple[str, str]], path: Path) -> Path:
    """Write a tab-separated label file (image path, text) for OpenOCR."""
    Path(path).write_text(chr(10).join(p + chr(9) + t for p, t in pairs) + chr(10))
    return Path(path)


def make_config(
    out_yml: Path,
    train_gt: Path,
    val_gt: Path,
    out_dir: Path,
    epochs: int = 25,
    batch: int = 96,
    orig_dict: bool = False,
    pretrained: str | None = None,
) -> Path:
    """Derive a fine-tuning config from OpenOCR's RepSVTR config."""
    cfg = yaml.safe_load((OPENOCR / "configs/rec/svtrv2/repsvtr_ch.yml").read_text())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if orig_dict:
        dict_path = str(OPENOCR / "tools/utils/ppocr_keys_v1.txt")
        max_len, space = 25, True
    else:
        dict_path = str(Path(out_dir) / "plate_dict.txt")
        Path(dict_path).write_text(chr(10).join(ALPHABET) + chr(10))
        max_len, space = 16, False
    cfg["Global"].update(
        {
            "epoch_num": epochs,
            "output_dir": str(out_dir),
            "pretrained_model": pretrained,
            "use_amp": False,
            "cal_metric_during_train": False,
            "character_dict_path": dict_path,
            "max_text_length": max_len,
            "use_space_char": space,
            "save_epoch_step": [100000, 100000],
        }
    )
    cfg["PostProcess"]["character_dict_path"] = dict_path
    for part in ("Train", "Eval"):
        ds = cfg[part]["dataset"]
        ds["data_dir"] = ""
        for op in ds["transforms"]:
            if "CTCLabelEncode" in op:
                op["CTCLabelEncode"].update(
                    {
                        "character_dict_path": dict_path,
                        "use_space_char": space,
                        "max_text_length": max_len,
                    }
                )
    cfg["Train"]["dataset"]["label_file_list"] = [str(train_gt)]
    cfg["Eval"]["dataset"]["label_file_list"] = [str(val_gt)]
    cfg["Train"]["loader"].update({"batch_size_per_card": batch, "num_workers": 2})
    cfg["Eval"]["loader"].update({"num_workers": 2})
    Path(out_yml).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return Path(out_yml)


def load_trainer(cfg_yml: Path, mode: str = "train_eval"):
    """Build OpenOCR's Trainer (model, loss, post-process, data loaders)."""
    if str(OPENOCR) not in sys.path:
        sys.path.insert(0, str(OPENOCR))
    from tools.engine.config import Config
    from tools.engine.trainer import Trainer

    cfg = Config(str(cfg_yml))
    cfg.cfg["Global"]["distributed"] = False
    return Trainer(cfg, mode=mode, task="rec")


def load_pretrained_sliced(model, ckpt_path: str) -> dict:
    """Load pretrained weights, slicing the CTC classifier to the plate alphabet."""
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    lines = open(
        OPENOCR / "tools/utils/ppocr_keys_v1.txt", encoding="utf-8"
    ).readlines()
    chars = [ln.rstrip(chr(10) + chr(13)) for ln in lines] + [" "]
    rows = [0] + [1 + chars.index(c) for c in ALPHABET]
    msd = model.state_dict()
    keep, skipped = {}, []
    for k, v in sd.items():
        if k in ("decoder.fc.weight", "decoder.fc.bias"):
            keep[k] = v[rows].clone()
        elif k in msd and msd[k].shape == v.shape:
            keep[k] = v
        else:
            skipped.append(k)
    missing, _ = model.load_state_dict(keep, strict=False)
    return {
        "loaded": len(keep),
        "skipped": len(skipped),
        "missing": len(missing),
        "skipped_examples": skipped[:5],
    }


def make_eval_loader(trainer, gt_path: Path):
    """Build an evaluation loader over another label file."""
    from tools.data import build_dataloader

    cfg = copy.deepcopy(trainer.cfg)
    cfg["Eval"]["dataset"]["label_file_list"] = [str(gt_path)]
    return build_dataloader(cfg, "Eval", trainer.logger, task="rec")


def _to_float(x):
    if torch.is_tensor(x):
        return x.float()
    if isinstance(x, dict):
        return {k: _to_float(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_float(v) for v in x)
    return x


@torch.no_grad()
def predict(model, post, loader, device) -> tuple[list[str], list[str]]:
    """Return normalised predictions and ground-truth strings for a loader."""
    model.eval()
    preds, gts = [], []
    for batch in loader:
        bt = [t.to(device) for t in batch]
        bn = [t.numpy() for t in batch]
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(bt[0], data=bt[1:])
        text, label = post(_to_float(out), bn)
        preds += [normalize_plate(t[0]) for t in text]
        gts += [normalize_plate(lab[0]) for lab in label]
    return preds, gts


def score(preds: list[str], gts: list[str]) -> dict:
    """Exact-match accuracy (percent) and mean character error rate."""
    exact = float(np.mean([p == g for p, g in zip(preds, gts)])) * 100
    return {
        "n": len(gts),
        "exact": round(exact, 1),
        "cer": round(float(np.mean([cer(p, g) for p, g in zip(preds, gts)])), 4),
    }


def train(
    trainer,
    epochs: int = 25,
    lr: float = 2e-4,
    head_mult: float = 5.0,
    wd: float = 0.05,
    warmup: int = 2,
    ema_decay: float = 0.998,
    patience: int = 5,
    out_dir: Path = Path("/content/ocr_runs/ft"),
    log=print,
) -> list[dict]:
    """Fine-tune with AdamW, cosine schedule, EMA weights and val-CER early stopping."""
    model, device = trainer.model, trainer.device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = [p for n, p in model.named_parameters() if not n.startswith("decoder")]
    dec = [p for n, p in model.named_parameters() if n.startswith("decoder")]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": lr}, {"params": dec, "lr": lr * head_mult}],
        weight_decay=wd,
    )
    steps = len(trainer.train_dataloader)
    total, warm = epochs * steps, warmup * steps

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    ema = copy.deepcopy(model).eval()
    best, bad, hist, step = 1e9, 0, [], 0
    for ep in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for batch in trainer.train_dataloader:
            bt = [t.to(device) for t in batch]
            with torch.autocast("cuda", dtype=torch.float16):
                preds = model(bt[0], data=bt[1:])
            loss = trainer.loss_class(_to_float(preds), bt)["loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            d = min(ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(
                    ema.state_dict().values(), model.state_dict().values()
                ):
                    if pe.dtype.is_floating_point:
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    else:
                        pe.copy_(pm)
            losses.append(float(loss))
        preds_v, gts_v = predict(
            ema, trainer.post_process_class, trainer.valid_dataloader, device
        )
        sc = score(preds_v, gts_v)
        rec = {
            "epoch": ep + 1,
            "train_loss": round(float(np.mean(losses)), 4),
            "val_exact": sc["exact"],
            "val_cer": sc["cer"],
            "secs": round(time.time() - t0),
        }
        hist.append(rec)
        log(json.dumps(rec))
        if sc["cer"] < best - 1e-4:
            best, bad = sc["cer"], 0
            torch.save(ema.state_dict(), out_dir / "best_ema.pt")
        else:
            bad += 1
            if bad >= patience:
                log("early stopping at epoch " + str(ep + 1))
                break
    (out_dir / "history.json").write_text(json.dumps(hist))
    ema.load_state_dict(torch.load(out_dir / "best_ema.pt", map_location=device))
    trainer.model = ema
    return hist
