"""Teacher VLM wrapper: transcribes plate crops with Qwen3-VL in 4-bit."""

from __future__ import annotations

import cv2
import torch
from PIL import Image

from anpr.ocr.consensus import parse_vlm_output

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = (
    "This image shows an Indian vehicle number plate. Transcribe exactly the "
    "characters printed on the plate, using uppercase letters and digits only "
    "and no spaces. If the plate has two lines, write the top line followed by "
    "the bottom line. If the plate cannot be read, answer UNREADABLE. "
    "Ignore the IND mark, stickers, dealer names and any other printed "
    "text. Output only the plate text."
)


def load_teacher(model_id: str = MODEL_ID):
    """Load a Qwen-VL model in 4-bit (fits a 16 GB T4) with its processor."""
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model.eval(), processor


def load_image(path: str, min_height: int = 128, max_side: int = 448) -> Image.Image:
    """Load a crop as RGB with its size bounded for the VLM.

    Small crops are upscaled to min_height so the VLM sees enough pixels, and the
    longest side is capped at max_side to bound visual tokens and GPU memory.
    """
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    scale = max(1.0, min_height / h)
    if max(h, w) * scale > max_side:
        scale = max_side / max(h, w)
    if abs(scale - 1.0) > 1e-6:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=interp)
    return Image.fromarray(im)


@torch.inference_mode()
def transcribe(
    model, processor, paths: list[str], batch_size: int = 8, max_new_tokens: int = 24
) -> list[dict]:
    """Transcribe crops greedily and report token-level confidence.

    Args:
        model: Loaded vision-language model.
        processor: Matching processor.
        paths: Crop image paths.
        batch_size: Crops per forward pass.
        max_new_tokens: Generation limit per crop.

    Returns:
        One dict per crop with raw reply, parsed text, mean and minimum token
        probability.
    """
    pad_id = processor.tokenizer.pad_token_id
    message = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
        }
    ]
    prompt = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    results = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        inputs = processor(
            text=[prompt] * len(chunk),
            images=[load_image(p) for p in chunk],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen = out.sequences[:, inputs["input_ids"].shape[1] :]
        probs = torch.exp(
            model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )
        )
        for b in range(len(chunk)):
            keep = gen[b] != pad_id
            p = probs[b][keep]
            raw = processor.tokenizer.decode(gen[b][keep], skip_special_tokens=True)
            results.append(
                {
                    "raw": raw,
                    "text": parse_vlm_output(raw),
                    "conf_mean": float(p.mean()) if p.numel() else 0.0,
                    "conf_min": float(p.min()) if p.numel() else 0.0,
                }
            )
    return results
