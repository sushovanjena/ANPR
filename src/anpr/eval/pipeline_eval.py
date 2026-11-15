"""End-to-end evaluation of plate events against ground-truth plate lists.

Ground truth for a video is the set of plate strings that are legible in it.
Events are scored on the final decision per vehicle, so the numbers reflect
what a downstream user would actually receive, including the consensus step.
"""

from __future__ import annotations

import json
from pathlib import Path


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_events(path: str | Path) -> list[dict]:
    """Read a JSON-lines event file."""
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def evaluate_events(events: list[dict], gt_plates: list[str]) -> dict:
    """Score events against the plates that are legible in the video.

    One decision per track is used (the ``final`` event if present, otherwise
    the ``confirmed`` one).

    Args:
        events: Event dicts as written by the sinks.
        gt_plates: Plate strings visible in the video.

    Returns:
        Counts and rates: ``recall`` (ground-truth plates read exactly),
        ``precision`` (valid, reported plates that are correct), ``cer``
        (character error rate of each ground-truth plate against its closest
        prediction), ``unread_vehicles`` (tracks without a plate) and
        ``review_rate`` (reads that failed the plate grammar).
    """
    per_track: dict[tuple[str, int], dict] = {}
    for e in events:
        key = (e["camera_id"], e["track_id"])
        if key not in per_track or e["event"] == "final":
            per_track[key] = e
    decisions = list(per_track.values())
    preds = [d["plate"] for d in decisions if d["plate"]]
    valid = [d["plate"] for d in decisions if d["plate"] and d["valid"]]
    gt = list(dict.fromkeys(gt_plates))

    exact = sum(g in preds for g in gt)
    correct_valid = sum(p in gt for p in valid)
    cers = []
    for g in gt:
        best = min((edit_distance(g, p) for p in preds), default=len(g))
        cers.append(best / max(len(g), 1))
    return {
        "vehicles": len(decisions),
        "gt_plates": len(gt),
        "recall": exact / len(gt) if gt else 0.0,
        "precision": correct_valid / len(valid) if valid else 0.0,
        "cer": sum(cers) / len(cers) if cers else 0.0,
        "unread_vehicles": sum(d["plate"] is None for d in decisions),
        "review_rate": (len(preds) - len(valid)) / len(preds) if preds else 0.0,
    }
