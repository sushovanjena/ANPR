"""Per-track temporal consensus over noisy plate reads.

A vehicle is seen in many frames, and each frame yields one OCR string with a
confidence. Instead of trusting the best single frame, we pool the reads:
each read votes for its grammar-corrected string, weighted by confidence and a
bonus when the read already fits the plate grammar. Disagreeing strings that
share a layout are then resolved character by character.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from anpr.ocr import grammar


@dataclass(frozen=True)
class Read:
    """One OCR result for a plate crop."""

    text: str
    conf: float
    frame: int = 0


@dataclass
class TrackReads:
    """Accumulates reads for one tracked vehicle."""

    reads: list[Read] = field(default_factory=list)

    def add(self, text: str, conf: float, frame: int = 0) -> None:
        """Store a read; empty strings are ignored."""
        if grammar.normalise(text):
            self.reads.append(Read(text, float(conf), frame))


@dataclass(frozen=True)
class Consensus:
    """Final plate decision for a track."""

    text: str
    score: float
    valid: bool
    n_reads: int


def _char_vote(reads: list[tuple[str, float]]) -> str:
    """Confidence-weighted majority vote per character position."""
    length = len(reads[0][0])
    out = []
    for i in range(length):
        tally: dict[str, float] = defaultdict(float)
        for text, w in reads:
            tally[text[i]] += w
        out.append(max(tally, key=tally.get))
    return "".join(out)


def decide(
    track: TrackReads,
    valid_bonus: float = 1.0,
    min_conf: float = 0.2,
) -> Consensus | None:
    """Combine a track's reads into one plate string.

    Args:
        track: Reads collected over the lifetime of a vehicle track.
        valid_bonus: Extra weight given to reads that fit the grammar after
            correction, so that clean reads outvote garbled ones.
        min_conf: Reads below this confidence are ignored.

    Returns:
        The decision, or ``None`` if no usable read exists. ``valid`` is False
        when the winning string does not fit the Indian plate grammar, which
        the caller should treat as "needs review" and not a confident result.
    """
    weighted: list[tuple[str, float, bool]] = []
    for r in track.reads:
        if r.conf < min_conf:
            continue
        text, ok = grammar.correct(r.text)
        weighted.append((text, r.conf + (valid_bonus if ok else 0.0), ok))
    if not weighted:
        return None

    by_text: dict[str, float] = defaultdict(float)
    for text, w, _ in weighted:
        by_text[text] += w
    best_text = max(by_text, key=by_text.get)

    # Character vote among reads with the same length as the leader.
    same_len = [(t, w) for t, w, _ in weighted if len(t) == len(best_text)]
    voted = _char_vote(same_len) if len(same_len) >= 3 else best_text
    voted, ok = grammar.correct(voted)
    if not ok:
        voted = best_text
        ok = grammar.is_valid(best_text)

    total = sum(w for _, w, _ in weighted)
    support = sum(w for t, w, _ in weighted if t == voted)
    return Consensus(voted, support / total if total else 0.0, ok, len(weighted))
