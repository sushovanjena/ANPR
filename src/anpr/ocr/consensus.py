"""Text normalisation and agreement logic for plate transcriptions."""

from __future__ import annotations

import re
from collections import Counter

UNREADABLE = "UNREADABLE"
COUNTRY_MARK = "IND"
MIN_LEN, MAX_LEN = 4, 12
_STANDARD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$")
_STANDARD_INNER = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}")
_BH_SERIES = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
_TO_DIGIT = str.maketrans("OIZSBGQD", "01258600")


def normalize_plate(text: str) -> str:
    """Upper-case a transcription and keep only letters and digits."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def matches_indian_format(plate: str) -> bool:
    """Return True if the plate fits a standard Indian registration pattern."""
    return bool(_STANDARD.match(plate) or _BH_SERIES.match(plate))


def strip_country_mark(line: str) -> str:
    """Remove the IND mark printed on HSRP plates from one normalised line."""
    if line == COUNTRY_MARK:
        return ""
    if line.startswith(COUNTRY_MARK) and len(line) - len(COUNTRY_MARK) >= MIN_LEN:
        return line[len(COUNTRY_MARK) :]
    return line


def split_reply(raw: str) -> list[str]:
    """Split a VLM reply into normalised lines with the IND mark removed.

    Returns an empty list for UNREADABLE replies.
    """
    text = raw.strip()
    if UNREADABLE in text.upper():
        return []
    if ":" in text:
        text = text.split(":")[-1]
    lines = [strip_country_mark(normalize_plate(line)) for line in text.splitlines()]
    return [line for line in lines if line]


def parse_vlm_output(raw: str) -> str:
    """Join the lines of a VLM reply into one normalised plate string.

    Length is not used to reject a reply, so extra printed words stay visible for
    review (see clean_label).

    Returns:
        The joined text, or an empty string for UNREADABLE, blank or too-short
        replies.
    """
    plate = "".join(split_reply(raw))
    return plate if len(plate) >= MIN_LEN else ""


def extract_indian_plate(text: str) -> str:
    """Return the standard-pattern plate contained in text, or an empty string."""
    if matches_indian_format(text):
        return text
    found = _STANDARD_INNER.search(text)
    return found.group(0) if found else ""


def repair_confusables(plate: str) -> str:
    """Fix letters misread for digits (O for 0, I for 1, ...) in numeric slots.

    Only the district code and the last four characters are touched. The last
    four must become digits and the result must fit the standard pattern;
    otherwise the input is returned unchanged.
    """
    if matches_indian_format(plate) or len(plate) < 8:
        return plate
    head, middle, tail = plate[:2], plate[2:-4], plate[-4:]
    tail_fixed = tail.translate(_TO_DIGIT)
    if not tail_fixed.isdigit():
        return plate
    candidate = head + middle[:2].translate(_TO_DIGIT) + middle[2:] + tail_fixed
    return candidate if matches_indian_format(candidate) else plate


def _best_slice(lines: list[str]) -> str:
    """Longest run of consecutive lines that forms a standard plate."""
    best = ""
    for i in range(len(lines)):
        for j in range(i + 1, len(lines) + 1):
            candidate = repair_confusables("".join(lines[i:j]))
            if matches_indian_format(candidate) and len(candidate) > len(best):
                best = candidate
    return best


def clean_label(raw: str) -> tuple[str, str]:
    """Turn a raw VLM reply into a (label, note) pair.

    Args:
        raw: Raw model reply.

    Returns:
        The label and a note. The note is empty for a clean read, otherwise
        confusables repaired, extra text removed, empty, non-standard (kept as
        read, for review) or long, unparsed.
    """
    lines = split_reply(raw)
    text = "".join(lines)
    if len(text) < MIN_LEN:
        return "", "empty"
    repaired = repair_confusables(text)
    if matches_indian_format(repaired):
        return repaired, ("confusables repaired" if repaired != text else "")
    best = _best_slice(lines) if len(lines) > 1 else ""
    if best:
        return best, "extra text removed"
    if len(text) > MAX_LEN:
        core = extract_indian_plate(text)
        if core:
            return core, "extra text removed"
        return text, "long, unparsed"
    return text, "non-standard"


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    """Character error rate of a prediction against a reference string."""
    if not ref:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, ref) / len(ref)


def vote(readings: dict[str, str]) -> tuple[str, int]:
    """Majority vote over the non-empty readings of several voters.

    Args:
        readings: Mapping from voter name to normalised plate text.

    Returns:
        The winning text and how many voters produced it, or ("", 0) when no
        voter produced a reading. Ties go to the voter listed first.
    """
    counts = Counter(t for t in readings.values() if t)
    if not counts:
        return "", 0
    best = max(counts.values())
    for text in readings.values():
        if counts.get(text) == best:
            return text, best
    return "", 0
