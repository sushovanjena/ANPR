import random

from anpr.ocr.consensus import matches_indian_format
from anpr.ocr.synth import degrade, random_plate_text, render_plate


def test_random_text_is_valid_indian_format():
    rng = random.Random(0)
    assert all(matches_indian_format(random_plate_text(rng)) for _ in range(300))


def test_render_and_degrade_return_images():
    rng = random.Random(1)
    for _ in range(10):
        img = degrade(render_plate(random_plate_text(rng), rng), rng)
        assert img.ndim == 3 and img.shape[2] == 3 and min(img.shape[:2]) >= 12


def test_generic_text_is_plain_alphanumeric():
    from anpr.ocr.synth import random_generic_text

    rng = random.Random(2)
    assert all(
        4 <= len(t) <= 9 and t.isalnum() and t.isupper() or t.isdigit()
        for t in (random_generic_text(rng) for _ in range(200))
    )
