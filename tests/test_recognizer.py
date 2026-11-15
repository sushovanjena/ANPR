import numpy as np

from anpr.ocr.recognizer import PlateReader, ctc_greedy, split_two_lines


def onehot(seq, n=37):
    x = np.full((len(seq), n), -5.0)
    for t, k in enumerate(seq):
        x[t, k] = 5.0
    return x


def test_ctc_collapses_repeats_and_blanks():
    # blank=0; "AA" separated by a blank is two characters, repeats collapse.
    logits = onehot([0, 11, 11, 0, 11, 12, 0])  # class 11 = 'A', 12 = 'B'
    text, conf = ctc_greedy(logits)
    assert text == "AAB" and conf > 0.9


def test_ctc_all_blank_is_empty():
    assert ctc_greedy(onehot([0, 0, 0])) == ("", 0.0)


def test_split_two_lines_covers_image():
    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    top, bottom = split_two_lines(crop)
    assert top.shape[0] + bottom.shape[0] >= 40


def test_wide_plate_reads_once():
    calls = []

    def ocr(c):
        calls.append(c.shape)
        return "MH12AB1234", 0.9

    out = PlateReader(ocr)(np.zeros((30, 120, 3), dtype=np.uint8))
    assert out[0] == "MH12AB1234" and len(calls) == 1


def test_square_plate_prefers_valid_two_line_join():
    state = {"n": 0}

    def ocr(c):
        # Whole-crop read is garbage; top then bottom lines read cleanly.
        state["n"] += 1
        return [("MH12AB", 0.4), ("MH12", 0.8), ("AB1234", 0.85)][
            min(state["n"] - 1, 2)
        ]

    # The first call is the whole crop.
    text, conf = PlateReader(ocr)(np.zeros((60, 60, 3), dtype=np.uint8))
    assert text == "MH12AB1234" and abs(conf - 0.8) < 1e-9
