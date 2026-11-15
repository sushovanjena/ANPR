from anpr.ocr import grammar


def test_valid_plates():
    for t in ["MH12AB1234", "DL1CAB1234", "KA01AB1234", "22BH1234AA"]:
        assert grammar.is_valid(t)


def test_invalid_state_rejected():
    assert not grammar.is_valid("ZZ12AB1234")


def test_normalise_strips_noise():
    assert grammar.normalise(" mh-12 ab.1234 ") == "MH12AB1234"


def test_correct_confusions():
    assert grammar.correct("KA0IAB1Z34") == ("KA01AB1234", True)
    assert grammar.correct("M H 12 A8 1234") == ("MH12AB1234", True)


def test_correct_leaves_unfixable_alone():
    text, ok = grammar.correct("ZZZ")
    assert text == "ZZZ" and not ok


def test_split_fields():
    f = grammar.split_fields("MH12AB1234")
    assert f == {"state": "MH", "district": "12", "series": "AB", "number": "1234"}
    assert grammar.split_fields("XX") is None
