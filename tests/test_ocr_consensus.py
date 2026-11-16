from anpr.ocr.consensus import (
    cer,
    clean_label,
    edit_distance,
    matches_indian_format,
    normalize_plate,
    parse_vlm_output,
    repair_confusables,
    vote,
)

NL = chr(10)


def test_normalize_plate():
    assert normalize_plate("mh 12-ab.1234") == "MH12AB1234"


def test_parse_handles_labels_unreadable_and_short():
    assert parse_vlm_output("Plate: MH 12 AB 1234") == "MH12AB1234"
    assert parse_vlm_output("UNREADABLE") == ""
    assert parse_vlm_output("unreadable") == ""
    assert parse_vlm_output("A1") == ""


def test_parse_keeps_long_replies_for_review():
    assert parse_vlm_output("x" * 30) == "X" * 30


def test_parse_drops_country_mark():
    assert parse_vlm_output("INDMH01AE8017") == "MH01AE8017"
    assert parse_vlm_output("UP65" + NL + "INDCW9680") == "UP65CW9680"
    assert parse_vlm_output("IND" + NL + "MH12AB1234") == "MH12AB1234"


def test_indian_format():
    assert matches_indian_format("MH12AB1234")
    assert matches_indian_format("DL3SBS9880")
    assert matches_indian_format("22BH1234AA")
    assert not matches_indian_format("1234ABCD")


def test_repair_confusables_only_when_it_helps():
    assert repair_confusables("MHO1AE8017") == "MH01AE8017"
    assert repair_confusables("MH01AE8017") == "MH01AE8017"
    assert repair_confusables("IOCGO470") == "IOCGO470"
    assert repair_confusables("KA03MYOO") == "KA03MYOO"


def test_clean_label_cases():
    assert clean_label("KL55X5721ROYALENFIELD") == ("KL55X5721", "extra text removed")
    assert clean_label("UNREADABLE") == ("", "empty")
    assert clean_label("KL31PTEMP7884") == ("KL31PTEMP7884", "long, unparsed")
    assert clean_label("INDMHO1AE8017") == ("MH01AE8017", "confusables repaired")
    assert clean_label("KA03MYOO") == ("KA03MYOO", "non-standard")
    assert clean_label("MH12AB1234") == ("MH12AB1234", "")


def test_clean_label_picks_the_plate_line():
    assert clean_label("GREWAL" + NL + "PBI9N4746") == (
        "PB19N4746",
        "extra text removed",
    )
    assert clean_label("MH14DX0087" + NL + "mycar") == (
        "MH14DX0087",
        "extra text removed",
    )
    assert clean_label("UP65" + NL + "INDCW9680") == ("UP65CW9680", "")


def test_edit_distance_and_cer():
    assert edit_distance("KITTEN", "SITTING") == 3
    assert cer("MH12", "MH12") == 0.0
    assert cer("", "") == 0.0
    assert cer("A", "") == 1.0


def test_vote_majority_and_tie_break():
    assert vote({"a": "X1", "b": "X1", "c": "Y2"}) == ("X1", 2)
    assert vote({"a": "X1", "b": "Y2"}) == ("X1", 1)
    assert vote({"a": "", "b": ""}) == ("", 0)
