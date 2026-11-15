from anpr.pipeline.consensus import TrackReads, decide


def make(*pairs):
    tr = TrackReads()
    for i, (t, c) in enumerate(pairs):
        tr.add(t, c, i)
    return tr


def test_majority_wins_over_single_confident_garble():
    tr = make(("MH12AB1234", 0.7), ("MH12AB1234", 0.6), ("MH12AB1284", 0.95))
    d = decide(tr)
    assert d.text == "MH12AB1234" and d.valid


def test_look_alike_read_is_repaired_before_voting():
    tr = make(("MH12AB1234", 0.8), ("MH1ZAB1234", 0.8), ("MH12A81234", 0.8))
    assert decide(tr).text == "MH12AB1234"


def test_character_vote_fixes_split_errors():
    tr = make(
        ("MH12AB1234", 0.8),
        ("MH12AB1238", 0.6),
        ("MH12AB1234", 0.6),
        ("MH12AB1264", 0.5),
    )
    assert decide(tr).text == "MH12AB1234"


def test_low_confidence_reads_ignored():
    assert decide(make(("MH12AB1234", 0.05))) is None


def test_invalid_consensus_is_flagged():
    d = decide(make(("HELLO", 0.9), ("HELLO", 0.9)))
    assert d.text == "HELLO" and not d.valid


def test_empty_reads_skipped():
    tr = TrackReads()
    tr.add("---", 0.9)
    assert tr.reads == []
