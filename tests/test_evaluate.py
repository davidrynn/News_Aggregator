from newsagg import config, evaluate
from newsagg.scoring import Verdict

CFG = config.load()


def label(kind, nw=4, rel=4, decidable=True):
    return {"kind": kind, "newsworthiness": nw, "relevance": rel, "topic": "world",
            "decidable_from_input": decidable}


def verdict(kind, nw=4, rel=4):
    return Verdict(kind=kind, newsworthiness=nw, relevance=rel, topic="world", reason="")


def test_compare_scores_agreement_and_confusion():
    pairs = [
        (label("original_reporting"), verdict("original_reporting")),
        (label("aggregation", decidable=False), verdict("original_reporting")),   # undetectable rewrite
        (label("lifestyle", nw=1, rel=2), verdict("lifestyle", nw=3, rel=2)),
        (label("original_reporting", nw=2), verdict("original_reporting", nw=4)),
    ]
    r = evaluate.compare(pairs, CFG)
    assert r.kind_accuracy == 0.75
    assert r.kind_accuracy_decidable == 1.0
    assert r.newsworthiness_mae == 1.0                     # (0 + 0 + 2 + 2) / 4
    assert r.keep_agreement == 0.5                         # items 2 and 4 disagree
    assert r.confusion[("aggregation", "original_reporting")] == 1
