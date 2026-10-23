"""Measure the scorer against the hand-labeled golden set (tests/golden.jsonl).

The spec's launch target is 80% agreement on kind. The confusion table shows which
mistakes the model makes, for example aggregation judged as original reporting.
"""
import json
import time
from collections import Counter
from dataclasses import dataclass

from newsagg import llm, scoring
from newsagg.config import ROOT, Config

GOLDEN = ROOT / "tests" / "golden.jsonl"
KINDS = ["original_reporting", "wire_copy", "aggregation", "opinion", "press_release", "clickbait", "lifestyle"]


@dataclass
class Report:
    n: int
    kind_accuracy: float
    kind_accuracy_decidable: float   # only items where the input was enough to decide
    newsworthiness_mae: float        # mean absolute error, in points on the 1-5 scale
    relevance_mae: float
    keep_agreement: float
    confusion: Counter               # (true kind, model kind) -> count


def compare(pairs: list[tuple[dict, scoring.Verdict]], cfg: Config) -> Report:
    """pairs: (golden label row, model verdict)."""
    def share(hits: list[bool]) -> float:
        return sum(hits) / len(hits) if hits else 0.0

    def golden_keep(g: dict) -> bool:
        return scoring.keep(scoring.Verdict(**{k: g[k] for k in scoring.Verdict.model_fields if k in g},
                                            reason=""), cfg)

    return Report(
        n=len(pairs),
        kind_accuracy=share([g["kind"] == v.kind for g, v in pairs]),
        kind_accuracy_decidable=share([g["kind"] == v.kind for g, v in pairs
                                       if g.get("decidable_from_input", True)]),
        newsworthiness_mae=sum(abs(g["newsworthiness"] - v.newsworthiness) for g, v in pairs) / max(len(pairs), 1),
        relevance_mae=sum(abs(g["relevance"] - v.relevance) for g, v in pairs) / max(len(pairs), 1),
        keep_agreement=share([golden_keep(g) == scoring.keep(v, cfg) for g, v in pairs]),
        confusion=Counter((g["kind"], v.kind) for g, v in pairs),
    )


def run(cfg: Config) -> None:
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs, failed = [], 0
    start = time.monotonic()
    with llm.client(cfg) as client:
        for g in golden:
            try:
                raw = llm.chat_json(scoring.SYSTEM_PROMPT, g["user_message"], scoring.SCHEMA, cfg, client)
                pairs.append((g, scoring.Verdict.model_validate_json(raw)))
            except Exception as e:  # report and carry on; one bad item shouldn't stop the run
                failed += 1
                print(f"  item {g.get('id')} failed: {e}")
    r = compare(pairs, cfg)
    print(f"model {cfg.provider.model}: {r.n} items in {time.monotonic() - start:.0f}s, {failed} failed")
    print(f"  kind agreement            {r.kind_accuracy:6.1%}   (target 80%)")
    print(f"  kind, decidable items     {r.kind_accuracy_decidable:6.1%}")
    print(f"  newsworthiness error      {r.newsworthiness_mae:6.2f} points")
    print(f"  relevance error           {r.relevance_mae:6.2f} points")
    print(f"  keep/drop agreement       {r.keep_agreement:6.1%}")
    print("  confusion (rows = true kind, columns = model kind):")
    print("  " + " " * 19 + "".join(f"{k[:9]:>10}" for k in KINDS))
    for true in KINDS:
        print(f"  {true:19}" + "".join(f"{r.confusion[(true, k)]:>10}" for k in KINDS))
