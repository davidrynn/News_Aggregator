"""Experiment: does scoring several stories per call keep the golden-set results?

Run from the project root:
  PYTHONPATH=. .venv/Scripts/python experiments/batch_eval.py [--provider deepinfra|google|openrouter] [--model ID] [--schema] [sizes...]
Default sizes 1 5 10 (1 = today's one call per story, timed under the same conditions).
Without --provider it scores on local Ollama as config.toml says. --schema sends a response
schema (for models that accept one); otherwise plain JSON mode with the shape in the message.
Keys come from the environment or the git-ignored .env (DEEPINFRA_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY).
Changes nothing in the pipeline.

The system prompt is sent unchanged (it is the spec's, revised only by Fable). Batching lives
in the user message: the Topics and Regions lines once, then the numbered articles, and the
reply schema asks for one verdict per number, as triage does.
"""
import json
import sys
import time

from pydantic import BaseModel, ValidationError

from newsagg import config, evaluate, llm, scoring
from newsagg.config import Provider

CLOUDS = {
    "deepinfra": dict(url="https://api.deepinfra.com/v1/openai", api_key_env="DEEPINFRA_API_KEY",
                      model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    "google": dict(url="https://generativelanguage.googleapis.com/v1beta/openai", api_key_env="GEMINI_API_KEY",
                   model="gemma-4-26b-a4b-it"),   # free tier
    "openrouter": dict(url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
                       model="meta-llama/llama-3.1-8b-instruct",
                       extra={"provider": {"require_parameters": True}}),   # only hosts that honour the schema
}


class NumberedVerdict(scoring.Verdict):
    n: int


class Verdicts(BaseModel):
    verdicts: list[NumberedVerdict]


SCHEMA = Verdicts.model_json_schema()


def batch_message(user_messages: list[str]) -> str:
    """Topics and Regions once (identical for every item), then each article numbered."""
    header = "\n".join(user_messages[0].splitlines()[:2])
    bodies = ["\n".join(m.splitlines()[3:]) for m in user_messages]
    return (f"{header}\n\nScore each of the {len(bodies)} numbered articles below separately, "
            f"one verdict per number.\n\n"
            + "\n\n".join(f"Article {n}\n{b}" for n, b in enumerate(bodies, 1)))


def score_batches(golden: list[dict], size: int, cfg, client) -> tuple[list, int, int]:
    """Return (golden, verdict) pairs, items missing or invalid in replies, failed calls.

    Each verdict in a batch is checked on its own, so one bad verdict loses one story, not the batch."""
    pairs, missing, failed = [], 0, 0
    for i in range(0, len(golden), size):
        chunk = golden[i:i + size]
        try:
            if size == 1:
                raw = llm.chat_json(scoring.SYSTEM_PROMPT, chunk[0]["user_message"], scoring.SCHEMA, cfg, client)
                got = {1: scoring.Verdict.model_validate_json(raw)}
            else:
                raw = llm.chat_json(scoring.SYSTEM_PROMPT, batch_message([g["user_message"] for g in chunk]),
                                    SCHEMA, cfg, client)
                got = {}
                for item in json.loads(raw).get("verdicts", []):
                    try:
                        v = NumberedVerdict.model_validate(item)
                        got[v.n] = v
                    except ValidationError as e:
                        print(f"  size {size}, batch {i // size}: one verdict invalid: {str(e)[:150]}")
        except Exception as e:  # report and carry on, as evaluate.run does
            failed += 1
            print(f"  size {size}, batch {i // size}: failed: {str(e)[:200]}")
            continue
        for n, g in enumerate(chunk, 1):
            if n in got:
                pairs.append((g, got[n]))
            else:
                missing += 1
    return pairs, missing, failed


def main() -> None:
    args = sys.argv[1:]
    def option(name):
        return args[args.index(name) + 1] if name in args else None
    cfg = config.load()
    if option("--provider"):
        cloud = dict(CLOUDS[option("--provider")])
        cloud["model"] = option("--model") or cloud["model"]
        cfg.provider = Provider(name="openai", json_schema="--schema" in args, **cloud)
    values = {option("--provider"), option("--model")}
    sizes = [int(a) for a in args if not a.startswith("--") and a not in values] or [1, 5, 10]
    golden = [json.loads(line) for line in evaluate.GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    with llm.client(cfg) as client:
        llm.chat_json(scoring.SYSTEM_PROMPT, golden[0]["user_message"], scoring.SCHEMA, cfg, client)  # load the model
        for size in sizes:
            start = time.monotonic()
            pairs, missing, failed = score_batches(golden, size, cfg, client)
            secs = time.monotonic() - start
            r = evaluate.compare(pairs, cfg)
            rows.append((size, r, missing, failed, secs))
            print(f"size {size:2}: done in {secs:.0f}s")

    print(f"\nmodel {cfg.provider.model} via {cfg.provider.url}, "
          f"{'schema' if cfg.provider.json_schema else 'JSON mode'}, {len(golden)} golden items")
    print(f"{'size':>4} {'scored':>6} {'missing':>7} {'failed':>6} {'kind':>6} {'decid.':>6} "
          f"{'keep':>6} {'nw err':>6} {'rel err':>7} {'s/story':>7}")
    for size, r, missing, failed, secs in rows:
        print(f"{size:>4} {r.n:>6} {missing:>7} {failed:>6} {r.kind_accuracy:>6.1%} {r.kind_accuracy_decidable:>6.1%} "
              f"{r.keep_agreement:>6.1%} {r.newsworthiness_mae:>6.2f} {r.relevance_mae:>7.2f} {secs / len(golden):>7.2f}")


if __name__ == "__main__":
    main()
