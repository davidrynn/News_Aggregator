import json

import httpx

from newsagg import config, triage

# The fake transport answers in Ollama's format, whatever provider config.toml names.
CFG = config.load().model_copy(update={"provider": config.Provider()})


def fake_client(replies: list):
    """An httpx client whose Ollama /api/chat answers come from `replies` in order;
    an int reply is an HTTP error status."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        reply = replies[len(calls) - 1]
        if isinstance(reply, int):
            return httpx.Response(reply)
        return httpx.Response(200, json={"message": {"content": json.dumps(reply)}})

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)), calls


def test_prompt_matches_experiment():
    assert triage.SYSTEM_PROMPT.startswith("You triage news headlines for a news page")
    assert triage.SYSTEM_PROMPT.endswith("Rate every headline. Return JSON.")


def test_user_message_numbers_headlines():
    assert triage.user_message(["A", "B"]) == "1. A\n2. B"


def test_rates_in_batches_of_50_in_order():
    titles = [f"t{i}" for i in range(60)]
    client, calls = fake_client([
        {"ratings": [{"n": n, "score": 5 if n == 1 else 2} for n in range(1, 51)]},
        {"ratings": [{"n": n, "score": 4} for n in range(1, 11)]},
    ])
    scores = triage.rate(titles, CFG, client)
    assert len(calls) == 2 and calls[1]["messages"][1]["content"].startswith("1. t50\n")
    assert scores[0] == 5 and scores[1:50] == [2] * 49 and scores[50:] == [4] * 10


def test_skipped_headlines_and_failed_batches_get_3():
    client, _ = fake_client([{"ratings": [{"n": 2, "score": 1}]}, 500])
    titles = ["a", "b", "c"] + [f"t{i}" for i in range(48)] + ["z"]
    scores = triage.rate(titles, CFG, client)
    assert scores[:3] == [3, 1, 3] and scores[50] == 3


def test_out_of_range_rating_fails_the_batch():
    client, _ = fake_client([{"ratings": [{"n": 1, "score": 9}]}])
    assert triage.rate(["a"], CFG, client) == [3]
