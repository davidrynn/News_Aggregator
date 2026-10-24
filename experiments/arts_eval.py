"""Experiment: would the current prompts keep arts news if `arts` were one of the topics?

Run from the project root: PYTHONPATH=. .venv/Scripts/python experiments/arts_eval.py
Reads candidate arts feeds, passes each item through the pipeline's own stages (page
filter, triage, scoring) and scores it twice: with today's topics and with "arts"
added. Prints one line per headline for judging by eye. Changes nothing in the pipeline.
"""
import collections
import dataclasses

from newsagg import config, enrich, filters, llm, rss, scoring, triage

FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml",
    "https://www.theguardian.com/artanddesign/rss",
    "https://www.theguardian.com/books/rss",
    "https://www.theguardian.com/culture/rss",
    "https://feeds.npr.org/1008/rss.xml",
    "https://www.theartnewspaper.com/rss.xml",
    "https://www.artnews.com/feed/",
    "https://hyperallergic.com/feed/",
]
PER_FEED = 15       # newest items per feed, to keep the run short


def main() -> None:
    cfg = config.load()
    arts_cfg = cfg.model_copy(update={"topics": cfg.topics.model_copy(
        update={"include": cfg.topics.include + ["arts"]})})

    by_feed = collections.defaultdict(list)
    for url in FEEDS:
        items = rss.fetch([url], cfg.output.max_age_hours)
        by_feed[url] = sorted(items, key=lambda a: a.seen_at, reverse=True)[:PER_FEED]
        print(f"{len(items):3} items in 36 h  {url}")
    articles = [a for items in by_feed.values() for a in items]

    results = enrich.enrich([[a] for a in articles])
    rows, dropped = [], collections.Counter()
    for a, result in zip(articles, results):
        info = result[1] if result else enrich.PageInfo(description=a.summary)
        if result and filters.page_type(info):
            dropped[a.domain] += 1          # review / opinion pages never reach the model
            continue
        if not info.description and a.summary:
            info = dataclasses.replace(info, description=a.summary)
        rows.append((a, info))
    print(f"\npage filter dropped {sum(dropped.values())} as opinion/review: {dict(dropped)}")

    with llm.client(cfg) as client:
        ratings = triage.rate([a.title for a, _ in rows], cfg, client)
        print(f"\n{'tri':>3} {'now':>11} {'+arts':>11}  outlet: headline   (kind/newsworthiness/relevance, * = kept)")
        kept = collections.Counter()
        for (a, info), r in zip(rows, ratings):
            cells = []
            for name, c in (("now", cfg), ("arts", arts_cfg)):
                v = scoring.score(a, info, c, client)
                if v is None:
                    cells.append("failed")
                    continue
                k = scoring.keep(v, c)
                kept[name] += k and r >= cfg.scoring.min_triage
                cells.append(f"{'*' if k else ' '}{v.kind[:8]}/{v.newsworthiness}/{v.relevance}")
            print(f"{r:>3} {cells[0]:>11} {cells[1]:>11}  {a.domain}: {a.title[:90]}")
    print(f"\nkept and past triage: now {kept['now']}, with arts {kept['arts']} of {len(rows)}")


if __name__ == "__main__":
    main()
