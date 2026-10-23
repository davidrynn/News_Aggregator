"""Command line entry point: python -m newsagg <command>."""
import argparse
import json
import logging
import time

from newsagg import blocklist, config, enrich, evaluate, filters, gdelt, grouping, llm, scoring


def cmd_run(cfg: config.Config) -> None:
    start = time.monotonic()
    done = gdelt.load_done()
    fetched = gdelt.fetch(cfg.feeds.gdelt_lookback, done)
    articles = fetched.articles
    print(f"read {len(fetched.read)} new GDELT files "
          f"({len(done & set(fetched.window))} of {len(fetched.window)} already read)")
    print(f"fetched {len(articles)} articles from {len({a.domain for a in articles})} outlets "
          f"in {time.monotonic() - start:.1f}s")
    bl = blocklist.load(cfg)
    kept, drops = filters.apply(articles, bl, cfg.filters.min_words, cfg.filters.max_per_outlet)
    print(f"rules kept {len(kept)}; dropped {dict(drops)} (blocklist has {len(bl)} entries)")

    start = time.monotonic()
    stories = grouping.group(kept, cfg)
    multi = [s for s in stories if len(s.outlets) >= 3]
    print(f"grouped into {len(stories)} stories ({len(multi)} with 3+ outlets) "
          f"in {time.monotonic() - start:.0f}s")

    # Most outlets first, then newest (decided 2026-10-23); only the top max_per_run go on.
    stories.sort(key=lambda s: (len(s.outlets), s.lead.seen_at), reverse=True)
    stories = stories[: cfg.scoring.max_per_run]
    start = time.monotonic()
    results = enrich.enrich([s.members for s in stories])
    enriched, failed, dropped = [], 0, 0
    for story, result in zip(stories, results):
        if result is None:              # no page fetched: score on the headline alone
            failed += 1
            enriched.append((story, story.lead, enrich.PageInfo()))
        elif filters.page_type(result[1]):
            dropped += 1
        else:
            enriched.append((story, *result))
    print(f"enriched {len(stories)} stories in {time.monotonic() - start:.0f}s: "
          f"{failed} pages failed, {dropped} dropped as opinion/review/satire")

    start = time.monotonic()
    verdicts_path = config.ROOT / "data" / "verdicts.jsonl"
    kept_stories, failed = [], 0
    with llm.client(cfg) as client, open(verdicts_path, "a", encoding="utf-8") as log_file:
        for story, article, info in enriched:
            verdict = scoring.score(article, info, cfg, client)
            if verdict is None:
                failed += 1
                continue
            # Every verdict is logged; the golden test set is labeled from this file.
            log_file.write(json.dumps({
                "url": article.url, "outlet": article.domain, "outlets": len(story.outlets),
                "user_message": scoring.user_message(article, info, cfg),
                **verdict.model_dump(), "kept": scoring.keep(verdict, cfg),
            }) + "\n")
            if scoring.keep(verdict, cfg):
                kept_stories.append((story, article, info, verdict))
    elapsed = time.monotonic() - start
    print(f"scored {len(enriched) - failed} stories in {elapsed:.0f}s "
          f"({elapsed / max(len(enriched), 1):.1f}s each, {failed} failed); kept {len(kept_stories)}")

    # Saved last, so a run that crashes re-reads its files next time.
    gdelt.save_done(done, fetched)


def cmd_serve(cfg: config.Config) -> None:
    print(f"serve: would serve {cfg.out_dir} at http://localhost:8080 (not built yet)")


def cmd_outlets(cfg: config.Config) -> None:
    print("outlets: reputation report (not built yet)")


COMMANDS = {"run": cmd_run, "serve": cmd_serve, "outlets": cmd_outlets, "eval": evaluate.run}


def main() -> None:
    parser = argparse.ArgumentParser(prog="newsagg")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--model", help="override provider.model, e.g. llama3.1:8b")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every request at INFO
    cfg = config.load()
    if args.model:
        cfg = cfg.model_copy(update={"provider": cfg.provider.model_copy(update={"model": args.model})})
    COMMANDS[args.command](cfg)


if __name__ == "__main__":
    main()
