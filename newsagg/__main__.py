"""Command line entry point: python -m newsagg <command>."""
import argparse
import functools
import http.server
import json
import logging
import dataclasses
import time
from datetime import datetime

from newsagg import blocklist, config, enrich, evaluate, filters, gdelt, grouping, llm, output, pages, rss, scoring
from newsagg.runlock import AlreadyRunning, run_lock
from newsagg.seen import Seen

MIN_ARTICLES = 20   # fewer than this suggests a GDELT outage: keep the last news.json
LOGS = config.ROOT / "logs"

log = logging.getLogger("newsagg.run")


def cmd_run(cfg: config.Config) -> None:
    run_start = start = time.monotonic()
    done = gdelt.load_done()
    fetched = gdelt.fetch(cfg.feeds.gdelt_lookback, done)
    articles = fetched.articles
    log.info(f"read {len(fetched.read)} new GDELT files "
             f"({len(done & set(fetched.window))} of {len(fetched.window)} already read)")
    log.info(f"fetched {len(articles)} articles from {len({a.domain for a in articles})} outlets "
             f"in {time.monotonic() - start:.1f}s")

    start = time.monotonic()
    known = {output.story_id(a.url) for a in articles}
    from_feeds = []
    for a in rss.fetch(cfg.feeds.extra, cfg.output.max_age_hours):
        if (key := output.story_id(a.url)) not in known:   # also in GDELT, or in two feeds
            known.add(key)
            from_feeds.append(a)
    articles += from_feeds
    log.info(f"RSS added {len(from_feeds)} articles from {len({a.domain for a in from_feeds})} outlets "
             f"({len(cfg.feeds.extra)} feeds) in {time.monotonic() - start:.1f}s")
    bl = blocklist.load(cfg)
    seen = Seen()
    seen.prune()
    kept, drops = filters.apply(articles, bl, cfg.filters.min_words, cfg.filters.max_per_outlet, seen)
    log.info(f"rules kept {len(kept)}; dropped {dict(drops)} (blocklist has {len(bl)} entries)")

    start = time.monotonic()
    stories = grouping.group(kept, cfg)
    multi = [s for s in stories if len(s.outlets) >= 3]
    log.info(f"grouped into {len(stories)} stories ({len(multi)} with 3+ outlets) "
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
            enriched.append((story, story.lead, enrich.PageInfo(description=story.lead.summary)))
        elif filters.page_type(result[1]):
            dropped += 1
        else:
            article, info = result
            if not info.description and article.summary:     # the feed's description fills in
                info = dataclasses.replace(info, description=article.summary)
            enriched.append((story, article, info))
    log.info(f"enriched {len(stories)} stories in {time.monotonic() - start:.0f}s: "
             f"{failed} pages failed, {dropped} dropped as opinion/review/satire")

    start = time.monotonic()
    verdicts_path = config.ROOT / "data" / "verdicts.jsonl"
    kept_stories, failed = [], 0
    with llm.client(cfg) as client, open(verdicts_path, "a", encoding="utf-8") as log_file:
        for story, article, info in enriched:
            verdict = scoring.score(article, info, cfg, client)
            if verdict is None:
                failed += 1     # not marked seen: tried again next run
                continue
            # The verdict covers the whole story, so none of its articles is scored again.
            seen.add(a.url for a in story.members)
            # Every verdict is logged; the golden test set is labeled from this file.
            log_file.write(json.dumps({
                "url": article.url, "outlet": article.domain, "outlets": len(story.outlets),
                "user_message": scoring.user_message(article, info, cfg),
                **verdict.model_dump(), "kept": scoring.keep(verdict, cfg),
            }) + "\n")
            if scoring.keep(verdict, cfg):
                kept_stories.append((story, article, info, verdict))
    elapsed = time.monotonic() - start
    log.info(f"scored {len(enriched) - failed} stories in {elapsed:.0f}s "
             f"({elapsed / max(len(enriched), 1):.1f}s each, {failed} failed); kept {len(kept_stories)}")

    if len(articles) < MIN_ARTICLES:
        log.info(f"only {len(articles)} articles this run; keeping the previous news.json")
    else:
        published = output.publish(kept_stories, cfg)
        log.info(f"wrote {cfg.out_dir / 'news.json'} with {len(published)} stories")

    # Saved last, so a run that crashes re-reads its files next time.
    gdelt.save_done(done, fetched)
    seen.close()
    log.info(f"run finished in {time.monotonic() - run_start:.0f}s")


def log_to_file() -> None:
    """Also write log lines to logs/run-YYYY-MM-DD.log (local date): Task Scheduler runs
    have no console, so this file is the only record of what happened."""
    LOGS.mkdir(exist_ok=True)
    handler = logging.FileHandler(LOGS / f"run-{datetime.now():%Y-%m-%d}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)


def run_once(cfg: config.Config, publish: bool = False) -> None:
    try:
        with run_lock():
            cmd_run(cfg)
            if publish:
                pages.publish(cfg)
    except AlreadyRunning:
        log.warning("another run is still going; skipping this one")
    except Exception:
        log.exception("run failed")   # the full traceback goes into the log file
        raise SystemExit(1)


def cmd_serve(cfg: config.Config, port: int = 8080) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(cfg.out_dir))
    with http.server.ThreadingHTTPServer(("localhost", port), handler) as server:
        print(f"serving {cfg.out_dir} at http://localhost:{port} (Ctrl+C to stop)")
        server.serve_forever()


def cmd_outlets(cfg: config.Config) -> None:
    print("outlets: reputation report (not built yet)")


def cmd_publish(cfg: config.Config) -> None:
    """Push the current out/ to GitHub Pages without running the pipeline."""
    pages.publish(cfg)


COMMANDS = {"run": run_once, "publish": cmd_publish, "serve": cmd_serve, "outlets": cmd_outlets, "eval": evaluate.run}


def main() -> None:
    parser = argparse.ArgumentParser(prog="newsagg")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--model", help="override provider.model, e.g. llama3.1:8b")
    parser.add_argument("--port", type=int, default=8080, help="port for serve")
    parser.add_argument("--publish", action="store_true", help="run: push the result to GitHub Pages")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every request at INFO
    if args.command == "run":
        log_to_file()   # first, so even a bad config.toml leaves a record
    try:
        cfg = config.load()
    except Exception:
        log.exception("could not load config.toml")
        raise SystemExit(1)
    if args.model:
        cfg = cfg.model_copy(update={"provider": cfg.provider.model_copy(update={"model": args.model})})
    if args.command == "serve":
        cmd_serve(cfg, args.port)
    elif args.command == "run":
        run_once(cfg, args.publish)
    else:
        COMMANDS[args.command](cfg)


if __name__ == "__main__":
    main()
