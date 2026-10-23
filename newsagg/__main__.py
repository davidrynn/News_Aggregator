"""Command line entry point: python -m newsagg <command>."""
import argparse
import logging
import time

from newsagg import blocklist, config, filters, gdelt, grouping


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
    for s in stories[:8]:
        print(f"  {len(s.outlets):3} outlets  {s.lead.title[:80]}")

    # Saved last, so a run that crashes re-reads its files next time.
    gdelt.save_done(done, fetched)


def cmd_serve(cfg: config.Config) -> None:
    print(f"serve: would serve {cfg.out_dir} at http://localhost:8080 (not built yet)")


def cmd_outlets(cfg: config.Config) -> None:
    print("outlets: reputation report (not built yet)")


COMMANDS = {"run": cmd_run, "serve": cmd_serve, "outlets": cmd_outlets}


def main() -> None:
    parser = argparse.ArgumentParser(prog="newsagg")
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every request at INFO
    COMMANDS[args.command](config.load())


if __name__ == "__main__":
    main()
