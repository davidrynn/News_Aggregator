# How newsagg works: a plain-language overview

Written 2026-10-26, after the move to cloud models. The spec has the full detail; this is the big picture.

## What a run does

Each run is one assembly line:

1. **Collect.** About 11,000 article links from GDELT, a global news index, plus about 550 from 37 outlets' own RSS feeds.
2. **Filter with rules.** This drops articles already seen, blocklisted sites, clickbait patterns and items that are too short. No AI is used yet.
3. **Group.** Articles about the same event become one "story". Each headline is turned into a list of numbers that captures its meaning, called an *embedding*, and headlines whose numbers are close are grouped. Since 2026-10-26 this uses `thenlper/gte-base` at DeepInfra, about $0.001 a run. Before that it was `nomic-embed-text` on the PC.
4. **Triage.** An AI model glances at each headline and rates it 1–5. Only stories rated 3 or more go on, because looking closely at everything would be too slow and too costly.
5. **Score.** For up to 400 stories, the AI reads the headline and description and gives:
   - **kind:** original reporting, wire copy, opinion, clickbait…
   - **newsworthiness**, 1–5
   - **relevance**, 1–5
   - **topic**

   A story is kept if its kind is real journalism and both scores are 3 or more.
6. **Assemble the page.** Duplicates are merged, with the AI double-checking borderline pairs. Stories are ranked, places are reserved for science, arts and the other topics, and the 65-story page is written.
7. **Publish** to GitHub Pages (`--publish`).

## Why the AI moved to the cloud

Steps 4–6 used to run on **llama3.1:8b** on the PC's graphics card. That was free, but it froze the computer during other work. On 2026-10-26 the AI moved to a cheap cloud service, without making the page worse.

## How the model was chosen

The tool for this is the **golden set**: 74 real articles that Fable, the most capable Claude model, labelled carefully. Any model can be scored against those labels. The number that matters most is **keep/drop agreement**: how often the model makes the same keep-or-drop call as Fable. The local Llama scored 78%.

| Try | What we learned |
|---|---|
| Same Llama, at DeepInfra | Worse (75%), and it sometimes broke the reply format. DeepInfra can't *force* that Llama to follow our reply shape (a "schema"), only ask it to. Ollama forced it locally. |
| Same Llama, at OpenRouter, with a forced schema | Back to local quality, which showed the schema was the problem. But Llama falls apart when given 5 stories at once. |
| Bigger models at DeepInfra: Gemma, DeepSeek | Better than Llama, and fine with 5 stories at once. **DeepSeek V4 Flash: 85% keep/drop. Chosen.** |
| Google's free tier | Gemma there was unusable. Flash-Lite matched DeepSeek for $0, but it's slow and rate-limited. Kept as a backup in `config.toml`. |

## Four ideas worth knowing

- **Provider:** which company runs the model. It is one setting in `config.toml`, so switching is a config change, not a code change.
- **Schema:** the exact shape the reply must have (fields, allowed values). When the service enforces it, the model cannot reply in the wrong shape. When it only asks, small models drift.
- **Batching:** send 5 stories in one request instead of 1. The long instructions are sent a fifth as often, which cuts cost. Small models can't cope with this; bigger ones can.
- **Parallel calls:** send 4 requests at once instead of waiting for each reply. Most of each call's time is spent waiting on the provider's side, so overlapping the waits cut a run from 29 minutes to about 7.

## Where things stand (2026-10-26)

- **The live page** comes from DeepSeek scoring. Arts is real arts news now, and Futurism dropped off.
- **Triage on DeepSeek** was checked. It is stricter than Llama, letting about 42% of headlines through against Llama's 53%, but it still caught every story Fable would keep.
- **Cost:** about $2–3 a month at 3 runs a day. The DeepInfra dashboard shows the actual spend.
- **Revert points:** git tags `good-2026-10-26` for the code and `site-2026-10-26` for the page.
- **Embeddings moved to the cloud too** (gte-base). Tested against nomic on 3,000 headlines: of the pairs it groups without asking, the same-event judge confirmed 92%, against 72% for nomic. The "same story" threshold had to be re-tuned (0.87 became 0.90), because each embedding model spreads its numbers differently.
- **Nothing runs on the PC any more.** Since 2026-10-27 GitHub Actions runs the pipeline once a day (11:00 UTC) on GitHub's free machines, and the "Run workflow" button on the repo's Actions tab starts one anytime, even from a phone. The recipe is `.github/workflows/run.yml`.
- **How a fresh machine remembers:** each GitHub machine starts empty, so the run's memory (articles already seen, the stories on the page, the scoring log) is kept on a `state` branch of the repo. A run loads it first and saves it last.
- **Running locally:** do `python -m newsagg load-state` first, so the PC's copy catches up with GitHub's.

## What's next

1. Watch whether DeepSeek gives out too many 5s for newsworthiness. Any change to the scorer prompt is Fable's.
2. Later: outlet reputation.
