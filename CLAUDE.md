# newsagg

- Spec: https://claude.ai/artifact/RnssrXcnzvGoDpEsJU9z2a (source of truth; read it first)
- C: is nearly full. Ollama models must live in D:\ollama\models via OLLAMA_MODELS.
  Never pull models or write large files to C:.
- Keep cutoff (min_newsworthiness, min_relevance) is 3 of 5 on purpose. Don't change it without asking.
- Model provider is one setting in config.toml; keep the pipeline provider-agnostic.
- Claude models: Opus 5.5 for the build. Fable 5.1 only for the scorer system prompt and
  labeling tests/golden.jsonl. The system prompt in the spec is the source; copy it verbatim.
- Learning mode: the owner is learning from this build. Work in small runnable steps,
  explain what and why before and after each one, and pause for questions between steps.
