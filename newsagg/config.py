"""Load config.toml into typed, validated settings.

Every tunable number lives in config.toml. Pydantic checks types and ranges when the
file is loaded, so a typo like `min_relevance = "three"` fails at startup instead of
deep inside a run.
"""
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config.toml"


class Provider(BaseModel):
    name: str = "ollama"
    model: str = "llama3.1:8b"
    url: str = "http://localhost:11434"
    # Embeddings have their own provider settings: the grouping thresholds belong to one
    # embedding model, so it can stay put when the chat model changes.
    embed_name: str = "ollama"                  # "ollama" or "openai" (any OpenAI-compatible API)
    embed_model: str = "nomic-embed-text"
    embed_url: str = "http://localhost:11434"
    embed_api_key_env: str = ""                 # "openai": the environment variable (or .env line) holding the key
    embed_prefix: str = "clustering: "          # put before every text; nomic was trained with task prefixes
    api_key_env: str = ""       # "openai" provider: the environment variable (or .env line) holding the key
    json_schema: bool = False   # "openai" provider: the model accepts a response schema (else plain JSON mode)
    concurrency: int = Field(1, ge=1, le=16)   # model calls in flight at once; 1 for local Ollama
    extra: dict = {}            # "openai" provider: added to every request body (e.g. OpenRouter's routing options)


class Topics(BaseModel):
    include: list[str]
    regions: list[str]
    language: str = "english"


class Feeds(BaseModel):
    gdelt_lookback: int = Field(8, ge=1, le=96)
    extra: list[str] = []


class Blocklist(BaseModel):
    iffy_url: str
    refresh_days: int = Field(7, ge=1)
    extra: list[str] = []
    allow: list[str] = []


class Scoring(BaseModel):
    min_newsworthiness: int = Field(3, ge=1, le=5)
    min_relevance: int = Field(3, ge=1, le=5)
    keep_kinds: list[str] = ["original_reporting", "wire_copy"]
    max_per_run: int = Field(400, ge=1)
    batch_size: int = Field(1, ge=1, le=20)     # stories per scoring call; 1 for an 8B model, which drops verdicts in batches
    triage: bool = False                       # rate headlines first; score only those rated min_triage+
    min_triage: int = Field(3, ge=1, le=5)


class Filters(BaseModel):
    min_words: int = Field(20, ge=0)
    max_per_outlet: int = Field(10, ge=1)


class Grouping(BaseModel):
    threshold: float = Field(0.87, gt=0, le=1)              # headlines: grouped at or above this
    page_threshold: float = Field(0.87, gt=0, le=1)         # page merge (headline + description): merged outright
    candidate_threshold: float = Field(0.75, gt=0, le=1)    # page merge: between this and page_threshold, the model decides
    wire_outlets: list[str] = ["apnews.com", "reuters.com", "afp.com", "bloomberg.com"]


class Reputation(BaseModel):
    half_life_days: float = Field(30, gt=0)
    min_samples: int = Field(20, ge=0)
    drop_below: float = Field(0.15, ge=0, le=1)


class Output(BaseModel):
    max_stories: int = Field(60, ge=1)
    max_age_hours: float = Field(36, gt=0)
    coverage_weight: float = Field(0.5, ge=0)
    feed_bonus: float = Field(1.5, ge=0)      # ranking bonus for stories led by an article from feeds.extra
    min_per_topic: dict[str, int] = {}        # topic -> stories it is guaranteed on the page, if it has them
    dir: str = "out"


class Config(BaseModel):
    provider: Provider
    topics: Topics
    feeds: Feeds
    blocklist: Blocklist
    scoring: Scoring
    filters: Filters = Filters()
    grouping: Grouping = Grouping()
    reputation: Reputation
    output: Output

    @property
    def out_dir(self) -> Path:
        return ROOT / self.output.dir


def load(path: Path = DEFAULT_PATH) -> Config:
    with open(path, "rb") as f:  # tomllib requires binary mode
        return Config.model_validate(tomllib.load(f))
