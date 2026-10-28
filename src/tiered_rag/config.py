"""Runtime configuration, read once from the environment.

Kept free of heavy imports (no langchain / torch) so it stays cheap to import
and unit-testable without models, Redis, or API keys present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get(env: dict[str, str], key: str, default: str = "") -> str:
    value = env.get(key)
    return value.strip() if value and value.strip() else default


def _int(env: dict[str, str], key: str, default: int) -> int:
    raw = _get(env, key)
    return int(raw) if raw else default


def _float(env: dict[str, str], key: str, default: float) -> float:
    raw = _get(env, key)
    return float(raw) if raw else default


def _path(env: dict[str, str], key: str, default: str) -> Path:
    raw = _get(env, key, default)
    p = Path(raw).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _levels(env: dict[str, str]) -> tuple[str, ...]:
    raw = _get(env, "TIERED_RAG_LEVELS", "public,secret")
    parsed = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not parsed:
        raise ValueError("TIERED_RAG_LEVELS must list at least one level")
    return parsed


@dataclass(frozen=True)
class Settings:
    qwen_api_key: str
    qwen_base_url: str
    qwen_model: str

    embedding_model: str
    reranker_model: str

    hf_home: Path | None
    hf_endpoint: str

    chroma_dir: Path
    chroma_collection: str
    upload_dir: Path

    redis_url: str

    bm25_weight: float
    vector_weight: float
    retrieve_k: int
    rerank_top_n: int

    chunk_size: int
    chunk_overlap: int

    header_pattern: str
    footer_pattern: str
    drop_cjk_line_breaks: bool

    levels: tuple[str, ...]
    history_window: int
    chunk_separators: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        if env is None:
            # Real environment wins over .env, so container-injected config is
            # never silently overridden by a stale file baked into an image.
            try:
                from dotenv import load_dotenv

                load_dotenv(PROJECT_ROOT / ".env")
            except ImportError:
                pass
            env = dict(os.environ)

        hf_home = _get(env, "HF_HOME")
        bm25_weight = _float(env, "TIERED_RAG_BM25_WEIGHT", 0.5)

        return cls(
            qwen_api_key=_get(env, "QWEN_API_KEY"),
            qwen_base_url=_get(
                env, "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            qwen_model=_get(env, "QWEN_MODEL", "qwen-plus"),
            embedding_model=_get(env, "TIERED_RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            reranker_model=_get(env, "TIERED_RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            hf_home=Path(hf_home).expanduser() if hf_home else None,
            hf_endpoint=_get(env, "HF_ENDPOINT", "https://huggingface.co"),
            chroma_dir=_path(env, "TIERED_RAG_CHROMA_DIR", "data/chroma_db"),
            chroma_collection=_get(env, "TIERED_RAG_CHROMA_COLLECTION", "rag_collection_v2"),
            upload_dir=_path(env, "TIERED_RAG_UPLOAD_DIR", "data/pdfs"),
            redis_url=_get(env, "TIERED_RAG_REDIS_URL", "redis://localhost:6379/0"),
            bm25_weight=bm25_weight,
            vector_weight=_float(env, "TIERED_RAG_VECTOR_WEIGHT", 1.0 - bm25_weight),
            retrieve_k=_int(env, "TIERED_RAG_RETRIEVE_K", 20),
            rerank_top_n=_int(env, "TIERED_RAG_RERANK_TOP_N", 3),
            chunk_size=_int(env, "TIERED_RAG_CHUNK_SIZE", 800),
            chunk_overlap=_int(env, "TIERED_RAG_CHUNK_OVERLAP", 20),
            chunk_separators=[
                s for s in _get(env, "TIERED_RAG_CHUNK_SEPARATORS", "\n\n,。,！,？").split(",")
            ],
            header_pattern=_get(env, "TIERED_RAG_HEADER_PATTERN"),
            footer_pattern=_get(env, "TIERED_RAG_FOOTER_PATTERN"),
            drop_cjk_line_breaks=_get(env, "TIERED_RAG_DROP_CJK_BREAKS", "true").lower()
            not in ("0", "false", "no"),
            levels=_levels(env),
            history_window=_int(env, "TIERED_RAG_HISTORY_WINDOW", 10),
        )

    def require_llm_credentials(self) -> None:
        if not self.qwen_api_key:
            raise RuntimeError(
                "QWEN_API_KEY is not set. Copy .env.example to .env and fill it in."
            )

    def allowed_levels(self, user_level: str) -> list[str]:
        """Ordered tiers: a user at level N may read tiers 0..N.

        The caller must treat ``user_level`` as already authenticated. See
        docs/decisions.md for why it is currently taken from the request body.
        """
        if user_level not in self.levels:
            raise ValueError(f"unknown level {user_level!r}, expected one of {self.levels}")
        return list(self.levels[: self.levels.index(user_level) + 1])
