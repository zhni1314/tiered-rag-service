"""Hybrid retrieval: BM25 + dense vectors, fused then reranked by a cross-encoder."""

from __future__ import annotations

import threading

import jieba
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .config import Settings

_EMPTY = Document(page_content="暂无数据", metadata={"level": "public"})


def _build_engine(
    docs: list[Document],
    allowed: list[str],
    vectorstore: Chroma,
    cross_encoder: HuggingFaceCrossEncoder,
    settings: Settings,
) -> ContextualCompressionRetriever:
    if not docs:
        docs = [_EMPTY]

    bm25 = BM25Retriever.from_documents(docs, preprocess_func=jieba.lcut, k=settings.retrieve_k)
    vector = vectorstore.as_retriever(
        search_kwargs={"k": settings.retrieve_k, "filter": {"level": {"$in": allowed}}}
    )
    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[settings.bm25_weight, settings.vector_weight],
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=settings.rerank_top_n)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid)


class RetrieverPool:
    """One compiled retrieval engine per permission level.

    Uploads rebuild the whole pool, so reads are guarded by a lock: the FastAPI
    event loop may serve a chat while an ingestion finishes in a worker thread.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        cross_encoder: HuggingFaceCrossEncoder,
        settings: Settings,
    ) -> None:
        self._vectorstore = vectorstore
        self._cross_encoder = cross_encoder
        self._settings = settings
        self._engines: dict[str, ContextualCompressionRetriever] = {}
        self._docs: list[Document] = []
        self._lock = threading.Lock()

    @property
    def documents(self) -> list[Document]:
        with self._lock:
            return list(self._docs)

    def rebuild(self, docs: list[Document]) -> None:
        settings = self._settings
        engines = {
            level: _build_engine(
                [d for d in docs if d.metadata.get("level") in settings.allowed_levels(level)],
                settings.allowed_levels(level),
                self._vectorstore,
                self._cross_encoder,
                settings,
            )
            for level in settings.levels
        }
        with self._lock:
            self._docs = list(docs)
            self._engines = engines

    def add(self, docs: list[Document]) -> None:
        with self._lock:
            combined = self._docs + docs
        self.rebuild(combined)

    def get(self, level: str):
        with self._lock:
            engine = self._engines.get(level)
        if engine is None:
            raise KeyError(f"no retrieval engine compiled for level {level!r}")
        return engine
