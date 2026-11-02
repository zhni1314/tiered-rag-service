import threading
import time

import pytest
from langchain_core.documents import Document

from tiered_rag.config import Settings


def _pool():
    pytest.importorskip("langchain_classic")
    pytest.importorskip("langchain_community")
    from tiered_rag.retrieval import RetrieverPool

    pool = RetrieverPool(None, None, Settings.from_env({"QWEN_API_KEY": "sk-test"}))

    def slow_rebuild(docs):
        # Stands in for BM25/engine construction, which takes seconds in production.
        time.sleep(0.02)
        with pool._lock:
            pool._docs = list(docs)
            pool._engines = {}

    pool.rebuild = slow_rebuild
    return pool


def _docs(source: str) -> list[Document]:
    return [Document(page_content=f"{source}-{i}", metadata={"source": source}) for i in range(5)]


def test_two_concurrent_reuploads_keep_both_files():
    pool = _pool()
    pool.rebuild(_docs("other"))

    threads = [
        threading.Thread(target=pool.replace_source, args=(source, _docs(source)))
        for source in ("a.pdf", "b.pdf")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    sources = {doc.metadata["source"] for doc in pool.documents}
    assert sources == {"other", "a.pdf", "b.pdf"}


def test_replace_source_removes_the_previous_version_of_that_file():
    pool = _pool()
    pool.rebuild(_docs("a.pdf"))

    pool.replace_source("a.pdf", _docs("a.pdf")[:2])

    assert [doc.page_content for doc in pool.documents] == ["a.pdf-0", "a.pdf-1"]
