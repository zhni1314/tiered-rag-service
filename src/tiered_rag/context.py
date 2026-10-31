"""Prompt context assembly: reranked children in, parent text out."""

from __future__ import annotations

from collections.abc import Mapping

from langchain_core.documents import Document


def assemble_context(
    docs: list[Document],
    parents: Mapping[str, str],
    max_parents: int = 3,
    budget: int = 3200,
) -> str:
    """Reranked children in, prompt context out.

    Children are deduped by parent while keeping rerank order, so the top-ranked
    hit is never dropped in favour of a lower-ranked sibling. A chunk without
    ``parent_id`` (index built before parents existed) acts as its own parent,
    which keeps old data answerable instead of silently producing an empty
    context. Blocks that do not fit the budget are dropped whole: a half
    sentence invites the model to finish it from outside the documents.
    """
    seen: set[str] = set()
    blocks: list[tuple[str, str]] = []
    for doc in docs:
        pid = doc.metadata.get("parent_id") or f"raw::{doc.page_content}"
        if pid in seen:
            continue
        seen.add(pid)
        blocks.append((pid, doc.page_content))
        if len(blocks) >= max_parents:
            break

    parts: list[str] = []
    used = 0
    for pid, fallback in blocks:
        text = (parents.get(pid) or "").strip() or fallback.strip()
        if not text:
            continue
        if used + len(text) > budget:
            continue
        parts.append(text)
        used += len(text) + 2  # the "\n\n" separator
    return "\n\n".join(parts)
