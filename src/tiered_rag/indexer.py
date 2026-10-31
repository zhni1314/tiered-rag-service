"""PDF -> text -> parent/child chunks -> LangChain Documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import Settings
from .text_cleaning import strip_repeating_layout

# Only control characters and path separators are unsafe; CJK file names are
# normal here, so they must survive sanitising.
_UNSAFE = re.compile(r"[\x00-\x1f/\\]+")


def safe_filename(original: str, upload_dir: Path) -> Path:
    """Keep an upload inside ``upload_dir``.

    ``UploadFile.filename`` is entirely client-controlled: a value like
    ``../../x.pdf`` would otherwise write anywhere on the host.
    """
    stem = Path(original or "").name
    if not stem.lower().endswith(".pdf"):
        raise ValueError("only .pdf files are accepted")

    cleaned = _UNSAFE.sub("_", stem)
    if cleaned in (".", "..") or not cleaned.strip("._"):
        raise ValueError("unsafe file name")

    dest = (upload_dir / cleaned).resolve()
    if dest.parent != upload_dir.resolve():
        raise ValueError("unsafe upload path")
    return dest


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() for page in reader.pages]
    return "\n".join(text for text in pages if text)


def chunk_to_documents(text: str, level: str, settings: Settings) -> list[Document]:
    """Flat chunking. Used by the evaluation harness only; ingestion is two-level."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.chunk_separators,
    )
    return [
        Document(page_content=chunk, metadata={"level": level, "source": "upload"})
        for chunk in splitter.split_text(text)
    ]


@dataclass(frozen=True)
class Ingested:
    children: list[Document]
    parents: dict[str, str]


def parent_id(source: str, text: str) -> str:
    """Stable id: the same text from the same file always maps to the same parent.

    "::" is the separator because Windows file names cannot contain ":".
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{source}::{digest}"


def chunk_parent_child(text: str, level: str, source: str, settings: Settings) -> Ingested:
    """Small children for retrieval, whole parents for the prompt.

    The parent body is returned separately instead of being copied into every
    child's metadata: one 800-char parent can own five children, and Chroma
    would otherwise store that text five times and load it all at startup.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.chunk_separators,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        separators=settings.chunk_separators,
    )

    children: list[Document] = []
    parents: dict[str, str] = {}
    for index, parent_text in enumerate(parent_splitter.split_text(text)):
        pid = parent_id(source, parent_text)
        parents[pid] = parent_text
        for child_text in child_splitter.split_text(parent_text):
            children.append(
                Document(
                    page_content=child_text,
                    metadata={
                        "level": level,
                        "parent_id": pid,
                        "source": source,
                        "parent_index": index,
                    },
                )
            )
    return Ingested(children=children, parents=parents)


def ingest(path: Path, level: str, settings: Settings) -> Ingested:
    raw = extract_pdf_text(path)
    if not raw.strip():
        return Ingested(children=[], parents={})
    cleaned = strip_repeating_layout(
        raw,
        header_pattern=settings.header_pattern,
        footer_pattern=settings.footer_pattern,
        drop_cjk_line_breaks=settings.drop_cjk_line_breaks,
    )
    return chunk_parent_child(cleaned, level, path.name, settings)


def without_source(docs: list[Document], source: str) -> list[Document]:
    """Drop every chunk previously indexed from ``source``."""
    return [doc for doc in docs if doc.metadata.get("source") != source]
