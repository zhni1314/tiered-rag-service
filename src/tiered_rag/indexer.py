"""PDF -> text -> chunks -> LangChain Documents."""

from __future__ import annotations

import re
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.chunk_separators,
    )
    return [
        Document(page_content=chunk, metadata={"level": level, "source": "upload"})
        for chunk in splitter.split_text(text)
    ]


def ingest(path: Path, level: str, settings: Settings) -> list[Document]:
    raw = extract_pdf_text(path)
    if not raw.strip():
        return []
    cleaned = strip_repeating_layout(
        raw,
        header_pattern=settings.header_pattern,
        footer_pattern=settings.footer_pattern,
        drop_cjk_line_breaks=settings.drop_cjk_line_breaks,
    )
    return chunk_to_documents(cleaned, level, settings)
