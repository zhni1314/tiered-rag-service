from pathlib import Path

import pytest
from langchain_core.documents import Document

from tiered_rag.config import Settings
from tiered_rag.indexer import (
    chunk_parent_child,
    chunk_to_documents,
    safe_filename,
    without_source,
)


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    (tmp_path / "pdfs").mkdir()
    return tmp_path / "pdfs"


@pytest.mark.parametrize(
    "raw",
    [
        "guide.pdf",
        "实验指导书-20251113.pdf",          # CJK names are the common case here
        "a b c.pdf",
        "UPPER.PDF",
    ],
)
def test_accepts_ordinary_names(raw, upload_dir):
    assert safe_filename(raw, upload_dir).parent == upload_dir.resolve()


@pytest.mark.parametrize(
    "raw",
    [
        "../../etc/passwd.pdf",
        "..\\..\\windows\\system32\\x.pdf",
        "/absolute/path.pdf",
        "a/../b.pdf",
    ],
)
def test_traversal_cannot_escape_the_upload_dir(raw, upload_dir):
    dest = safe_filename(raw, upload_dir)
    assert dest.parent == upload_dir.resolve()
    assert upload_dir.resolve() not in dest.parent.parents


@pytest.mark.parametrize("raw", ["notes.txt", "report", "", "dir/"])
def test_rejects_non_pdf(raw, upload_dir):
    with pytest.raises(ValueError):
        safe_filename(raw, upload_dir)


def test_control_characters_are_neutralised(upload_dir):
    dest = safe_filename("x\x00y.pdf", upload_dir)
    assert "\x00" not in dest.name


def _settings() -> Settings:
    return Settings.from_env({"QWEN_API_KEY": "sk-test"})


def test_every_chunk_carries_its_permission_level():
    docs = chunk_to_documents("句子。" * 400, "secret", _settings())
    assert docs
    assert {d.metadata["level"] for d in docs} == {"secret"}


def test_chunk_size_is_respected():
    docs = chunk_to_documents("很长的一段中文。" * 500, "public", _settings())
    assert all(len(d.page_content) <= 800 for d in docs)


def _ingest_sample(source: str = "实验册.pdf"):
    return chunk_parent_child("句子内容。" * 200, "public", source, _settings())


def test_children_are_small_and_point_at_a_stored_parent():
    result = _ingest_sample()

    assert len(result.parents) > 1
    for child in result.children:
        meta = child.metadata
        assert len(child.page_content) <= 200
        assert meta["parent_id"] in result.parents
        assert meta["source"] == "实验册.pdf"
        assert meta["level"] == "public"
        assert isinstance(meta["parent_index"], int)
        # The parent body must live in the store, not be copied into metadata.
        assert "parent_text" not in meta


def test_every_parent_is_reachable_from_its_children():
    result = _ingest_sample()

    referenced = {child.metadata["parent_id"] for child in result.children}
    assert referenced == set(result.parents)


def test_parent_ids_are_stable_and_scoped_to_the_file():
    first = _ingest_sample("a.pdf")
    again = _ingest_sample("a.pdf")
    other = _ingest_sample("b.pdf")

    assert set(first.parents) == set(again.parents)
    assert set(first.parents).isdisjoint(other.parents)


def _doc(text: str, source: str) -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_without_source_drops_old_chunks_and_keeps_the_rest():
    docs = [_doc("a-1", "a.pdf"), _doc("b-1", "b.pdf"), _doc("a-2", "a.pdf")]

    kept = without_source(docs, "a.pdf")

    assert [doc.page_content for doc in kept] == ["b-1"]
    assert [doc.page_content for doc in docs] == ["a-1", "b-1", "a-2"]


def test_without_source_is_a_no_op_for_an_unknown_file():
    docs = [_doc("a-1", "a.pdf")]
    assert without_source(docs, "missing.pdf") == docs
