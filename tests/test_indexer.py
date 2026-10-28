from pathlib import Path

import pytest

from tiered_rag.config import Settings
from tiered_rag.indexer import chunk_to_documents, safe_filename


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
