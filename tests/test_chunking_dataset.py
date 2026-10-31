import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "scripts" / "chunking_dataset.json"
CORPUS = ROOT / "examples" / "sample_corpus"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _dataset() -> dict:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def _corpus() -> dict[str, str]:
    corpus = {}
    for pdf in sorted(CORPUS.glob("*.pdf")):
        text = "".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
        corpus[pdf.name] = _normalize(text)
    return corpus


def test_dataset_keeps_the_anti_cheating_shape():
    dataset = _dataset()
    questions = dataset["questions"]

    assert dataset["name"] == "sample-corpus-chunking-v2"
    assert len(questions) >= 60
    assert len({item["id"] for item in questions}) == len(questions)
    categories = {item["category"] for item in questions}
    assert categories == {"exact", "paraphrase", "cross_document"}
    assert sum(item["category"] == "cross_document" for item in questions) >= 15

    for item in questions:
        assert item["question"].strip()
        assert item["gold"], item["id"]
        if item["category"] == "cross_document":
            # A cross-document question must really depend on two documents.
            assert len(item["gold"]) >= 2, item["id"]


def test_every_gold_string_occurs_exactly_once_in_the_corpus():
    corpus = _corpus()
    assert len(corpus) == 10

    for item in _dataset()["questions"]:
        for gold in item["gold"]:
            occurrences = sum(text.count(_normalize(gold)) for text in corpus.values())
            assert occurrences == 1, f"{item['id']}: {gold!r} occurs {occurrences} times"


def test_cross_document_gold_spans_at_least_two_files():
    corpus = _corpus()

    for item in _dataset()["questions"]:
        if item["category"] != "cross_document":
            continue
        files = {
            name
            for gold in item["gold"]
            for name, text in corpus.items()
            if _normalize(gold) in text
        }
        assert len(files) >= 2, f"{item['id']} only touches {files}"
