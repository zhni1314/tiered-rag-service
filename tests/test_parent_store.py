import json

import pytest

from tiered_rag.parents import ParentStore


def test_missing_file_loads_as_empty(tmp_path):
    store = ParentStore(tmp_path / "parents.json")
    assert store.load() == 0
    assert dict(store) == {}


def test_update_persists_and_reloads(tmp_path):
    path = tmp_path / "nested" / "parents.json"
    ParentStore(path).update("a.pdf", {"a.pdf::1": "父文一"})

    reloaded = ParentStore(path)
    assert reloaded.load() == 1
    assert reloaded["a.pdf::1"] == "父文一"


def test_reuploading_a_source_replaces_only_its_own_parents(tmp_path):
    store = ParentStore(tmp_path / "parents.json")
    store.update("a.pdf", {"a.pdf::1": "旧"})
    store.update("b.pdf", {"b.pdf::1": "乙"})
    store.update("a.pdf", {"a.pdf::2": "新"})

    assert "a.pdf::1" not in store
    assert store["a.pdf::2"] == "新"
    assert store["b.pdf::1"] == "乙"


def test_update_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "parents.json"
    ParentStore(path).update("a.pdf", {"a.pdf::1": "父文"})

    assert not (tmp_path / "parents.json.tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"a.pdf::1": "父文"}


def test_corrupt_file_fails_loudly(tmp_path):
    path = tmp_path / "parents.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unreadable"):
        ParentStore(path).load()


def test_wrong_shape_fails_loudly(tmp_path):
    path = tmp_path / "parents.json"
    path.write_text(json.dumps(["a.pdf::1"]), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected shape"):
        ParentStore(path).load()
