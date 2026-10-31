from langchain_core.documents import Document

from tiered_rag.context import assemble_context


def _doc(text: str, **metadata: str) -> Document:
    return Document(page_content=text, metadata=metadata)


def test_parents_keep_rerank_order_and_drop_duplicate_siblings():
    docs = [
        _doc("子块1-1", parent_id="p1"),
        _doc("子块2-1", parent_id="p2"),
        _doc("子块1-2", parent_id="p1"),
    ]
    parents = {"p1": "父块一", "p2": "父块二"}
    assert assemble_context(docs, parents) == "父块一\n\n父块二"


def test_parent_text_is_preferred_over_the_matching_child():
    docs = [_doc("短子块", parent_id="p1")]
    assert assemble_context(docs, {"p1": "完整父块正文"}) == "完整父块正文"


def test_missing_parent_falls_back_to_child_text():
    docs = [_doc("子块正文", parent_id="p1")]
    assert assemble_context(docs, {}) == "子块正文"


def test_blank_parent_text_falls_back_to_child_text():
    docs = [_doc("子块正文", parent_id="p1")]
    assert assemble_context(docs, {"p1": "   "}) == "子块正文"


def test_legacy_chunks_without_parent_id_are_not_dropped():
    docs = [_doc("旧数据甲"), _doc("旧数据乙")]
    assert assemble_context(docs, {}) == "旧数据甲\n\n旧数据乙"


def test_cap_limits_how_many_parents_are_used():
    docs = [_doc("x", parent_id=f"p{i}") for i in range(5)]
    parents = {f"p{i}": f"父{i}" for i in range(5)}
    assert assemble_context(docs, parents, max_parents=2) == "父0\n\n父1"


def test_budget_drops_whole_blocks_instead_of_cutting_text():
    docs = [_doc("x", parent_id=f"p{i}") for i in range(3)]
    parents = {f"p{i}": chr(ord("a") + i) * 1000 for i in range(3)}
    assert assemble_context(docs, parents, max_parents=3, budget=2500) == (
        "a" * 1000 + "\n\n" + "b" * 1000
    )


def test_empty_input_returns_empty_string():
    assert assemble_context([], {}) == ""
