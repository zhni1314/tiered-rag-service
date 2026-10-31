from tiered_rag.config import Settings


def _settings(**overrides: str) -> Settings:
    env = {"QWEN_API_KEY": "sk-test", **overrides}
    return Settings.from_env(env)


def test_defaults_match_the_tuned_pipeline():
    s = _settings()
    assert s.levels == ("public", "secret")
    assert s.chunk_size == 800
    assert s.child_chunk_size == 200
    assert s.max_parents == 3
    assert s.context_budget_chars == 3200
    # More children than parents: several children can share one parent.
    assert s.rerank_top_n == 10
    assert s.embedding_model == "BAAI/bge-small-zh-v1.5"


def test_parent_store_defaults_inside_the_chroma_dir():
    s = _settings()
    assert s.parent_store == s.chroma_dir / "parent_store.json"


def test_parent_store_can_be_pointed_elsewhere():
    s = _settings(TIERED_RAG_PARENT_STORE="data/custom/parents.json")
    assert s.parent_store.is_absolute()
    assert s.parent_store.parts[-2:] == ("custom", "parents.json")


def test_vector_weight_defaults_to_the_remainder_of_bm25():
    assert _settings(TIERED_RAG_BM25_WEIGHT="0.3").vector_weight == 0.7


def test_explicit_vector_weight_is_respected():
    s = _settings(TIERED_RAG_BM25_WEIGHT="0.3", TIERED_RAG_VECTOR_WEIGHT="0.9")
    assert s.vector_weight == 0.9


def test_levels_are_ordered_and_grant_access_downward():
    s = _settings(TIERED_RAG_LEVELS="public,internal,secret")
    assert s.allowed_levels("public") == ["public"]
    assert s.allowed_levels("internal") == ["public", "internal"]
    assert s.allowed_levels("secret") == ["public", "internal", "secret"]


def test_unknown_level_is_rejected_rather_than_defaulted():
    import pytest

    with pytest.raises(ValueError):
        _settings().allowed_levels("root")


def test_missing_api_key_fails_loudly_at_startup():
    import pytest

    s = Settings.from_env({})
    with pytest.raises(RuntimeError, match="QWEN_API_KEY"):
        s.require_llm_credentials()


def test_relative_paths_are_resolved_against_the_project_root():
    s = _settings(TIERED_RAG_CHROMA_DIR="data/chroma_db")
    assert s.chroma_dir.is_absolute()
    assert s.chroma_dir.parts[-2:] == ("data", "chroma_db")
