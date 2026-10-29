from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "evaluate_script",
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py",
)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


@pytest.mark.parametrize(
    "values,fraction,expected",
    [
        ([], 0.5, None),
        ([8.0], 0.95, 8.0),
        ([30.0, 10.0, 20.0], 0.5, 20.0),
        ([0.0, 100.0], 0.95, 95.0),
    ],
)
def test_percentile(values, fraction, expected):
    assert evaluate.percentile(values, fraction) == expected


def test_dataset_references_only_its_synthetic_documents():
    dataset, digest = evaluate.load_dataset()
    ids = {item["id"] for item in dataset["documents"]}
    assert len(dataset["questions"]) == 12
    assert len(ids) == 24
    assert len(digest) == 64
    for item in dataset["questions"]:
        assert set(item["gold_ids"]) <= ids
        assert item["reference"]


def test_error_output_does_not_include_message_or_credentials():
    exc = RuntimeError("Authorization: Bearer private-test-secret")
    exc.status_code = 401
    result = evaluate.error_info(exc)
    assert result == {"type": "RuntimeError", "http_status": 401}
    assert "private-test-secret" not in json.dumps(result)


def test_missing_scores_are_counted_not_replaced_with_zero():
    rows = [
        {"kind": "quality", "variant": "dense", "ok": True, "scores": {"faithfulness": 0.8}},
        {"kind": "quality", "variant": "dense", "ok": False, "scores": {}},
    ]
    scores = evaluate.summarize(rows)["ragas"]["dense"]
    assert scores["faithfulness"] == {"mean": 0.8, "valid": 1, "failed_or_skipped": 1}
    assert scores["context_recall"]["mean"] is None
    assert scores["generation_failed"] == 1


def test_cache_summary_excludes_prefill_and_failed_latency():
    rows = [
        {"kind": "warmup", "ok": True, "latency_ms": 9000},
        {
            "kind": "cache",
            "group": "hit",
            "ok": True,
            "latency_ms": 10,
            "external_requests": 0,
            "cache_events": [{"hit": True}],
        },
        {
            "kind": "cache",
            "group": "hit",
            "ok": False,
            "latency_ms": 500,
            "external_requests": 1,
            "cache_events": [{"hit": False}],
        },
    ]
    group = evaluate.summarize(rows)["cache"]["hit"]
    assert group["attempted"] == 2
    assert group["failed"] == 1
    assert group["p50_ms"] == 10
    assert group["observed_hit_rate"] == 0.5


def test_cache_hit_requires_same_prompt_and_no_external_request():
    seed = {"ok": True, "cache_events": [{"input_hash": "original"}], "response": "answer"}
    row = {
        "ok": True,
        "external_requests": 0,
        "response": "answer",
        "cache_events": [{"hit": True, "input_hash": "changed"}],
    }
    evaluate.check_cache_expectation(row, "hit", seed)
    assert not row["ok"]
    assert row["error"]["type"] == "CacheExpectationMismatch"


def test_recording_pool_returns_the_same_contexts_it_records():
    from langchain_core.documents import Document

    docs = [Document(page_content=f"text-{i}", metadata={"eval_id": str(i)}) for i in range(4)]
    engine = SimpleNamespace(invoke=lambda query: docs)
    recorder = evaluate.RecordingPool(engine, 3)
    returned = recorder.get("public").invoke("question")
    assert recorder.contexts == [d.page_content for d in returned]
    assert recorder.context_ids == ["0", "1", "2"]
    with pytest.raises(ValueError):
        recorder.get("secret")


def test_request_budget_blocks_before_transport_and_counts_usage():
    import httpx

    async def scenario():
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(
                200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 2}}
            )

        meter = evaluate.RequestMeter(1)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
            event_hooks={"request": [meter.on_request], "response": [meter.on_response]},
        ) as client:
            await client.post("https://example.test/chat/completions", json={})
            with pytest.raises(evaluate.RequestBudgetExceeded):
                await client.post("https://example.test/chat/completions", json={})
        assert len(calls) == 1
        assert meter.requests == 1
        assert meter.prompt_tokens == 10
        assert meter.completion_tokens == 2

    asyncio.run(scenario())


@pytest.fixture
def memory_redis():
    redis = pytest.importorskip("redis")
    pytest.importorskip("langchain_community.cache")

    class MemoryRedis(redis.Redis):
        def __init__(self):
            super().__init__()
            self.data = {"history:existing-session": {"entry": "keep"}}
            self.deleted = []
            self.ttls = {}

        def hgetall(self, key):
            return self.data.get(key, {})

        def pipeline(self):
            parent = self

            class Pipeline:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def hset(self, key, mapping):
                    parent.data[key] = mapping

                def expire(self, key, ttl):
                    parent.ttls[key] = ttl

                def execute(self):
                    return []

            return Pipeline()

        def delete(self, *keys):
            self.deleted.extend(keys)
            for key in keys:
                self.data.pop(key, None)

    return MemoryRedis()


def test_cache_preserves_existing_keys_and_cannot_flush(memory_redis):
    from langchain_core.outputs import Generation

    cache = evaluate.make_cache(memory_redis, "tiered-rag-eval:test:")
    assert cache.lookup("prompt", "model") is None
    cache.update("prompt", "model", [Generation(text="answer")])
    assert cache.lookup("prompt", "model")[0].text == "answer"
    assert [event["hit"] for event in cache.events] == [False, True]
    cache.namespace = "other"
    assert cache.lookup("prompt", "model") is None
    with pytest.raises(RuntimeError):
        cache.clear()
    cache.cleanup()
    assert memory_redis.data == {"history:existing-session": {"entry": "keep"}}
    assert all(key.startswith("tiered-rag-eval:test:") for key in memory_redis.deleted)
    assert set(memory_redis.ttls.values()) == {3600}


def test_real_chain_cache_protocol_with_mock_http(memory_redis, monkeypatch):
    pytest.importorskip("langchain_openai")
    pytest.importorskip("langchain_classic")
    import httpx
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI

    from tiered_rag.config import Settings

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    engine = SimpleNamespace(
        invoke=lambda query: [
            Document(page_content="公开测试内容。", metadata={"eval_id": "test"}),
        ]
    )
    question = {"id": "q", "question": "公开测试内容是什么？"}

    async def scenario():
        def respond(request):
            body = json.loads(request.content)
            assert "公开测试内容。" in body["messages"][-1]["content"]
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "测试回答"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
            )

        meter = evaluate.RequestMeter(20)
        cache = evaluate.make_cache(memory_redis, "tiered-rag-eval:integration:")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
            event_hooks={"request": [meter.on_request], "response": [meter.on_response]},
        ) as client:
            llm = ChatOpenAI(
                model="test-model",
                api_key="not-a-real-key",
                base_url="https://example.test/v1",
                http_async_client=client,
                max_retries=0,
                temperature=0,
                streaming=False,
            )
            rows = []
            await evaluate.run_cache(
                [question],
                SimpleNamespace(seed=42, repeats=2),
                Settings.from_env({}),
                engine,
                llm,
                cache,
                meter,
                rows.append,
            )
        result = evaluate.summarize(rows)["cache"]
        assert meter.requests == 7
        assert all(row["ok"] for row in rows)
        assert result["disabled"]["valid"] == 2
        assert result["miss"]["observed_hit_rate"] == 0
        assert result["hit"]["observed_hit_rate"] == 1
        assert result["hit"]["external_requests"] == 0
        cache.cleanup()

    asyncio.run(scenario())


def test_quality_uses_exact_contexts_and_keeps_metric_failures(monkeypatch):
    pytest.importorskip("langchain_classic")
    pytest.importorskip("langchain_community")
    from langchain_core.documents import Document
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from tiered_rag.config import Settings

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    seen = {}

    class Metric:
        def __init__(self, name, value):
            self.name, self.value = name, value

        async def ascore(self, **kwargs):
            seen[self.name] = kwargs
            if self.name == "context_recall":
                raise ValueError("private-test-secret should not appear in records")
            return SimpleNamespace(value=self.value)

    async def scenario():
        engine = SimpleNamespace(
            invoke=lambda query: [
                Document(page_content="实际检索片段", metadata={"eval_id": "evidence"}),
            ]
        )
        question = {
            "id": "q",
            "question": "问题",
            "reference": "参考答案",
            "gold_ids": ["evidence"],
        }
        metrics = {
            name: Metric(name, float("nan") if name == "answer_relevancy" else 0.5)
            for name in evaluate.METRICS
        }
        rows = []
        await evaluate.run_quality(
            [question],
            SimpleNamespace(variants=["dense"]),
            Settings.from_env({}),
            {"dense": engine},
            FakeListChatModel(responses=["实际回答"]),
            metrics,
            evaluate.RequestMeter(10),
            rows.append,
        )
        assert seen["faithfulness"]["retrieved_contexts"] == ["实际检索片段"]
        assert seen["faithfulness"]["response"] == "实际回答"
        assert seen["context_recall"]["reference"] == "参考答案"
        assert rows[0]["scores"]["answer_relevancy"] is None
        assert rows[0]["scores"]["context_recall"] is None
        assert rows[0]["scores"]["faithfulness"] == 0.5
        assert "private-test-secret" not in json.dumps(rows)
        assert evaluate.summarize(rows)["ragas"]["dense"]["context_recall"]["valid"] == 0

    asyncio.run(scenario())


def test_ragas_instructor_json_protocol_offline(monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    pytest.importorskip("ragas")
    import httpx
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    async def scenario():
        replies = iter(
            [
                {"statements": ["例会在周三。"]},
                {
                    "statements": [
                        {"statement": "例会在周三。", "reason": "资料明确说明。", "verdict": 1}
                    ]
                },
            ]
        )

        def respond(request):
            content = json.dumps(next(replies), ensure_ascii=False)
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
            )

        meter = evaluate.RequestMeter(2)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
            event_hooks={"request": [meter.on_request], "response": [meter.on_response]},
        ) as http:
            client = AsyncOpenAI(
                api_key="not-a-real-key",
                base_url="https://example.test/v1",
                http_client=http,
                max_retries=0,
            )
            judge = llm_factory("test-model", client=client, temperature=0, max_tokens=2048)
            result = await Faithfulness(llm=judge).ascore(
                user_input="例会在周几？",
                response="例会在周三。",
                retrieved_contexts=["例会在周三。"],
            )
        assert result.value == 1.0
        assert meter.requests == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["responses", "embeddings"])
def test_unmetered_endpoints_are_blocked_before_sending(endpoint):
    import httpx

    async def scenario():
        def respond(request):
            pytest.fail("Unsupported endpoint reached the transport")

        meter = evaluate.RequestMeter(1)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond),
            event_hooks={"request": [meter.on_request]},
        ) as client:
            with pytest.raises(ValueError, match="Unsupported API endpoint"):
                await client.post(f"https://example.test/v1/{endpoint}", json={})
        assert meter.requests == 0

    asyncio.run(scenario())


def test_cleanup_failure_marks_report_failed(tmp_path, monkeypatch):
    langchain_openai = pytest.importorskip("langchain_openai")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: object())

    def cleanup():
        raise RuntimeError("private-test-secret")

    cache = SimpleNamespace(cleanup=cleanup)
    monkeypatch.setattr(evaluate, "make_cache", lambda *args: cache)
    monkeypatch.setattr(evaluate, "build_engines", lambda *args: ({"hybrid_rerank": object()}, 1))

    async def run_cache(questions, args, settings, engine, llm, cache, meter, emit):
        for group in ("disabled", "miss", "hit"):
            emit(
                {
                    "kind": "cache",
                    "question_id": "q",
                    "group": group,
                    "ok": True,
                    "latency_ms": 10,
                    "cache_events": [],
                    "external_requests": 0,
                }
            )

    monkeypatch.setattr(evaluate, "run_cache", run_cache)
    args = SimpleNamespace(
        mode="cache",
        questions=1,
        repeats=1,
        variants=[],
        max_api_requests=10,
        timeout=10,
        device="cpu",
        model="test-model",
        base_url="https://example.test/v1",
    )
    status = asyncio.run(
        evaluate.execute(
            args,
            "not-a-real-key",
            None,
            {},
            {"questions": [{"id": "q"}]},
            object(),
            tmp_path,
            {},
        )
    )
    summary_text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    report = json.loads(summary_text)
    assert status == 1
    assert report["status"] == "partial_or_failed"
    assert report["metadata"]["cache_cleanup_error"] == {"type": "RuntimeError"}
    assert "private-test-secret" not in summary_text
    assert "not-a-real-key" not in summary_text
