"""Run isolated, non-streaming cache, retrieval-recall and RAGAS experiments.

Everything runs on a synthetic corpus; paid API calls are budgeted, keys are
typed at runtime and telemetry stays off.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import getpass
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import statistics
import sys
import time
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(__file__).with_name("eval_dataset.json")
VARIANTS = ("dense", "bm25", "hybrid", "hybrid_rerank")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
RECALL_KS = (1, 3, 5, 8)


def configure_environment(hf_home: Path) -> None:
    os.environ.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "RAGAS_DO_NOT_TRACK": "true",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "ANONYMIZED_TELEMETRY": "false",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    logging.disable(logging.CRITICAL)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower), 3)


def error_info(exc: Exception) -> dict:
    result = {"type": type(exc).__name__}
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        result["http_status"] = status
    return result


def load_dataset() -> tuple[dict, str]:
    raw = DATASET.read_bytes()
    data = json.loads(raw)
    ids = {doc["id"] for doc in data["documents"]}
    if len(ids) != len(data["documents"]):
        raise ValueError("Duplicate document IDs")
    for question in data["questions"]:
        if not question["reference"] or not set(question["gold_ids"]) <= ids:
            raise ValueError("Invalid reference or gold IDs")
    return data, hashlib.sha256(raw).hexdigest()


def make_cache(redis_client, prefix: str):
    from langchain_community.cache import RedisCache

    class MeasuredRedisCache(RedisCache):
        def __init__(self):
            super().__init__(redis_client, ttl=3600)
            self.namespace = ""
            self.events = []
            self.keys = set()

        def _key(self, prompt, llm_string):
            return prefix + self.namespace + ":" + super()._key(prompt, llm_string)

        def lookup(self, prompt, llm_string):
            started = time.perf_counter()
            value = self._get_generations(self.redis.hgetall(self._key(prompt, llm_string)))
            self.events.append(
                {
                    "hit": value is not None,
                    "input_hash": hashlib.sha256((prompt + llm_string).encode()).hexdigest(),
                    "lookup_ms": (time.perf_counter() - started) * 1000,
                }
            )
            return value

        def update(self, prompt, llm_string, return_val):
            self._ensure_generation_type(return_val)
            key = self._key(prompt, llm_string)
            self.keys.add(key)
            with self.redis.pipeline() as pipe:
                self._configure_pipeline_for_update(key, pipe, return_val, self.ttl)
                pipe.execute()

        def clear(self, **kwargs):
            raise RuntimeError("Whole-database cache clearing is prohibited")

        def cleanup(self):
            if self.keys:
                self.redis.delete(*sorted(self.keys))
                self.keys.clear()

    return MeasuredRedisCache()


class RecordingPool:
    def __init__(self, engine, top_n: int):
        self.engine = engine
        self.top_n = top_n
        self.contexts = []
        self.context_ids = []
        self.retrieval_ms = None

    def get(self, level):
        if level != "public":
            raise ValueError("Evaluation only uses public synthetic data")
        return self

    def invoke(self, query):
        started = time.perf_counter()
        docs = self.engine.invoke(query)[: self.top_n]
        self.retrieval_ms = (time.perf_counter() - started) * 1000
        self.contexts = [doc.page_content for doc in docs]
        self.context_ids = [doc.metadata["eval_id"] for doc in docs]
        return docs


class RequestBudgetExceeded(RuntimeError):
    pass


class RequestMeter:
    def __init__(self, limit: int):
        self.limit = limit
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_responses = 0

    async def on_request(self, request):
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            raise ValueError("Unsupported API endpoint; request blocked before sending")
        if self.requests >= self.limit:
            raise RequestBudgetExceeded("API request limit reached")
        self.requests += 1

    async def on_response(self, response):
        if response.request.url.path.endswith("/chat/completions"):
            await response.aread()
            if response.is_success:
                usage = response.json().get("usage")
                if usage:
                    self.prompt_tokens += usage.get("prompt_tokens", 0)
                    self.completion_tokens += usage.get("completion_tokens", 0)
                    self.usage_responses += 1


async def measure(chain, recorder, question, meter, cache=None) -> dict:
    recorder.contexts, recorder.context_ids, recorder.retrieval_ms = [], [], None
    first_request = meter.requests
    event_start = len(cache.events) if cache else 0
    started = time.perf_counter()
    result = {"question_id": question["id"], "user_input": question["question"], "ok": False}
    try:
        result["response"] = await chain.ainvoke(
            {
                "question": question["question"],
                "history": "",
                "user_level": "public",
            }
        )
        result["ok"] = True
    except Exception as exc:
        result["error"] = error_info(exc)
    result.update(
        {
            "latency_ms": (time.perf_counter() - started) * 1000,
            "retrieval_ms": recorder.retrieval_ms,
            "external_requests": meter.requests - first_request,
            "retrieved_contexts": list(recorder.contexts),
            "context_ids": list(recorder.context_ids),
            "cache_events": list(cache.events[event_start:]) if cache else [],
        }
    )
    return result


def check_cache_expectation(row: dict, group: str, seed: dict) -> None:
    if not row["ok"]:
        return
    events = row["cache_events"]
    valid = {
        "disabled": not events and row["external_requests"] == 1,
        "miss": len(events) == 1 and not events[0]["hit"] and row["external_requests"] == 1,
        "hit": (
            len(events) == 1
            and events[0]["hit"]
            and row["external_requests"] == 0
            and seed["ok"]
            and bool(seed["cache_events"])
            and events[0]["input_hash"] == seed["cache_events"][0]["input_hash"]
            and row.get("response") == seed.get("response")
        ),
    }[group]
    if not valid:
        row.update(ok=False, error={"type": "CacheExpectationMismatch"})


def eval_settings(settings):
    """Keep the evaluated prompt identical to the recorded contexts.

    Production caps top-3 parents inside 3200 chars. If the judge scored the full
    top-k while the LLM saw a truncated subset, the quality numbers would describe
    a prompt that was never actually sent.
    """
    return dataclasses.replace(
        settings, max_parents=settings.rerank_top_n, context_budget_chars=10**7
    )


async def run_cache(questions, args, settings, engine, llm, cache, meter, emit):
    from tiered_rag.chain import build_chain

    settings = eval_settings(settings)
    recorder = RecordingPool(engine, settings.rerank_top_n)
    chains = {
        "disabled": build_chain(llm.model_copy(update={"cache": False}), recorder, settings),
        "enabled": build_chain(llm.model_copy(update={"cache": cache}), recorder, settings),
    }
    warmup = await measure(chains["disabled"], recorder, questions[0], meter)
    emit(dict(warmup, kind="warmup", phase="model_warmup"))
    if not warmup["ok"]:
        return
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        for question in questions:
            pair = f"{repeat}:{question['id']}"
            cache.namespace = pair + ":hot"
            seed = await measure(chains["enabled"], recorder, question, meter, cache)
            emit(dict(seed, kind="warmup", phase="cache_prefill", pair=pair))
            if not seed["ok"]:
                return
            groups = ["disabled", "miss", "hit"]
            rng.shuffle(groups)
            for group in groups:
                cache.namespace = pair + (":cold" if group == "miss" else ":hot")
                chain = chains["disabled" if group == "disabled" else "enabled"]
                row = await measure(
                    chain,
                    recorder,
                    question,
                    meter,
                    None if group == "disabled" else cache,
                )
                check_cache_expectation(row, group, seed)
                emit(dict(row, kind="cache", group=group, pair=pair))
                if not row["ok"]:
                    return


async def run_quality(questions, args, settings, engines, llm, metrics, meter, emit):
    from tiered_rag.chain import build_chain

    settings = eval_settings(settings)
    for question in questions:
        for variant in args.variants:
            recorder = RecordingPool(engines[variant], settings.rerank_top_n)
            chain = build_chain(llm.model_copy(update={"cache": False}), recorder, settings)
            row = await measure(chain, recorder, question, meter)
            row.update(kind="quality", variant=variant, reference=question["reference"], scores={})
            row["gold_document_recall_at_3"] = len(
                set(question["gold_ids"]) & set(row["context_ids"])
            ) / len(question["gold_ids"])
            first_judge_request = meter.requests
            if row["ok"]:
                common = {
                    "user_input": question["question"],
                    "retrieved_contexts": row["retrieved_contexts"],
                }
                inputs = {
                    "faithfulness": dict(common, response=row["response"]),
                    "answer_relevancy": {
                        "user_input": question["question"],
                        "response": row["response"],
                    },
                    "context_precision": dict(common, reference=question["reference"]),
                    "context_recall": dict(common, reference=question["reference"]),
                }
                for name, metric in metrics.items():
                    try:
                        value = float((await metric.ascore(**inputs[name])).value)
                        row["scores"][name] = value if math.isfinite(value) else None
                        if not math.isfinite(value):
                            row.setdefault("metric_errors", {})[name] = {"type": "NonFiniteScore"}
                    except Exception as exc:
                        row["scores"][name] = None
                        row.setdefault("metric_errors", {})[name] = error_info(exc)
            row["judge_requests"] = meter.requests - first_judge_request
            emit(row)
            if not row["ok"] or meter.requests >= meter.limit:
                return
            if len(row.get("metric_errors", {})) == len(metrics):
                return


def rank_metrics(ranked_ids: list[str], gold_ids: list[str]) -> dict:
    gold = set(gold_ids)
    result = {"mrr": 0.0}
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold:
            result["mrr"] = round(1.0 / rank, 4)
            break
    for k in RECALL_KS:
        found = gold & set(ranked_ids[:k])
        result[f"recall@{k}"] = round(len(found) / len(gold), 4)
        result[f"full@{k}"] = len(found) == len(gold)
    return result


def run_recall(questions, args, engines, emit):
    for question in questions:
        for variant in args.variants:
            started = time.perf_counter()
            row = {
                "kind": "recall",
                "variant": variant,
                "question_id": question["id"],
                "category": question["category"],
                "user_input": question["question"],
                "gold_ids": list(question["gold_ids"]),
                "ok": False,
            }
            try:
                docs = engines[variant].invoke(question["question"])
            except Exception as exc:
                row["error"] = error_info(exc)
                emit(row)
                return
            row["ranked_ids"] = [str(doc.metadata.get("eval_id", "")) for doc in docs]
            row["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 1)
            row.update(rank_metrics(row["ranked_ids"], row["gold_ids"]))
            row["ok"] = True
            emit(row)


def summarize(rows: list[dict]) -> dict:
    summary = {"cache": {}, "ragas": {}, "recall": {}}
    for group in ("disabled", "miss", "hit"):
        attempts = [r for r in rows if r["kind"] == "cache" and r["group"] == group]
        good = [r for r in attempts if r["ok"]]
        latencies = [r["latency_ms"] for r in good]
        events = [e for r in attempts for e in r["cache_events"]]
        summary["cache"][group] = {
            "attempted": len(attempts),
            "valid": len(good),
            "failed": len(attempts) - len(good),
            "p50_ms": percentile(latencies, 0.5),
            "p95_ms": percentile(latencies, 0.95),
            "p95_small_sample_warning": len(good) < 100,
            "observed_hit_rate": sum(e["hit"] for e in events) / len(events) if events else None,
            "external_requests": sum(r["external_requests"] for r in attempts),
        }
    for variant in VARIANTS:
        attempts = [r for r in rows if r["kind"] == "quality" and r["variant"] == variant]
        if not attempts:
            continue
        result = {
            "attempted": len(attempts),
            "generation_failed": sum(not r["ok"] for r in attempts),
        }
        for name in METRICS:
            values = [r["scores"].get(name) for r in attempts]
            valid = [v for v in values if v is not None and math.isfinite(v)]
            result[name] = {
                "mean": statistics.mean(valid) if valid else None,
                "valid": len(valid),
                "failed_or_skipped": len(attempts) - len(valid),
            }
        summary["ragas"][variant] = result
    for variant in VARIANTS:
        attempts = [r for r in rows if r["kind"] == "recall" and r["variant"] == variant]
        if not attempts:
            continue
        good = [r for r in attempts if r["ok"]]
        result = {
            "attempted": len(attempts),
            "failed": len(attempts) - len(good),
            "retrieval_p50_ms": percentile([r["retrieval_ms"] for r in good], 0.5),
        }
        groups = {"all": good}
        for category in sorted({r["category"] for r in good}):
            groups[category] = [r for r in good if r["category"] == category]
        for label, subset in groups.items():
            if not subset:
                continue
            result[label] = {
                "n": len(subset),
                "mrr": round(statistics.mean(r["mrr"] for r in subset), 4),
            }
            for k in RECALL_KS:
                result[label][f"recall@{k}"] = round(
                    statistics.mean(r[f"recall@{k}"] for r in subset), 4
                )
                result[label][f"full@{k}"] = round(
                    sum(r[f"full@{k}"] for r in subset) / len(subset), 4
                )
        summary["recall"][variant] = result
    return summary


def preflight(args):
    from huggingface_hub import snapshot_download

    packages = [
        "langchain-core",
        "langchain-community",
        "langchain-classic",
        "langchain-openai",
        "langchain-huggingface",
        "chromadb",
        "sentence-transformers",
        "redis",
    ]
    if args.mode in ("check", "ragas", "all"):
        packages += ["ragas", "instructor", "openai"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    paths = {
        model: snapshot_download(
            model,
            cache_dir=str(args.hf_home / "hub"),
            local_files_only=True,
        )
        for model in (args.embedding_model, args.reranker_model)
    }
    redis_client = None
    if args.mode in ("check", "cache", "all"):
        from redis import Redis

        redis_client = Redis(
            host="127.0.0.1",
            port=args.redis_port,
            db=args.redis_db,
            protocol=2,
            socket_connect_timeout=3,
            socket_timeout=5,
            decode_responses=True,
        )
        try:
            redis_client.ping()
        except Exception:
            redis_client.close()
            print("Redis 未连接：请先启动本地 Redis；也可以先用 --mode ragas（不需要 Redis）。")
            raise
    print("依赖和本地模型缓存检查通过；未读取 .env，未请求模型 API。")
    return versions, paths, redis_client


def build_engines(settings, model_paths, dataset, run_dir, device, chunking="flat"):
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    from tiered_rag.indexer import chunk_parent_child, chunk_to_documents
    from tiered_rag.retrieval import RetrieverPool

    embeddings = HuggingFaceEmbeddings(
        model_name=model_paths[settings.embedding_model],
        model_kwargs={"device": device, "local_files_only": True},
    )
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=model_paths[settings.reranker_model],
        model_kwargs={"device": device},
    )
    client = chromadb.PersistentClient(
        path=str(run_dir / "index"),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    store = Chroma(client=client, collection_name="synthetic_eval", embedding_function=embeddings)
    documents = []
    for item in dataset["documents"]:
        if chunking == "parent_child":
            chunks = chunk_parent_child(
                item["text"], "public", f"{item['id']}.md", settings
            ).children
        else:
            chunks = chunk_to_documents(item["text"], "public", settings)
        for doc in chunks:
            doc.metadata.update(source="synthetic-handbook", eval_id=item["id"])
        documents.extend(chunks)
    store.add_documents(documents)
    pool = RetrieverPool(store, cross_encoder, settings)
    pool.rebuild(documents)
    reranked = pool.get("public")
    hybrid = reranked.base_retriever
    engines = {
        "bm25": hybrid.retrievers[0],
        "dense": hybrid.retrievers[1],
        "hybrid": hybrid,
        "hybrid_rerank": reranked,
    }
    for engine in engines.values():
        engine.invoke(dataset["questions"][0]["question"])
    return engines, len(documents)


async def execute(args, key, settings, model_paths, dataset, redis_client, run_dir, metadata):
    import httpx
    from langchain_openai import ChatOpenAI

    questions = dataset["questions"][: args.questions]
    rows = []
    cache = make_cache(redis_client, f"tiered-rag-eval:{run_dir.name}:") if redis_client else None
    meter = RequestMeter(args.max_api_requests)
    fatal_error = None
    with (run_dir / "records.jsonl").open("x", encoding="utf-8") as raw:

        def emit(row):
            rows.append(row)
            raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            raw.flush()
            status = "OK" if row["ok"] and not row.get("metric_errors") else "FAILED"
            print(f"{row['kind']} {row['question_id']}: {status}")

        try:
            print("正在加载本地模型并构建独立测试索引；启动和预热不计入请求耗时。")
            engines, chunks = build_engines(
                settings, model_paths, dataset, run_dir, args.device, args.chunking
            )
            metadata["chunks"] = chunks
            if args.mode == "recall":
                run_recall(questions, args, engines, emit)
            async with httpx.AsyncClient(
                timeout=args.timeout,
                event_hooks={"request": [meter.on_request], "response": [meter.on_response]},
            ) as http:
                llm = ChatOpenAI(
                    model=args.model,
                    api_key=key,
                    base_url=args.base_url,
                    temperature=0,
                    max_tokens=512,
                    max_retries=0,
                    timeout=args.timeout,
                    cache=False,
                    streaming=False,
                    http_async_client=http,
                )
                if args.mode in ("cache", "all"):
                    await run_cache(
                        questions,
                        args,
                        settings,
                        engines["hybrid_rerank"],
                        llm,
                        cache,
                        meter,
                        emit,
                    )
                if args.mode in ("ragas", "all") and meter.requests < meter.limit:
                    from openai import AsyncOpenAI
                    from ragas.embeddings import HuggingFaceEmbeddings as RagasEmbeddings
                    from ragas.llms import llm_factory
                    from ragas.metrics.collections import (
                        AnswerRelevancy,
                        ContextPrecisionWithReference,
                        ContextRecall,
                        Faithfulness,
                    )

                    judge_client = AsyncOpenAI(
                        api_key=key,
                        base_url=args.base_url,
                        http_client=http,
                        max_retries=0,
                        timeout=args.timeout,
                    )
                    judge = llm_factory(
                        args.judge_model,
                        client=judge_client,
                        temperature=0,
                        max_tokens=2048,
                    )
                    judge_embeddings = RagasEmbeddings(
                        model=model_paths[settings.embedding_model],
                        device=args.device,
                        local_files_only=True,
                    )
                    metrics = {
                        "faithfulness": Faithfulness(llm=judge),
                        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=judge_embeddings),
                        "context_precision": ContextPrecisionWithReference(llm=judge),
                        "context_recall": ContextRecall(llm=judge),
                    }
                    await run_quality(
                        questions,
                        args,
                        settings,
                        engines,
                        llm,
                        metrics,
                        meter,
                        emit,
                    )
        except Exception as exc:
            fatal_error = error_info(exc)
        finally:
            if cache:
                try:
                    cache.cleanup()
                except Exception as exc:
                    metadata["cache_cleanup_error"] = error_info(exc)
            summary = summarize(rows)
            expected_cache = (
                len(questions) * args.repeats * 3 if args.mode in ("cache", "all") else 0
            )
            expected_quality = (
                len(questions) * len(args.variants) if args.mode in ("ragas", "all") else 0
            )
            expected_recall = len(questions) * len(args.variants) if args.mode == "recall" else 0
            cache_rows = [r for r in rows if r["kind"] == "cache"]
            quality_rows = [r for r in rows if r["kind"] == "quality"]
            recall_rows = [r for r in rows if r["kind"] == "recall"]
            complete = (
                fatal_error is None
                and "cache_cleanup_error" not in metadata
                and len(cache_rows) == expected_cache
                and len(quality_rows) == expected_quality
                and len(recall_rows) == expected_recall
                and all(r["ok"] for r in rows)
                and all(not r.get("metric_errors") for r in quality_rows)
            )
            summary.update(
                {
                    "status": "complete" if complete else "partial_or_failed",
                    "metadata": metadata,
                    "fatal_error": fatal_error,
                    "api_usage": vars(meter),
                    "expected_cache_rows": expected_cache,
                    "expected_quality_rows": expected_quality,
                    "expected_recall_rows": expected_recall,
                }
            )
            (run_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            print(f"结果状态：{summary['status']}；请求尝试数：{meter.requests}")
            print(f"结果目录：{run_dir}")
            print("只需分享 summary.json；小样本结果不代表真实私有 PDF 的问答质量。")
    return 0 if complete else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("check", "cache", "ragas", "recall", "all"), default="check"
    )
    parser.add_argument(
        "--questions", type=int, default=3, help="First N synthetic questions (1-22)"
    )
    parser.add_argument("--repeats", type=int, default=3, help="Cache trials per question (1-20)")
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=["dense", "hybrid_rerank"]
    )
    parser.add_argument(
        "--chunking",
        choices=("flat", "parent_child"),
        default="flat",
        help="index chunking: flat 800-char chunks, or parent 800 / child 200",
    )
    parser.add_argument("--hf-home", type=Path, default=ROOT / ".hf_cache")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--judge-model", default="qwen-plus")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-api-requests", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.questions <= 22 or not 1 <= args.repeats <= 20:
        parser.error("questions must be 1-22; repeats must be 1-20")
    if args.max_api_requests < 1 or args.timeout <= 0:
        parser.error("timeout and max-api-requests must be positive")
    endpoint = urlsplit(args.base_url)
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or any(
            [
                endpoint.username,
                endpoint.password,
                endpoint.query,
                endpoint.fragment,
            ]
        )
    ):
        parser.error("base-url must be HTTPS without credentials, query or fragment")
    args.variants = list(dict.fromkeys(args.variants))
    return args


def main() -> int:
    args = parse_args()
    configure_environment(args.hf_home)
    sys.path.insert(0, str(ROOT / "src"))
    from tiered_rag.config import Settings

    redis_client = None
    try:
        dataset, dataset_hash = load_dataset()
        versions, paths, redis_client = preflight(args)
        if args.mode == "check":
            print(json.dumps(versions, ensure_ascii=False, indent=2))
            return 0
        settings = Settings.from_env(
            {
                "TIERED_RAG_EMBEDDING_MODEL": args.embedding_model,
                "TIERED_RAG_RERANKER_MODEL": args.reranker_model,
                "TIERED_RAG_LEVELS": "public",
            }
        )
        if args.mode == "recall":
            # One wide ranking, then truncated per k: rebuilding per k would rerun the
            # cross-encoder for every k and cost minutes for no extra information.
            settings = dataclasses.replace(settings, rerank_top_n=settings.retrieve_k)
            key = "offline-no-call"
            print("离线检索评测：只用本地向量与重排模型，不发送内容到 API，不产生费用。")
        else:
            print(
                f"将把自编手册片段发送给 {urlsplit(args.base_url).hostname}，"
                "会产生模型调用费用。"
            )
            print(
                f"问题数：{args.questions}；模型请求尝试上限：{args.max_api_requests}（非金额上限）。"
            )
            if not sys.stdin.isatty():
                print("请在自己的交互式终端运行，以安全输入密钥。")
                return 1
            if input("确认执行请输入 RUN，其余输入退出：").strip() != "RUN":
                return 0
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass("输入 API Key（不显示、不写入文件）：").strip()
            if not key:
                print("未输入密钥，已退出。")
                return 1
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = ROOT / "data" / "evaluation" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "run_id": run_id,
            "mode": args.mode,
            "dataset": dataset["name"],
            "dataset_sha256": dataset_hash,
            "versions": versions,
            "python": platform.python_version(),
            "os": platform.platform(),
            "device": args.device,
            "generation_model": args.model,
            "judge_model": args.judge_model,
            "api_host": urlsplit(args.base_url).hostname,
            "same_generation_and_judge_model": args.model == args.judge_model,
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
            "questions": args.questions,
            "repeats": args.repeats,
            "variants": args.variants,
            "seed": args.seed,
            "concurrency": 1,
            "temperature": 0,
            "generation_max_tokens": 512,
            "judge_max_tokens": 2048,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "child_chunk_size": settings.child_chunk_size,
            "chunking": args.chunking,
            "retrieve_k": settings.retrieve_k,
            "rerank_top_n": settings.rerank_top_n,
            "bm25_weight": settings.bm25_weight,
            "vector_weight": settings.vector_weight,
            "scope": "Non-streaming build_chain; empty history; no HTTP/UI/PDF extraction test",
            "limitations": [
                "Small synthetic public corpus; not production quality evidence",
                "Hand-authored references; independent human review still needed",
                "No authentication, history isolation or streaming cache validation",
                "Latency excludes model loading, local warmup, HTTP and history storage",
                "P95 uses linear interpolation and is unstable with few samples",
                "API request count includes warmup, prefills and judge retries; not billing proof",
            ],
        }
        if args.mode == "recall":
            metadata["recall_ks"] = list(RECALL_KS)
            metadata["limitations"].append(
                "recall mode: rerank_top_n here is the candidate width; each recall@k is "
                "truncated from that single ranking and equals an online top_n=k run; "
                "no generation or judge calls, so answer quality is not measured"
            )
        return asyncio.run(
            execute(
                args,
                key,
                settings,
                paths,
                dataset,
                redis_client,
                run_dir,
                metadata,
            )
        )
    except Exception as exc:
        print("执行停止（仅显示脱敏错误）：" + json.dumps(error_info(exc)))
        return 1
    finally:
        if redis_client:
            redis_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
