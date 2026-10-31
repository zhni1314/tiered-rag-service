"""Offline chunking benchmark: does the assembled prompt contain the answer?

Compares flat 800-char chunks with parent(800)/child(200) chunking on the public
sample corpus, under the same 3200-char prompt budget. Retrieval is BM25+dense
(no reranker by default), with an optional hybrid_rerank variant that adds the
cross-encoder reranker. The reranker only reorders the same candidates, so the
default run stays fast and free.

    python scripts/chunking_bench.py
    python scripts/chunking_bench.py --variants hybrid_rerank   # slow on CPU
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "scripts" / "chunking_dataset.json"

sys.path.insert(0, str(ROOT / "src"))

import jieba  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)

from tiered_rag.bootstrap import configure_huggingface  # noqa: E402
from tiered_rag.config import Settings  # noqa: E402
from tiered_rag.context import assemble_context  # noqa: E402
from tiered_rag.indexer import (  # noqa: E402
    chunk_parent_child,
    chunk_to_documents,
    extract_pdf_text,
)
from tiered_rag.text_cleaning import strip_repeating_layout  # noqa: E402

MODES = ("flat", "parent_child")


class LimitedRerank:
    """Rerank only the first ``limit`` fused candidates.

    LangChain's CrossEncoderReranker scores every candidate it receives, so the
    rerank bill is set by the fused list length, not by top_n.
    """

    def __init__(self, base, compressor, limit: int):
        self.base = base
        self.compressor = compressor
        self.limit = limit

    def invoke(self, query):
        return self.compressor.compress_documents(self.base.invoke(query)[: self.limit], query)


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def load_corpus(settings: Settings, folder: Path) -> list[tuple[str, str]]:
    docs = []
    for pdf in sorted(folder.glob("*.pdf")):
        cleaned = strip_repeating_layout(
            extract_pdf_text(pdf),
            header_pattern=settings.header_pattern,
            footer_pattern=settings.footer_pattern,
            drop_cjk_line_breaks=settings.drop_cjk_line_breaks,
        )
        docs.append((pdf.name, cleaned))
    return docs


def build_index(
    mode: str,
    corpus,
    settings,
    embeddings,
    run_dir: Path,
    cross_encoder=None,
    rerank_candidates: list[int] | None = None,
):
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import Chroma

    client = chromadb.PersistentClient(
        path=str(run_dir / mode / "index"),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    store = Chroma(client=client, collection_name=f"bench_{mode}", embedding_function=embeddings)

    documents, parents = [], {}
    for name, text in corpus:
        if mode == "parent_child":
            result = chunk_parent_child(text, "public", name, settings)
            documents.extend(result.children)
            parents.update(result.parents)
        else:
            chunks = chunk_to_documents(text, "public", settings)
            for chunk in chunks:
                chunk.metadata["source"] = name
            documents.extend(chunks)
    store.add_documents(documents)

    bm25 = BM25Retriever.from_documents(
        documents, preprocess_func=jieba.lcut, k=settings.retrieve_k
    )
    dense = store.as_retriever(
        search_kwargs={"k": settings.retrieve_k, "filter": {"level": {"$in": ["public"]}}}
    )
    hybrid = EnsembleRetriever(
        retrievers=[bm25, dense], weights=[settings.bm25_weight, settings.vector_weight]
    )
    engines = {"dense": dense, "hybrid": hybrid}
    if cross_encoder is not None:
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=settings.rerank_top_n)
        if rerank_candidates:
            for limit in rerank_candidates:
                engines[f"hybrid_rerank@{limit}"] = LimitedRerank(hybrid, compressor, limit)
        else:
            engines["hybrid_rerank"] = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=hybrid,
            )
    for engine in engines.values():
        engine.invoke("预热查询")
    return engines, documents, parents


def assemble(mode: str, docs, parents, settings) -> str:
    if mode == "parent_child":
        return assemble_context(
            docs,
            parents,
            max_parents=settings.max_parents,
            budget=settings.context_budget_chars,
        )
    # Flat chunks are their own parents; four of them roughly fill the same budget.
    return assemble_context(docs, {}, max_parents=4, budget=settings.context_budget_chars)


def summarize(rows: list[dict]) -> dict:
    def block(subset: list[dict]) -> dict:
        return {
            "n": len(subset),
            "hit_rate": round(sum(row["hit"] for row in subset) / len(subset), 4),
            "partial_rate": round(sum(row["partial"] for row in subset) / len(subset), 4),
        }

    return {
        **block(rows),
        "prompt_chars_mean": round(statistics.mean(row["prompt_chars"] for row in rows)),
        "retrieval_ms_median": round(statistics.median(row["retrieval_ms"] for row in rows), 2),
        "by_category": {
            category: block([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
    }


def run(args) -> int:
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    settings = Settings.from_env()
    if args.rerank_top_n is not None:
        settings = dataclasses.replace(settings, rerank_top_n=args.rerank_top_n)
    configure_huggingface(settings)

    from langchain_huggingface import HuggingFaceEmbeddings

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "data" / "evaluation" / f"chunking-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    corpus = load_corpus(settings, ROOT / args.corpus)
    print(f"corpus: {len(corpus)} docs, {sum(len(text) for _, text in corpus)} chars")
    variants = [name for name in args.variants if name != "hybrid_rerank"]
    if args.rerank_candidates:
        variants += [f"hybrid_rerank@{limit}" for limit in args.rerank_candidates]
    elif "hybrid_rerank" in args.variants:
        variants.append("hybrid_rerank")
    uses_reranker = any(name.startswith("hybrid_rerank") for name in variants)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"local_files_only": True},
    )
    cross_encoder = None
    if uses_reranker:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        cross_encoder = HuggingFaceCrossEncoder(
            model_name=settings.reranker_model,
            model_kwargs={"local_files_only": True},
        )
    summary = {
        "run_id": run_id,
        "dataset": dataset["name"],
        "corpus": args.corpus,
        "modes": args.modes,
        "variants": variants,
        "k_candidates": settings.rerank_top_n,
        "budget_chars": settings.context_budget_chars,
        "reranker": "included" if uses_reranker else "excluded",
        "results": {},
    }
    records: list[dict] = []

    for mode in args.modes:
        engines, documents, parents = build_index(
            mode,
            corpus,
            settings,
            embeddings,
            run_dir,
            cross_encoder,
            args.rerank_candidates,
        )
        summary["results"][mode] = {
            "chunks": len(documents),
            "parents": len(parents) or len(documents),
            "variants": {},
        }
        print(f"\n[{mode}] {len(documents)} chunks / {len(parents) or len(documents)} parents")
        for variant in variants:
            engine = engines[variant]
            rows = []
            for question in dataset["questions"]:
                started = time.perf_counter()
                ranked = engine.invoke(question["question"])[: settings.rerank_top_n]
                elapsed = (time.perf_counter() - started) * 1000
                prompt = normalize(assemble(mode, ranked, parents, settings))
                found = [normalize(gold) in prompt for gold in question["gold"]]
                row = {
                    "id": question["id"],
                    "category": question["category"],
                    "hit": all(found),
                    "partial": sum(found) / len(found),
                    "prompt_chars": len(prompt),
                    "retrieval_ms": round(elapsed, 2),
                }
                rows.append(row)
                record = {"mode": mode, "variant": variant, **row}
                records.append(record)
            summary["results"][mode]["variants"][variant] = summarize(rows)
            stats = summary["results"][mode]["variants"][variant]
            print(
                f"  {variant:7s} hit={stats['hit_rate']:.3f} "
                f"partial={stats['partial_rate']:.3f} "
                f"prompt={stats['prompt_chars_mean']:.0f}字 "
                f"retrieval={stats['retrieval_ms_median']:.0f} ms"
            )
            for category, value in stats["by_category"].items():
                print(f"    {category:15s} hit={value['hit_rate']:.3f} (n={value['n']})")

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    print(f"\n结果目录：{run_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DATASET), help="labelled questions JSON")
    parser.add_argument("--corpus", default="examples/sample_corpus", help="folder of PDFs")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("dense", "hybrid", "hybrid_rerank"),
        default=["dense", "hybrid"],
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="override TIERED_RAG_RERANK_TOP_N for this run (outputs kept after rerank)",
    )
    parser.add_argument(
        "--rerank-candidates",
        nargs="+",
        type=int,
        default=None,
        help="fused candidates fed to the reranker, e.g. 5 10 19",
    )
    args = parser.parse_args()
    if args.rerank_candidates and any(limit < 1 for limit in args.rerank_candidates):
        parser.error("--rerank-candidates must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
