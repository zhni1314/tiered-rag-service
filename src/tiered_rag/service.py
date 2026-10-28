"""FastAPI application: startup wiring plus HTTP routes."""

from __future__ import annotations

import logging
import os
import shutil
import time
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

from contextlib import asynccontextmanager  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from .bootstrap import configure_huggingface  # noqa: E402
from .config import Settings  # noqa: E402
from .history import HistoryStore  # noqa: E402
from .indexer import ingest, safe_filename  # noqa: E402
from .schemas import ChatRequest, ChatResponse, HistoryItem  # noqa: E402

log = logging.getLogger(__name__)


def _load_cross_encoder(model_name: str):
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return HuggingFaceCrossEncoder(model_name=model_name)


def _require_level(settings: Settings, level: str) -> str:
    if level not in settings.levels:
        raise HTTPException(status_code=400, detail=f"unknown level: {level}")
    return level


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings: Settings = application.state.settings
    settings.require_llm_credentials()
    # HF_HOME / HF_ENDPOINT must be set before the model loaders are imported.
    configure_huggingface(settings)

    from langchain.chat_models import init_chat_model
    from langchain_community.cache import RedisCache
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_core.globals import set_llm_cache
    from langchain_huggingface import HuggingFaceEmbeddings
    from redis import Redis

    from .chain import build_chain
    from .retrieval import RetrieverPool

    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    redis = Redis.from_url(settings.redis_url, protocol=2, decode_responses=True)
    set_llm_cache(RedisCache(redis))

    log.info("loading embedder=%s reranker=%s", settings.embedding_model, settings.reranker_model)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    cross_encoder = _load_cross_encoder(settings.reranker_model)

    vectorstore = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )

    stored = vectorstore.get()
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(stored.get("documents", []), stored.get("metadatas", []))
    ]
    if not docs:
        docs = [Document(page_content="欢迎使用 RAG 智能问答系统。", metadata={"level": "public"})]
        vectorstore.add_documents(docs)

    pool = RetrieverPool(vectorstore, cross_encoder, settings)
    pool.rebuild(docs)

    llm = init_chat_model(
        model=settings.qwen_model,
        model_provider="openai",
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
    )

    application.state.redis = redis
    application.state.history = HistoryStore(redis, settings.history_window)
    application.state.vectorstore = vectorstore
    application.state.pool = pool
    application.state.chain = build_chain(llm, pool, settings)

    log.info("ready: %d chunks, levels=%s", len(pool.documents), list(settings.levels))
    yield
    redis.close()


app = FastAPI(
    title="tiered-rag-service",
    description=(
        "Chinese-document QA: hybrid BM25+dense retrieval, cross-encoder rerank, "
        "conditional query rewriting, per-tier retrieval engines, Redis history and LLM cache."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.state.settings = Settings.from_env()


@app.get("/healthz")
async def healthz(request: Request) -> dict:
    return {"status": "ok", "chunks": len(request.app.state.pool.documents)}


@app.post("/upload", summary="Upload a PDF, parse and index it")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    level: str = Form("public"),
) -> dict:
    settings: Settings = request.app.state.settings
    _require_level(settings, level)

    try:
        dest = safe_filename(file.filename or "", settings.upload_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with dest.open("wb") as buffer:
        file.file.seek(0)
        shutil.copyfileobj(file.file, buffer)

    try:
        docs = await run_in_threadpool(ingest, dest, level, settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"parse failed: {exc}") from exc

    if not docs:
        return {"message": "no text extracted", "filename": dest.name, "chunks_added": 0}

    await run_in_threadpool(request.app.state.vectorstore.add_documents, docs)
    await run_in_threadpool(request.app.state.pool.add, docs)

    return {
        "message": f"indexed at level {level}",
        "filename": dest.name,
        "chunks_added": len(docs),
    }


@app.post("/chat", response_model=ChatResponse, summary="Ask a question")
async def chat(request: ChatRequest, http: Request) -> ChatResponse:
    settings: Settings = http.app.state.settings
    _require_level(settings, request.user_level)

    started = time.perf_counter()
    history_text = http.app.state.history.as_prompt_text(request.session_id)
    answer = await http.app.state.chain.ainvoke(
        {"question": request.question, "history": history_text, "user_level": request.user_level}
    )
    http.app.state.history.append_turn(request.session_id, request.question, answer)

    return ChatResponse(
        answer=answer,
        latency=round(time.perf_counter() - started, 4),
        level=request.user_level,
    )


@app.post("/chat/stream", summary="Ask a question, streamed as plain-text chunks")
async def chat_stream(request: ChatRequest, http: Request) -> StreamingResponse:
    settings: Settings = http.app.state.settings
    _require_level(settings, request.user_level)

    history_text = http.app.state.history.as_prompt_text(request.session_id)
    payload = {
        "question": request.question,
        "history": history_text,
        "user_level": request.user_level,
    }

    async def generate():
        parts: list[str] = []
        async for chunk in http.app.state.chain.astream(payload):
            parts.append(chunk)
            yield chunk
        http.app.state.history.append_turn(request.session_id, request.question, "".join(parts))

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/history/{session_id}", response_model=list[HistoryItem], summary="Session history")
async def get_history(request: Request, session_id: str) -> list[dict]:
    return request.app.state.history.all_items(session_id)
