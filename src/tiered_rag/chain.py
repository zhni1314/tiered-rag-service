"""The RAG runnable: (conditional) query rewrite -> tiered retrieval -> grounded answer."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .config import Settings
from .retrieval import RetrieverPool

log = logging.getLogger(__name__)

# Chinese anaphora that make a follow-up question unanswerable out of context.
_PRONOUNS = ("它", "他", "她", "这", "那", "该", "此", "上面", "刚才", "它们", "这些", "那些")

_ANSWER_SYSTEM = """你是一个极其严谨的助手。
【铁律】：
1. 必须严格根据【参考资料】回答用户问题。
2. 严禁使用你自己的常识、推理或扩写。
3. 如果【参考资料】中完全没有答案，请直接回答"资料中未提及"。

【历史对话】：
{history}
"""

_REWRITE_SYSTEM = (
    "你是一个查询重写助手。请结合【历史对话】，将用户的最新问题重写为一个独立、完整、"
    "没有代词的查询语句。只输出重写后的句子。\n【历史对话】：\n{history}"
)


def build_chain(llm, pool: RetrieverPool, settings: Settings):
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _ANSWER_SYSTEM),
            ("human", "【参考资料】：\n{context}\n\n【用户问题】：{question}"),
        ]
    )
    rewrite_chain = (
        ChatPromptTemplate.from_messages([("system", _REWRITE_SYSTEM), ("human", "{question}")])
        | llm
        | StrOutputParser()
    )

    def smart_rewrite(x: dict) -> str:
        """Skip the extra LLM call unless the turn actually needs context."""
        question, history = x["question"], x["history"]
        if not history.strip() or not any(p in question for p in _PRONOUNS):
            log.debug("rewrite skipped: %s", question)
            return question
        log.info("rewrite triggered: %s", question)
        return rewrite_chain.invoke({"question": question, "history": history})

    async def get_context(x: dict) -> str:
        engine = pool.get(x["user_level"])
        # The cross-encoder rerank is CPU-bound; keep it off the event loop.
        docs = await asyncio.to_thread(engine.invoke, x["rewritten_query"])
        return "\n".join(doc.page_content for doc in docs)

    return (
        RunnablePassthrough.assign(rewritten_query=RunnableLambda(smart_rewrite))
        | {
            "context": get_context,
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )
