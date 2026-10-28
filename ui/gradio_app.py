"""Thin Gradio UI over the HTTP API.

Run the backend first, then:  python ui/gradio_app.py
"""

from __future__ import annotations

import os

import gradio as gr
import requests

BACKEND_URL = os.getenv("TIERED_RAG_BACKEND_URL", "http://127.0.0.1:8000/chat/stream")
LEVELS = [s.strip() for s in os.getenv("TIERED_RAG_LEVELS", "public,secret").split(",") if s.strip()]


def extract_text(message) -> str:
    """Gradio hands back str / dict / list depending on version and input type."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return message.get("text", "")
    if isinstance(message, list):
        for item in message:
            if isinstance(item, dict) and "text" in item:
                return item["text"]
            if isinstance(item, str):
                return item
    return str(message)


def stream_answer(question: str, level: str, session_id: str):
    payload = {"session_id": session_id, "question": question, "user_level": level}
    try:
        response = requests.post(BACKEND_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        partial = ""
        for line in response.iter_lines():
            if line:
                partial += line.decode("utf-8")
                yield partial
    except requests.RequestException as exc:
        yield f"⚠️ 后端连接失败：{exc}"


with gr.Blocks(title="tiered-rag-service") as demo:
    gr.Markdown("# 🛡️ 分级文档问答（tiered-rag-service）")
    gr.Markdown(
        "FastAPI + LangChain + Chroma + BM25 + cross-encoder rerank；"
        "支持多轮对话、条件查询重写、分级检索链路与 SSE 流式输出。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=600)
            with gr.Row():
                msg = gr.Textbox(placeholder="请输入你的问题...", show_label=False, scale=4)
                level = gr.Dropdown(choices=LEVELS, value=LEVELS[0], label="模拟用户权限", scale=1)
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.ClearButton([msg, chatbot], value="清空对话")

    session_id = gr.State(value="web_user_001")

    def user_submit(message, chat_history):
        chat_history.append({"role": "user", "content": extract_text(message)})
        return "", chat_history

    def bot_stream(chat_history, chosen_level, sid):
        question = extract_text(chat_history[-1]["content"])
        chat_history.append({"role": "assistant", "content": ""})
        for partial in stream_answer(question, chosen_level, sid):
            chat_history[-1]["content"] = partial
            yield chat_history

    inputs = [msg, chatbot]
    msg.submit(user_submit, inputs, [msg, chatbot]).then(
        bot_stream, [chatbot, level, session_id], [chatbot]
    )
    submit_btn.click(user_submit, inputs, [msg, chatbot]).then(
        bot_stream, [chatbot, level, session_id], [chatbot]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
