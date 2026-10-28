"""Per-session chat history and the shared Redis client."""

from __future__ import annotations

import json
import time

from redis import Redis


class HistoryStore:
    def __init__(self, redis: Redis, window: int) -> None:
        self._redis = redis
        self._window = window

    @staticmethod
    def _key(session_id: str) -> str:
        return f"history:{session_id}"

    def as_prompt_text(self, session_id: str) -> str:
        raw = self._redis.lrange(self._key(session_id), 0, self._window - 1)
        lines = []
        for item in reversed(raw):
            msg = json.loads(item)
            lines.append(f"{msg['role']}: {msg['content']}\n")
        return "".join(lines)

    def _push(self, key: str, role: str, content: str, now: float) -> None:
        entry = {"role": role, "content": content, "timestamp": now}
        self._redis.lpush(key, json.dumps(entry, ensure_ascii=False))

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        key = self._key(session_id)
        now = time.time()
        self._push(key, "user", question, now)
        self._push(key, "assistant", answer, now)
        self._redis.ltrim(key, 0, self._window - 1)

    def all_items(self, session_id: str) -> list[dict]:
        raw = self._redis.lrange(self._key(session_id), 0, -1)
        return [json.loads(item) for item in reversed(raw)]
