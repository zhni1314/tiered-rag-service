"""Parent chunks kept outside Chroma, keyed by parent id."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path


class ParentStore(Mapping[str, str]):
    """Persistent id -> parent-text map.

    Chroma stores children only; the text the LLM reads lives here so it is
    written once per parent instead of once per child.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._texts: dict[str, str] = {}
        self._lock = threading.Lock()

    def __getitem__(self, key: str) -> str:
        return self._texts[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._texts)

    def __len__(self) -> int:
        return len(self._texts)

    def load(self) -> int:
        """Load from disk. A corrupt file is fatal: starting empty would make
        every answer fall back to tiny child chunks without anyone noticing."""
        if not self._path.exists():
            return 0
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"parent store {self._path} is unreadable: {exc}") from exc
        if not isinstance(data, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in data.items()
        ):
            raise RuntimeError(f"parent store {self._path} has an unexpected shape")
        with self._lock:
            self._texts = data
        return len(data)

    def update(self, source: str, parents: Mapping[str, str]) -> None:
        """Replace one file's parents and persist atomically."""
        prefix = f"{source}::"
        with self._lock:
            merged = {
                pid: text for pid, text in self._texts.items() if not pid.startswith(prefix)
            }
            merged.update(parents)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            try:
                tmp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, self._path)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            self._texts = merged
