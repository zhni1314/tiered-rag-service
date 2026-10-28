"""Dev entrypoint: ``python -m tiered_rag``."""

from __future__ import annotations

import logging

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)

if __name__ == "__main__":
    uvicorn.run("tiered_rag.service:app", host="127.0.0.1", port=8000, reload=True)
