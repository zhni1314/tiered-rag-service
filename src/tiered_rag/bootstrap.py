"""Sets up Hugging Face cache/env before any library that reads them is imported."""

from __future__ import annotations

import os

from .config import Settings


def configure_huggingface(settings: Settings) -> None:
    if settings.hf_home is not None:
        settings.hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(settings.hf_home)
    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
