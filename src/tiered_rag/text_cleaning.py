"""Pure text normalization applied between PDF extraction and chunking."""

from __future__ import annotations

import re
from functools import lru_cache

# Chinese sentences rarely use spaces, so a soft line wrap inside a sentence
# becomes a literal "\n" that splits tokens and hurts BM25 scoring.
_CJK_LINE_BREAK = re.compile(r"(?<=[一-鿿])\n(?=[一-鿿])")


@lru_cache(maxsize=64)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def strip_repeating_layout(
    text: str,
    header_pattern: str = "",
    footer_pattern: str = "",
    drop_cjk_line_breaks: bool = True,
) -> str:
    if header_pattern:
        text = _compile(header_pattern).sub("", text)
    if footer_pattern:
        text = _compile(footer_pattern).sub("", text)
    if drop_cjk_line_breaks:
        text = _CJK_LINE_BREAK.sub("", text)
    return text
