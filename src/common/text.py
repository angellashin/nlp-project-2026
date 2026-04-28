from __future__ import annotations

import html
import re
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    value = html.unescape(str(text))
    return WHITESPACE_RE.sub(" ", value).strip()


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."

