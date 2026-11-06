"""
Review summarization with Llama 3.1 (Ollama).
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List

from utils.llm import chat

_SYSTEM = (
    "You summarize customer product reviews. Be faithful to the reviews, "
    "do not invent details, and write in plain language. Reply with the summary "
    "text only: no introduction such as 'Here is a summary', no headings, no bullet points."
)

_PROMPTS = {
    "pros": (
        "Below are positive reviews of one product. Summarize what customers "
        "like in 2-3 sentences, focusing on recurring points.\n\nReviews:\n{text}"
    ),
    "cons": (
        "Below are negative reviews of one product. Summarize what customers "
        "complain about in 2-3 sentences, focusing on recurring problems.\n\nReviews:\n{text}"
    ),
    "general": (
        "Summarize the following product reviews in 2-3 sentences.\n\nReviews:\n{text}"
    ),
}


_PREAMBLE = re.compile(r"^\s*(here(?:'s| is| are)\b[^\n:]*:)\s*", re.IGNORECASE)


def _strip_preamble(text: str) -> str:
    """Drop a leading 'Here is a summary of ...:' line if the model adds one anyway."""
    return _PREAMBLE.sub("", text).strip()


def summarize(text: str, kind: str = "general", max_tokens: int = 160) -> str:
    """Summarize a block of review text. kind: 'pros' | 'cons' | 'general'."""
    return _strip_preamble(chat(
        _PROMPTS[kind].format(text=text),
        system=_SYSTEM,
        temperature=0.2,
        max_tokens=max_tokens,
    ))


def summarize_batch(
    texts: List[str], kind: str = "general", max_workers: int = 4
) -> List[str]:
    """Summarize multiple texts concurrently (match OLLAMA_NUM_PARALLEL)."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda t: summarize(t, kind), texts))
