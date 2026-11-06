"""
Thin Ollama wrapper for Llama 3.1.

Requires a running Ollama server with the model pulled:
    ollama pull llama3.1:8b
"""

import json
import re
from typing import Optional

import ollama

LLAMA_MODEL = "llama3.1:8b"

_client: Optional[ollama.Client] = None


def get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client()
    return _client


def chat(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    json_mode: bool = False,
    model: str = LLAMA_MODEL,
    num_ctx: Optional[int] = None,
) -> str:
    """Single-turn chat completion. Returns the response text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = get_client().chat(
        model=model,
        messages=messages,
        format="json" if json_mode else None,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
            **({"num_ctx": num_ctx} if num_ctx else {}),
        },
    )
    return resp["message"]["content"].strip()


def chat_json(prompt: str, **kwargs) -> dict:
    """Chat in JSON mode and parse the result; returns {} if unparseable."""
    raw = chat(prompt, json_mode=True, **kwargs)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}
