"""
Sentiment analysis utilities for product reviews, powered by Llama 3.1 (Ollama).
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pandas as pd

from utils.llm import LLAMA_MODEL, chat_json

LABELS = ("positive", "neutral", "negative")

_SYSTEM = (
    "You are a sentiment classifier for e-commerce product reviews. "
    "Classify the overall sentiment of the review as exactly one of: "
    "positive, neutral, negative. Also give your confidence between 0 and 1. "
    'Respond only with JSON: {"label": "...", "confidence": 0.0}'
)


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of a given text.

    Returns:
        Dictionary with keys: label (positive/negative/neutral), score (0-1)
    """
    out = chat_json(f"Review:\n{text[:1000]}", system=_SYSTEM, max_tokens=40, num_ctx=2048)
    label = str(out.get("label", "")).lower().strip()
    if label not in LABELS:
        label = "neutral"  # unparseable / off-schema answer
    try:
        score = min(1.0, max(0.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        score = 0.5
    return {"label": label, "score": score}


def batch_sentiment_analysis(
    texts: List[str],
    max_workers: int = 4,
    complaint_threshold: float = 0.85,
) -> pd.DataFrame:
    """
    Perform sentiment analysis on multiple texts (concurrent Ollama requests).

    Args:
        texts: List of texts to analyze
        max_workers: Parallel requests (match OLLAMA_NUM_PARALLEL)
        complaint_threshold: Minimum confidence of a negative label to flag as complaint

    Returns:
        DataFrame with columns: llama_label, llama_score, is_complaint
    """
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(analyze_sentiment, texts))

    labels = [r["label"] for r in results]
    scores = [r["score"] for r in results]
    complaints = [
        lbl == "negative" and sc >= complaint_threshold
        for lbl, sc in zip(labels, scores)
    ]
    return pd.DataFrame(
        {"llama_label": labels, "llama_score": scores, "is_complaint": complaints}
    )


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def score_with_cache(
    texts: List[str],
    cache_dir: str,
    max_workers: int = 4,
    chunk_size: int = 200,
    complaint_threshold: float = 0.85,
) -> pd.DataFrame:
    """
    Like batch_sentiment_analysis, but resumable and deduplicated.

    Each distinct text is scored once and saved to a parquet cache after every
    chunk, so a crash or rerun only scores texts not seen before. The cache file
    name includes a hash of the model and prompt, so changing either starts a
    fresh cache. is_complaint is recomputed from the threshold on every call.

    Returns a DataFrame aligned with `texts`: llama_label, llama_score, is_complaint.
    """
    from tqdm.auto import tqdm

    os.makedirs(cache_dir, exist_ok=True)
    version = hashlib.sha1(f"{LLAMA_MODEL}|{_SYSTEM}".encode()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"llama_sentiment_cache_{version}.parquet")

    cache = (
        pd.read_parquet(cache_path)
        if os.path.exists(cache_path)
        else pd.DataFrame(columns=["key", "text", "llama_label", "llama_score"])
    )
    done = set(cache["key"])

    keys = [_text_key(t) for t in texts]
    todo = {}
    for k, t in zip(keys, texts):
        if k not in done and k not in todo:
            todo[k] = t
    print(f"{len(texts):,} texts | {len(set(keys)):,} distinct | "
          f"{len(set(keys)) - len(todo):,} cached | {len(todo):,} to score")

    items = list(todo.items())
    for i in tqdm(range(0, len(items), chunk_size), desc="Scoring", disable=not items):
        chunk = items[i : i + chunk_size]
        res = batch_sentiment_analysis(
            [t for _, t in chunk], max_workers=max_workers,
            complaint_threshold=complaint_threshold,
        )
        new = pd.DataFrame({
            "key": [k for k, _ in chunk],
            "text": [t for _, t in chunk],
            "llama_label": res["llama_label"].values,
            "llama_score": res["llama_score"].values,
        })
        cache = pd.concat([cache, new], ignore_index=True)
        tmp_path = cache_path + ".tmp"              # write-then-rename so a crash or a
        cache.to_parquet(tmp_path, index=False)     # concurrent read never sees a half-written file
        os.replace(tmp_path, cache_path)

    lookup = cache.drop_duplicates("key").set_index("key")
    out = pd.DataFrame({
        "llama_label": lookup.loc[keys, "llama_label"].values,
        "llama_score": lookup.loc[keys, "llama_score"].astype(float).values,
    })
    out["is_complaint"] = (out["llama_label"] == "negative") & (out["llama_score"] >= complaint_threshold)
    return out


def get_sentiment_distribution(df: pd.DataFrame, sentiment_col: str) -> Dict:
    """
    Get distribution of sentiments in a dataset.

    Returns:
        Dictionary with sentiment distribution (counts and percentages)
    """
    counts = df[sentiment_col].value_counts()
    pcts = (counts / len(df) * 100).round(2)
    return {
        "counts": counts.to_dict(),
        "percentages": pcts.to_dict(),
        "total": len(df),
    }
