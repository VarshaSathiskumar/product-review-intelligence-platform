"""
Semantic embeddings with BGE-M3 (BAAI/bge-m3, 1024-dim dense vectors).
"""

from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BGE_M3_MODEL = "BAAI/bge-m3"
_model = None


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(BGE_M3_MODEL, device=_device())
        _model.max_seq_length = 512
    return _model


def embed_texts(
    texts: List[str], batch_size: int = 32, show_progress_bar: bool = False
) -> np.ndarray:
    """Return L2-normalised BGE-M3 embeddings, shape (len(texts), 1024)."""
    return get_embedder().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )


def cosine_similarity(a: str, b: str) -> float:
    """Cosine similarity between two texts (embeddings are normalised)."""
    ea, eb = embed_texts([a, b])
    return float(np.dot(ea, eb))
