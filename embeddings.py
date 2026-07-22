"""Local sentence-transformers embedding wrapper for AMS.

Multilingual MiniLM is a small (~420MB), fast, well-supported model with
solid Japanese coverage. First call downloads the weights; subsequent calls
hit the lru_cache and reuse the loaded model.
"""
from __future__ import annotations

import functools
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384

_load_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    with _load_lock:
        return SentenceTransformer(MODEL_NAME)


def embed(text: str) -> bytes:
    """Encode text to a normalized float32 byte string suitable for sqlite-vec."""
    model = get_model()
    vec = model.encode(
        text, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    return vec.tobytes()


def embed_for_row(key: str, value: str) -> bytes:
    return embed(f"{key}: {value}")
