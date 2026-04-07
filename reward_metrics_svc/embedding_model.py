from __future__ import annotations

from collections import OrderedDict
import importlib
import os
import threading
from typing import Dict, List, Optional

import numpy as np
import torch

SEMANTIC_MODEL_NAME = os.environ.get("SEMANTIC_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
SEMANTIC_BATCH_SIZE = int(os.environ.get("SEMANTIC_BATCH_SIZE", "32"))
SEMANTIC_DEVICE = os.environ.get("SEMANTIC_DEVICE", "auto").strip().lower()
SEMANTIC_EMBED_CACHE_SIZE = int(os.environ.get("SEMANTIC_EMBED_CACHE_SIZE", "8192"))

_MODEL = None
_MODEL_LOCK = threading.Lock()
_ENCODE_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
_EMBED_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_MODEL_DIM: Optional[int] = None


def _resolve_device() -> str:
    if SEMANTIC_DEVICE and SEMANTIC_DEVICE != "auto":
        return SEMANTIC_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        st_module = importlib.import_module("sentence_transformers")
        sentence_transformer_cls = getattr(st_module, "SentenceTransformer")
        _MODEL = sentence_transformer_cls(SEMANTIC_MODEL_NAME, device=_resolve_device())
    return _MODEL


def _cache_get(text: str) -> Optional[np.ndarray]:
    with _CACHE_LOCK:
        vector = _EMBED_CACHE.get(text)
        if vector is None:
            return None
        _EMBED_CACHE.move_to_end(text)
        return vector


def _cache_put(text: str, vector: np.ndarray) -> None:
    if SEMANTIC_EMBED_CACHE_SIZE <= 0:
        return

    with _CACHE_LOCK:
        _EMBED_CACHE[text] = vector
        _EMBED_CACHE.move_to_end(text)
        while len(_EMBED_CACHE) > SEMANTIC_EMBED_CACHE_SIZE:
            _EMBED_CACHE.popitem(last=False)


def embed_texts(texts: List[str]) -> np.ndarray:
    values = [str(text or "").strip() for text in list(texts or [])]
    if not values:
        return np.zeros((0, 0), dtype=np.float32)

    outputs: List[Optional[np.ndarray]] = [None] * len(values)
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    for idx, text in enumerate(values):
        if not text:
            continue
        cached = _cache_get(text)
        if cached is not None:
            outputs[idx] = cached
            continue
        missing_indices.append(idx)
        missing_texts.append(text)

    if missing_texts:
        model = _load_model()
        with _ENCODE_LOCK:
            encoded = model.encode(
                missing_texts,
                batch_size=max(1, SEMANTIC_BATCH_SIZE),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        encoded = np.asarray(encoded, dtype=np.float32)

        global _MODEL_DIM
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        if encoded.size > 0:
            _MODEL_DIM = int(encoded.shape[1])

        for idx, vec in zip(missing_indices, encoded):
            outputs[idx] = vec
            _cache_put(values[idx], vec)

    dim = _MODEL_DIM if _MODEL_DIM is not None else 0
    if dim <= 0:
        model = _load_model()
        try:
            dim = int(model.get_sentence_embedding_dimension())
            _MODEL_DIM = dim
        except Exception:
            dim = 0

    if dim <= 0:
        return np.zeros((len(values), 0), dtype=np.float32)

    final = np.zeros((len(values), dim), dtype=np.float32)
    for idx, vector in enumerate(outputs):
        if vector is None:
            continue
        final[idx] = np.asarray(vector, dtype=np.float32)
    return final


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("cosine_similarity_matrix expects 2D arrays")
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    if left.shape[1] != right.shape[1]:
        raise ValueError("Embedding dimensions must match")

    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)

    left_safe = left / np.clip(left_norm, 1e-8, None)
    right_safe = right / np.clip(right_norm, 1e-8, None)

    scores = np.matmul(left_safe, right_safe.T)
    return np.clip(scores, -1.0, 1.0)


def embedding_cache_size() -> int:
    with _CACHE_LOCK:
        return len(_EMBED_CACHE)
