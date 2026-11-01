"""Text embeddings used by dedup, the prompt cache, retrieval and the agent.

The default :class:`HashingEmbedder` needs no model download and is fully
deterministic across processes and machines (features are hashed with
``blake2b``, never Python's salted ``hash()``). Install the ``embeddings``
extra to use a sentence-transformers model instead.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from marketbrief.config import Settings


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to L2-normalised float32 vectors of shape (n, dim)."""

    dim: int
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers him his how i if in into is it its itself just me more most my
    no nor not now of off on once only or other our ours out over own said same says she should so
    some such than that the their theirs them then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you your
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:[.'\-][a-z0-9]+)*")

# Relative weight of each feature family. Words and bigrams carry topic and
# phrasing; character 4-grams add robustness to inflection and typos.
_WEIGHTS = {"w": 1.0, "b": 0.8, "c": 0.35}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed."""
    return [w for w in _WORD_RE.findall((text or "").lower()) if w not in STOPWORDS]


def _features(text: str) -> Counter[str]:
    words = tokenize(text)
    feats: Counter[str] = Counter()
    feats.update(f"w:{w}" for w in words)
    feats.update(f"b:{a} {b}" for a, b in zip(words, words[1:]))
    for w in words:
        padded = f"<{w}>"
        if len(padded) <= 4:
            feats[f"c:{padded}"] += 1
        else:
            feats.update(f"c:{padded[i:i + 4]}" for i in range(len(padded) - 3))
    return feats


class HashingEmbedder:
    """Feature-hashing embedder: words + word bigrams + char 4-grams.

    Each feature is hashed with ``blake2b`` into a signed bucket, weighted by
    sublinear term frequency ``1 + log(tf)`` and its family weight; the result
    is L2-normalised. Empty or stopword-only text embeds to the zero vector.
    """

    def __init__(self, dim: int = 1024) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = int(dim)
        self.name = f"hashing-blake2b-{self.dim}"
        self._bucket_cache: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def _bucket(self, feature: str) -> tuple[int, float]:
        hit = self._bucket_cache.get(feature)
        if hit is None:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            hit = (value % self.dim, 1.0 if (value >> 63) & 1 else -1.0)
            with self._lock:
                if len(self._bucket_cache) > 500_000:
                    self._bucket_cache.clear()
                self._bucket_cache[feature] = hit
        return hit

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for feat, tf in _features(text).items():
            idx, sign = self._bucket(feat)
            vec[idx] += sign * _WEIGHTS[feat[0]] * (1.0 + math.log(tf))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self.embed_one(text)
        return out


class SentenceTransformerEmbedder:
    """Dense embeddings from a sentence-transformers model (optional extra)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The 'sentence-transformers' embedding backend needs the optional extra: "
                "pip install -e .[embeddings]  (or set embeddings.backend: hashing)"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st-{model_name}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if idx:
            vecs = self._model.encode(
                [texts[i] for i in idx], normalize_embeddings=True, convert_to_numpy=True
            )
            out[idx] = np.asarray(vecs, dtype=np.float32)
        return out


_EMBEDDERS: dict[tuple[str, str | int], Embedder] = {}
_EMBEDDERS_LOCK = threading.Lock()


def get_embedder(settings: Settings) -> Embedder:
    """Shared embedder for the configured backend (one instance per config)."""
    cfg = settings.embeddings
    key: tuple[str, str | int] = (
        (cfg.backend, cfg.model_name) if cfg.backend == "sentence-transformers" else (cfg.backend, cfg.dim)
    )
    with _EMBEDDERS_LOCK:
        emb = _EMBEDDERS.get(key)
        if emb is None:
            if cfg.backend == "sentence-transformers":
                emb = SentenceTransformerEmbedder(cfg.model_name)
            elif cfg.backend == "hashing":
                emb = HashingEmbedder(cfg.dim)
            else:  # pragma: no cover - Literal-validated in Settings
                raise ValueError(f"unknown embeddings backend {cfg.backend!r}")
            _EMBEDDERS[key] = emb
        return emb


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows stay zero."""
    arr = _as_2d(x)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 0)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row of ``a`` with every row of ``b`` -> (len(a), len(b))."""
    a2, b2 = normalize_rows(a), normalize_rows(b)
    if a2.shape[1] != b2.shape[1]:
        raise ValueError(f"dimension mismatch: {a2.shape[1]} vs {b2.shape[1]}")
    return np.clip(a2 @ b2.T, -1.0, 1.0)
