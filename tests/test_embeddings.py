"""Embeddings: shape/normalisation, cross-process determinism, similarity sanity."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from marketbrief.config import Settings
from marketbrief.embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine_matrix,
    get_embedder,
    tokenize,
)

TEXTS = [
    "Aurora Silicon raises $410 million in a Series D funding round.",
    "Meridian Cloud has agreed to acquire Vireo Analytics for $1.2 billion.",
    "",
]


def test_shape_dtype_and_normalisation() -> None:
    emb = HashingEmbedder(dim=256)
    assert isinstance(emb, Embedder)
    vecs = emb.embed(TEXTS)
    assert vecs.shape == (3, 256) and vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms[:2], 1.0, atol=1e-5)
    assert norms[2] == 0.0  # empty text -> zero vector
    assert emb.embed(["the of and"]).any() == False  # noqa: E712 - stopwords only
    assert emb.embed([]).shape == (0, 256)


def test_tokenize_drops_stopwords_and_keeps_numbers() -> None:
    assert tokenize("The GPU-hour price is $2.15, said Northwind.") == ["gpu-hour", "price", "2.15", "northwind"]


def _fingerprint_script(dim: int) -> str:
    return (
        "import hashlib, sys\n"
        "from marketbrief.embeddings import HashingEmbedder\n"
        f"v = HashingEmbedder({dim}).embed({TEXTS!r})\n"
        "sys.stdout.write(hashlib.sha256(v.tobytes()).hexdigest())\n"
    )


def test_deterministic_across_processes() -> None:
    """blake2b feature hashing gives identical vectors regardless of PYTHONHASHSEED."""
    local = hashlib.sha256(HashingEmbedder(512).embed(TEXTS).tobytes()).hexdigest()
    outputs = set()
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", _fingerprint_script(512)],
            capture_output=True, text=True, env=env, timeout=60, check=True,
        )
        outputs.add(proc.stdout.strip())
    assert outputs == {local}


def test_similarity_ranks_related_text_higher() -> None:
    emb = HashingEmbedder()
    base = "Northwind Compute cuts H-class GPU-hour pricing by 22% as new supply arrives."
    rewrite = "Northwind Compute slashes H-class GPU-hour prices 22% as fresh supply arrives."
    same_company_other_story = "Northwind Compute reports third-quarter revenue up 48% to $312 million."
    unrelated = "Bastion Labs appoints a former hyperscaler executive as its new CEO."
    sims = cosine_matrix(emb.embed([base]), emb.embed([rewrite, same_company_other_story, unrelated]))[0]
    assert sims[0] > sims[1] > sims[2]
    assert sims[0] > 0.6 and sims[2] < 0.2


def test_cosine_matrix_handles_1d_and_zero_rows() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
    sims = cosine_matrix(a, b)
    assert sims.shape == (1, 3)
    assert np.allclose(sims, [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        cosine_matrix(np.ones((1, 3)), np.ones((1, 4)))


def test_get_embedder_caches_per_backend_and_dim() -> None:
    s1 = Settings()
    s2 = Settings()
    assert get_embedder(s1) is get_embedder(s2)
    s3 = Settings.model_validate({"embeddings": {"dim": 128}})
    e3 = get_embedder(s3)
    assert e3 is not get_embedder(s1) and e3.dim == 128


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None,
    reason="sentence-transformers is installed",
)
def test_sentence_transformer_backend_explains_missing_extra() -> None:
    with pytest.raises(ImportError, match=r"pip install -e \.\[embeddings\]"):
        SentenceTransformerEmbedder("any-model")
