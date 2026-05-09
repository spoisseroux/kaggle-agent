"""Local embedding service using sentence-transformers on RTX 5070.

Generates embeddings 3x faster than homelab GTX 1070.
Supports multiple models for different use cases.
"""
from __future__ import annotations

import os
import numpy as np
from typing import List, Literal
from functools import lru_cache
import logging

log = logging.getLogger(__name__)

# Model configurations
MODELS = {
    # Fast, general purpose (384 dim)
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "desc": "Fast general-purpose embeddings",
    },
    # Better quality (768 dim)
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "dim": 768,
        "desc": "High-quality general-purpose embeddings",
    },
    # Code-specific (768 dim)
    "code": {
        "name": "microsoft/codebert-base",
        "dim": 768,
        "desc": "Code-specific embeddings",
    },
}

ModelType = Literal["minilm", "mpnet", "code"]

class LocalEmbedder:
    """Generate embeddings locally on RTX 5070."""

    def __init__(
        self,
        model_type: ModelType = "mpnet",
        device: str | None = None,
        target_dim: int = 1536,  # Match homelab Qdrant
    ):
        """
        Initialize embedder.

        Args:
            model_type: Which model to use
            device: cuda, cpu, or None (auto-detect)
            target_dim: Pad/truncate to this dimension (for Qdrant compatibility)
        """
        self.model_type = model_type
        self.model_config = MODELS[model_type]
        self.target_dim = target_dim

        # Auto-detect device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        log.info(
            f"Initializing {self.model_config['name']} on {device} "
            f"(native: {self.model_config['dim']}d, target: {target_dim}d)"
        )

        self._model = None  # Lazy load

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_config["name"],
                device=self.device
            )
        return self._model

    def embed(
        self,
        texts: str | List[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts
            normalize: L2 normalize vectors (recommended for cosine similarity)
            show_progress: Show progress bar for batch encoding

        Returns:
            numpy array of shape (n_texts, target_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        # Pad or truncate to target dimension
        if embeddings.shape[1] != self.target_dim:
            embeddings = self._resize_embeddings(embeddings)

        return embeddings[0] if single else embeddings

    def _resize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Pad or truncate embeddings to target dimension."""
        native_dim = embeddings.shape[1]

        if native_dim < self.target_dim:
            # Pad with zeros
            padding = np.zeros((embeddings.shape[0], self.target_dim - native_dim))
            return np.hstack([embeddings, padding])
        elif native_dim > self.target_dim:
            # Truncate
            return embeddings[:, :self.target_dim]
        else:
            return embeddings

    def embed_code(
        self,
        code: str | List[str],
        add_context: bool = True,
    ) -> np.ndarray:
        """
        Embed code snippets with optional context.

        Args:
            code: Code snippet(s)
            add_context: Prepend "Code: " to help model understand context

        Returns:
            Embeddings
        """
        if isinstance(code, str):
            code = [code]
            single = True
        else:
            single = False

        if add_context:
            code = [f"Code: {c}" for c in code]

        result = self.embed(code, normalize=True)
        return result[0] if single else result


# Global embedder instance
_embedder: LocalEmbedder | None = None


@lru_cache(maxsize=1)
def get_embedder(model_type: ModelType = "mpnet") -> LocalEmbedder:
    """Get or create global embedder instance."""
    global _embedder
    if _embedder is None or _embedder.model_type != model_type:
        _embedder = LocalEmbedder(model_type=model_type)
    return _embedder


def embed_text(text: str | List[str], model: ModelType = "mpnet") -> np.ndarray:
    """Quick helper to embed text."""
    embedder = get_embedder(model)
    return embedder.embed(text)


def embed_code(code: str | List[str], model: ModelType = "mpnet") -> np.ndarray:
    """Quick helper to embed code."""
    embedder = get_embedder(model)
    return embedder.embed_code(code)


# Example usage
if __name__ == "__main__":
    import time

    embedder = LocalEmbedder(model_type="mpnet")

    # Test single text
    text = "TimeSeriesSplit validation for time-series data"
    t0 = time.time()
    emb = embedder.embed(text)
    elapsed = time.time() - t0

    print(f"Model: {embedder.model_config['name']}")
    print(f"Device: {embedder.device}")
    print(f"Embedding shape: {emb.shape}")
    print(f"Time: {elapsed:.3f}s")

    # Test code embedding
    code = """
def create_lag_features(df, lags=[1, 7, 14]):
    for lag in lags:
        df[f'Lag_{lag}'] = df['sales'].shift(lag)
    return df
"""
    t0 = time.time()
    code_emb = embedder.embed_code(code)
    elapsed = time.time() - t0

    print(f"\nCode embedding shape: {code_emb.shape}")
    print(f"Time: {elapsed:.3f}s")

    # Test batch
    texts = [
        "Lag features for time series",
        "Rolling window aggregations",
        "Exponential moving average",
    ] * 10  # 30 texts

    t0 = time.time()
    batch_embs = embedder.embed(texts, show_progress=True)
    elapsed = time.time() - t0

    print(f"\nBatch: {len(texts)} texts")
    print(f"Shape: {batch_embs.shape}")
    print(f"Time: {elapsed:.3f}s ({len(texts)/elapsed:.1f} texts/s)")
