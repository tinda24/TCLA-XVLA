from functools import lru_cache
from typing import Sequence

import numpy as np
import torch


@lru_cache(maxsize=4)
def _load_sentence_encoder(name_or_path: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for instruction-conditioned action autoencoder training. "
            "Install it in the current environment or set sentence_encoder_name_or_path to a local "
            "SentenceTransformer-compatible model after installation."
        ) from exc
    return SentenceTransformer(name_or_path, device=device)


def encode_texts_np(
    texts: Sequence[str],
    name_or_path: str,
    dim: int,
    device: str = "cpu",
) -> np.ndarray:
    model = _load_sentence_encoder(str(name_or_path), str(device))
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[None]
    if embeddings.shape[-1] != int(dim):
        raise ValueError(
            f"Sentence embedding dim mismatch: expected {int(dim)}, got {embeddings.shape[-1]} "
            f"from {name_or_path}."
        )
    return embeddings


def encode_texts_tensor(
    texts: Sequence[str],
    name_or_path: str,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    encoder_device: str = "cpu",
) -> torch.Tensor:
    embeddings = encode_texts_np(
        texts, name_or_path=name_or_path, dim=dim, device=encoder_device
    )
    return torch.as_tensor(embeddings, device=device, dtype=dtype)
