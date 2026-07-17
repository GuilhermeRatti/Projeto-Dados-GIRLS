"""Embeddings densos com cache automático em disco."""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_or_compute_embeddings(
    texts_raw: list[str],
    project_ids,
    model_name: str,
    cache_emb: Path,
    cache_ids: Path,
    batch_size: int = 64,
) -> np.ndarray:
    """Carrega embeddings do cache ou os computa do zero e persiste no cache.

    Documentos são prefixados com "passage:" conforme a spec do e5-base.
    """
    if cache_emb.exists() and cache_ids.exists():
        cached_ids = np.load(cache_ids, allow_pickle=True).tolist()
        if cached_ids == list(project_ids):
            doc_embs = np.load(cache_emb)
            print(f"Embeddings carregados do cache: {doc_embs.shape}")
            return doc_embs

    print(f"Calculando embeddings com {model_name}...")
    model = SentenceTransformer(model_name)
    passages = ["passage: " + t for t in texts_raw]
    doc_embs = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=_device(),
    )
    del model

    cache_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_emb, doc_embs)
    np.save(cache_ids, np.array(list(project_ids), dtype=object))
    print(f"Embeddings salvos: {doc_embs.shape}")
    return doc_embs


def compute_dense_scores(
    doc_embs: np.ndarray,
    queries: list[str],
    model_name: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Calcula similaridade máxima entre cada documento e as queries densas.

    Queries são prefixadas com "query:" conforme a spec do e5-base.
    """
    print("Calculando embeddings das queries...")
    model = SentenceTransformer(model_name)
    query_embs = model.encode(
        ["query: " + q for q in queries],
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=_device(),
    )
    del model

    sim_matrix = doc_embs @ query_embs.T   # (N_proj, N_queries)
    scores = sim_matrix.max(axis=1)         # melhor query por projeto
    print(f"Dense — sim média: {scores.mean():.4f}, max: {scores.max():.4f}")
    return scores
