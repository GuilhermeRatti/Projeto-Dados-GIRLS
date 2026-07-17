"""Reciprocal Rank Fusion (RRF) para combinar rankings BM25 e denso."""

import numpy as np


def rrf(scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Converte um vetor de scores em RRF scores (maior = mais relevante).

    rank 1 é o melhor; k=60 é o valor canônico da literatura.
    """
    ranks = scores.argsort()[::-1].argsort() + 1
    return 1.0 / (k + ranks)


def compute_rrf_total(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    k: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Funde BM25 e Dense via RRF. Retorna (rrf_bm25, rrf_dense, rrf_total)."""
    rrf_bm25  = rrf(bm25_scores, k)
    rrf_dense = rrf(dense_scores, k)
    rrf_total = rrf_bm25 + rrf_dense
    return rrf_bm25, rrf_dense, rrf_total
