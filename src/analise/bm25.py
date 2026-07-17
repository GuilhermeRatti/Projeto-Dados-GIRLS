"""Scoring BM25 sobre o corpus de projetos."""

import numpy as np
from rank_bm25 import BM25Okapi

from .texto import norm


def compute_bm25_scores(corpus_tok: list[list[str]], keywords: list[str]) -> np.ndarray:
    """Calcula scores BM25 para o corpus dado um conjunto de keywords.

    Todos os tokens das keywords são usados em uma única query conjunta,
    o que captura projetos que mencionam qualquer combinação delas.
    """
    bm25 = BM25Okapi(corpus_tok)
    kw_tokens = list({tok for kw in keywords for tok in norm(kw).split()})
    scores = bm25.get_scores(kw_tokens)
    print(f"BM25 — projetos com score > 0: {(scores > 0).sum()}")
    return scores
