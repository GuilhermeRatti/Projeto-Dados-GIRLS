"""Orquestrador do pipeline de análise de gênero.

Fluxo:
    dados brutos → BM25 + embeddings densos → RRF fusion
    → scores_genero.csv → (threshold) → projetos_genero.csv + revisao.csv
    → distribuicao_scores.png + analise_genero.png
"""

import warnings

import numpy as np
import pandas as pd

from .bm25 import compute_bm25_scores
from .classificacao import is_stem
from .config import (
    BATCH_SIZE,
    CACHE_EMB,
    CACHE_IDS,
    DATA_FILE,
    DENSE_QUERIES,
    EMBED_MODEL,
    FIG_ANALISE,
    FIG_SCORES,
    K_RRF,
    KEYWORDS_BM25,
    OUTPUT_FILE,
    REVIEW_FILE,
    SCORES_FILE,
    THRESHOLD_RRF,
)
from .embeddings import compute_dense_scores, load_or_compute_embeddings
from .fusao import compute_rrf_total
from .texto import norm, project_text
from .visualizacao import plot_analise_genero, plot_distribuicao_scores

warnings.filterwarnings("ignore")


def run(threshold: float = THRESHOLD_RRF) -> None:
    """Executa o pipeline completo. Ajuste `threshold` em config.py."""

    # ── 1. Carregar e preparar corpus ─────────────────────────────────────────
    print(f"Lendo dados de '{DATA_FILE}'...")
    df = pd.read_csv(DATA_FILE)
    print(f"Projetos carregados: {len(df)}")

    texts_raw  = df.apply(project_text, axis=1).tolist()
    texts_norm = [norm(t) for t in texts_raw]
    corpus_tok = [t.split() for t in texts_norm]

    # ── 2. BM25 ──────────────────────────────────────────────────────────────
    bm25_scores = compute_bm25_scores(corpus_tok, KEYWORDS_BM25)

    # ── 3. Embeddings densos ──────────────────────────────────────────────────
    doc_embs = load_or_compute_embeddings(
        texts_raw, df["id_projeto"], EMBED_MODEL, CACHE_EMB, CACHE_IDS, BATCH_SIZE,
    )
    dense_scores = compute_dense_scores(doc_embs, DENSE_QUERIES, EMBED_MODEL, BATCH_SIZE)

    # ── 4. RRF fusion ─────────────────────────────────────────────────────────
    _, _, rrf_total = compute_rrf_total(bm25_scores, dense_scores, K_RRF)

    df_scores = df[["id_projeto", "titulo", "situacao",
                    "unidade_lista", "sigla_unidade_lista"]].copy()
    df_scores["score_bm25"]  = bm25_scores
    df_scores["score_dense"] = dense_scores
    df_scores["score_rrf"]   = rrf_total
    df_scores.sort_values("score_rrf", ascending=False, inplace=True)

    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(SCORES_FILE, index=False, encoding="utf-8-sig")
    print(f"Scores salvos em '{SCORES_FILE}'")
    print(df_scores.head(10)[["id_projeto", "titulo", "score_bm25",
                               "score_dense", "score_rrf"]].to_string())

    # ── 5. Distribuição de scores ─────────────────────────────────────────────
    plot_distribuicao_scores(df_scores, threshold, FIG_SCORES)
    print("\n→ Ajuste THRESHOLD_RRF em src/analise/config.py e reexecute se necessário.")

    if threshold is None:
        print("⚠  THRESHOLD_RRF não definido — encerrando após o plot de distribuição.")
        return

    # ── 6. Preview do corte ───────────────────────────────────────────────────
    sel = df_scores[df_scores["score_rrf"] >= threshold]
    print(f"\nProjetos acima do threshold ({threshold}): {len(sel)} de {len(df_scores)}")
    print("\nTop 15 (mais relevantes):")
    print(sel.head(15)[["id_projeto", "titulo", "score_rrf"]].to_string())
    print("\nBorda do corte (±15% do threshold):")
    border = df_scores[
        df_scores["score_rrf"].between(threshold * 0.85, threshold * 1.15)
    ].sort_values("score_rrf")
    print(border[["id_projeto", "titulo", "score_rrf"]].to_string())

    # ── 7. Filtrar e classificar projetos ─────────────────────────────────────
    ids_sel = set(df_scores.loc[df_scores["score_rrf"] >= threshold, "id_projeto"])
    df_gen  = df[df["id_projeto"].isin(ids_sel)].copy()

    score_map = df_scores.set_index("id_projeto")
    df_gen["score_rrf"]   = df_gen["id_projeto"].map(score_map["score_rrf"])
    df_gen["score_dense"] = df_gen["id_projeto"].map(score_map["score_dense"])
    df_gen["score_bm25"]  = df_gen["id_projeto"].map(score_map["score_bm25"])
    df_gen["is_stem"]     = df_gen.apply(is_stem, axis=1)
    df_gen.sort_values("score_rrf", ascending=False, inplace=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_gen.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nProjetos de gênero salvos: '{OUTPUT_FILE}' ({len(df_gen)} projetos)")

    review_cols = [
        "score_rrf", "score_bm25", "score_dense",
        "id_projeto", "titulo", "situacao", "is_stem",
        "unidade_lista",
        "palavra_chave_1", "palavra_chave_2", "palavra_chave_3", "palavra_chave_4",
        "resumo",
    ]
    df_gen[review_cols].to_csv(REVIEW_FILE, index=False, encoding="utf-8-sig")
    print(f"CSV de revisão salvo: '{REVIEW_FILE}'")

    # ── 8. Visualizações ──────────────────────────────────────────────────────
    plot_analise_genero(df_gen, threshold, FIG_ANALISE)

    # ── 9. Sumário ────────────────────────────────────────────────────────────
    _print_summary(df_gen)


def _print_summary(df_gen: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"Total de projetos com temática de gênero/mulheres: {len(df_gen)}")
    print(f"  STEM (estreito): {df_gen['is_stem'].sum()} "
          f"({df_gen['is_stem'].mean() * 100:.1f}%)")
    print(f"  Não-STEM:        {(~df_gen['is_stem']).sum()}")
    print("\nSituação:")
    for sit, cnt in df_gen["situacao"].value_counts().items():
        print(f"  {sit:<40} {cnt:>4}  ({cnt / len(df_gen) * 100:.1f}%)")
    ativos = df_gen[df_gen["situacao"].isin([
        "Ativo", "Inadimplente",
        "Aguardando envio de relatório",
        "Relatório aguardando aprovação",
    ])]
    print(f"\nAtivos + em andamento: {len(ativos)} ({len(ativos) / len(df_gen) * 100:.1f}%)")
    print("\nTop 5 unidades:")
    for unit, cnt in df_gen["unidade_lista"].value_counts().head(5).items():
        print(f"  {unit:<50} {cnt}")
    print("=" * 60)
