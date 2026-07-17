"""Funções de visualização dos resultados do pipeline de análise de gênero."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_distribuicao_scores(
    df_scores: pd.DataFrame, threshold: float, output: Path
) -> None:
    """Histogramas dos 3 scores com o threshold marcado no RRF.

    Use esse gráfico para escolher ou ajustar o THRESHOLD_RRF em config.py.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Distribuição de Scores — Projetos de Extensão UFES", fontsize=13)

    for ax, (col, label) in zip(axes, [
        ("score_bm25",  "BM25 Score"),
        ("score_dense", "Dense Cosine Similarity"),
        ("score_rrf",   "RRF Score (fusão)"),
    ]):
        ax.hist(df_scores[col], bins=80, color="#2e7d6e", edgecolor="none", alpha=0.85)
        ax.set_xlabel(label)
        ax.set_ylabel("Nº de projetos")
        ax.set_title(label)
        if threshold and col == "score_rrf":
            ax.axvline(threshold, color="crimson", lw=2, ls="--",
                       label=f"threshold={threshold}")
            ax.legend()

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.show()
    print(f"Gráfico salvo em '{output}'")


def plot_analise_genero(
    df_gen: pd.DataFrame, threshold: float, output: Path
) -> None:
    """6 subplots de análise dos projetos de gênero filtrados."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(
        f"Projetos de Extensão UFES — Mulheres e Gênero  "
        f"(n={len(df_gen)}, threshold RRF={threshold})",
        fontsize=13, y=1.01,
    )

    # 1) Situação
    ax = axes[0, 0]
    sit = df_gen["situacao"].value_counts()
    bars = ax.barh(sit.index, sit.values, color=sns.color_palette("muted", len(sit)))
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Nº de projetos")
    ax.set_title("Situação dos projetos")
    ax.invert_yaxis()

    # 2) STEM vs. não-STEM
    ax = axes[0, 1]
    stem_counts = df_gen["is_stem"].value_counts()
    ax.pie(
        [stem_counts.get(False, 0), stem_counts.get(True, 0)],
        labels=["Não-STEM", "STEM (Exatas/Eng/TI)"],
        colors=["#6baed6", "#2c7bb6"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("STEM vs. Não-STEM")

    # 3) Situação × STEM
    ax = axes[0, 2]
    cross = pd.crosstab(df_gen["situacao"], df_gen["is_stem"])
    cross.columns = ["Não-STEM", "STEM"]
    cross.plot(kind="barh", ax=ax, color=["#6baed6", "#2c7bb6"], edgecolor="none")
    ax.set_title("Situação × STEM")
    ax.set_xlabel("Nº de projetos")
    ax.legend(loc="lower right")

    # 4) Top 15 unidades
    ax = axes[1, 0]
    top_units = df_gen["unidade_lista"].value_counts().head(15)
    hbars = ax.barh(top_units.index[::-1], top_units.values[::-1], color="#2e7d6e")
    ax.bar_label(hbars, padding=3)
    ax.set_title("Top 15 unidades")
    ax.set_xlabel("Nº de projetos")
    ax.tick_params(axis="y", labelsize=7)

    # 5) Evolução temporal (ano de início)
    ax = axes[1, 1]
    df_gen = df_gen.copy()
    df_gen["ano_inicio"] = pd.to_datetime(df_gen["data_inicio"], errors="coerce").dt.year
    timeline = df_gen["ano_inicio"].value_counts().sort_index()
    ax.bar(timeline.index, timeline.values, color="#2e7d6e")
    ax.set_title("Projetos por ano de início")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Nº de projetos")

    # 6) Top 20 palavras-chave
    ax = axes[1, 2]
    kws = pd.concat([
        df_gen["palavra_chave_1"], df_gen["palavra_chave_2"],
        df_gen["palavra_chave_3"], df_gen["palavra_chave_4"],
    ]).dropna().str.lower().str.strip()
    top_kw = kws.value_counts().head(20)
    ax.barh(top_kw.index[::-1], top_kw.values[::-1], color="#2e7d6e")
    ax.set_title("Top 20 palavras-chave")
    ax.set_xlabel("Frequência")
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Gráfico salvo em '{output}'")
