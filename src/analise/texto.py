"""Funções de normalização e preparação de texto dos projetos."""

import re
import unicodedata

import pandas as pd


def norm(text: str) -> str:
    """Lowercase + remove acentos + mantém letras/números/espaço."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def project_text(row) -> str:
    """Concatena os campos textuais de um projeto em uma string única."""
    fields = [
        row.get("titulo", ""),
        row.get("resumo", ""),
        row.get("palavra_chave_1", ""),
        row.get("palavra_chave_2", ""),
        row.get("palavra_chave_3", ""),
        row.get("palavra_chave_4", ""),
        row.get("apresentacao", ""),
    ]
    return " ".join(str(f) for f in fields if pd.notna(f))
