"""Classificação de projetos como STEM (Exatas, Engenharia, TI)."""

from .config import STEM_CENTROS, STEM_KW_EXCL, STEM_KW_UNIT
from .texto import norm


def is_stem(row) -> bool:
    """Retorna True se o projeto pertence a uma unidade de STEM estreito.

    Critério: centro (CT ou CCE) ou keywords STEM no nome da unidade,
    excluindo explicitamente Educação Física e Desportos.
    """
    unit   = norm(str(row.get("unidade_lista", "")))
    sigla  = str(row.get("sigla_unidade_lista", ""))
    centro = sigla.split("/")[-1].strip().upper() if "/" in sigla else sigla.strip().upper()

    if centro in STEM_CENTROS:
        return True
    if any(kw in unit for kw in STEM_KW_EXCL):
        return False
    return any(kw in unit for kw in STEM_KW_UNIT)
