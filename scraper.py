"""
Ponto de entrada do scraper de projetos de extensão UFES.

Uso:
    python scraper.py                  # coleta todos os projetos
    python scraper.py --retry-missing  # refaz projetos com resumo nulo

Os dados são salvos em data/brutos/. Ver src/coleta/scraper.py para detalhes.
"""

import asyncio
import sys

from src.coleta.scraper import main, retry_missing

if __name__ == "__main__":
    if "--retry-missing" in sys.argv:
        asyncio.run(retry_missing())
    else:
        asyncio.run(main())
