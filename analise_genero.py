"""
Ponto de entrada do pipeline de análise de gênero.

Uso:
    python analise_genero.py

Para ajustar o threshold ou os parâmetros do modelo, edite:
    src/analise/config.py
"""

from src.analise import run

if __name__ == "__main__":
    run()
