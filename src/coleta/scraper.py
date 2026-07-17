"""
Scraper para os projetos de extensão da UFES.
Fonte: https://projetos.ufes.br/#/consulta-projetos
API: https://pib-api.prod.uks.ufes.br/

Coleta todos os projetos (ativos e inativos) e enriquece com dados das abas
Informações e Extensão de cada projeto.
"""

import asyncio
import logging
import time
from pathlib import Path

import aiohttp
import pandas as pd

API_BASE    = "https://pib-api.prod.uks.ufes.br"
CONCURRENCY = 15  # requisições simultâneas

ROOT        = Path(__file__).resolve().parents[2]
DIR_BRUTOS  = ROOT / "data" / "brutos"
OUTPUT_CSV  = DIR_BRUTOS / "projetos_extensao.csv"
OUTPUT_JSON = DIR_BRUTOS / "projetos_extensao.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: dict = None,
    retries: int = 3,
):
    for attempt in range(retries):
        try:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status == 200:
                    return await r.json(content_type=None)
                if r.status == 404:
                    return None
                log.warning("HTTP %s para %s", r.status, url)
                return None
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                log.warning("Erro ao buscar %s: %s", url, e)
                return None


async def fetch_project_detail(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    project_id: int,
):
    async with sem:
        detail, extensao = await asyncio.gather(
            fetch_json(session, f"{API_BASE}/projetos/{project_id}"),
            fetch_json(session, f"{API_BASE}/projetos/{project_id}/extensao"),
        )
        return project_id, detail, extensao


def extract_fields(base: dict, detail: dict | None, extensao: dict | None) -> dict:
    row = {
        "id_projeto":              base.get("idProjeto"),
        "titulo":                  base.get("titulo"),
        "classificacao_primaria":  base.get("classificacaoPrimaria"),
        "classificacao_secundaria": base.get("classificacaoSecundaria"),
        "data_inicio":             base.get("dataInicio"),
        "data_conclusao":          base.get("dataConclusao"),
        "situacao":                base.get("situacao"),
        "coordenador":             base.get("nomeCoordenador"),
        "unidade_lista":           base.get("nomeUnidade"),
        "sigla_unidade_lista":     base.get("siglaUnidade"),
        # aba Informações
        "resumo":         None,
        "palavra_chave_1": None,
        "palavra_chave_2": None,
        "palavra_chave_3": None,
        "palavra_chave_4": None,
        # aba Extensão
        "apresentacao":          None,
        "unidade_extensao":      None,
        "sigla_unidade_extensao": None,
    }

    if detail:
        row["resumo"]         = detail.get("resumo")
        row["palavra_chave_1"] = detail.get("palavraChave01")
        row["palavra_chave_2"] = detail.get("palavraChave02")
        row["palavra_chave_3"] = detail.get("palavraChave03")
        row["palavra_chave_4"] = detail.get("palavraChave04")

    if extensao:
        row["apresentacao"] = extensao.get("apresentacao")
        unidade = extensao.get("unidade")
        if isinstance(unidade, dict):
            row["unidade_extensao"]      = unidade.get("nome")
            row["sigla_unidade_extensao"] = unidade.get("siglaUnidade")

    return row


async def main():
    start = time.time()
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    headers   = {"Accept": "application/json"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        # 1. Lista completa de projetos de extensão (todas as situações)
        log.info("Buscando lista de projetos...")
        projects = await fetch_json(
            session,
            f"{API_BASE}/projetos",
            params={
                "numero": "", "titulo": "", "membroEquipe": "",
                "funcaoParticipante": "", "situacao": "", "palavraChave": "",
                "tipoProjeto": 39725,  # Extensão
                "subtipoProjeto": "", "principalAreaTematicaExtensao": "",
                "areaTematicaExtensaoAfim": "", "linhaExtensao": "",
                "grandeAreaConhecimento": "", "periodoInicio": "",
                "periodoTermino": "", "observacoes": "", "tipoPublico": "",
                "hierarquica": "true",
            },
        )

        if not projects:
            log.error("Nenhum projeto retornado.")
            return

        log.info("Total de projetos encontrados: %d", len(projects))

        # 2. Detalhes em paralelo (aba Informações + aba Extensão)
        sem   = asyncio.Semaphore(CONCURRENCY)
        tasks = [fetch_project_detail(session, sem, p["idProjeto"]) for p in projects]

        results_map = {}
        done = 0
        for coro in asyncio.as_completed(tasks):
            pid, detail, extensao = await coro
            results_map[pid] = (detail, extensao)
            done += 1
            if done % 200 == 0:
                log.info("Progresso: %d/%d", done, len(tasks))

        log.info("Detalhes coletados. Montando dataset...")

    # 3. Monta e salva o dataset
    rows = [
        extract_fields(base, *results_map.get(base["idProjeto"], (None, None)))
        for base in projects
    ]
    df = pd.DataFrame(rows)

    DIR_BRUTOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)

    elapsed = time.time() - start
    log.info("Concluído em %.1fs. %d projetos salvos em '%s' e '%s'.",
             elapsed, len(df), OUTPUT_CSV, OUTPUT_JSON)

    print("\n--- Resumo ---")
    print(f"Total de projetos: {len(df)}")
    print("Por situação:")
    print(df["situacao"].value_counts().to_string())
    print(f"\nColunas: {list(df.columns)}")
    print(df.head(3).to_string())


async def retry_missing():
    """Re-busca projetos cujo resumo ficou nulo por falha de conexão."""
    df = pd.read_csv(OUTPUT_CSV)
    missing_ids = df.loc[df["resumo"].isna(), "id_projeto"].tolist()
    log.info("Re-buscando %d projetos com dados ausentes...", len(missing_ids))

    sem       = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    headers   = {"Accept": "application/json"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [fetch_project_detail(session, sem, pid) for pid in missing_ids]
        done  = 0
        for coro in asyncio.as_completed(tasks):
            pid, detail, extensao = await coro
            idx = df.index[df["id_projeto"] == pid]
            if detail:
                df.loc[idx, "resumo"]         = detail.get("resumo")
                df.loc[idx, "palavra_chave_1"] = detail.get("palavraChave01")
                df.loc[idx, "palavra_chave_2"] = detail.get("palavraChave02")
                df.loc[idx, "palavra_chave_3"] = detail.get("palavraChave03")
                df.loc[idx, "palavra_chave_4"] = detail.get("palavraChave04")
            if extensao:
                df.loc[idx, "apresentacao"] = extensao.get("apresentacao")
                unidade = extensao.get("unidade")
                if isinstance(unidade, dict):
                    df.loc[idx, "unidade_extensao"]      = unidade.get("nome")
                    df.loc[idx, "sigla_unidade_extensao"] = unidade.get("siglaUnidade")
            done += 1
            if done % 100 == 0:
                log.info("Progresso: %d/%d", done, len(missing_ids))

    still_missing = df["resumo"].isna().sum()
    log.info("Concluído. Ainda sem resumo: %d (genuinamente vazios na base)", still_missing)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)
    log.info("Arquivos atualizados: %s, %s", OUTPUT_CSV, OUTPUT_JSON)
