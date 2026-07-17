"""Constantes de configuração e caminhos do pipeline de análise de gênero."""

from pathlib import Path

# ── Diretórios ────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[2]
DIR_BRUTOS      = ROOT / "data" / "brutos"
DIR_PROCESSADOS = ROOT / "data" / "processados"
DIR_CACHE       = ROOT / "data" / "cache"
DIR_FIGURAS     = ROOT / "outputs" / "figuras"

# ── Parâmetros do pipeline ────────────────────────────────────────────────────
THRESHOLD_RRF = 0.009
EMBED_MODEL   = "intfloat/multilingual-e5-base"
K_RRF         = 60   # constante padrão do RRF
BATCH_SIZE    = 64   # batch de encoding (reduzir se faltar VRAM)

# ── Arquivos de entrada ───────────────────────────────────────────────────────
DATA_FILE = DIR_BRUTOS / "projetos_extensao.csv"

# ── Cache de embeddings ───────────────────────────────────────────────────────
CACHE_EMB = DIR_CACHE / "embeddings.npy"
CACHE_IDS = DIR_CACHE / "ids.npy"

# ── Saídas ────────────────────────────────────────────────────────────────────
SCORES_FILE = DIR_PROCESSADOS / "scores_genero.csv"
OUTPUT_FILE = DIR_PROCESSADOS / "projetos_genero.csv"
REVIEW_FILE = DIR_PROCESSADOS / "projetos_genero_revisao.csv"
FIG_SCORES  = DIR_FIGURAS / "distribuicao_scores.png"
FIG_ANALISE = DIR_FIGURAS / "analise_genero.png"

# ── Keywords BM25 (português, com variações sem acento) ───────────────────────
KEYWORDS_BM25 = [
    "mulher", "mulheres", "feminino", "feminina", "femininas", "femininos",
    "genero", "gênero",
    "feminismo", "feminista", "feministas",
    # "empoderamento", "empoderar",
    # "violencia", "violência", "domestica", "doméstica",
    "machismo", "sexismo", "sexista",
    "meninas", "garota", "garotas",
    "maternidade", "gestante", "gestantes", "maternal",
    "trans", "transgenero", "transgênero",
    "direitos da mulher", "saude da mulher", "saúde da mulher",
    "diversidade de genero", "diversidade de gênero",
    "identidade de genero", "identidade de gênero",
    "inclusao de genero", "inclusão de gênero",
    "politicas de genero", "políticas de gênero",
    "girl", "women", "gender",  # alguns projetos usam inglês
]

# ── Queries para embeddings densos (e5-base usa prefixo "query:") ─────────────
DENSE_QUERIES = [
    "projetos voltados para mulheres e inclusão de gênero na universidade",
    "empoderamento feminino equidade de gênero ações afirmativas direitos",
    "combate à violência contra a mulher violência doméstica de gênero",
    "participação feminina em ciência tecnologia engenharia matemática STEM",
    "saúde da mulher direitos femininos políticas públicas de gênero",
    "meninas mulheres educação científica tecnológica empoderamento",
]

# ── Classificação STEM estreita (Exatas + Engenharia + TI) ───────────────────
STEM_CENTROS = {"CT", "CCE"}  # Centro Tecnológico, Centro de Ciências Exatas
STEM_KW_UNIT = [
    "engenharia", "computação", "computacao", "informática", "informatica",
    "matemática", "matematica", "física", "fisica", "química", "quimica",
    "estatística", "estatistica", "ciências exatas", "ciencias exatas",
    "sistemas de informação", "sistemas de informacao",
    "tecnologia da informação", "tecnologia da informacao",
]
# "Educação Física" não é STEM — excluir explicitamente
STEM_KW_EXCL = ["educação física", "educacao fisica", "desportos", "ginástica"]
