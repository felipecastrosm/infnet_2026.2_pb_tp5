"""Preparação dos dados para uso com o LLM local (corpus, fichas, ranking e recuperação de contexto).

Transforma o que já foi coletado em um corpus de documentos curtos e autocontidos,
prontos para:
- perguntas e respostas com recuperação de contexto (RAG): o LLM responde com base
  nos trechos mais relevantes, citando a fonte;
- resumo e classificação de sentimento das notícias (um documento = uma matéria);
- geração de relatórios: as "fichas de local" descrevem em texto os números do clima.

Tipos de documento:
- `noticia`: matérias da ABSOLAR e do Canal Solar, divididas em trechos de até
  ~220 palavras com sobreposição, para caber na janela de contexto;
- `ficha_local`: um resumo em texto, gerado a partir dos números da NASA POWER e do
  CRESESB, de cada capital (irradiação, melhor/pior mês, geração estimada).

Também traz a recuperação de contexto usada nas perguntas e respostas com o LLM local
(src/llm/tarefas.py): uma busca por palavras-chave com pesos TF-IDF (`buscar`),
reforçada por regras de metadados (`buscar_contexto`): a ficha da cidade citada na
pergunta entra sempre, e perguntas comparativas ("qual capital tem mais sol?") recebem
o ranking das capitais. O modelo de prompt fica em `montar_prompt`.

Uso (na raiz do projeto):
    python -m src.data_processing.llm_prep      # gera data/processed/corpus_llm.jsonl
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data_processing import climate, text

SAIDA = Path(__file__).resolve().parents[2] / "data" / "processed" / "corpus_llm.jsonl"
MAX_PALAVRAS = 220
SOBREPOSICAO = 40

PROMPT_QA = """Você é um assistente do Solar Viável, que ajuda pessoas leigas a avaliar
energia solar fotovoltaica no Brasil. Responda em português, de forma simples, usando
SOMENTE as informações do contexto abaixo. Cite as fontes entre colchetes, por exemplo
[1]. Se o contexto não tiver a resposta, diga que não sabe.

Contexto:
{contexto}

Pergunta: {pergunta}
Resposta:"""


def dividir_em_trechos(texto: str, max_palavras: int = MAX_PALAVRAS, sobreposicao: int = SOBREPOSICAO) -> list[str]:
    """Divide um texto em janelas de palavras com sobreposição (para não cortar ideias)."""
    palavras = texto.split()
    if len(palavras) <= max_palavras:
        return [" ".join(palavras)] if palavras else []
    passo = max_palavras - sobreposicao
    return [" ".join(palavras[i : i + max_palavras]) for i in range(0, len(palavras) - sobreposicao, passo)]


def _id(*partes) -> str:
    return hashlib.md5("|".join(map(str, partes)).encode()).hexdigest()[:12]


def documentos_noticias(noticias: pd.DataFrame) -> list[dict]:
    docs = []
    for row in noticias.itertuples():
        corpo = row.texto or row.resumo
        data = row.data.date().isoformat() if pd.notna(row.data) else ""
        for i, trecho in enumerate(dividir_em_trechos(corpo)):
            docs.append(
                {
                    "id": _id(row.url, i),
                    "tipo": "noticia",
                    "fonte": row.fonte,
                    "titulo": row.titulo,
                    "url": row.url,
                    "data": data,
                    "cidade": "",
                    # o título vai junto em cada trecho para o trecho fazer sentido sozinho
                    "texto": f"{row.titulo}. {trecho}",
                }
            )
    return docs


def _num(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def fichas_locais(clima: pd.DataFrame, sundata: pd.DataFrame, potencia_kwp: float = 5.0, desempenho: float = 0.80) -> list[dict]:
    """Uma ficha em linguagem natural por capital, combinando NASA POWER e CRESESB."""
    docs = []
    for cidade, df in clima.groupby("cidade"):
        mensal = df.set_index("date")["ALLSKY_SFC_SW_DWN"].resample("MS").mean()
        frases = [
            f"Potencial solar da capital {cidade}. Entre {df['date'].min():%d/%m/%Y} e {df['date'].max():%d/%m/%Y}, a irradiação solar "
            f"média medida por satélite (NASA POWER) foi de {_num(df['ALLSKY_SFC_SW_DWN'].mean())} kWh/m² por dia "
            "(o mesmo que horas de sol pleno por dia), "
            f"com temperatura média de {_num(df['T2M'].mean(), 1)} °C e cobertura de nuvens média de "
            f"{_num(df['CLOUD_AMT'].mean(), 0)}%.",
            f"O mês mais ensolarado do período foi {mensal.idxmax():%m/%Y} ({_num(mensal.max())} kWh/m²/dia) e o "
            f"menos ensolarado foi {mensal.idxmin():%m/%Y} ({_num(mensal.min())} kWh/m²/dia).",
        ]
        ref = sundata[(sundata["cidade"] == cidade) & (sundata["plano"] == "Maior média anual")]
        if not ref.empty:
            r = ref.iloc[0]
            meses = pd.Series({m: r[m] for m in climate.MESES})
            ger = climate.geracao_mensal(meses.tolist(), potencia_kwp, desempenho, tarifa=0)
            frases.append(
                f"Na base de longo prazo do CRESESB (estação {r['estacao']}, a {_num(r['distancia_km'], 1)} km), "
                f"com os painéis inclinados a {int(r['inclinacao_graus'])}° voltados para o norte, a irradiação média "
                f"anual é de {_num(r['media_anual'])} kWh/m²/dia, variando de {_num(meses.min())} ({meses.idxmin()}) "
                f"a {_num(meses.max())} ({meses.idxmax()})."
            )
            frases.append(
                f"Um sistema de {_num(potencia_kwp, 1)} kWp com taxa de desempenho de {_num(desempenho)} geraria cerca de "
                f"{_num(ger['geracao_kwh'].sum() / 12, 0)} kWh por mês em média ({_num(ger['geracao_kwh'].sum(), 0)} kWh por ano)."
            )
        docs.append(
            {
                "id": _id("ficha", cidade),
                "tipo": "ficha_local",
                "fonte": "NASA POWER + CRESESB",
                "titulo": f"Ficha de potencial solar: {cidade}",
                "url": "",
                "data": f"{df['date'].max():%Y-%m-%d}",
                "cidade": cidade,
                "texto": " ".join(frases),
            }
        )
    return docs


def ranking_capitais(clima: pd.DataFrame, sundata: pd.DataFrame, potencia_kwp: float = 5.0, desempenho: float = 0.80) -> list[dict]:
    """Um documento com as capitais já ordenadas (para perguntas comparativas).

    Modelos pequenos erram ao comparar dez fichas soltas; com a ordem calculada
    aqui (a partir dos dados), o modelo só precisa ler o primeiro e o último colocado.
    """
    media = clima.groupby("cidade")["ALLSKY_SFC_SW_DWN"].mean().sort_values(ascending=False)
    frases = [
        f"Ranking das {len(media)} capitais pela irradiação solar média medida por satélite (NASA POWER) entre "
        f"{clima['date'].min():%d/%m/%Y} e {clima['date'].max():%d/%m/%Y}, da maior para a menor: "
        + "; ".join(f"{i}º {c}: {_num(v)} kWh/m²/dia" for i, (c, v) in enumerate(media.items(), start=1))
        + f". A capital com maior irradiação é {media.index[0]} e a com menor irradiação é {media.index[-1]}."
    ]
    ref = sundata[sundata["plano"] == "Maior média anual"].copy()
    if not ref.empty:
        ref["geracao_mes"] = [
            climate.geracao_mensal([r[m] for m in climate.MESES], potencia_kwp, desempenho, tarifa=0)["geracao_kwh"].sum() / 12
            for _, r in ref.iterrows()
        ]
        ref = ref.sort_values("media_anual", ascending=False)
        frases.append(
            f"Ranking pela irradiação de longo prazo do CRESESB com os painéis inclinados, e geração média mensal de um "
            f"sistema de {_num(potencia_kwp, 1)} kWp: "
            + "; ".join(
                f"{i}º {r.cidade}: {_num(r.media_anual)} kWh/m²/dia, {_num(r.geracao_mes, 0)} kWh por mês"
                for i, r in enumerate(ref.itertuples(), start=1)
            )
            + "."
        )
    return [
        {
            "id": _id("ranking", "capitais"),
            "tipo": "ranking",
            "fonte": "NASA POWER + CRESESB",
            "titulo": "Ranking das capitais por potencial solar",
            "url": "",
            "data": f"{clima['date'].max():%Y-%m-%d}",
            "cidade": "",
            "texto": " ".join(frases),
        }
    ]


def construir_corpus(noticias: pd.DataFrame, clima: pd.DataFrame, sundata: pd.DataFrame) -> pd.DataFrame:
    corpus = pd.DataFrame(documentos_noticias(noticias) + fichas_locais(clima, sundata) + ranking_capitais(clima, sundata))
    corpus["n_palavras"] = corpus["texto"].str.split().str.len()
    return corpus


def buscar(corpus: pd.DataFrame, pergunta: str, k: int = 4) -> pd.DataFrame:
    """Recuperador de linha de base: pontua cada trecho pelas palavras da pergunta (TF-IDF simples)."""
    termos = set(text.tokenizar(pergunta))
    if corpus.empty or not termos:
        return corpus.head(0).assign(pontuacao=[])
    tokens = corpus["texto"].map(lambda t: Counter(text.tokenizar(t)))
    n_docs = len(corpus)
    idf = {t: math.log(1 + n_docs / (1 + sum(t in c for c in tokens))) for t in termos}
    pontuacao = tokens.map(lambda c: sum(math.log1p(c[t]) * idf[t] for t in termos if t in c))
    resultado = corpus.assign(pontuacao=pontuacao.round(3))
    return resultado[resultado["pontuacao"] > 0].nlargest(k, "pontuacao")


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento.lower())


_COMPARATIVA = re.compile(
    r"\b(qual|quais|onde|ranking|compar\w*|maior|menor|melhor|pior|mais|menos)\b.*"
    r"\b(capita\w*|cidade\w*|loca\w*|regia\w*|estado\w*|sol|ensolarad\w*|irradia\w*|gera\w*)\b"
)


def cidades_citadas(corpus: pd.DataFrame, pergunta: str) -> list[str]:
    """Cidades das fichas cujo nome aparece na pergunta (sem acento e sem a UF)."""
    p = _normalizar(pergunta)
    cidades = [c for c in corpus["cidade"].unique() if c]
    return [c for c in cidades if re.search(rf"\b{re.escape(_normalizar(c.split(' (')[0]))}\b", p)]


def eh_comparativa(pergunta: str) -> bool:
    return bool(_COMPARATIVA.search(_normalizar(pergunta)))


def buscar_contexto(corpus: pd.DataFrame, pergunta: str, k: int = 4) -> pd.DataFrame:
    """Recuperação usada pelo LLM: TF-IDF + regras de metadados (cidade e comparação).

    - a ficha de cada cidade citada na pergunta entra primeiro (a busca por palavras
      sozinha às vezes prefere uma notícia que só menciona a cidade);
    - se a pergunta compara locais e não cita cidade, entra o ranking das capitais
      (ordem já calculada a partir dos dados), para o modelo não precisar comparar;
    - o restante das `k` vagas vem da busca por palavras-chave.
    """
    if corpus.empty:
        return buscar(corpus, pergunta, k)
    fichas = corpus[corpus["tipo"] == "ficha_local"]
    citadas = cidades_citadas(corpus, pergunta)
    if citadas:
        prioritarios = fichas[fichas["cidade"].isin(citadas)]
    elif eh_comparativa(pergunta):
        prioritarios = corpus[corpus["tipo"] == "ranking"]
    else:
        return buscar(corpus, pergunta, k)
    vagas = max(k - len(prioritarios), 1)
    busca = buscar(corpus[~corpus["id"].isin(prioritarios["id"])], pergunta, vagas)
    pontuacao = float(busca["pontuacao"].max()) if len(busca) else 1.0
    return pd.concat([prioritarios.assign(pontuacao=pontuacao), busca], ignore_index=True)


def montar_prompt(pergunta: str, trechos: pd.DataFrame) -> str:
    contexto = "\n\n".join(
        f"[{i}] {row.titulo} ({row.fonte}{', ' + row.data if row.data else ''})\n{row.texto}"
        for i, row in enumerate(trechos.itertuples(), start=1)
    )
    return PROMPT_QA.format(contexto=contexto or "(nenhum trecho encontrado)", pergunta=pergunta)


if __name__ == "__main__":
    from src import repositorio

    corpus = construir_corpus(
        repositorio.carregar_noticias(),
        repositorio.carregar_clima_base(),
        repositorio.carregar_sundata(),
    )
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_json(SAIDA, orient="records", lines=True, force_ascii=False)
    print(f"{len(corpus)} documentos ({corpus['tipo'].value_counts().to_dict()}) salvos em {SAIDA}")
