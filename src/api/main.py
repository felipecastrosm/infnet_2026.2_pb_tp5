"""API do Solar Viável (FastAPI).

Expõe os dados do projeto (clima da NASA POWER, irradiação de longo prazo do CRESESB,
notícias do setor e medições enviadas por usuários) e os cálculos de geração, para que
o app Streamlit (ou qualquer outro cliente) consulte e envie dados por HTTP.

Desde o TP4, a API também executa um **LLM local** (HuggingFace Transformers): resumo e
sentimento de textos (`POST /processar_texto`), perguntas e respostas sobre o corpus
(`POST /llm/perguntar`) e relatório de viabilidade (`POST /llm/relatorio`). Toda rota tem
Response Model e todo erro sai no mesmo formato (`schemas.RespostaErro`).

Rodar (na raiz do projeto):
    uvicorn src.api.main:app --reload
Documentação interativa: http://127.0.0.1:8000/docs (Swagger) e /redoc.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import repositorio
from src.api import schemas
from src.data_acquisition import nasa_power_client
from src.data_processing import climate, llm_prep, text, viabilidade
from src.llm import agente, memoria, tarefas
from src.llm import erros as erros_llm
from src.llm.servico import servico as servico_llm

log = logging.getLogger("solar_viavel.api")

DESCRICAO = """
API do **Solar Viável**: dados e cálculos para avaliar o potencial de geração solar
fotovoltaica de residências e pequenos negócios no Brasil (ODS 7).

* **Locais e clima**: série diária da NASA POWER para 10 capitais, mais as medições
  enviadas pelos usuários; consulta ao vivo de qualquer coordenada.
* **Longo prazo**: irradiação média mensal do CRESESB/SunData (coletada com Selenium),
  no plano horizontal e inclinado.
* **Estimativa**: geração (kWh) e economia (R$) para um sistema de potência informada.
* **Notícias**: matérias do setor coletadas da ABSOLAR e do Canal Solar (Beautiful Soup).
* **Medições**: envio (POST), consulta e remoção de dados do usuário.
* **LLM local** (HuggingFace Transformers, sem nuvem): resumo e sentimento de textos
  (`POST /processar_texto`), perguntas e respostas sobre as notícias e fichas das capitais
  com citação das fontes (`POST /llm/perguntar`) e relatório de viabilidade cujos números
  são conferidos com a API (`POST /llm/relatorio`). Estado do modelo em `GET /llm/status`.
* **Agente consultor (TP5)**: o LLM escolhe, passo a passo, quais análises rodar
  (potencial, dimensionamento, finanças, sazonalidade, CO2, benchmark, notícias) e escreve
  uma recomendação com números conferidos (`POST /agente/recomendar`).
* **Conversa com memória (TP5)**: perguntas de acompanhamento ("e em Manaus?") entendidas
  pelo contexto, com janela recente, resumo e fatos lembrados (`/conversas`).
* **Painel (TP5)**: indicadores do ciclo de ciência de dados e insights (`GET /painel`).

**Erros**: todos no formato `RespostaErro` (`status`, `erro`, `mensagem`, `sugestao`,
`detalhes`). Principais códigos: 404 (local inexistente), 409 (conflito), 413 (texto
longo demais para o modelo), 422 (dados inválidos), 502 (saída inválida do modelo ou
falha da NASA POWER), 503 (modelo carregando, ocupado, sem memória ou desativado, com
`Retry-After` quando faz sentido) e 504 (geração passou do prazo).
"""

TAGS = [
    {"name": "Locais e clima", "description": "Consulta da série diária de clima e irradiância."},
    {"name": "Longo prazo (CRESESB)", "description": "Médias mensais de longo prazo, plano horizontal e inclinado."},
    {"name": "Estimativa", "description": "Cálculo de geração e economia."},
    {"name": "Notícias", "description": "Notícias do setor solar extraídas por web scraping."},
    {"name": "Medições do usuário", "description": "Envio e gestão de dados próprios."},
    {"name": "Processamento de texto (LLM)", "description": "Resumo e sentimento com o LLM local."},
    {"name": "LLM", "description": "Estado do modelo local, perguntas e respostas (RAG) e relatório de viabilidade."},
    {"name": "Agente consultor (TP5)", "description": "Agente que escolhe as análises com o LLM e recomenda uma ação."},
    {"name": "Conversa com memória (TP5)", "description": "Chat que lembra o contexto entre as perguntas."},
    {"name": "Painel (TP5)", "description": "Dados do dashboard: ciclo de ciência de dados, benchmark das capitais e notícias com IA."},
]


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Ao subir a API, começa a carregar o LLM em segundo plano (a API já atende enquanto isso)."""
    llm = servico_llm()
    if llm.config.carregar_na_inicializacao:
        llm.carregar_em_segundo_plano()
    yield


app = FastAPI(title="Solar Viável API", version="2.0.0", description=DESCRICAO, openapi_tags=TAGS, lifespan=ciclo_de_vida)


# ---------------------------------------------------------------------------
# Tratamento de erros: toda falha vira uma RespostaErro (mesmo formato em todas as rotas)
# ---------------------------------------------------------------------------
CODIGOS_HTTP = {
    400: "requisicao_invalida",
    404: "nao_encontrado",
    405: "metodo_nao_permitido",
    409: "conflito",
    413: "conteudo_grande_demais",
    502: "falha_em_servico_externo",
    503: "servico_indisponivel",
}

TRADUCOES_VALIDACAO = {
    "missing": "campo obrigatório",
    "string_too_short": "texto curto demais (mínimo de {min_length} caracteres)",
    "string_too_long": "texto longo demais (máximo de {max_length} caracteres)",
    "too_short": "lista com poucos itens (mínimo de {min_length})",
    "too_long": "lista com itens demais (máximo de {max_length})",
    "greater_than": "precisa ser maior que {gt}",
    "greater_than_equal": "precisa ser maior ou igual a {ge}",
    "less_than": "precisa ser menor que {lt}",
    "less_than_equal": "precisa ser menor ou igual a {le}",
    "literal_error": "valor não permitido; use {expected}",
    "int_parsing": "precisa ser um número inteiro",
    "float_parsing": "precisa ser um número",
    "bool_parsing": "precisa ser true ou false",
    "date_from_datetime_parsing": "data inválida (use AAAA-MM-DD)",
    "date_parsing": "data inválida (use AAAA-MM-DD)",
    "json_invalid": "JSON malformado",
    "model_attributes_type": "o corpo da requisição precisa ser um objeto JSON",
    "string_type": "precisa ser um texto",
    "union_tag_invalid": "tipo não reconhecido",
}


def _erro(request: Request, status_code: int, erro: str, mensagem: str, sugestao=None, detalhes=None, headers=None) -> JSONResponse:
    corpo = schemas.RespostaErro(
        status=status_code, erro=erro, mensagem=mensagem, sugestao=sugestao, detalhes=detalhes, caminho=request.url.path
    )
    return JSONResponse(corpo.model_dump(), status_code=status_code, headers=headers)


@app.exception_handler(erros_llm.ErroLLM)
async def tratar_erro_llm(request: Request, exc: erros_llm.ErroLLM):
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    log.warning("%s %s -> %s %s", request.method, request.url.path, exc.status_code, exc.erro)
    return _erro(request, exc.status_code, exc.erro, exc.mensagem, exc.sugestao, headers=headers)


@app.exception_handler(StarletteHTTPException)
async def tratar_http(request: Request, exc: StarletteHTTPException):
    sugestao = "Veja as rotas disponíveis em /docs." if exc.status_code in (404, 405) and not isinstance(exc.detail, str) else None
    mensagem = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    if exc.status_code == 404 and mensagem == "Not Found":
        mensagem, sugestao = f"A rota {request.url.path} não existe.", "Veja as rotas disponíveis em /docs."
    if exc.status_code == 405 and mensagem == "Method Not Allowed":
        mensagem, sugestao = f"Método {request.method} não é aceito em {request.url.path}.", "Veja os métodos de cada rota em /docs."
    return _erro(request, exc.status_code, CODIGOS_HTTP.get(exc.status_code, "erro_http"), mensagem, sugestao, headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def tratar_validacao(request: Request, exc: RequestValidationError):
    detalhes = []
    for e in exc.errors():
        modelo = TRADUCOES_VALIDACAO.get(e["type"])
        try:
            problema = modelo.format(**e.get("ctx", {})) if modelo else e["msg"]
        except (KeyError, IndexError):
            problema = e["msg"]
        if e["type"] == "value_error":
            problema = str(e.get("ctx", {}).get("error", e["msg"]))
        detalhes.append(schemas.DetalheValidacao(campo=".".join(map(str, e["loc"])), problema=problema, tipo=e["type"]))
    resumo = "; ".join(f"{d.campo.split('.', 1)[-1]}: {d.problema}" for d in detalhes[:3])
    return _erro(
        request,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "dados_invalidos",
        f"Os dados enviados são inválidos ({resumo}).",
        "Corrija os campos indicados em 'detalhes'. O formato esperado está em /docs.",
        detalhes=[d.model_dump() for d in detalhes],
    )


@app.exception_handler(Exception)
async def tratar_inesperado(request: Request, exc: Exception):
    log.exception("Erro inesperado em %s %s", request.method, request.url.path)
    return _erro(
        request,
        500,
        "erro_interno",
        f"Erro inesperado no servidor ({type(exc).__name__}).",
        "Tente novamente; se persistir, veja o log da API.",
    )


ERROS_COMUNS = {422: {"model": schemas.RespostaErro, "description": "Dados inválidos"}}
ERROS_LLM = {
    **ERROS_COMUNS,
    413: {"model": schemas.RespostaErro, "description": "Texto longo demais para o modelo"},
    500: {"model": schemas.RespostaErro, "description": "Falha inesperada na inferência"},
    502: {"model": schemas.RespostaErro, "description": "O modelo devolveu uma saída inválida"},
    503: {"model": schemas.RespostaErro, "description": "Modelo carregando, ocupado, sem memória, desativado ou com falha na carga"},
    504: {"model": schemas.RespostaErro, "description": "A geração passou do prazo (SOLAR_LLM_TIMEOUT_S)"},
}
ERRO_404 = {404: {"model": schemas.RespostaErro, "description": "Local não encontrado"}}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _clima_base() -> pd.DataFrame:
    return repositorio.carregar_clima_base()


@lru_cache(maxsize=1)
def _sundata() -> pd.DataFrame:
    return repositorio.carregar_sundata()


@lru_cache(maxsize=1)
def _noticias() -> pd.DataFrame:
    return repositorio.carregar_noticias()


@lru_cache(maxsize=1)
def _corpus() -> pd.DataFrame:
    return repositorio.carregar_corpus_llm()


@lru_cache(maxsize=1)
def _noticias_ia() -> pd.DataFrame:
    return repositorio.carregar_noticias_ia()


def _dados_agente() -> agente.DadosAgente:
    return agente.DadosAgente(sundata=_sundata(), clima=_clima_base(), noticias_ia=_noticias_ia())


def _clima(incluir_medicoes: bool = True) -> pd.DataFrame:
    partes = [_clima_base().assign(origem="nasa")]
    medicoes = repositorio.carregar_medicoes() if incluir_medicoes else pd.DataFrame()
    if not medicoes.empty:
        partes.append(medicoes)
    return pd.concat(partes, ignore_index=True)


def _registros(df: pd.DataFrame) -> list:
    """DataFrame -> lista de dicts serializável (NaN vira null, datas viram AAAA-MM-DD)."""
    df = df.copy()
    for col in df.select_dtypes("datetime").columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _filtrar_periodo(df: pd.DataFrame, inicio: Optional[date], fim: Optional[date]) -> pd.DataFrame:
    if inicio:
        df = df[df["date"].dt.date >= inicio]
    if fim:
        df = df[df["date"].dt.date <= fim]
    return df


def _exigir_cidades(df: pd.DataFrame, cidades: List[str]) -> None:
    desconhecidas = sorted(set(cidades) - set(df["cidade"]))
    if desconhecidas:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Local não encontrado: {', '.join(desconhecidas)}. Veja GET /locais.")


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------
@app.get("/", response_model=schemas.Raiz, summary="Boas-vindas e links úteis")
def raiz():
    return {"api": "Solar Viável", "versao": app.version, "documentacao": "/docs", "saude": "/saude", "llm": "/llm/status"}


@app.get("/saude", response_model=schemas.Saude, summary="Verifica se a API está no ar")
def saude():
    return {"status": "ok", "llm": servico_llm().estado.value}


@app.get("/locais", response_model=List[schemas.Local], tags=["Locais e clima"], summary="Lista os locais disponíveis")
def listar_locais(incluir_medicoes: bool = True):
    """Capitais coletadas da NASA POWER e locais enviados por usuários, com o período e a irradiância média."""
    df = _clima(incluir_medicoes)
    resumo = (
        df.groupby(["cidade", "origem"], as_index=False)
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            dias=("date", "count"),
            inicio=("date", "min"),
            fim=("date", "max"),
            irradiacao_media=("ALLSKY_SFC_SW_DWN", "mean"),
        )
        .round({"irradiacao_media": 3})
        .sort_values("irradiacao_media", ascending=False)
    )
    return _registros(resumo)


@app.get("/clima", response_model=List[schemas.RegistroClima], responses=ERRO_404, tags=["Locais e clima"], summary="Série diária de clima")
def consultar_clima(
    cidade: List[str] = Query(default=[], description="um ou mais locais (repita o parâmetro); vazio = todos"),
    inicio: Optional[date] = Query(None, description="data inicial, AAAA-MM-DD"),
    fim: Optional[date] = Query(None, description="data final, AAAA-MM-DD"),
    incluir_medicoes: bool = Query(True, description="inclui as medições enviadas por usuários"),
):
    """Irradiância, temperatura, umidade, vento e nuvens por dia. Ex.: `/clima?cidade=Recife (PE)&inicio=2026-01-01`."""
    df = _clima(incluir_medicoes)
    if cidade:
        _exigir_cidades(df, cidade)
        df = df[df["cidade"].isin(cidade)]
    return _registros(_filtrar_periodo(df, inicio, fim))


@app.get(
    "/clima/resumo",
    response_model=List[Dict[str, Union[str, float, None]]],
    responses=ERRO_404,
    tags=["Locais e clima"],
    summary="Estatísticas por local",
)
def resumo_clima(
    cidade: List[str] = Query(default=[]),
    inicio: Optional[date] = None,
    fim: Optional[date] = None,
):
    """Média, mínimo e máximo de cada variável por local, ordenado pela irradiância média."""
    df = _clima()
    if cidade:
        _exigir_cidades(df, cidade)
        df = df[df["cidade"].isin(cidade)]
    df = _filtrar_periodo(df, inicio, fim)
    if df.empty:
        return []
    return _registros(climate.resumo_por_cidade(df).reset_index())


@app.get(
    "/clima/ao-vivo",
    response_model=List[schemas.RegistroClima],
    responses={502: {"model": schemas.RespostaErro, "description": "Falha ao consultar a NASA POWER"}},
    tags=["Locais e clima"],
    summary="Consulta ao vivo da NASA POWER",
)
def clima_ao_vivo(
    latitude: float = Query(..., ge=-35, le=6),
    longitude: float = Query(..., ge=-75, le=-30),
    dias: int = Query(180, ge=7, le=365),
    nome: str = Query("Meu local", max_length=80),
):
    """Busca na NASA POWER os últimos `dias` dias de qualquer coordenada do Brasil (não é salvo)."""
    fim = date.today() - timedelta(days=2)  # a NASA tem defasagem de poucos dias
    try:
        df = nasa_power_client.get_power_data(latitude, longitude, fim - timedelta(days=dias), fim)
    except Exception as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Falha ao consultar a NASA POWER: {exc}") from exc
    df = df.reset_index()
    for col in climate.VARIAVEIS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.assign(cidade=f"{nome} (ao vivo)", latitude=latitude, longitude=longitude, origem="nasa_ao_vivo")
    return _registros(df[[*climate.COLUNAS_MODELO, "origem"]])


@app.get(
    "/irradiacao-longo-prazo",
    response_model=List[schemas.IrradiacaoLongoPrazo],
    responses=ERRO_404,
    tags=["Longo prazo (CRESESB)"],
    summary="Irradiação média mensal de longo prazo",
)
def irradiacao_longo_prazo(
    cidade: List[str] = Query(default=[]),
    plano: Optional[Literal["Plano Horizontal", "Ângulo igual a latitude", "Maior média anual", "Maior mínimo mensal"]] = None,
):
    """Dados do CRESESB/SunData para a estação de referência de cada capital (mais próxima em terra)."""
    df = _sundata()
    if cidade:
        _exigir_cidades(df, cidade)
        df = df[df["cidade"].isin(cidade)]
    if plano:
        df = df[df["plano"] == plano]
    return [
        {
            **{k: r[k] for k in ("cidade", "estacao", "municipio", "uf", "distancia_km", "plano", "inclinacao_graus", "media_anual", "delta")},
            "mensal": {m: r[m] for m in climate.MESES},
        }
        for r in _registros(df)
    ]


@app.get("/estimativa", response_model=schemas.Estimativa, responses=ERRO_404, tags=["Estimativa"], summary="Geração e economia estimadas")
def estimativa(
    cidade: str,
    potencia_kwp: float = Query(5.0, gt=0, le=1000, description="potência instalada"),
    desempenho: float = Query(0.80, ge=0.5, le=0.95, description="taxa de desempenho (perdas)"),
    tarifa: float = Query(0.90, gt=0, le=5, description="R$/kWh"),
    fonte: Literal["nasa", "cresesb"] = Query("cresesb", description="nasa = últimos 12 meses medidos; cresesb = longo prazo, painel inclinado"),
):
    """Geração = irradiância (horas de sol pleno) x potência x desempenho. Ex.: `/estimativa?cidade=Fortaleza (CE)&potencia_kwp=4`."""
    if fonte == "cresesb":
        df = _sundata()
        linha = df[(df["cidade"] == cidade) & (df["plano"] == "Maior média anual")]
        if linha.empty:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Sem dados do CRESESB para '{cidade}'. Use fonte=nasa.")
        r = linha.iloc[0]
        mensal = climate.geracao_mensal([r[m] for m in climate.MESES], potencia_kwp, desempenho, tarifa)
        ano = float(mensal["geracao_kwh"].sum())
        irradiacao = float(r["media_anual"])
        obs = f"Estação {r['estacao']} ({r['distancia_km']} km), painéis a {r['inclinacao_graus']}° voltados para o norte."
    else:
        df = _clima()
        serie = df.loc[df["cidade"] == cidade, "ALLSKY_SFC_SW_DWN"]
        if serie.empty:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Local não encontrado: {cidade}. Veja GET /locais.")
        irradiacao = float(serie.mean())
        ano = climate.estimar_geracao(irradiacao, potencia_kwp, desempenho) * 365
        obs = f"Média de {serie.count()} dias de irradiância no plano horizontal."
    dia = ano / 365
    return {
        "cidade": cidade,
        "fonte": fonte,
        "irradiacao_media": round(irradiacao, 3),
        "potencia_kwp": potencia_kwp,
        "desempenho": desempenho,
        "tarifa": tarifa,
        "geracao_dia_kwh": round(dia, 2),
        "geracao_mes_kwh": round(ano / 12, 1),
        "geracao_ano_kwh": round(ano, 0),
        "economia_mes_rs": round(ano / 12 * tarifa, 2),
        "economia_ano_rs": round(ano * tarifa, 2),
        "observacao": obs,
    }


@app.get("/noticias", response_model=List[schemas.Noticia], tags=["Notícias"], summary="Notícias do setor solar")
def listar_noticias(
    fonte: List[str] = Query(default=[], description="ABSOLAR e/ou Canal Solar"),
    busca: Optional[str] = Query(None, description="filtra por termo no título/resumo/texto"),
    limite: int = Query(50, ge=1, le=500),
    incluir_texto: bool = Query(False, description="inclui o texto completo das matérias"),
):
    df = _noticias()
    if fonte:
        df = df[df["fonte"].isin(fonte)]
    if busca:
        conteudo = df["titulo"] + " " + df["resumo"] + " " + df["texto"]
        df = df[conteudo.str.contains(busca, case=False, regex=False)]
    df = df.assign(n_palavras=df["texto"].str.split().str.len()).sort_values("data", ascending=False).head(limite)
    if not incluir_texto:
        df = df.drop(columns="texto")
    return _registros(df)


@app.get("/noticias/palavras", response_model=List[schemas.PalavraFrequente], tags=["Notícias"], summary="Palavras mais frequentes")
def palavras_frequentes(fonte: List[str] = Query(default=[]), top: int = Query(20, ge=1, le=100)):
    df = _noticias()
    if fonte:
        df = df[df["fonte"].isin(fonte)]
    return _registros(text.frequencia_palavras(text.montar_corpus(df), top))


@app.post(
    "/medicoes",
    response_model=schemas.RespostaMedicoes,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": schemas.RespostaErro, "description": "Nome reservado a uma capital"}},
    tags=["Medições do usuário"],
    summary="Envia medições de um local",
)
def enviar_medicoes(lote: schemas.LoteMedicoes):
    """Recebe medições diárias (validadas) e as guarda; o local passa a aparecer em `/locais` e `/clima`.

    Datas repetidas para o mesmo local substituem as anteriores. Os nomes das capitais
    coletadas são reservados.
    """
    if lote.cidade in set(_clima_base()["cidade"]):
        raise HTTPException(status.HTTP_409_CONFLICT, f"'{lote.cidade}' é um local da base NASA; use outro nome.")
    df = pd.DataFrame([m.model_dump() for m in lote.medicoes])
    df["date"] = pd.to_datetime(df["date"])
    df = df.assign(cidade=lote.cidade, latitude=lote.latitude, longitude=lote.longitude)
    recebidas = repositorio.salvar_medicoes(df)
    return {"cidade": lote.cidade, "recebidas": recebidas, "mensagem": f"{recebidas} medição(ões) salva(s)."}


@app.get("/medicoes", response_model=List[schemas.RegistroClima], tags=["Medições do usuário"], summary="Lista as medições enviadas")
def listar_medicoes(cidade: Optional[str] = None):
    df = repositorio.carregar_medicoes()
    if cidade:
        df = df[df["cidade"] == cidade]
    return _registros(df)


@app.delete(
    "/medicoes/{cidade}",
    response_model=schemas.RemocaoMedicoes,
    responses={404: {"model": schemas.RespostaErro, "description": "Nenhuma medição para o local"}},
    tags=["Medições do usuário"],
    summary="Remove as medições de um local",
)
def remover_medicoes(cidade: str):
    removidas = repositorio.remover_medicoes(cidade)
    if not removidas:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Nenhuma medição para '{cidade}'.")
    return {"cidade": cidade, "removidas": removidas}


@app.get("/llm/corpus", response_model=schemas.CorpusLLM, tags=["LLM"], summary="Estatísticas do corpus usado pelo LLM")
def corpus_llm():
    df = _corpus()
    return {
        "documentos": len(df),
        "palavras": int(df["n_palavras"].sum()) if len(df) else 0,
        "por_tipo": df["tipo"].value_counts().to_dict(),
        "por_fonte": df["fonte"].value_counts().to_dict(),
    }


@app.get("/llm/contexto", response_model=schemas.ContextoLLM, tags=["LLM"], summary="Recupera contexto para uma pergunta")
def contexto_llm(pergunta: str = Query(..., min_length=3), k: int = Query(4, ge=1, le=10)):
    """Mostra a etapa de recuperação: os trechos que `POST /llm/perguntar` enviaria ao modelo."""
    trechos = llm_prep.buscar_contexto(_corpus(), pergunta, k)
    return {"pergunta": pergunta, "trechos": _registros(trechos), "prompt": llm_prep.montar_prompt(pergunta, trechos)}


# ---------------------------------------------------------------------------
# LLM local (HuggingFace Transformers)
# ---------------------------------------------------------------------------
ROTAS_LLM = [
    "POST /processar_texto",
    "POST /llm/perguntar",
    "POST /llm/relatorio",
    "POST /agente/recomendar",
    "POST /conversas/{id}/mensagens",
]


@app.get("/llm/status", response_model=schemas.StatusLLM, tags=["LLM"], summary="Estado do modelo local")
def status_llm():
    """Mostra se o modelo está carregado, em que hardware (cuda, mps ou cpu), quanta memória usa e os limites configurados."""
    return {**servico_llm().status(), "tarefas": ROTAS_LLM}


@app.post(
    "/llm/carregar",
    response_model=schemas.StatusLLM,
    status_code=status.HTTP_202_ACCEPTED,
    responses={503: {"model": schemas.RespostaErro, "description": "LLM desativado nesta instalação"}},
    tags=["LLM"],
    summary="Inicia a carga do modelo",
)
def carregar_llm():
    """Começa a carregar o modelo em segundo plano (útil com SOLAR_LLM_CARREGAR=0). Acompanhe em `GET /llm/status`."""
    llm = servico_llm()
    if llm.estado.value == "desativado":
        raise erros_llm.ModeloDesativado("O LLM está desativado nesta instalação (SOLAR_LLM_DESATIVADO=1).")
    llm.carregar_em_segundo_plano()
    return {**llm.status(), "tarefas": ROTAS_LLM}


def _uso(bruto: dict, em_cache: bool) -> dict:
    return {**bruto, "tempo_s": 0.0 if em_cache else bruto["tempo_s"], "em_cache": em_cache}


@app.post(
    "/processar_texto",
    response_model=schemas.RespostaProcessamento,
    responses=ERROS_LLM,
    tags=["Processamento de texto (LLM)"],
    summary="Resume ou classifica o sentimento de um texto",
)
def processar_texto(pedido: schemas.PedidoProcessarTexto):
    """Processa um texto com o LLM local.

    * `tarefa=resumo`: resumo fiel em um parágrafo de até `max_palavras` palavras.
    * `tarefa=sentimento`: positivo, neutro ou negativo, com a probabilidade de cada rótulo
      (calculada a partir das probabilidades do próprio modelo) e uma justificativa curta.
      Com `ponto_de_vista=setor_solar`, o sentimento é avaliado para quem quer instalar
      energia solar (ex.: notícias do setor).

    A resposta traz o resultado, o modelo usado (nome, hardware) e o custo (tokens e tempo).
    """
    llm = servico_llm()
    if pedido.tarefa == "resumo":
        saida = tarefas.resumir(llm, pedido.texto, pedido.max_palavras)
    else:
        saida = tarefas.classificar_sentimento(llm, pedido.texto, pedido.ponto_de_vista, pedido.justificar)
    return {
        "tarefa": pedido.tarefa,
        "entrada": {
            "n_caracteres": len(pedido.texto),
            "n_palavras": len(pedido.texto.split()),
            "n_tokens": saida["n_tokens_entrada"],
            "inicio": pedido.texto[:120],
        },
        "resultado": saida["resultado"],
        "modelo": llm.info_modelo(),
        "uso": _uso(saida["uso"], saida["em_cache"]),
        "processado_em": datetime.now(),
    }


@app.post(
    "/llm/perguntar",
    response_model=schemas.RespostaPergunta,
    responses=ERROS_LLM,
    tags=["LLM"],
    summary="Responde perguntas com base nas notícias e nos dados das capitais (RAG)",
)
def perguntar(pedido: schemas.PedidoPergunta):
    """Recupera os trechos mais relevantes do corpus (notícias + fichas e ranking das capitais),
    envia-os numerados ao modelo e devolve a resposta com as fontes. O modelo é instruído a
    usar só o contexto, citar [n] e dizer quando não encontrar a informação."""
    llm = servico_llm()
    llm.info_modelo()  # falha cedo (503) se o modelo não estiver pronto
    saida = tarefas.responder_pergunta(llm, _corpus(), pedido.pergunta, pedido.k)
    return {
        "pergunta": pedido.pergunta,
        "resposta": saida["resposta"],
        "fontes": saida["fontes"],
        "sem_contexto": saida["sem_contexto"],
        "modelo": llm.info_modelo(),
        "uso": _uso(saida["uso"], saida["em_cache"]),
    }


@app.post(
    "/llm/relatorio",
    response_model=schemas.RespostaRelatorio,
    responses={**ERROS_LLM, **ERRO_404},
    tags=["LLM"],
    summary="Relatório de viabilidade em linguagem simples",
)
def relatorio(pedido: schemas.PedidoRelatorio):
    """Gera um relatório para leigos a partir de `GET /estimativa` (fonte CRESESB), da
    irradiação mês a mês e da ficha do local. Os números vêm da API: depois da geração,
    cada número do texto é conferido contra os dados de entrada (`verificacao`)."""
    llm = servico_llm()
    llm.info_modelo()
    est = estimativa(pedido.cidade, pedido.potencia_kwp, pedido.desempenho, pedido.tarifa, "cresesb")
    sundata = _sundata()
    linha = sundata[(sundata["cidade"] == pedido.cidade) & (sundata["plano"] == "Maior média anual")].iloc[0]
    mensal = {m: float(linha[m]) for m in climate.MESES}
    corpus = _corpus()
    fichas = corpus.loc[(corpus["tipo"] == "ficha_local") & (corpus["cidade"] == pedido.cidade), "texto"]
    saida = tarefas.gerar_relatorio(llm, est, mensal, fichas.iloc[0] if len(fichas) else "")
    return {
        "cidade": pedido.cidade,
        "relatorio": saida["relatorio"],
        "verificacao": saida["verificacao"],
        "estimativa": est,
        "modelo": llm.info_modelo(),
        "uso": _uso(saida["uso"], saida["em_cache"]),
    }


# ---------------------------------------------------------------------------
# TP5: agente consultor
# ---------------------------------------------------------------------------
@app.get("/agente/ferramentas", response_model=List[schemas.FerramentaAgente], tags=["Agente consultor (TP5)"], summary="Ferramentas do agente")
def ferramentas_agente():
    """As ações que o agente pode escolher, com as dependências entre elas (guardrails)."""
    return agente.ferramentas_publicas()


@app.post(
    "/agente/recomendar",
    response_model=schemas.RespostaAgente,
    responses={**ERROS_LLM, **ERRO_404},
    tags=["Agente consultor (TP5)"],
    summary="Analisa o caso do usuário e recomenda uma ação",
)
def recomendar(pedido: schemas.PedidoAgente):
    """Executa o agente: a cada passo o LLM local escolhe a próxima análise (pela probabilidade
    de cada ação), a ferramenta calcula sobre os dados coletados e o resultado vira uma
    observação. No fim, o LLM escreve a recomendação e cada número do texto é conferido.

    A resposta traz os passos (com as probabilidades de cada decisão), os resultados de cada
    ferramenta, os alertas por regras, a recomendação e o `id` para conversar sobre ela."""
    llm = servico_llm()
    llm.info_modelo()
    perfil = pedido.model_dump(exclude={"objetivo"})
    try:
        saida = agente.executar(llm, _dados_agente(), perfil, pedido.objetivo)
    except KeyError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Sem dados de longo prazo para '{pedido.cidade}'. Use uma das capitais de GET /locais."
        )
    return {**saida, "modelo": llm.info_modelo(), "uso": _uso(saida["uso"], saida["em_cache"])}


@app.get(
    "/agente/recomendacoes/{id_}",
    response_model=schemas.RespostaAgente,
    responses={404: {"model": schemas.RespostaErro, "description": "Recomendação não encontrada"}},
    tags=["Agente consultor (TP5)"],
    summary="Consulta uma recomendação já feita",
)
def obter_recomendacao(id_: str):
    rec = agente.obter_recomendacao(id_)
    if rec is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Recomendação '{id_}' não encontrada.")
    llm = servico_llm()
    return {**rec, "modelo": {**llm.info_modelo()}, "uso": _uso(rec["uso"], True)}


# ---------------------------------------------------------------------------
# TP5: conversa com memória
# ---------------------------------------------------------------------------
ERRO_CONVERSA = {404: {"model": schemas.RespostaErro, "description": "Conversa não encontrada"}}


@app.post(
    "/conversas",
    response_model=schemas.Conversa,
    status_code=status.HTTP_201_CREATED,
    responses={404: {"model": schemas.RespostaErro, "description": "Recomendação não encontrada"}},
    tags=["Conversa com memória (TP5)"],
    summary="Inicia uma conversa (opcionalmente sobre uma recomendação do agente)",
)
def criar_conversa(pedido: Optional[schemas.PedidoConversa] = None):
    rec = None
    if pedido and pedido.recomendacao_id:
        rec = agente.obter_recomendacao(pedido.recomendacao_id)
        if rec is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Recomendação '{pedido.recomendacao_id}' não encontrada.")
    return memoria.repositorio_conversas().criar(rec).publico()


@app.get("/conversas/{id_}", response_model=schemas.Conversa, responses=ERRO_CONVERSA, tags=["Conversa com memória (TP5)"], summary="Histórico e memória de uma conversa")
def obter_conversa(id_: str):
    return memoria.repositorio_conversas().obter(id_).publico()


@app.delete("/conversas/{id_}", response_model=schemas.RemocaoConversa, responses=ERRO_CONVERSA, tags=["Conversa com memória (TP5)"], summary="Apaga uma conversa")
def apagar_conversa(id_: str):
    """Apaga o histórico e a memória (da API e do disco)."""
    memoria.repositorio_conversas().apagar(id_)
    return {"id": id_, "apagada": True}


@app.post(
    "/conversas/{id_}/mensagens",
    response_model=schemas.RespostaMensagem,
    responses={**ERROS_LLM, **ERRO_CONVERSA},
    tags=["Conversa com memória (TP5)"],
    summary="Envia uma pergunta e recebe a resposta, usando a memória da conversa",
)
def enviar_mensagem(id_: str, pedido: schemas.PedidoMensagem):
    """A pergunta é reescrita com a memória (ex.: "e em Manaus?" vira uma pergunta completa),
    os trechos são buscados com a pergunta reescrita e o modelo responde vendo as últimas
    trocas, o resumo das anteriores e os fatos lembrados. A resposta mostra o estado da memória."""
    repo = memoria.repositorio_conversas()
    conversa = repo.obter(id_)
    llm = servico_llm()
    llm.info_modelo()
    saida = memoria.conversar(llm, _corpus(), conversa, pedido.pergunta, pedido.k)
    repo.atualizar(conversa)
    return {
        **saida,
        "conversa_id": conversa.id,
        "memoria": memoria.estado_memoria(conversa),
        "modelo": llm.info_modelo(),
        "uso": _uso(saida["uso"], False),
    }


# ---------------------------------------------------------------------------
# TP5: painel
# ---------------------------------------------------------------------------
@app.get("/painel", response_model=schemas.Painel, tags=["Painel (TP5)"], summary="Indicadores do ciclo de ciência de dados e insights")
def painel(
    consumo_kwh_mes: float = Query(300, gt=0, le=20000),
    tarifa: float = Query(0.95, gt=0, le=5),
    custo_kwp: float = Query(viabilidade.CUSTO_KWP_PADRAO, ge=1000, le=15000),
):
    """Tudo que o painel do app mostra de uma vez: quanto dado foi coletado e processado em
    cada etapa, o benchmark das capitais para o perfil informado e o sentimento das notícias
    classificado pelo LLM."""
    clima, sundata, noticias, corpus, nia = _clima_base(), _sundata(), _noticias(), _corpus(), _noticias_ia()
    ausentes = float(clima["ALLSKY_SFC_SW_DWN"].isna().mean() * 100)
    llm = servico_llm()
    ciclo = [
        {
            "etapa": "Coleta",
            "descricao": "API REST da NASA POWER, Selenium no SunData (CRESESB) e Beautiful Soup na ABSOLAR e no Canal Solar",
            "indicadores": {
                "fontes": 4,
                "dias_de_clima": int(clima["date"].nunique()),
                "registros_nasa": len(clima),
                "capitais": int(clima["cidade"].nunique()),
                "series_cresesb": len(sundata),
                "noticias": len(noticias),
            },
        },
        {
            "etapa": "Processamento",
            "descricao": "limpeza (-999 vira ausente), padronização, trechos de texto, fichas das capitais e ranking",
            "indicadores": {
                "ausentes_irradiacao_pct": round(ausentes, 1),
                "documentos_corpus": len(corpus),
                "palavras_corpus": int(corpus["n_palavras"].sum()) if len(corpus) else 0,
            },
        },
        {
            "etapa": "Análise",
            "descricao": "geração mês a mês, dimensionamento, payback com Fio B, CO2 evitado e benchmark das capitais",
            "indicadores": {"capitais_comparadas": len(viabilidade.capitais(sundata)), "ferramentas_do_agente": len(agente.FERRAMENTAS)},
        },
        {
            "etapa": "IA",
            "descricao": "LLM local: resumo e sentimento das notícias, agente que decide as análises e chat com memória",
            "indicadores": {
                "noticias_classificadas": len(nia),
                "modelo": llm.config.modelo,
                "estado_modelo": llm.estado.value,
            },
        },
        {
            "etapa": "Apresentação",
            "descricao": "painel Streamlit consumindo a API FastAPI, com recomendações, gráficos e conversa",
            "indicadores": {"rotas_api": len([r for r in app.routes if getattr(r, "include_in_schema", False)])},
        },
    ]
    bench = viabilidade.comparar_capitais(sundata, consumo_kwh_mes, tarifa, custo_kwp)
    por_fonte, por_mes = [], []
    if len(nia):
        por_fonte = nia.groupby(["fonte", "sentimento"]).size().reset_index(name="noticias").to_dict(orient="records")
        mes = nia.dropna(subset=["data"]).assign(mes=lambda d: d["data"].dt.strftime("%Y-%m"))
        por_mes = mes.groupby(["mes", "sentimento"]).size().reset_index(name="noticias").to_dict(orient="records")
    return {
        "ciclo": ciclo,
        "capitais": bench["capitais"],
        "perfil": {"consumo_kwh_mes": consumo_kwh_mes, "tarifa": tarifa, "custo_kwp": custo_kwp},
        "sentimento_noticias": nia["sentimento"].value_counts().to_dict() if len(nia) else {},
        "sentimento_por_fonte": por_fonte,
        "sentimento_por_mes": por_mes,
        "noticias_processadas": len(nia),
        "modelo_noticias": str(nia["modelo"].iloc[0]) if len(nia) else None,
        "premissas": viabilidade.PREMISSAS,
    }


@app.get("/noticias/ia", response_model=List[schemas.NoticiaIA], tags=["Notícias"], summary="Notícias resumidas e classificadas pelo LLM")
def noticias_ia(sentimento: Optional[Literal["positivo", "neutro", "negativo"]] = None, limite: int = Query(100, ge=1, le=500)):
    """Resultado do processamento em lote (`python -m src.llm.enriquecer_noticias`): resumo de cada
    notícia e sentimento do ponto de vista de quem quer instalar energia solar."""
    df = _noticias_ia()
    if sentimento:
        df = df[df["sentimento"] == sentimento]
    df = df.sort_values("data", ascending=False).head(limite)
    return [
        {
            "titulo": r.titulo,
            "fonte": r.fonte,
            "data": r.data.strftime("%Y-%m-%d") if pd.notna(r.data) else "",
            "url": r.url,
            "resumo_ia": r.resumo_ia,
            "sentimento": r.sentimento,
            "confianca": float(r.confianca),
            "justificativa": r.justificativa or None,
        }
        for r in df.itertuples()
    ]
