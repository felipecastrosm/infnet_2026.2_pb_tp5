"""Modelos (Pydantic) de entrada e saída da API: validam os dados e aparecem no /docs.

Toda rota declara um `response_model`: o FastAPI valida o que a rota devolve contra
esse modelo (campos, tipos e limites) antes de responder, e a documentação OpenAPI
mostra o formato exato de cada resposta. Os erros também têm formato fixo
(`RespostaErro`), seja qual for a causa.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Respostas gerais e de erro
# ---------------------------------------------------------------------------
class DetalheValidacao(BaseModel):
    campo: str = Field(description="onde está o problema, ex.: body.texto ou query.cidade")
    problema: str = Field(description="explicação em português")
    tipo: str = Field(description="código do erro de validação (Pydantic)")


class RespostaErro(BaseModel):
    """Formato único de erro de toda a API."""

    status: int = Field(description="código HTTP")
    erro: str = Field(description="código curto e estável do erro, bom para tratar no cliente")
    mensagem: str = Field(description="o que aconteceu, em português")
    sugestao: Optional[str] = Field(None, description="o que fazer para resolver")
    detalhes: Optional[List[DetalheValidacao]] = Field(None, description="lista de problemas (erros de validação)")
    caminho: str = Field(description="rota que gerou o erro")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": 503,
                    "erro": "modelo_carregando",
                    "mensagem": "O modelo 'Qwen/Qwen3-1.7B' ainda está sendo carregado.",
                    "sugestao": "O modelo está sendo carregado na memória. Tente de novo em alguns segundos.",
                    "detalhes": None,
                    "caminho": "/processar_texto",
                }
            ]
        }
    }


class Raiz(BaseModel):
    api: str
    versao: str
    documentacao: str
    saude: str
    llm: str


class Saude(BaseModel):
    status: Literal["ok"]
    llm: str = Field(description="estado do LLM local (ver GET /llm/status)")


class RemocaoMedicoes(BaseModel):
    cidade: str
    removidas: int


class PalavraFrequente(BaseModel):
    palavra: str
    frequencia: int


class CorpusLLM(BaseModel):
    documentos: int
    palavras: int
    por_tipo: Dict[str, int]
    por_fonte: Dict[str, int]


class Local(BaseModel):
    cidade: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    origem: str = Field(description="nasa (capitais coletadas) ou usuario (medições enviadas via POST)")
    dias: int
    inicio: dt.date
    fim: dt.date
    irradiacao_media: float = Field(description="kWh/m²/dia no período")


class RegistroClima(BaseModel):
    date: dt.date
    cidade: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    ALLSKY_SFC_SW_DWN: Optional[float] = Field(None, description="irradiância solar, kWh/m²/dia")
    T2M: Optional[float] = Field(None, description="temperatura a 2 m, °C")
    RH2M: Optional[float] = Field(None, description="umidade relativa, %")
    WS10M: Optional[float] = Field(None, description="vento a 10 m, m/s")
    CLOUD_AMT: Optional[float] = Field(None, description="cobertura de nuvens, %")
    origem: str = "nasa"


class Medicao(BaseModel):
    """Uma medição diária enviada pelo usuário (ex.: de um piranômetro ou do inversor)."""

    date: dt.date
    ALLSKY_SFC_SW_DWN: float = Field(ge=0, le=12, description="irradiância solar, kWh/m²/dia (0 a 12)")
    T2M: Optional[float] = Field(None, ge=-20, le=55)
    RH2M: Optional[float] = Field(None, ge=0, le=100)
    WS10M: Optional[float] = Field(None, ge=0, le=60)
    CLOUD_AMT: Optional[float] = Field(None, ge=0, le=100)


class LoteMedicoes(BaseModel):
    cidade: str = Field(min_length=2, max_length=80, description="nome do local, ex.: 'Meu telhado - Niterói'")
    latitude: Optional[float] = Field(None, ge=-35, le=6)
    longitude: Optional[float] = Field(None, ge=-75, le=-30)
    medicoes: List[Medicao] = Field(min_length=1, max_length=5000)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cidade": "Meu telhado - Niterói",
                    "latitude": -22.88,
                    "longitude": -43.10,
                    "medicoes": [
                        {"date": "2026-09-01", "ALLSKY_SFC_SW_DWN": 5.2, "T2M": 24.1},
                        {"date": "2026-09-02", "ALLSKY_SFC_SW_DWN": 3.8},
                    ],
                }
            ]
        }
    }


class RespostaMedicoes(BaseModel):
    cidade: str
    recebidas: int
    mensagem: str


class Estimativa(BaseModel):
    cidade: str
    fonte: str
    irradiacao_media: float = Field(description="kWh/m²/dia usada no cálculo")
    potencia_kwp: float
    desempenho: float
    tarifa: float
    geracao_dia_kwh: float
    geracao_mes_kwh: float
    geracao_ano_kwh: float
    economia_mes_rs: float
    economia_ano_rs: float
    observacao: str


class IrradiacaoLongoPrazo(BaseModel):
    cidade: str
    estacao: str
    municipio: str
    uf: str
    distancia_km: float
    plano: str
    inclinacao_graus: int
    mensal: dict = Field(description="irradiação diária média de cada mês, kWh/m²/dia")
    media_anual: float
    delta: float = Field(description="diferença entre o maior e o menor mês")


class Noticia(BaseModel):
    fonte: str
    titulo: str
    categoria: str
    resumo: str
    url: str
    data: Optional[dt.date] = None
    n_palavras: int
    texto: Optional[str] = None


class TrechoContexto(BaseModel):
    id: str
    tipo: str
    fonte: str
    titulo: str
    url: str
    data: str
    cidade: str
    texto: str
    pontuacao: float


class ContextoLLM(BaseModel):
    pergunta: str
    trechos: List[TrechoContexto]
    prompt: str = Field(description="prompt montado com os trechos (o mesmo conteúdo que POST /llm/perguntar envia ao modelo)")


# ---------------------------------------------------------------------------
# LLM local
# ---------------------------------------------------------------------------
def _texto_com_conteudo(valor: str) -> str:
    valor = re.sub(r"\s+", " ", valor).strip()
    if len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", valor)) < 3:
        raise ValueError("o texto precisa ter pelo menos 3 palavras")
    return valor


class InfoModelo(BaseModel):
    nome: str = Field(description="modelo da HuggingFace executado localmente")
    dispositivo: str = Field(description="cuda, mps (GPU da Apple) ou cpu")
    dtype: str = Field(description="tipo numérico dos pesos (float16, float32...)")
    parametros_milhoes: float


class UsoModelo(BaseModel):
    tokens_entrada: int = Field(ge=0)
    tokens_saida: int = Field(ge=0)
    tempo_s: float = Field(ge=0, description="tempo de processamento (0 se veio do cache)")
    tokens_por_s: Optional[float] = None
    em_cache: bool = Field(description="true = mesma entrada já processada; resposta reaproveitada")


class LimitesLLM(BaseModel):
    timeout_s: float
    espera_fila_s: float
    max_tokens_entrada: int


class StatusLLM(BaseModel):
    estado: Literal["desativado", "nao_carregado", "carregando", "pronto", "erro"]
    pronto: bool
    modelo: str
    dispositivo: Optional[str] = None
    dtype: Optional[str] = None
    parametros_milhoes: Optional[float] = None
    carregado_em: Optional[dt.datetime] = None
    tempo_carregamento_s: Optional[float] = None
    memoria_modelo_mb: Optional[float] = Field(None, description="memória ocupada pelo modelo na GPU (cuda/mps)")
    memoria_processo_mb: Optional[float] = Field(None, description="memória RAM do processo da API")
    requisicoes_atendidas: int
    limites: LimitesLLM
    erro: Optional[str] = Field(None, description="motivo, se a carga falhou")
    tarefas: List[str] = Field(description="rotas que usam o modelo")


class PedidoProcessarTexto(BaseModel):
    texto: str = Field(min_length=20, max_length=20000, description="texto em português (20 a 20.000 caracteres)")
    tarefa: Literal["resumo", "sentimento"] = Field("resumo", description="resumo ou sentimento")
    max_palavras: int = Field(80, ge=20, le=250, description="tamanho máximo do resumo, em palavras")
    ponto_de_vista: Literal["geral", "setor_solar"] = Field(
        "geral", description="sentimento: tom geral do texto ou do ponto de vista do setor solar / de quem quer instalar"
    )
    justificar: bool = Field(True, description="sentimento: pede ao modelo uma frase justificando o rótulo")

    _validar_texto = field_validator("texto")(_texto_com_conteudo)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texto": "A fonte solar respondeu por 73% da expansão da matriz elétrica brasileira entre janeiro e agosto "
                    "de 2026, com 3,5 GW adicionados. A capacidade total de geração centralizada do país já supera 220 GW.",
                    "tarefa": "resumo",
                    "max_palavras": 40,
                },
                {
                    "texto": "Os cortes de geração (curtailment) já somam 73,4 milhões de MWh desde 2021 e podem causar "
                    "prejuízo de até R$ 13,6 bilhões às usinas renováveis.",
                    "tarefa": "sentimento",
                    "ponto_de_vista": "setor_solar",
                },
            ]
        }
    }


class DadosEntrada(BaseModel):
    n_caracteres: int
    n_palavras: int
    n_tokens: int = Field(description="tokens do texto para o tokenizador do modelo")
    inicio: str = Field(description="primeiros caracteres do texto recebido")


class ResultadoResumo(BaseModel):
    tipo: Literal["resumo"]
    resumo: str
    n_palavras_resumo: int
    taxa_compressao: float = Field(ge=0, description="palavras do resumo / palavras do texto")
    cortado: bool = Field(description="true se a geração atingiu o limite de tokens (o final incompleto é descartado)")


class ProbabilidadesSentimento(BaseModel):
    positivo: float = Field(ge=0, le=1)
    neutro: float = Field(ge=0, le=1)
    negativo: float = Field(ge=0, le=1)


class ResultadoSentimento(BaseModel):
    tipo: Literal["sentimento"]
    sentimento: Literal["positivo", "neutro", "negativo"]
    confianca: float = Field(ge=0, le=1, description="probabilidade do rótulo escolhido")
    probabilidades: ProbabilidadesSentimento
    ponto_de_vista: Literal["geral", "setor_solar"]
    justificativa: Optional[str] = None


class RespostaProcessamento(BaseModel):
    tarefa: Literal["resumo", "sentimento"]
    entrada: DadosEntrada
    resultado: Annotated[Union[ResultadoResumo, ResultadoSentimento], Field(discriminator="tipo")]
    modelo: InfoModelo
    uso: UsoModelo
    processado_em: dt.datetime


class PedidoPergunta(BaseModel):
    pergunta: str = Field(min_length=5, max_length=500)
    k: int = Field(4, ge=1, le=8, description="quantos trechos do corpus enviar ao modelo")

    _validar_pergunta = field_validator("pergunta")(_texto_com_conteudo)

    model_config = {"json_schema_extra": {"examples": [{"pergunta": "Quanto um sistema de 5 kWp gera por mês em Curitiba?", "k": 4}]}}


class FonteCitada(BaseModel):
    n: int = Field(description="número do trecho no contexto, usado nas citações [n]")
    titulo: str
    fonte: str
    tipo: str
    url: str
    data: str
    citada: bool = Field(description="o modelo citou este trecho na resposta")


class RespostaPergunta(BaseModel):
    pergunta: str
    resposta: str
    fontes: List[FonteCitada]
    sem_contexto: bool = Field(description="true se nenhum trecho do corpus foi encontrado (o modelo não é chamado)")
    modelo: InfoModelo
    uso: UsoModelo


class PedidoRelatorio(BaseModel):
    cidade: str = Field(min_length=2, max_length=80, description="uma das capitais (GET /locais)")
    potencia_kwp: float = Field(5.0, gt=0, le=1000)
    desempenho: float = Field(0.80, ge=0.5, le=0.95)
    tarifa: float = Field(0.90, gt=0, le=5, description="R$/kWh")

    model_config = {"json_schema_extra": {"examples": [{"cidade": "Curitiba (PR)", "potencia_kwp": 4, "tarifa": 0.95}]}}


class VerificacaoNumeros(BaseModel):
    numeros_citados: List[float]
    nao_encontrados: List[float] = Field(description="números do texto que não existem nos dados de entrada")
    confiavel: bool = Field(description="true se todos os números conferem com a API")


class RespostaRelatorio(BaseModel):
    cidade: str
    relatorio: str = Field(description="relatório em Markdown")
    verificacao: VerificacaoNumeros
    estimativa: Estimativa = Field(description="os números de origem (GET /estimativa)")
    modelo: InfoModelo
    uso: UsoModelo
