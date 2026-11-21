"""Memória conversacional do assistente (TP5).

No TP4 cada pergunta era respondida sozinha: "e em Manaus?" ou "e se fosse 3 kWp?" não
faziam sentido para o modelo. Aqui a conversa ganha memória em três camadas, combinadas
a cada pergunta:

| Camada | O que guarda | Como é atualizada |
|---|---|---|
| Janela recente | as últimas `JANELA` trocas, na íntegra | a cada resposta |
| Resumo | o que saiu da janela, condensado em um parágrafo | pelo LLM, quando a janela enche |
| Fatos | cidade, consumo, potência, tarifa e a recomendação do agente | por regras (expressões regulares e a lista de capitais), sem LLM |

Antes de buscar contexto, a pergunta de acompanhamento é **reescrita** como pergunta
completa ("e em Manaus?" vira "Quanto um sistema de 5 kWp gera por mês em Manaus?"): é
essa pergunta que vai para a recuperação de trechos (RAG), porque a busca não tem memória.
Se a reescrita do modelo vier inválida, entra uma regra simples (pergunta + fatos).

As conversas ficam em memória e também em arquivos JSON (`data/processed/conversas/`),
então sobrevivem a um reinício da API. Conversas sem uso há mais de `VALIDADE_H` horas
são descartadas.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src import repositorio
from src.data_processing import llm_prep
from src.llm import erros, tarefas
from src.llm.servico import ServicoLLM

JANELA = 3  # trocas (pergunta + resposta) enviadas na íntegra ao modelo
VALIDADE_H = 24
MAX_CONVERSAS = 500


def pasta_conversas() -> Path:
    return Path(os.environ.get("SOLAR_CONVERSAS_DIR", repositorio.PROCESSED_DIR / "conversas"))


@dataclass
class Turno:
    pergunta: str
    resposta: str
    pergunta_reescrita: str
    fontes: list
    criado_em: str


@dataclass
class Conversa:
    id: str
    criada_em: str
    atualizada_em: str
    turnos: list = field(default_factory=list)
    resumo: str = ""
    turnos_resumidos: int = 0
    fatos: dict = field(default_factory=dict)
    recomendacao_id: Optional[str] = None
    ficha_recomendacao: str = ""

    def janela(self) -> list[Turno]:
        return self.turnos[max(self.turnos_resumidos, len(self.turnos) - JANELA):]

    def publico(self) -> dict:
        return {
            "id": self.id,
            "criada_em": self.criada_em,
            "atualizada_em": self.atualizada_em,
            "turnos": [asdict(t) if isinstance(t, Turno) else t for t in self.turnos],
            "memoria": estado_memoria(self),
        }


def estado_memoria(c: Conversa) -> dict:
    return {
        "fatos": c.fatos,
        "resumo": c.resumo,
        "total_turnos": len(c.turnos),
        "turnos_na_janela": len(c.janela()),
        "turnos_resumidos": c.turnos_resumidos,
        "tamanho_janela": JANELA,
        "recomendacao_id": c.recomendacao_id,
    }


# ---------------------------------------------------------------------------
# Repositório de conversas (memória + disco)
# ---------------------------------------------------------------------------
class RepositorioConversas:
    def __init__(self):
        self._conversas: dict[str, Conversa] = {}
        self._trava = threading.Lock()

    def _arquivo(self, id_: str) -> Path:
        return pasta_conversas() / f"{id_}.json"

    def _salvar(self, c: Conversa) -> None:
        pasta = pasta_conversas()
        pasta.mkdir(parents=True, exist_ok=True)
        self._arquivo(c.id).write_text(json.dumps(asdict(c), ensure_ascii=False, indent=1), encoding="utf-8")

    def _expirar(self) -> None:
        limite = (datetime.now() - timedelta(hours=VALIDADE_H)).isoformat()
        for id_ in [i for i, c in self._conversas.items() if c.atualizada_em < limite]:
            self.apagar(id_, silencioso=True)
        while len(self._conversas) > MAX_CONVERSAS:
            mais_antiga = min(self._conversas.values(), key=lambda c: c.atualizada_em)
            self.apagar(mais_antiga.id, silencioso=True)

    def criar(self, recomendacao: Optional[dict] = None) -> Conversa:
        agora = datetime.now().isoformat(timespec="seconds")
        c = Conversa(id=uuid.uuid4().hex[:12], criada_em=agora, atualizada_em=agora)
        if recomendacao:
            from src.llm import agente

            c.recomendacao_id = recomendacao["id"]
            c.ficha_recomendacao = agente.ficha_recomendacao(recomendacao)
            p = recomendacao["perfil"]
            dim = recomendacao["resultados"].get("dimensionar_sistema", {})
            c.fatos = {
                "cidade": p["cidade"],
                "cidade_recomendacao": p["cidade"],
                "cidades_citadas": [p["cidade"]],
                "consumo_kwh_mes": p["consumo_kwh_mes"],
                "tarifa": p["tarifa"],
                "potencia_kwp": dim.get("potencia_kwp", p.get("potencia_kwp")),
            }
        with self._trava:
            self._expirar()
            self._conversas[c.id] = c
            self._salvar(c)
        return c

    def obter(self, id_: str) -> Conversa:
        with self._trava:
            c = self._conversas.get(id_)
            if c is None and self._arquivo(id_).exists():
                bruto = json.loads(self._arquivo(id_).read_text(encoding="utf-8"))
                bruto["turnos"] = [Turno(**t) for t in bruto["turnos"]]
                c = self._conversas[id_] = Conversa(**bruto)
        if c is None:
            raise ConversaNaoEncontrada(f"Conversa '{id_}' não encontrada (pode ter expirado após {VALIDADE_H} h sem uso).")
        return c

    def atualizar(self, c: Conversa) -> None:
        c.atualizada_em = datetime.now().isoformat(timespec="seconds")
        with self._trava:
            self._conversas[c.id] = c
            self._salvar(c)

    def apagar(self, id_: str, silencioso: bool = False) -> None:
        existia = self._conversas.pop(id_, None) is not None
        arquivo = self._arquivo(id_)
        if arquivo.exists():
            arquivo.unlink()
            existia = True
        if not existia and not silencioso:
            raise ConversaNaoEncontrada(f"Conversa '{id_}' não encontrada.")

    def limpar(self) -> None:
        with self._trava:
            for id_ in list(self._conversas):
                self.apagar(id_, silencioso=True)


class ConversaNaoEncontrada(erros.ErroLLM):
    status_code = 404
    erro = "conversa_nao_encontrada"
    sugestao = "Crie uma nova conversa com POST /conversas."


_repositorio = RepositorioConversas()


def repositorio_conversas() -> RepositorioConversas:
    return _repositorio


# ---------------------------------------------------------------------------
# Fatos (memória de entidades, por regras)
# ---------------------------------------------------------------------------
_NUM = r"(\d+(?:[.,]\d+)?)"
_POTENCIA = re.compile(_NUM + r"\s*kwp\b", re.I)
_CONSUMO = re.compile(_NUM + r"\s*kwh\s*(?:/|por|ao|no|mensais|mensal|de consumo)\s*(?:m[eê]s|mensais)?", re.I)
_CONSUMO_2 = re.compile(r"consum\w*\s*(?:é|e|de|:)?\s*(?:cerca de\s*)?" + _NUM + r"\s*kwh", re.I)
_TARIFA = re.compile(r"r\$\s*" + _NUM + r"\s*(?:/|por)\s*kwh", re.I)


def _f(bruto: str) -> float:
    return float(bruto.replace(",", "."))


def extrair_fatos(corpus: pd.DataFrame, texto: str) -> dict:
    fatos = {}
    cidades = llm_prep.cidades_citadas(corpus, texto)
    if cidades:
        fatos["cidade"] = cidades[-1]
    if m := _POTENCIA.search(texto):
        fatos["potencia_kwp"] = _f(m.group(1))
    m = _CONSUMO_2.search(texto) or (_CONSUMO.search(texto) if "kwh/m" not in texto.lower() else None)
    if m:
        fatos["consumo_kwh_mes"] = _f(m.group(1))
    if m := _TARIFA.search(texto):
        fatos["tarifa"] = _f(m.group(1))
    return fatos


def _num(v: float) -> str:
    """4.0 -> "4"; 3.5 -> "3,5" (sem zeros sobrando)."""
    return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def descrever_fatos(fatos: dict) -> str:
    partes = []
    if "cidade" in fatos:
        partes.append(f"cidade em foco: {fatos['cidade']}")
    if fatos.get("cidade_recomendacao") and fatos.get("cidade_recomendacao") != fatos.get("cidade"):
        partes.append(f"a recomendação do agente é para {fatos['cidade_recomendacao']}")
    if len(fatos.get("cidades_citadas", [])) > 1:
        partes.append(f"cidades já citadas: {', '.join(fatos['cidades_citadas'])}")
    if fatos.get("potencia_kwp"):
        partes.append(f"potência do sistema: {_num(fatos['potencia_kwp'])} kWp")
    if fatos.get("consumo_kwh_mes"):
        partes.append(f"consumo: {_num(fatos['consumo_kwh_mes'])} kWh por mês")
    if fatos.get("tarifa"):
        partes.append(f"tarifa: R$ {tarifa_fmt(fatos['tarifa'])} por kWh")
    return "; ".join(partes)


def tarifa_fmt(v: float) -> str:
    return tarefas._fmt(v, 2)


# ---------------------------------------------------------------------------
# Reescrita da pergunta, resposta e resumo
# ---------------------------------------------------------------------------
SISTEMA_REESCRITA = (
    "Você reescreve perguntas de acompanhamento de uma conversa sobre energia solar. Transforme a última pergunta do "
    "usuário em uma pergunta completa, que possa ser entendida sem o histórico, trocando referências como 'lá', "
    "'e em', 'isso' ou 'nesse caso' pelo que elas significam. Mantenha a intenção e os números do usuário. "
    "Não responda a pergunta. Escreva só a pergunta reescrita, em uma frase."
)


def _encurtar(texto: str, palavras: int) -> str:
    p = texto.split()
    return " ".join(p[:palavras]) + ("..." if len(p) > palavras else "")


def precisa_reescrever(conversa: Conversa, pergunta: str) -> bool:
    return bool(conversa.turnos) or bool(conversa.fatos)


_ACOMPANHAMENTO = re.compile(
    r"^(e|mas|entao|então|e se|e quanto|e a|e o|e os|e as)\b|\b(la|lá|ali|dela|dele|delas|deles|nela|nele|isso|esse|essa|"
    r"esses|essas|nesse|nessa|neste|nesta|mesmo|mesma|tambem|também|anterior|ela|ele|elas|eles)\b",
    re.I,
)


def parece_acompanhamento(pergunta: str) -> bool:
    """Perguntas que dependem do histórico: começam com "e..." ou usam referências ("lá", "dela", "isso")."""
    return bool(_ACOMPANHAMENTO.search(pergunta.strip())) or len(pergunta.split()) <= 3


def regra_reescrita(pergunta: str, fatos_pergunta: dict, fatos: dict) -> str:
    """Plano B: acrescenta à pergunta os fatos da memória que ela não traz (só em perguntas de acompanhamento)."""
    if not parece_acompanhamento(pergunta):
        return pergunta
    extras = []
    if "cidade" in fatos and "cidade" not in fatos_pergunta:
        extras.append(f"em {fatos['cidade']}")
    if fatos.get("potencia_kwp") and "potencia_kwp" not in fatos_pergunta and re.search(r"ger|produz|econom|sistema", pergunta, re.I):
        extras.append(f"com um sistema de {_num(fatos['potencia_kwp'])} kWp")
    return pergunta if not extras else f"{pergunta.rstrip('?. ')} ({', '.join(extras)})?"


def _pedido_reescrita(fatos: str, resumo: str, historico: str, pergunta: str) -> str:
    return (
        (f"Fatos da conversa: {fatos}\n" if fatos else "")
        + (f"Resumo da conversa: {resumo}\n" if resumo else "")
        + (f"Histórico recente:\n{historico}\n\n" if historico else "\n")
        + f"Última pergunta: {pergunta}\n\nPergunta reescrita:"
    )


# Exemplos (few-shot): modelos pequenos seguem muito melhor um formato mostrado do que descrito.
EXEMPLOS_REESCRITA = [
    (
        "cidade em foco: Curitiba (PR); potência do sistema: 5 kWp",
        "Usuário: Quanto um sistema de 5 kWp gera por mês em Curitiba?\nAssistente: Gera cerca de 534 kWh por mês [1].",
        "E em Manaus?",
        "Quanto um sistema de 5 kWp gera por mês em Manaus?",
    ),
    (
        "cidade em foco: Fortaleza (CE); cidades já citadas: Recife (PE), Fortaleza (CE)",
        "Usuário: Como é o sol em Recife?\nAssistente: Recife recebe 5,48 kWh/m² por dia [1].\n"
        "Usuário: E em Fortaleza?\nAssistente: Fortaleza recebe 5,78 kWh/m² por dia [1].",
        "Qual delas é melhor para energia solar?",
        "Entre Recife e Fortaleza, qual capital tem mais sol para energia solar?",
    ),
    (
        "",
        "Usuário: O que é a Lei 14.300?\nAssistente: É o marco legal da geração distribuída [1].",
        "E o que muda para quem instala agora?",
        "O que a Lei 14.300 muda para quem instala energia solar agora?",
    ),
]


def _normalizar(texto: str) -> str:
    return re.sub(r"\W+", " ", texto.lower()).strip()


def reescrever(servico: ServicoLLM, conversa: Conversa, pergunta: str, fatos_pergunta: dict) -> tuple[str, str, list]:
    """Devolve (pergunta reescrita, método, gerações)."""
    if not precisa_reescrever(conversa, pergunta):
        return pergunta, "original", []
    historico = "\n".join(
        f"Usuário: {t.pergunta}\nAssistente: {_encurtar(t.resposta, 45)}" for t in conversa.janela()[-2:]
    )
    mensagens = [{"role": "system", "content": SISTEMA_REESCRITA}]
    for fatos, hist, ultima, reescrita in EXEMPLOS_REESCRITA:
        mensagens += [
            {"role": "user", "content": _pedido_reescrita(fatos, "", hist, ultima)},
            {"role": "assistant", "content": reescrita},
        ]
    mensagens.append({"role": "user", "content": _pedido_reescrita(descrever_fatos(conversa.fatos), conversa.resumo, historico, pergunta)})
    g = servico.gerar(mensagens, max_novos_tokens=80)
    texto = tarefas._limpar_saida(g.texto).split("\n")[0].strip().strip('"').strip()
    texto = re.sub(r"^(pergunta reescrita|pergunta)\s*:\s*", "", texto, flags=re.I)
    n = len(texto.split())
    regra = regra_reescrita(pergunta, fatos_pergunta, conversa.fatos)
    if 3 <= n <= 60 and not g.cortada and not (_normalizar(texto) == _normalizar(pergunta) and regra != pergunta):
        return texto, "modelo", [g]
    # saída inválida, ou o modelo só repetiu a pergunta quando a memória tinha algo a acrescentar
    return regra, "regra", [g]


SISTEMA_CHAT = tarefas.SISTEMA_QA + (
    " Você conversa com o usuário e se lembra do que foi dito antes: use os fatos e o resumo da conversa para "
    "entender a pergunta, mas os números da resposta devem vir do contexto."
)


def _docs_calculados(simular, corpus: pd.DataFrame, conversa: Conversa, pergunta_busca: str) -> pd.DataFrame:
    """Simulações feitas na hora com os fatos lembrados (cidade, potência), via `simular`."""
    if simular is None:
        return pd.DataFrame()
    cidades = llm_prep.cidades_citadas(corpus, pergunta_busca) or ([conversa.fatos["cidade"]] if "cidade" in conversa.fatos else [])
    if not cidades:
        return pd.DataFrame()
    docs = simular(cidades, conversa.fatos.get("potencia_kwp"))
    return pd.DataFrame(
        [
            {"id": f"sim-{i}", "tipo": "simulacao", "fonte": "Cálculo do Solar Viável (CRESESB)", "url": "",
             "data": datetime.now().strftime("%Y-%m-%d"), "n_palavras": len(d["texto"].split()), "pontuacao": 1.0, **d}
            for i, d in enumerate(docs)
        ]
    )


def _contexto_conversa(corpus: pd.DataFrame, conversa: Conversa, pergunta_busca: str, k: int, simular=None) -> pd.DataFrame:
    calculados = _docs_calculados(simular, corpus, conversa, pergunta_busca)
    trechos = llm_prep.buscar_contexto(corpus, pergunta_busca, k)
    if len(calculados):
        # a simulação substitui a ficha genérica da mesma cidade (calculada para 5 kWp)
        trechos = trechos[~((trechos["tipo"] == "ficha_local") & trechos["cidade"].isin(calculados["cidade"]))]
        trechos = pd.concat([calculados, trechos], ignore_index=True).head(k + 1)
    if conversa.ficha_recomendacao:
        rec = pd.DataFrame(
            [
                {
                    "id": f"rec-{conversa.recomendacao_id}",
                    "tipo": "recomendacao",
                    "fonte": "Agente consultor do Solar Viável",
                    "titulo": "Recomendação do agente consultor",
                    "url": "",
                    "data": conversa.criada_em[:10],
                    "cidade": conversa.fatos.get("cidade", ""),
                    "texto": conversa.ficha_recomendacao,
                    "n_palavras": len(conversa.ficha_recomendacao.split()),
                    "pontuacao": float(trechos["pontuacao"].max()) if len(trechos) else 1.0,
                }
            ]
        )
        # a recomendação vem primeiro, a não ser que a pergunta seja sobre outra cidade
        # (ex.: "e em Fortaleza?" numa conversa sobre Curitiba): aí a ficha da cidade nova vem antes
        outras = llm_prep.cidades_citadas(corpus, pergunta_busca)
        if outras and conversa.fatos.get("cidade_recomendacao") not in outras:
            trechos = pd.concat([trechos.head(max(k - 1, 1)), rec], ignore_index=True)
        else:
            trechos = pd.concat([rec, trechos.head(max(k - 1, 1))], ignore_index=True)
    return trechos


def _resumir_conversa(servico: ServicoLLM, conversa: Conversa) -> list:
    """Condensa no resumo as trocas que saíram da janela."""
    saindo = conversa.turnos[conversa.turnos_resumidos : len(conversa.turnos) - JANELA]
    if not saindo:
        return []
    trocas = "\n".join(f"Usuário: {t.pergunta}\nAssistente: {_encurtar(t.resposta, 60)}" for t in saindo)
    mensagens = [
        {
            "role": "system",
            "content": "Você mantém um resumo curto de uma conversa sobre energia solar. Guarde o que o usuário quer saber, "
            "as cidades, números e conclusões importantes. Escreva em português, em no máximo 70 palavras.",
        },
        {
            "role": "user",
            "content": (f"Resumo atual: {conversa.resumo}\n\n" if conversa.resumo else "")
            + f"Novas trocas:\n{trocas}\n\nResumo atualizado:",
        },
    ]
    g = servico.gerar(mensagens, max_novos_tokens=140)
    resumo = tarefas._ate_ultima_frase(tarefas._limpar_saida(g.texto)) if g.cortada else tarefas._limpar_saida(g.texto)
    if resumo:
        conversa.resumo = tarefas._limitar_palavras(tarefas._sem_repeticao(resumo), 90)
        conversa.turnos_resumidos += len(saindo)
    return [g]


def conversar(servico: ServicoLLM, corpus: pd.DataFrame, conversa: Conversa, pergunta: str, k: int = 4, simular=None) -> dict:
    """Responde usando a memória. `simular(cidades, potencia_kwp)` (opcional) devolve documentos
    calculados na hora para as cidades da pergunta (ver viabilidade.fichas_simulacao)."""
    pergunta = tarefas.normalizar_espacos(pergunta)
    servico.exigir_tamanho(pergunta, "A pergunta")
    inicio = time.perf_counter()
    fatos_pergunta = extrair_fatos(corpus, pergunta)
    reescrita, metodo, geracoes = reescrever(servico, conversa, pergunta, fatos_pergunta)
    conversa.fatos.update(fatos_pergunta)  # a pergunta nova atualiza a memória de fatos
    citadas = [c for c in conversa.fatos.get("cidades_citadas", []) if c not in llm_prep.cidades_citadas(corpus, pergunta)]
    citadas += llm_prep.cidades_citadas(corpus, pergunta)
    if citadas:
        conversa.fatos["cidades_citadas"] = citadas[-4:]
    trechos = _contexto_conversa(corpus, conversa, reescrita, k, simular)
    fontes = [
        {"n": i, "titulo": r.titulo, "fonte": r.fonte, "tipo": r.tipo, "url": r.url, "data": r.data, "citada": False}
        for i, r in enumerate(trechos.itertuples(), start=1)
    ]
    memoria = descrever_fatos(conversa.fatos)
    sistema = SISTEMA_CHAT
    if memoria or conversa.resumo:
        sistema += "\n\nMemória da conversa:" + (f"\nFatos: {memoria}." if memoria else "") + (
            f"\nResumo do que já foi conversado: {conversa.resumo}" if conversa.resumo else ""
        )
    historico = []
    for t in conversa.janela():
        historico += [{"role": "user", "content": t.pergunta}, {"role": "assistant", "content": t.resposta}]
    if trechos.empty:
        resposta = "Não encontrei essa informação nos dados do Solar Viável (notícias do setor e fichas das capitais)."
    else:
        pedido = (
            f"Contexto:\n{tarefas._contexto_numerado(trechos)}\n\nPergunta: {pergunta}"
            + (f"\n(Entenda a pergunta como: {reescrita})" if reescrita != pergunta else "")
            + "\n\nResponda em até 4 frases, usando só o contexto, e cite os trechos usados entre colchetes, como [1]."
        )
        g = servico.gerar([{"role": "system", "content": sistema}, *historico, {"role": "user", "content": pedido}], max_novos_tokens=320)
        geracoes.append(g)
        resposta = tarefas._limpar_saida(g.texto)
        if g.cortada:
            resposta = tarefas._ate_ultima_frase(resposta)
        resposta = re.sub(r"^(\[\d+\]\s*)+", "", tarefas._sem_repeticao(resposta, max_frases=6))
        resposta = re.sub(r"(\[\d+\])([.!?]?)(\s*\1)+", r"\1\2", resposta).strip()
        if not resposta:
            raise erros.SaidaInvalida("O modelo devolveu uma resposta vazia.")
    citadas = {int(n) for n in re.findall(r"\[(\d+)\]", resposta)}
    for f in fontes:
        f["citada"] = f["n"] in citadas
    conversa.turnos.append(Turno(pergunta, resposta, reescrita, fontes, datetime.now().isoformat(timespec="seconds")))
    resumiu = False
    if len(conversa.turnos) - conversa.turnos_resumidos > JANELA:
        geracoes += _resumir_conversa(servico, conversa)
        resumiu = True
    return {
        "pergunta": pergunta,
        "pergunta_reescrita": reescrita,
        "metodo_reescrita": metodo,
        "resposta": resposta,
        "fontes": fontes,
        "resumo_atualizado": resumiu,
        "uso": tarefas._uso(geracoes, inicio),
    }
