"""Tarefas de linguagem natural do Solar Viável, executadas pelo LLM local.

Cada função monta o prompt (em formato de chat), chama o serviço (`servico.py`) e
transforma a saída do modelo em um dicionário com campos fixos, que a API valida com
os Response Models de `src/api/schemas.py`.

| Tarefa      | Como o modelo é usado                                   | Saída estruturada |
|-------------|---------------------------------------------------------|-------------------|
| resumo      | geração de texto (gulosa), limite de palavras           | resumo, nº de palavras, taxa de compressão |
| sentimento  | pontuação dos rótulos (log-probabilidades) + justificativa curta | rótulo, confiança, probabilidades |
| perguntar   | RAG: trechos do corpus numerados + regras de citação   | resposta, fontes citadas |
| relatorio   | geração a partir do JSON de `/estimativa` + ficha do local | relatório em Markdown + checagem dos números |

Princípio mantido desde o TP3: **os números vêm da API**. O modelo explica e resume;
no relatório, cada número do texto é conferido contra os dados de entrada.

Os resultados ficam em um cache em memória (mesma entrada → mesma resposta, sem
executar o modelo de novo).
"""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from typing import Optional

import pandas as pd

from src.data_processing import llm_prep
from src.llm import erros
from src.llm.servico import ServicoLLM

ROTULOS_SENTIMENTO = {
    "positivo": ["positivo", "Positivo"],
    "neutro": ["neutro", "Neutro"],
    "negativo": ["negativo", "Negativo"],
}

PONTOS_DE_VISTA = {
    "geral": "o tom geral do texto",
    "setor_solar": (
        "o ponto de vista de quem quer instalar energia solar e do setor de energia solar no Brasil. "
        "positivo = o fato relatado traz um benefício concreto (crescimento, queda de custo, nova usina, "
        "economia, proteção ao consumidor). negativo = o fato relatado traz um problema concreto (custo maior, "
        "prejuízo, restrição, risco, fraude, tarifa mais cara). neutro = texto informativo, técnico, regulatório "
        "ou sobre debates, propostas e prazos ainda sem efeito definido. Muitas notícias do setor são neutras: "
        "não marque positivo só porque o tema é energia solar"
    ),
}

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 256


def _chave(servico: ServicoLLM, tarefa: str, **params) -> str:
    return json.dumps({"modelo": servico.config.modelo, "tarefa": tarefa, **params}, sort_keys=True, ensure_ascii=False, default=str)


def _do_cache(chave: str) -> Optional[dict]:
    if chave in _CACHE:
        _CACHE.move_to_end(chave)
        return {**_CACHE[chave], "em_cache": True}
    return None


def _guardar(chave: str, resultado: dict) -> dict:
    _CACHE[chave] = resultado
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return {**resultado, "em_cache": False}


def limpar_cache() -> None:
    _CACHE.clear()


def normalizar_espacos(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()


def _limpar_saida(texto: str) -> str:
    """Remove raciocínio (<think>), rótulos repetidos e espaços sobrando."""
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S)
    texto = re.sub(r"^\s*(resumo|resposta|justificativa)\s*:\s*", "", texto, flags=re.I)
    return texto.strip()


def _sem_repeticao(texto: str, max_frases: Optional[int] = None) -> str:
    """Remove frases repetidas (laços de repetição de modelos pequenos) e limita o número de frases."""
    vistas, frases = set(), []
    for frase in re.split(r"(?<=[.!?\]])\s+", texto.strip()):
        chave = re.sub(r"\W+", " ", frase.lower()).strip()
        if chave and chave not in vistas:
            vistas.add(chave)
            frases.append(frase)
    if max_frases:
        frases = frases[:max_frases]
    return " ".join(frases)


def _ate_ultima_frase(texto: str) -> str:
    """Se a geração foi cortada no meio de uma frase, descarta o pedaço incompleto."""
    fim = max(texto.rfind(". "), texto.rfind("! "), texto.rfind("? "), texto.rfind(".\n"))
    if texto.rstrip().endswith((".", "!", "?")):
        return texto.strip()
    return texto[: fim + 1].strip() if fim > 0 else texto.strip()


def _limitar_palavras(texto: str, max_palavras: int) -> str:
    """Mantém frases inteiras até o limite de palavras (modelos pequenos às vezes passam do pedido)."""
    if _contar_palavras(texto) <= max_palavras:
        return texto
    frases = re.split(r"(?<=[.!?])\s+", texto)
    escolhidas = []
    for frase in frases:
        if escolhidas and _contar_palavras(" ".join(escolhidas + [frase])) > max_palavras:
            break
        escolhidas.append(frase)
    return " ".join(escolhidas)


def _uso(geracoes: list, inicio: float) -> dict:
    entrada = sum(g.tokens_entrada for g in geracoes)
    saida = sum(g.tokens_saida for g in geracoes)
    tempo = time.perf_counter() - inicio
    return {
        "tokens_entrada": entrada,
        "tokens_saida": saida,
        "tempo_s": round(tempo, 2),
        "tokens_por_s": round(saida / tempo, 1) if tempo > 0 and saida else None,
    }


def _contar_palavras(texto: str) -> int:
    return len(texto.split())


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
def resumir(servico: ServicoLLM, texto: str, max_palavras: int = 80) -> dict:
    texto = normalizar_espacos(texto)
    chave = _chave(servico, "resumo", texto=texto, max_palavras=max_palavras)
    if (em_cache := _do_cache(chave)) is not None:
        return em_cache
    servico.exigir_tamanho(texto)
    inicio = time.perf_counter()
    mensagens = [
        {
            "role": "system",
            "content": "Você resume textos em português do Brasil, de forma fiel e objetiva. "
            "Use apenas informações presentes no texto, sem opiniões e sem inventar dados.",
        },
        {
            "role": "user",
            "content": f"Resuma o texto abaixo em um único parágrafo de no máximo {max_palavras} palavras.\n\nTexto:\n{texto}",
        },
    ]
    geracao = servico.gerar(mensagens, max_novos_tokens=int(max_palavras * 2.5) + 32)
    resumo = _limpar_saida(geracao.texto)
    if geracao.cortada:
        resumo = _ate_ultima_frase(resumo)
    resumo = _limitar_palavras(_sem_repeticao(resumo), int(max_palavras * 1.15))
    if _contar_palavras(resumo) < 3:
        raise erros.SaidaInvalida("O modelo não produziu um resumo aproveitável para este texto.")
    n_entrada, n_resumo = _contar_palavras(texto), _contar_palavras(resumo)
    resultado = {
        "tipo": "resumo",
        "resumo": resumo,
        "n_palavras_resumo": n_resumo,
        "taxa_compressao": round(n_resumo / n_entrada, 3) if n_entrada else 0.0,
        "cortado": geracao.cortada,
    }
    return _guardar(chave, {"resultado": resultado, "uso": _uso([geracao], inicio), "n_tokens_entrada": geracao.tokens_entrada})


# ---------------------------------------------------------------------------
# Sentimento
# ---------------------------------------------------------------------------
def _mensagens_sentimento(texto: str, ponto_de_vista: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "Você classifica o sentimento de textos em português. "
            f"Considere {PONTOS_DE_VISTA[ponto_de_vista]}. "
            "Responda com uma única palavra: positivo, neutro ou negativo.",
        },
        {"role": "user", "content": f"Texto:\n{texto}\n\nSentimento (positivo, neutro ou negativo):"},
    ]


def classificar_sentimento(servico: ServicoLLM, texto: str, ponto_de_vista: str = "geral", justificar: bool = True) -> dict:
    texto = normalizar_espacos(texto)
    chave = _chave(servico, "sentimento", texto=texto, ponto_de_vista=ponto_de_vista, justificar=justificar)
    if (em_cache := _do_cache(chave)) is not None:
        return em_cache
    n_tokens = servico.exigir_tamanho(texto)
    inicio = time.perf_counter()
    mensagens = _mensagens_sentimento(texto, ponto_de_vista)
    probabilidades = servico.pontuar(mensagens, ROTULOS_SENTIMENTO)
    sentimento = max(probabilidades, key=probabilidades.get)
    geracoes = []
    justificativa = None
    if justificar:
        g = servico.gerar(
            mensagens
            + [
                {"role": "assistant", "content": sentimento},
                {"role": "user", "content": f"Em uma única frase de até 30 palavras, explique por que o texto é {sentimento}."},
            ],
            max_novos_tokens=90,
        )
        geracoes.append(g)
        justificativa = _ate_ultima_frase(_limpar_saida(g.texto)) or None
    uso = _uso(geracoes, inicio)
    uso["tokens_entrada"] += n_tokens  # a pontuação também lê o texto inteiro
    resultado = {
        "tipo": "sentimento",
        "sentimento": sentimento,
        "confianca": round(probabilidades[sentimento], 4),
        "probabilidades": {r: round(p, 4) for r, p in probabilidades.items()},
        "ponto_de_vista": ponto_de_vista,
        "justificativa": justificativa,
    }
    return _guardar(chave, {"resultado": resultado, "uso": uso, "n_tokens_entrada": n_tokens})


# ---------------------------------------------------------------------------
# Perguntas e respostas sobre o corpus (RAG)
# ---------------------------------------------------------------------------
SISTEMA_QA = (
    "Você é o assistente do Solar Viável, que ajuda pessoas leigas a avaliar energia solar "
    "fotovoltaica no Brasil. Responda em português do Brasil, de forma simples e curta (até 5 frases), "
    "usando SOMENTE as informações do contexto. Cite a fonte de cada informação com o número do "
    "trecho entre colchetes, por exemplo [1]. Copie os números exatamente como aparecem no contexto. "
    "Se o contexto não tiver a resposta, responda exatamente: Não encontrei essa informação nos dados do Solar Viável."
)


def _contexto_numerado(trechos: pd.DataFrame) -> str:
    return "\n\n".join(
        f"[{i}] {row.titulo} ({row.fonte}{', ' + row.data if row.data else ''})\n{row.texto}"
        for i, row in enumerate(trechos.itertuples(), start=1)
    )


def responder_pergunta(servico: ServicoLLM, corpus: pd.DataFrame, pergunta: str, k: int = 4) -> dict:
    pergunta = normalizar_espacos(pergunta)
    chave = _chave(servico, "perguntar", pergunta=pergunta, k=k, n_corpus=len(corpus))
    if (em_cache := _do_cache(chave)) is not None:
        return em_cache
    trechos = llm_prep.buscar_contexto(corpus, pergunta, k)
    fontes = [
        {"n": i, "titulo": r.titulo, "fonte": r.fonte, "tipo": r.tipo, "url": r.url, "data": r.data, "citada": False}
        for i, r in enumerate(trechos.itertuples(), start=1)
    ]
    inicio = time.perf_counter()
    if trechos.empty:
        resposta = "Não encontrei essa informação nos dados do Solar Viável (notícias do setor e fichas das capitais)."
        return _guardar(chave, {"resposta": resposta, "fontes": [], "sem_contexto": True, "uso": _uso([], inicio)})
    mensagens = [
        {"role": "system", "content": SISTEMA_QA},
        {
            "role": "user",
            "content": f"Contexto:\n{_contexto_numerado(trechos)}\n\nPergunta: {pergunta}\n\n"
            "Responda em até 4 frases, usando só o contexto, e cite os trechos usados entre colchetes, como [1].",
        },
    ]
    geracao = servico.gerar(mensagens, max_novos_tokens=320)
    resposta = _limpar_saida(geracao.texto)
    if geracao.cortada:
        resposta = _ate_ultima_frase(resposta)
    resposta = re.sub(r"^(\[\d+\]\s*)+", "", _sem_repeticao(resposta, max_frases=6))
    resposta = re.sub(r"(\[\d+\])([.!?]?)(\s*\1)+", r"\1\2", resposta).strip()  # "[1]. [1]" -> "[1]."
    if not resposta:
        raise erros.SaidaInvalida("O modelo devolveu uma resposta vazia.")
    citadas = {int(n) for n in re.findall(r"\[(\d+)\]", resposta)}
    for f in fontes:
        f["citada"] = f["n"] in citadas
    return _guardar(chave, {"resposta": resposta, "fontes": fontes, "sem_contexto": False, "uso": _uso([geracao], inicio)})


# ---------------------------------------------------------------------------
# Relatório de viabilidade
# ---------------------------------------------------------------------------
_NUMERO = re.compile(r"(?<![\w/])\d{1,3}(?:\.\d{3})+(?:,\d+)?(?![\d/])|(?<![\w/.,])\d+(?:[.,]\d+)?(?![\d/])")


_UNIDADE = re.compile(r"\s*(%|por cento|anos?\b|mes(es)?\b|kwh|kwp|kw\b|mwh|gw|watts?\b|w\b|reais|kg)", re.I)


def _valor(bruto: str) -> float:
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", bruto):
        return float(bruto.replace(".", "").replace(",", "."))
    return float(bruto.replace(",", "."))


def extrair_numeros(texto: str) -> list[float]:
    """Números de um texto em formato brasileiro ("1.234,5") ou com ponto decimal ("4.31")."""
    return [_valor(m.group()) for m in _NUMERO.finditer(texto)]


def _numeros_a_conferir(texto: str) -> list[float]:
    """Números do texto que são dados (e precisam existir na entrada).

    Ficam de fora inteiros pequenos sem unidade (numeração, "3 frases", meses) e anos.
    Um número com unidade ou valor em reais ("5 anos", "R$ 3", "10%") sempre é conferido.
    """
    numeros = []
    for m in _NUMERO.finditer(texto):
        valor = _valor(m.group())
        com_unidade = bool(_UNIDADE.match(texto, m.end())) or texto[max(0, m.start() - 3) : m.start()].strip().endswith("R$")
        inteiro = "," not in m.group() and "." not in m.group()
        if not com_unidade and inteiro and (valor <= 12 or 1900 <= valor <= 2100):
            continue
        numeros.append(valor)
    return numeros


def _numeros_permitidos(dados: dict, ficha: str) -> set[float]:
    valores = set(extrair_numeros(ficha))

    def visitar(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            valores.add(float(v))
        elif isinstance(v, str):
            valores.update(extrair_numeros(v))
        elif isinstance(v, dict):
            for x in v.values():
                visitar(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                visitar(x)

    visitar(dados)
    return valores


def _confere(numero: float, permitidos: set[float]) -> bool:
    for p in permitidos:
        # tolerância de arredondamento: 1% (ou 0,05); valores grandes podem aparecer sem casas decimais
        if abs(numero - p) <= max(0.011 * abs(p), 0.051) or (abs(p) >= 50 and round(p) == numero):
            return True
    return False


def verificar_numeros(texto: str, dados: dict, ficha: str = "") -> dict:
    """Confere se cada número do texto gerado existe nos dados de entrada (tolerância de arredondamento)."""
    permitidos = _numeros_permitidos(dados, ficha)
    citados = sorted(set(_numeros_a_conferir(texto)))
    nao_encontrados = [n for n in citados if not _confere(n, permitidos)]
    return {"numeros_citados": citados, "nao_encontrados": nao_encontrados, "confiavel": not nao_encontrados}


def _fmt(v: float, casas: int = 2) -> str:
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def dados_relatorio(estimativa: dict, mensal: Optional[dict]) -> str:
    """Os números que o modelo pode usar, já formatados em português."""
    linhas = [
        f"- Local: {estimativa['cidade']}",
        f"- Irradiação solar média usada no cálculo: {_fmt(estimativa['irradiacao_media'])} kWh/m² por dia (horas de sol pleno)",
        f"- Potência do sistema: {_fmt(estimativa['potencia_kwp'], 1)} kWp",
        f"- Taxa de desempenho (perdas): {_fmt(estimativa['desempenho'])}",
        f"- Tarifa de energia: R$ {_fmt(estimativa['tarifa'])} por kWh",
        f"- Geração estimada: {_fmt(estimativa['geracao_dia_kwh'])} kWh por dia, {_fmt(estimativa['geracao_mes_kwh'], 1)} kWh por mês, {_fmt(estimativa['geracao_ano_kwh'], 0)} kWh por ano",
        f"- Economia estimada: R$ {_fmt(estimativa['economia_mes_rs'])} por mês, R$ {_fmt(estimativa['economia_ano_rs'])} por ano",
        f"- Observação do cálculo: {estimativa['observacao']}",
    ]
    if mensal:
        melhor = max(mensal, key=mensal.get)
        pior = min(mensal, key=mensal.get)
        linhas.append(
            f"- Irradiação de longo prazo por mês (kWh/m²/dia): " + ", ".join(f"{m} {_fmt(v)}" for m, v in mensal.items())
        )
        linhas.append(f"- Melhor mês: {melhor} ({_fmt(mensal[melhor])}); pior mês: {pior} ({_fmt(mensal[pior])})")
    return "\n".join(linhas)


SISTEMA_RELATORIO = (
    "Você escreve relatórios curtos de viabilidade de energia solar para pessoas leigas, em português do Brasil. "
    "Use SOMENTE os números fornecidos nos dados, copiados exatamente como estão. Não faça contas, não estime "
    "valores novos (como preço do sistema ou tempo de retorno) e não invente informações."
)


def ficha_sem_geracao(ficha: str) -> str:
    """Tira da ficha a frase de geração (calculada para 5 kWp), que confundiria um relatório de outra potência."""
    return " ".join(f for f in re.split(r"(?<=\.)\s+(?=[A-ZÀ-Ú])", ficha) if not f.startswith("Um sistema de"))


def gerar_relatorio(servico: ServicoLLM, estimativa: dict, mensal: Optional[dict] = None, ficha: str = "") -> dict:
    dados = dados_relatorio(estimativa, mensal)
    ficha = ficha_sem_geracao(ficha)
    chave = _chave(servico, "relatorio", dados=dados, ficha=ficha)
    if (em_cache := _do_cache(chave)) is not None:
        return em_cache
    inicio = time.perf_counter()
    mensagens = [
        {"role": "system", "content": SISTEMA_RELATORIO},
        {
            "role": "user",
            "content": f"Dados do cálculo:\n{dados}\n\n"
            + (f"Ficha do local:\n{ficha}\n\n" if ficha else "")
            + "Escreva o relatório em Markdown com estas seções, cada uma com 2 ou 3 frases:\n"
            "## Potencial solar do local\n## Geração e economia esperadas\n## Variação ao longo do ano\n## Próximos passos\n"
            "Em próximos passos, recomende pedir orçamentos a instaladores e conferir as regras de geração "
            "distribuída da ANEEL e da distribuidora.",
        },
    ]
    geracao = servico.gerar(mensagens, max_novos_tokens=520)
    relatorio = _limpar_saida(geracao.texto)
    if geracao.cortada:
        relatorio = _ate_ultima_frase(relatorio)
    if _contar_palavras(relatorio) < 20:
        raise erros.SaidaInvalida("O modelo não produziu um relatório aproveitável.")
    verificacao = verificar_numeros(relatorio, {"estimativa": estimativa, "mensal": mensal or {}}, ficha)
    return _guardar(chave, {"relatorio": relatorio, "verificacao": verificacao, "uso": _uso([geracao], inicio)})
