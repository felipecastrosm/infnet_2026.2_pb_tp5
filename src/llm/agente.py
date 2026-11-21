"""Agente consultor do Solar Viável (TP5): decide quais análises rodar e recomenda uma ação.

O usuário informa a cidade, o consumo, a tarifa e, se quiser, um objetivo em texto livre
("quero saber em quanto tempo o investimento se paga", "me preocupa o inverno"). O
agente segue um ciclo **pensar, agir, observar**:

1. **Decidir** a próxima ação entre as ferramentas disponíveis. Quem decide é o LLM local,
   por *decodificação restrita*: em vez de pedir um texto livre (que um modelo pequeno pode
   quebrar), mede a probabilidade que o modelo dá a cada nome de ferramenta como resposta,
   dado o objetivo e o que já foi observado. A saída é sempre uma ação válida, com a
   confiança do modelo, que o painel mostra.
2. **Agir**: executar a ferramenta escolhida. As ferramentas são cálculos determinísticos
   sobre os dados coletados (`src/data_processing/viabilidade.py`): o modelo nunca faz contas.
3. **Observar**: o resultado vira uma frase curta, que entra no contexto da próxima decisão.

Regras de segurança (guardrails): só aparecem ferramentas cujas dependências já rodaram
(não dá para calcular o payback sem dimensionar o sistema); `finalizar` só aparece depois
das três análises obrigatórias; quando só há uma opção, ela é escolhida sem chamar o modelo
(economiza tempo); há um limite de passos.

No fim, o LLM escreve a recomendação em Markdown a partir dos fatos calculados, de alertas
gerados por regras (benchmarks de payback, consumo mínimo, Fio B) e das notícias do setor já
resumidas pelo modelo. Cada número do texto é conferido com os resultados das ferramentas
(`tarefas.verificar_numeros`).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

from src.data_processing import viabilidade as v
from src.llm import erros, tarefas
from src.llm.servico import ServicoLLM

FINALIZAR = "finalizar"
OBRIGATORIAS = ("potencial_solar", "dimensionar_sistema", "analise_financeira")
MAX_PASSOS = 9


@dataclass
class DadosAgente:
    """Dados coletados no projeto que as ferramentas usam."""

    sundata: pd.DataFrame
    clima: pd.DataFrame
    noticias_ia: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class Ferramenta:
    nome: str
    titulo: str
    descricao: str  # o que o modelo lê para decidir
    depende: tuple = ()
    executar: Callable = None
    resumir: Callable = None


_fmt = tarefas._fmt


def _anos(valor) -> str:
    return "não se paga em 25 anos" if valor is None else f"{_fmt(valor, 1)} anos"


# ---------------------------------------------------------------------------
# Ferramentas: execução (sobre os dados) e resumo (o que o modelo observa)
# ---------------------------------------------------------------------------
def _exec_potencial(d: DadosAgente, p: dict, r: dict) -> dict:
    return v.potencial_solar(d.sundata, d.clima, p["cidade"])


def _nivel(posicao: int, total: int) -> str:
    if posicao <= 3:
        return "entre os mais altos do Brasil"
    if posicao > total - 3:
        return "entre os mais baixos das capitais"
    return "intermediário entre as capitais"


def _res_potencial(x: dict) -> str:
    nasa = (
        f" No último ano, a NASA mediu {_fmt(x['nasa_ultimo_ano'])} kWh/m²/dia no plano horizontal "
        f"({_fmt(x['diferenca_nasa_pct'], 1)}% em relação à média histórica)."
        if x["nasa_ultimo_ano"] is not None and x["diferenca_nasa_pct"] is not None
        else ""
    )
    return (
        f"{x['cidade']} recebe em média {_fmt(x['irradiacao_media'])} kWh/m²/dia com os painéis inclinados a "
        f"{x['inclinacao_graus']}° (CRESESB), {x['posicao_ranking']}º lugar entre {x['total_capitais']} capitais "
        f"(potencial {_nivel(x['posicao_ranking'], x['total_capitais'])}). "
        f"Melhor mês: {x['melhor_mes']} ({_fmt(x['mensal'][x['melhor_mes']])}); pior mês: {x['pior_mes']} "
        f"({_fmt(x['mensal'][x['pior_mes']])}).{nasa}"
    )


def _exec_dimensionar(d: DadosAgente, p: dict, r: dict) -> dict:
    return v.dimensionar_sistema(d.sundata, p["cidade"], p["consumo_kwh_mes"], p["ligacao"], p["desempenho"], p.get("potencia_kwp"))


def _res_dimensionar(x: dict) -> str:
    return (
        f"Sistema de {_fmt(x['potencia_kwp'])} kWp ({x['modulos']} módulos de 550 W, cerca de {_fmt(x['area_m2'], 1)} m² de telhado), "
        f"potência {x['origem_potencia']}. Gera {_fmt(x['geracao_kwh_mes'], 1)} kWh por mês em média, "
        f"{_fmt(x['cobertura_consumo_pct'], 1)}% do consumo de {_fmt(x['consumo_kwh_mes'], 0)} kWh/mês "
        f"(a distribuidora cobra no mínimo {x['disponibilidade_kwh_mes']} kWh/mês na ligação {x['ligacao']})."
    )


def _exec_financeira(d: DadosAgente, p: dict, r: dict) -> dict:
    return v.analise_financeira(r["dimensionar_sistema"], p["tarifa"], p["custo_kwp"], p.get("reajuste_tarifa", 0.0))


def _res_financeira(x: dict) -> str:
    tir = f", taxa interna de retorno de {_fmt(x['tir_pct'], 1)}% ao ano" if x["tir_pct"] is not None else ""
    return (
        f"Investimento de R$ {_fmt(x['investimento_rs'])}. Economia de R$ {_fmt(x['economia_mes_rs'])} por mês no 1º ano, "
        f"já descontados o Fio B ({_fmt(x['desconto_fio_b_ano1_pct'], 1)}% a menos) e a manutenção. Retorno do "
        f"investimento (payback) em {_anos(x['payback_anos'])}{tir}. Em 25 anos, economia de "
        f"R$ {_fmt(x['economia_25_anos_rs'])} ({_fmt(x['retorno_sobre_investimento'], 1)} vezes o investimento). "
        f"Viabilidade {x['classificacao']}."
    )


def _exec_sazonalidade(d: DadosAgente, p: dict, r: dict) -> dict:
    return v.sazonalidade(d.sundata, p["cidade"], r["dimensionar_sistema"])


def _res_sazonalidade(x: dict) -> str:
    deficit = ", ".join(x["meses_com_deficit"]) or "nenhum"
    return (
        f"A geração varia {_fmt(x['variacao_pct'], 1)}% entre o melhor mês ({x['maior_geracao_mes']}, "
        f"{_fmt(x['maior_geracao_kwh'], 1)} kWh) e o pior ({x['menor_geracao_mes']}, {_fmt(x['menor_geracao_kwh'], 1)} kWh). "
        f"Meses em que a geração fica abaixo do consumo: {deficit}. Sobram {_fmt(x['creditos_kwh_ano'], 0)} kWh de "
        "créditos por ano nos meses de excedente, que abatem a conta dos meses mais fracos."
    )


def _exec_impacto(d: DadosAgente, p: dict, r: dict) -> dict:
    return v.impacto_ambiental(r["dimensionar_sistema"])


def _res_impacto(x: dict) -> str:
    return (
        f"Evita {_fmt(x['co2_evitado_kg_ano'], 0)} kg de CO2 por ano e {_fmt(x['co2_evitado_t_25_anos'], 1)} toneladas em 25 anos, "
        f"gerando {_fmt(x['energia_limpa_mwh_25_anos'], 1)} MWh de energia limpa. {_fmt(x['consumo_atendido_por_solar_pct'], 1)}% "
        "do consumo passa a ser atendido por energia solar. A matriz elétrica brasileira já é pouco poluente "
        f"({_fmt(x['fator_emissao_t_mwh'] * 1000, 1)} kg de CO2 por MWh), então o maior ganho social é o acesso a energia mais barata (ODS 7)."
    )


def _exec_comparar(d: DadosAgente, p: dict, r: dict) -> dict:
    return v.comparar_capitais(d.sundata, p["consumo_kwh_mes"], p["tarifa"], p["custo_kwp"], p["ligacao"], p["desempenho"], p["cidade"])


def _res_comparar(x: dict) -> str:
    pos, total = x["posicao_payback"], x["total_capitais"]
    nivel = "entre os mais rápidos" if pos <= 3 else "entre os mais lentos" if pos > total - 3 else "intermediário"
    return (
        f"Com o mesmo consumo e tarifa, {x['cidade']} fica em {pos}º lugar em tempo de retorno entre {total} capitais "
        f"(1º = retorno mais rápido; retorno {nivel}). Melhor: {x['melhor']}; pior: {x['pior']}; payback mediano das "
        f"capitais: {_anos(x['payback_mediano_anos'])}."
    )


def _exec_noticias(d: DadosAgente, p: dict, r: dict) -> dict:
    return v.noticias_setor(d.noticias_ia)


def _res_noticias(x: dict) -> str:
    if not x["disponivel"]:
        return "As notícias ainda não foram processadas pelo modelo."
    sent = ", ".join(f"{n} {s}" for s, n in x["sentimentos_relevantes"].items()) or "nenhuma"
    titulos = "; ".join(f"\"{n['titulo']}\" ({n['sentimento']})" for n in x["destaques"])
    return (
        f"Das {x['total']} notícias do setor, {x['relevantes']} tratam de tarifa, regras ou custos (sentimento: {sent}). "
        f"Destaques: {titulos}."
    )


FERRAMENTAS: dict[str, Ferramenta] = {
    f.nome: f
    for f in [
        Ferramenta("potencial_solar", "Potencial solar do local", "irradiação solar da cidade e posição entre as capitais",
                   (), _exec_potencial, _res_potencial),
        Ferramenta("dimensionar_sistema", "Dimensionamento do sistema", "potência, número de painéis e área de telhado para o consumo",
                   ("potencial_solar",), _exec_dimensionar, _res_dimensionar),
        Ferramenta("analise_financeira", "Análise financeira", "investimento, economia na conta de luz e tempo de retorno (payback)",
                   ("dimensionar_sistema",), _exec_financeira, _res_financeira),
        Ferramenta("sazonalidade", "Geração mês a mês", "variação da geração ao longo do ano, meses fracos de inverno e créditos",
                   ("dimensionar_sistema",), _exec_sazonalidade, _res_sazonalidade),
        Ferramenta("impacto_ambiental", "Impacto ambiental (ODS)", "CO2 evitado e energia limpa gerada, impacto ambiental e ODS",
                   ("dimensionar_sistema",), _exec_impacto, _res_impacto),
        Ferramenta("comparar_capitais", "Comparação com as capitais", "comparação do retorno com as outras capitais (benchmark)",
                   ("potencial_solar",), _exec_comparar, _res_comparar),
        Ferramenta("noticias_setor", "Notícias do setor", "notícias recentes sobre tarifas, regras da ANEEL e custos do setor solar",
                   ("potencial_solar",), _exec_noticias, _res_noticias),
    ]
}
DESCRICAO_FINALIZAR = "encerrar a análise e escrever a recomendação, quando já houver informação suficiente para o objetivo"


def ferramentas_publicas() -> list[dict]:
    return [
        {"nome": f.nome, "titulo": f.titulo, "descricao": f.descricao, "depende_de": list(f.depende), "obrigatoria": f.nome in OBRIGATORIAS}
        for f in FERRAMENTAS.values()
    ] + [{"nome": FINALIZAR, "titulo": "Finalizar", "descricao": DESCRICAO_FINALIZAR, "depende_de": list(OBRIGATORIAS), "obrigatoria": False}]


# ---------------------------------------------------------------------------
# Decisão (o modelo escolhe a próxima ação)
# ---------------------------------------------------------------------------
SISTEMA_DECISAO = (
    "Você é o agente consultor do Solar Viável, que avalia se vale a pena instalar energia solar. "
    "Você trabalha em passos: a cada passo escolhe UMA ação da lista para atender ao objetivo do usuário. "
    "Prefira ações que tenham relação direta com o objetivo e não repita análises já feitas. "
    "Responda somente com o nome da ação."
)


def descrever_perfil(p: dict) -> str:
    potencia = f", potência desejada {_fmt(p['potencia_kwp'])} kWp" if p.get("potencia_kwp") else ""
    return (
        f"{p['cidade']}, consumo de {_fmt(p['consumo_kwh_mes'], 0)} kWh por mês, tarifa de R$ {_fmt(p['tarifa'])} por kWh, "
        f"ligação {p['ligacao']}, custo de instalação de R$ {_fmt(p['custo_kwp'], 0)} por kWp{potencia}"
    )


def disponiveis(feitas: list[str]) -> list[str]:
    opcoes = [f.nome for f in FERRAMENTAS.values() if f.nome not in feitas and all(dep in feitas for dep in f.depende)]
    if all(o in feitas for o in OBRIGATORIAS):
        opcoes.append(FINALIZAR)
    return opcoes


def mensagens_decisao(objetivo: str, perfil: dict, observacoes: list[tuple[str, str]], opcoes: list[str]) -> list[dict]:
    feitas = "\n".join(f"- {nome}: {texto}" for nome, texto in observacoes) or "- nenhuma ainda"
    lista = "\n".join(
        f"- {o}: {FERRAMENTAS[o].descricao if o in FERRAMENTAS else DESCRICAO_FINALIZAR}" for o in opcoes
    )
    return [
        {"role": "system", "content": SISTEMA_DECISAO},
        {
            "role": "user",
            "content": f"Objetivo do usuário: {objetivo}\nPerfil: {descrever_perfil(perfil)}.\n\n"
            f"Análises já feitas:\n{feitas}\n\nAções disponíveis:\n{lista}\n\nPróxima ação:",
        },
    ]


# ---------------------------------------------------------------------------
# Alertas por regras (benchmarks) e síntese
# ---------------------------------------------------------------------------
def alertas(resultados: dict) -> list[str]:
    a = []
    dim, fin = resultados["dimensionar_sistema"], resultados["analise_financeira"]
    pot = resultados["potencial_solar"]
    if fin["payback_anos"] is None or fin["payback_anos"] > 8:
        a.append("O tempo de retorno passa de 8 anos, acima do que costuma ser considerado atrativo para residências.")
    if dim["consumo_kwh_mes"] <= dim["disponibilidade_kwh_mes"] + 70:
        a.append(
            f"O consumo é baixo: a distribuidora cobra no mínimo {dim['disponibilidade_kwh_mes']} kWh por mês mesmo com "
            "geração própria, o que reduz a economia possível."
        )
    if dim["cobertura_consumo_pct"] < 80:
        a.append("A potência escolhida cobre menos de 80% do consumo; a conta continua relevante.")
    if dim["cobertura_consumo_pct"] > 130:
        a.append("A potência escolhida gera bem mais do que o consumo; o excedente vira créditos que podem expirar em 5 anos.")
    if dim["area_m2"] > 40:
        a.append(f"O sistema ocupa cerca de {_fmt(dim['area_m2'], 1)} m²: confirme se o telhado tem essa área livre e voltada para o norte.")
    if pot["diferenca_nasa_pct"] is not None and abs(pot["diferenca_nasa_pct"]) >= 8:
        sentido = "mais" if pot["diferenca_nasa_pct"] > 0 else "menos"
        a.append(f"O último ano foi {sentido} ensolarado que a média histórica; a análise usa a média de longo prazo, mais segura.")
    a.append("A Lei 14.300/2022 cobra uma parte crescente do Fio B sobre a energia injetada na rede; a economia já considera isso.")
    return a


SISTEMA_SINTESE = (
    "Você é o consultor do Solar Viável e escreve recomendações sobre energia solar para pessoas leigas, em português do Brasil. "
    "Use SOMENTE os fatos fornecidos e copie os números exatamente como estão. Não faça contas, não estime valores novos "
    "e não invente informações. Seja direto e prático."
)


def mensagens_sintese(objetivo: str, perfil: dict, observacoes: list[tuple[str, str]], classificacao: str, avisos: list[str]) -> list[dict]:
    fatos = "\n".join(f"- {FERRAMENTAS[n].titulo}: {t}" for n, t in observacoes)
    avisos_txt = "\n".join(f"- {x}" for x in avisos)
    secoes = ["## Veredito", "## Por que", "## Ações recomendadas", "## Pontos de atenção"]
    if any(n == "impacto_ambiental" for n, _ in observacoes):
        secoes.append("## Impacto ESG e ODS")
    return [
        {"role": "system", "content": SISTEMA_SINTESE},
        {
            "role": "user",
            "content": f"Objetivo do usuário: {objetivo}\nPerfil: {descrever_perfil(perfil)}.\n\n"
            f"Fatos calculados pelas ferramentas:\n{fatos}\n\nClassificação da viabilidade: {classificacao}\n\n"
            f"Alertas:\n{avisos_txt}\n\n"
            "Escreva a recomendação em Markdown com estas seções:\n" + "\n".join(secoes) + "\n"
            f"No veredito, comece com a viabilidade {classificacao} e responda ao objetivo do usuário em até 2 frases. "
            "Em ações recomendadas, liste 3 ações numeradas e práticas (por exemplo, pedir orçamentos a instaladores, "
            "conferir o telhado, acompanhar as regras da distribuidora). Cada seção com no máximo 3 frases ou itens.",
        },
    ]


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
_RECOMENDACOES: "OrderedDict[str, dict]" = OrderedDict()
_MAX_RECOMENDACOES = 100


def obter_recomendacao(id_: str) -> Optional[dict]:
    return _RECOMENDACOES.get(id_)


def limpar() -> None:
    _RECOMENDACOES.clear()


def _guardar(resultado: dict) -> dict:
    _RECOMENDACOES[resultado["id"]] = resultado
    while len(_RECOMENDACOES) > _MAX_RECOMENDACOES:
        _RECOMENDACOES.popitem(last=False)
    return resultado


def executar(servico: ServicoLLM, dados: DadosAgente, perfil: dict, objetivo: str = "", max_passos: int = MAX_PASSOS) -> dict:
    if perfil["cidade"] not in v.capitais(dados.sundata):
        raise KeyError(perfil["cidade"])
    objetivo = tarefas.normalizar_espacos(objetivo) or "avaliação completa: vale a pena instalar energia solar?"
    servico.exigir_tamanho(objetivo, "O objetivo")
    chave = tarefas._chave(servico, "agente", perfil=perfil, objetivo=objetivo)
    if (em_cache := tarefas._do_cache(chave)) is not None:
        return _guardar({**em_cache, "id": uuid.uuid4().hex[:12], "criado_em": datetime.now().isoformat(timespec="seconds")})

    inicio = time.perf_counter()
    resultados: dict[str, dict] = {}
    observacoes: list[tuple[str, str]] = []
    passos, geracoes, decisoes_modelo = [], [], 0
    for n in range(1, max_passos + 1):
        opcoes = disponiveis(list(resultados))
        t0 = time.perf_counter()
        if len(opcoes) == 1:
            acao, decisao = opcoes[0], {"modo": "regra", "motivo": "única ação disponível neste passo", "probabilidades": {opcoes[0]: 1.0}}
        else:
            probs = servico.pontuar(mensagens_decisao(objetivo, perfil, observacoes, opcoes), {o: [o] for o in opcoes})
            acao = max(probs, key=probs.get)
            decisoes_modelo += 1
            decisao = {
                "modo": "modelo",
                "motivo": f"o modelo deu {probs[acao] * 100:.0f}% de probabilidade a esta ação".replace(".", ","),
                "probabilidades": {k: round(p, 4) for k, p in sorted(probs.items(), key=lambda kv: -kv[1])},
            }
        tempo_decisao = time.perf_counter() - t0
        if acao == FINALIZAR:
            passos.append({"n": n, "acao": FINALIZAR, "titulo": "Finalizar e recomendar", "decisao": decisao, "resumo": "", "tempo_s": round(tempo_decisao, 2)})
            break
        f = FERRAMENTAS[acao]
        t1 = time.perf_counter()
        resultados[acao] = f.executar(dados, perfil, resultados)
        resumo = f.resumir(resultados[acao])
        observacoes.append((acao, resumo))
        passos.append(
            {
                "n": n,
                "acao": acao,
                "titulo": f.titulo,
                "decisao": decisao,
                "resumo": resumo,
                "tempo_s": round(tempo_decisao + time.perf_counter() - t1, 2),
            }
        )
    else:
        # limite de passos: garante as obrigatórias antes de recomendar
        for acao in OBRIGATORIAS:
            if acao not in resultados:
                resultados[acao] = FERRAMENTAS[acao].executar(dados, perfil, resultados)
                observacoes.append((acao, FERRAMENTAS[acao].resumir(resultados[acao])))

    classificacao = resultados["analise_financeira"]["classificacao"]
    avisos = alertas(resultados)
    g = servico.gerar(mensagens_sintese(objetivo, perfil, observacoes, classificacao, avisos), max_novos_tokens=650)
    geracoes.append(g)
    texto = tarefas._limpar_saida(g.texto)
    texto = re.sub(r"^#\s+", "## ", texto, flags=re.M)  # títulos sempre no mesmo nível
    texto = re.sub(r"\n-{3,}\s*\n", "\n", texto).strip()  # separadores "---" entre seções
    if g.cortada:
        texto = tarefas._ate_ultima_frase(texto)
    if tarefas._contar_palavras(texto) < 20:
        raise erros.SaidaInvalida("O modelo não produziu uma recomendação aproveitável.")
    # o fluxo de caixa (anos 0 a 25) ficaria de fora: aceitaria qualquer número pequeno como "conferido"
    conferir = {n: {k: x for k, x in r.items() if k != "fluxo_caixa"} for n, r in resultados.items()}
    fatos_texto = "\n".join(t for _, t in observacoes) + "\n" + "\n".join(avisos) + "\n" + descrever_perfil(perfil)
    verificacao = tarefas.verificar_numeros(texto, {"resultados": conferir, "perfil": perfil}, fatos_texto)

    uso = tarefas._uso(geracoes, inicio)
    uso["decisoes_do_modelo"] = decisoes_modelo
    resultado = {
        "perfil": perfil,
        "objetivo": objetivo,
        "passos": passos,
        "resultados": json.loads(json.dumps(resultados, default=float)),
        "classificacao": classificacao,
        "alertas": avisos,
        "recomendacao": texto,
        "verificacao": verificacao,
        "premissas": v.PREMISSAS,
        "uso": uso,
    }
    salvo = tarefas._guardar(chave, resultado)
    return _guardar({**salvo, "id": uuid.uuid4().hex[:12], "criado_em": datetime.now().isoformat(timespec="seconds")})


def ficha_recomendacao(rec: dict) -> str:
    """A recomendação em texto, para o chat com memória usar como fonte."""
    fatos = " ".join(p["resumo"] for p in rec["passos"] if p["resumo"])
    return (
        f"Recomendação do agente consultor para {descrever_perfil(rec['perfil'])}. Objetivo: {rec['objetivo']}. "
        f"Viabilidade {rec['classificacao']}. {fatos} Alertas: {' '.join(rec['alertas'])}"
    )
