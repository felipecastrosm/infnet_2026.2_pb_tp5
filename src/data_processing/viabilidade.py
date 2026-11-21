"""Análise de viabilidade de um sistema fotovoltaico (cálculos determinísticos, sem LLM).

São as "ferramentas" do agente consultor (src/llm/agente.py) e a base do painel
(GET /painel). Todo número que o agente ou o painel mostram sai daqui, calculado a
partir dos dados coletados no projeto:

- CRESESB/SunData (Selenium): irradiação de longo prazo mês a mês, painel inclinado;
- NASA POWER (API REST): irradiação medida no último ano, para comparar com o ano típico;
- notícias da ABSOLAR e do Canal Solar (Beautiful Soup), resumidas e classificadas pelo
  LLM local em lote (src/llm/enriquecer_noticias.py).

O modelo de linguagem não faz contas: ele escolhe quais análises rodar e explica os
resultados (ver agente.py). As premissas ficam em `PREMISSAS`, com a fonte de cada uma,
e são devolvidas junto com os resultados para o usuário saber de onde vem cada número.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.data_processing import climate

# ---------------------------------------------------------------------------
# Premissas (todas editáveis pelo usuário, exceto as físicas)
# ---------------------------------------------------------------------------
CUSTO_KWP_PADRAO = 3000.0  # R$ por kWp instalado
FATOR_EMISSAO_T_MWH = 0.0385  # tCO2 por MWh do Sistema Interligado Nacional
DEGRADACAO_ANO = 0.005  # perda de eficiência dos módulos por ano
VIDA_UTIL_ANOS = 25
POTENCIA_MODULO_KWP = 0.55  # módulo típico de 550 W
AREA_MODULO_M2 = 2.6  # área de um módulo de 550 W (cerca de 2,3 x 1,1 m)
DISPONIBILIDADE_KWH = {"monofasica": 30, "bifasica": 50, "trifasica": 100}
ANO_INSTALACAO = 2027
MANUTENCAO_ANO = 0.01  # limpeza e manutenção: 1% do investimento por ano
PARCELA_FIO_B = 0.28  # parte da tarifa que remunera a rede de distribuição (TUSD Fio B)
SIMULTANEIDADE = 0.30  # parte da geração consumida na hora, que não passa pela rede
FIO_B_COBRADO = {2023: 0.15, 2024: 0.30, 2025: 0.45, 2026: 0.60, 2027: 0.75, 2028: 0.90}  # Lei 14.300/2022; 100% depois

PREMISSAS = {
    "custo_kwp": {
        "valor": CUSTO_KWP_PADRAO,
        "unidade": "R$/kWp",
        "fonte": "Greener, 1º semestre de 2025: sistema residencial de 4 kWp por cerca de R$ 11.360 "
        "(R$ 2,84/Wp), mais a alta de 7% no 1º semestre de 2026 (pv magazine). Arredondado para R$ 3.000/kWp.",
    },
    "fator_emissao": {
        "valor": FATOR_EMISSAO_T_MWH,
        "unidade": "tCO2/MWh",
        "fonte": "Fator médio anual de emissão do Sistema Interligado Nacional em 2023 (MCTI): 38,5 kg de CO2 por MWh.",
    },
    "degradacao": {"valor": DEGRADACAO_ANO, "unidade": "por ano", "fonte": "Garantia de desempenho típica de módulos (0,5% ao ano)."},
    "vida_util": {"valor": VIDA_UTIL_ANOS, "unidade": "anos", "fonte": "Garantia de desempenho típica de módulos fotovoltaicos."},
    "modulo": {"valor": POTENCIA_MODULO_KWP, "unidade": "kWp por módulo", "fonte": "Módulo comercial comum de 550 W, cerca de 2,6 m²."},
    "fio_b": {
        "valor": {"parcela_da_tarifa": PARCELA_FIO_B, "simultaneidade": SIMULTANEIDADE, "ano_instalacao": ANO_INSTALACAO},
        "unidade": "fração",
        "fonte": "Lei 14.300/2022: a energia injetada na rede paga uma parte crescente do Fio B (75% em 2027, 90% em 2028 "
        "e 100% depois). Fio B estimado em 28% da tarifa e 30% da geração consumida na hora (valores típicos residenciais).",
    },
    "manutencao": {"valor": MANUTENCAO_ANO, "unidade": "do investimento por ano", "fonte": "Estimativa usual de operação e manutenção (limpeza, inspeção)."},
    "custo_disponibilidade": {
        "valor": DISPONIBILIDADE_KWH,
        "unidade": "kWh/mês",
        "fonte": "ANEEL (REN 1.000/2021): consumo mínimo cobrado mesmo com geração própria, conforme o tipo de ligação.",
    },
}

FAIXAS_VIABILIDADE = [  # (payback máximo em anos, classificação)
    (3.5, "excelente"),
    (5.0, "boa"),
    (8.0, "moderada"),
    (math.inf, "baixa"),
]


def _ref(sundata: pd.DataFrame, cidade: str) -> pd.Series:
    linha = sundata[(sundata["cidade"] == cidade) & (sundata["plano"] == "Maior média anual")]
    if linha.empty:
        raise KeyError(cidade)
    return linha.iloc[0]


def capitais(sundata: pd.DataFrame) -> list[str]:
    return sorted(sundata.loc[sundata["plano"] == "Maior média anual", "cidade"].unique())


# ---------------------------------------------------------------------------
# Ferramenta 1: potencial solar do local
# ---------------------------------------------------------------------------
def potencial_solar(sundata: pd.DataFrame, clima: pd.DataFrame, cidade: str) -> dict:
    """Irradiação de longo prazo (CRESESB), último ano medido (NASA) e posição entre as capitais."""
    r = _ref(sundata, cidade)
    horizontal = sundata[(sundata["cidade"] == cidade) & (sundata["plano"] == "Plano Horizontal")]
    media_h = float(horizontal["media_anual"].iloc[0]) if len(horizontal) else None
    ranking = sundata[sundata["plano"] == "Maior média anual"].sort_values("media_anual", ascending=False)["cidade"].tolist()
    mensal = {m: float(r[m]) for m in climate.MESES}
    nasa = clima.loc[clima["cidade"] == cidade, "ALLSKY_SFC_SW_DWN"]
    nasa_media = float(nasa.mean()) if len(nasa) else None
    return {
        "cidade": cidade,
        "irradiacao_media": round(float(r["media_anual"]), 2),
        "inclinacao_graus": int(r["inclinacao_graus"]),
        "ganho_inclinacao_pct": round((float(r["media_anual"]) / media_h - 1) * 100, 1) if media_h else None,
        "irradiacao_horizontal": media_h,
        "mensal": mensal,
        "melhor_mes": max(mensal, key=mensal.get),
        "pior_mes": min(mensal, key=mensal.get),
        "posicao_ranking": ranking.index(cidade) + 1,
        "total_capitais": len(ranking),
        "nasa_ultimo_ano": round(nasa_media, 2) if nasa_media is not None else None,
        "nasa_dias": int(nasa.count()),
        "diferenca_nasa_pct": round((nasa_media / media_h - 1) * 100, 1) if nasa_media and media_h else None,
        "estacao": r["estacao"],
        "distancia_km": float(r["distancia_km"]),
    }


# ---------------------------------------------------------------------------
# Ferramenta 2: dimensionamento
# ---------------------------------------------------------------------------
def dimensionar_sistema(
    sundata: pd.DataFrame,
    cidade: str,
    consumo_kwh_mes: float,
    ligacao: str = "bifasica",
    desempenho: float = 0.80,
    potencia_kwp: Optional[float] = None,
    arredondar_modulos: bool = True,
) -> dict:
    """Potência para compensar o consumo (descontado o mínimo cobrado pela distribuidora).

    Com `potencia_kwp`, usa a potência informada em vez de calcular. Com
    `arredondar_modulos=False`, usa a potência exata (útil para comparar locais sem o
    "degrau" de um módulo a mais ou a menos).
    """
    r = _ref(sundata, cidade)
    disponibilidade = DISPONIBILIDADE_KWH[ligacao]
    compensavel_mes = max(consumo_kwh_mes - disponibilidade, 0.0)
    kwh_por_kwp_ano = float(climate.geracao_mensal([r[m] for m in climate.MESES], 1.0, desempenho, 0)["geracao_kwh"].sum())
    necessario = compensavel_mes * 12 / kwh_por_kwp_ano if kwh_por_kwp_ano else 0.0
    if potencia_kwp:
        modulos = max(1, round(potencia_kwp / POTENCIA_MODULO_KWP))
        kwp = float(potencia_kwp)
        origem = "informada pelo usuário"
    elif arredondar_modulos:
        modulos = max(1, math.ceil(necessario / POTENCIA_MODULO_KWP - 1e-9))
        kwp = round(modulos * POTENCIA_MODULO_KWP, 2)
        origem = "calculada para compensar o consumo"
    else:
        kwp = round(necessario, 2)
        modulos = max(1, math.ceil(necessario / POTENCIA_MODULO_KWP - 1e-9))
        origem = "exata para compensar o consumo"
    geracao_ano = kwh_por_kwp_ano * kwp
    return {
        "potencia_kwp": kwp,
        "potencia_necessaria_kwp": round(necessario, 2),
        "origem_potencia": origem,
        "modulos": modulos,
        "area_m2": round(modulos * AREA_MODULO_M2, 1),
        "consumo_kwh_mes": consumo_kwh_mes,
        "disponibilidade_kwh_mes": disponibilidade,
        "ligacao": ligacao,
        "compensavel_kwh_mes": round(compensavel_mes, 1),
        "geracao_kwh_ano": round(geracao_ano, 0),
        "geracao_kwh_mes": round(geracao_ano / 12, 1),
        "produtividade_kwh_kwp_ano": round(kwh_por_kwp_ano, 0),
        "cobertura_consumo_pct": round(min(geracao_ano / (consumo_kwh_mes * 12), 9.99) * 100, 1) if consumo_kwh_mes else 0.0,
        "desempenho": desempenho,
    }


# ---------------------------------------------------------------------------
# Ferramenta 3: análise financeira
# ---------------------------------------------------------------------------
def _tir(fluxos: list[float]) -> Optional[float]:
    """Taxa interna de retorno por bisseção (None se não houver troca de sinal)."""

    def vpl(taxa):
        return sum(f / (1 + taxa) ** i for i, f in enumerate(fluxos))

    baixo, alto = -0.99, 10.0
    if vpl(baixo) * vpl(alto) > 0:
        return None
    for _ in range(200):
        meio = (baixo + alto) / 2
        if vpl(baixo) * vpl(meio) <= 0:
            alto = meio
        else:
            baixo = meio
    return (baixo + alto) / 2


def classificar(payback_anos: Optional[float]) -> str:
    if payback_anos is None:
        return "baixa"
    return next(rotulo for limite, rotulo in FAIXAS_VIABILIDADE if payback_anos <= limite)


def valor_kwh_compensado(tarifa: float, ano_calendario: int) -> float:
    """Quanto vale, em média, 1 kWh gerado: o consumido na hora vale a tarifa cheia; o injetado paga parte do Fio B."""
    cobrado = FIO_B_COBRADO.get(ano_calendario, 1.0 if ano_calendario > 2028 else 0.0)
    return tarifa * (SIMULTANEIDADE + (1 - SIMULTANEIDADE) * (1 - PARCELA_FIO_B * cobrado))


def analise_financeira(
    dimensionamento: dict,
    tarifa: float,
    custo_kwp: float = CUSTO_KWP_PADRAO,
    reajuste_tarifa: float = 0.0,
) -> dict:
    """Investimento, economia, payback, TIR e fluxo de caixa em 25 anos.

    Considera a degradação dos módulos e a cobrança gradual do Fio B sobre a energia
    injetada na rede (Lei 14.300/2022), a partir de `ANO_INSTALACAO`.
    """
    investimento = dimensionamento["potencia_kwp"] * custo_kwp
    compensavel_ano = dimensionamento["compensavel_kwh_mes"] * 12
    fluxo, acumulado, payback = [-investimento], -investimento, None
    serie = [{"ano": 0, "economia_rs": 0.0, "acumulado_rs": round(-investimento, 2)}]
    economia_total = 0.0
    for ano in range(1, VIDA_UTIL_ANOS + 1):
        geracao = dimensionamento["geracao_kwh_ano"] * (1 - DEGRADACAO_ANO) ** (ano - 1)
        tarifa_ano = tarifa * (1 + reajuste_tarifa) ** (ano - 1)
        economia = min(geracao, compensavel_ano) * valor_kwh_compensado(tarifa_ano, ANO_INSTALACAO + ano - 1) - investimento * MANUTENCAO_ANO
        anterior, acumulado = acumulado, acumulado + economia
        if payback is None and acumulado >= 0 and economia > 0:
            payback = ano - 1 + (-anterior / economia)
        economia_total += economia
        fluxo.append(economia)
        serie.append({"ano": ano, "economia_rs": round(economia, 2), "acumulado_rs": round(acumulado, 2)})
    economia_ano1 = fluxo[1]
    tir = _tir(fluxo)
    return {
        "investimento_rs": round(investimento, 2),
        "custo_kwp": custo_kwp,
        "tarifa": tarifa,
        "reajuste_tarifa": reajuste_tarifa,
        "economia_ano1_rs": round(economia_ano1, 2),
        "economia_mes_rs": round(economia_ano1 / 12, 2),
        "desconto_fio_b_ano1_pct": round((1 - valor_kwh_compensado(1.0, ANO_INSTALACAO)) * 100, 1),
        "payback_anos": round(payback, 1) if payback is not None else None,
        "economia_25_anos_rs": round(economia_total, 2),
        "lucro_25_anos_rs": round(economia_total - investimento, 2),
        "retorno_sobre_investimento": round(economia_total / investimento, 1) if investimento else None,
        "tir_pct": round(tir * 100, 1) if tir is not None else None,
        "classificacao": classificar(payback),
        "fluxo_caixa": serie,
    }


# ---------------------------------------------------------------------------
# Ferramenta 4: sazonalidade (geração x consumo mês a mês)
# ---------------------------------------------------------------------------
def sazonalidade(sundata: pd.DataFrame, cidade: str, dimensionamento: dict) -> dict:
    r = _ref(sundata, cidade)
    mensal = climate.geracao_mensal([r[m] for m in climate.MESES], dimensionamento["potencia_kwp"], dimensionamento["desempenho"], 0)
    consumo = dimensionamento["consumo_kwh_mes"]
    mensal["consumo_kwh"] = consumo
    mensal["saldo_kwh"] = (mensal["geracao_kwh"] - consumo).round(1)
    excedente = mensal.loc[mensal["saldo_kwh"] > 0, "mes"].tolist()
    deficit = mensal.loc[mensal["saldo_kwh"] < 0, "mes"].tolist()
    return {
        "meses": mensal[["mes", "geracao_kwh", "consumo_kwh", "saldo_kwh"]].to_dict(orient="records"),
        "meses_com_excedente": excedente,
        "meses_com_deficit": deficit,
        "maior_geracao_mes": mensal.loc[mensal["geracao_kwh"].idxmax(), "mes"],
        "maior_geracao_kwh": float(mensal["geracao_kwh"].max()),
        "menor_geracao_mes": mensal.loc[mensal["geracao_kwh"].idxmin(), "mes"],
        "menor_geracao_kwh": float(mensal["geracao_kwh"].min()),
        "variacao_pct": round(float(mensal["geracao_kwh"].max() / mensal["geracao_kwh"].min() - 1) * 100, 1),
        "creditos_kwh_ano": round(float(mensal.loc[mensal["saldo_kwh"] > 0, "saldo_kwh"].sum()), 0),
    }


# ---------------------------------------------------------------------------
# Ferramenta 5: impacto ambiental (ODS 7 e 13)
# ---------------------------------------------------------------------------
def impacto_ambiental(dimensionamento: dict, fator_emissao: float = FATOR_EMISSAO_T_MWH) -> dict:
    geracao_ano = dimensionamento["geracao_kwh_ano"]
    total_25 = sum(geracao_ano * (1 - DEGRADACAO_ANO) ** i for i in range(VIDA_UTIL_ANOS))
    renovavel = min(geracao_ano, dimensionamento["consumo_kwh_mes"] * 12)
    return {
        "co2_evitado_kg_ano": round(geracao_ano / 1000 * fator_emissao * 1000, 0),
        "co2_evitado_t_25_anos": round(total_25 / 1000 * fator_emissao, 1),
        "energia_limpa_mwh_25_anos": round(total_25 / 1000, 1),
        "consumo_atendido_por_solar_pct": round(renovavel / (dimensionamento["consumo_kwh_mes"] * 12) * 100, 1)
        if dimensionamento["consumo_kwh_mes"]
        else 0.0,
        "fator_emissao_t_mwh": fator_emissao,
        "ods": [
            "ODS 7.1: acesso a energia a preço acessível (economia na conta de luz)",
            "ODS 7.2: aumento da participação de energia renovável no consumo",
            "ODS 13.2: redução de emissões de gases de efeito estufa",
        ],
    }


# ---------------------------------------------------------------------------
# Ferramenta 6: benchmark entre as capitais
# ---------------------------------------------------------------------------
def comparar_capitais(
    sundata: pd.DataFrame,
    consumo_kwh_mes: float,
    tarifa: float,
    custo_kwp: float = CUSTO_KWP_PADRAO,
    ligacao: str = "bifasica",
    desempenho: float = 0.80,
    cidade: Optional[str] = None,
) -> dict:
    """O mesmo perfil de consumo nas 10 capitais: potência, payback e CO2 evitado.

    Usa a potência exata (sem arredondar para módulos inteiros), para que a diferença entre
    as capitais venha só do sol de cada uma.
    """
    linhas = []
    for c in capitais(sundata):
        dim = dimensionar_sistema(sundata, c, consumo_kwh_mes, ligacao, desempenho, arredondar_modulos=False)
        fin = analise_financeira(dim, tarifa, custo_kwp)
        amb = impacto_ambiental(dim)
        pot = _ref(sundata, c)
        linhas.append(
            {
                "cidade": c,
                "irradiacao_media": round(float(pot["media_anual"]), 2),
                "potencia_kwp": dim["potencia_kwp"],
                "investimento_rs": fin["investimento_rs"],
                "payback_anos": fin["payback_anos"],
                "economia_25_anos_rs": fin["economia_25_anos_rs"],
                "co2_evitado_kg_ano": amb["co2_evitado_kg_ano"],
                "classificacao": fin["classificacao"],
            }
        )
    tabela = sorted(linhas, key=lambda x: (x["payback_anos"] is None, x["payback_anos"] or 0))
    resultado = {
        "capitais": tabela,
        "melhor": tabela[0]["cidade"],
        "pior": tabela[-1]["cidade"],
        "payback_mediano_anos": round(float(pd.Series([t["payback_anos"] for t in tabela if t["payback_anos"]]).median()), 1),
    }
    if cidade:
        posicao = next(i for i, t in enumerate(tabela, start=1) if t["cidade"] == cidade)
        resultado.update({"cidade": cidade, "posicao_payback": posicao, "total_capitais": len(tabela)})
    return resultado


# ---------------------------------------------------------------------------
# Ferramenta 7: contexto do setor (notícias resumidas e classificadas pelo LLM)
# ---------------------------------------------------------------------------
TEMAS_DECISAO = ["tarifa", "conta de luz", "geração distribuída", "compensação", "fio b", "aneel", "preço", "custo", "financiamento", "consumidor"]


def noticias_setor(noticias_ia: pd.DataFrame, temas: Optional[list[str]] = None, n: int = 3) -> dict:
    """Notícias ligadas à decisão de instalar (tarifa, regras, custos), com resumo e sentimento do LLM."""
    if noticias_ia.empty:
        return {"disponivel": False, "total": 0, "sentimentos": {}, "destaques": []}
    temas = temas or TEMAS_DECISAO
    conteudo = (noticias_ia["titulo"] + " " + noticias_ia["resumo_ia"].fillna("")).str.lower()
    relevancia = conteudo.map(lambda t: sum(t.count(tema) for tema in temas))
    df = noticias_ia.assign(relevancia=relevancia).sort_values(["relevancia", "data"], ascending=False)
    destaques = df[df["relevancia"] > 0].head(n)
    return {
        "disponivel": True,
        "total": len(noticias_ia),
        "sentimentos": noticias_ia["sentimento"].value_counts().to_dict(),
        "relevantes": int((relevancia > 0).sum()),
        "sentimentos_relevantes": df.loc[df["relevancia"] > 0, "sentimento"].value_counts().to_dict(),
        "destaques": [
            {
                "titulo": r.titulo,
                "fonte": r.fonte,
                "data": str(r.data)[:10] if pd.notna(r.data) else "",
                "url": r.url,
                "sentimento": r.sentimento,
                "resumo_ia": r.resumo_ia,
            }
            for r in destaques.itertuples()
        ],
    }
