"""Painel do Solar Viável (TP5): o ciclo completo de ciência de dados em uma tela.

- **Consultor com IA**: o usuário informa cidade, consumo e tarifa; o agente (LLM local)
  escolhe as análises, a API calcula sobre os dados coletados e o modelo escreve a
  recomendação. Logo abaixo, um chat com memória continua a conversa sobre o resultado.
- **Panorama das capitais**: benchmark de retorno e irradiação para um perfil de consumo.
- **Setor solar com IA**: as 54 notícias coletadas, resumidas e classificadas pelo LLM.
- **Como funciona**: da coleta à apresentação, com os números de cada etapa.

Dados: `GET /painel`, `GET /noticias/ia`, `GET /irradiacao-longo-prazo`,
`POST /agente/recomendar` e `/conversas` da API.
"""

import re

import altair as alt
import pandas as pd
import streamlit as st

from src import api_client, ui
from src.data_processing import climate

CORES_CLASSE = {"excelente": "#15803d", "boa": "#0d9488", "moderada": "#d97706", "baixa": "#dc2626"}
CORES_SENTIMENTO = {"positivo": "#16a34a", "neutro": "#94a3b8", "negativo": "#dc2626"}
LIGACOES = {"monofasica": "Monofásica (mín. 30 kWh)", "bifasica": "Bifásica (mín. 50 kWh)", "trifasica": "Trifásica (mín. 100 kWh)"}
OBJETIVOS = [
    "",
    "Em quanto tempo o investimento se paga?",
    "Me preocupa o inverno: a geração cai muito?",
    "Qual o impacto ambiental da instalação?",
    "Minha cidade é boa para solar em comparação com outras capitais?",
    "O que as notícias dizem sobre tarifas e regras para quem instala?",
]
SUGESTOES_CHAT = [
    "Por que a viabilidade ficou assim?",
    "Quais meses geram menos energia?",
    "E em Fortaleza, seria melhor?",
]


@st.cache_data(ttl=300, show_spinner=False)
def dados_painel(consumo: float, tarifa: float) -> dict:
    return api_client.get("/painel", consumo_kwh_mes=consumo, tarifa=tarifa)


@st.cache_data(ttl=300, show_spinner=False)
def noticias_ia() -> pd.DataFrame:
    return pd.DataFrame(api_client.get("/noticias/ia", limite=500))


def moeda(v: float, casas: int = 0) -> str:
    """Valor em reais para st.metric (que não interpreta Markdown, então o cifrão vai sem escape)."""
    return f"R$ {ui.brl(v, casas)}"


def titulos_menores(markdown: str) -> str:
    """Os títulos "## Seção" da recomendação viram "#### Seção" (cabem melhor no cartão)."""
    return re.sub(r"^## ", "#### ", markdown, flags=re.M)


# ---------------------------------------------------------------------------
# Cabeçalho e ciclo de ciência de dados
# ---------------------------------------------------------------------------
st.title("Solar Viável")
st.markdown(
    "**Vale a pena instalar energia solar?** Informe sua cidade e seu consumo: um agente de IA que roda "
    "nesta máquina analisa dados reais de irradiação (NASA e CRESESB), calcula investimento, retorno e CO₂ "
    "evitado e explica o resultado em linguagem simples. Alinhado ao **ODS 7** e ao **ODS 13**."
)
status = ui.painel_llm(compacto=True)
llm_pronto = bool(status and status["pronto"])

try:
    painel = dados_painel(300.0, 0.95)
except api_client.ErroAPI as exc:
    ui.erro_api(exc)

etapas = {e["etapa"]: e for e in painel["ciclo"]}
coleta, proc, ia = etapas["Coleta"]["indicadores"], etapas["Processamento"]["indicadores"], etapas["IA"]["indicadores"]
CARTOES = [
    ("1 · Coleta", ui.brl(coleta["registros_nasa"] + coleta["series_cresesb"] + coleta["noticias"]), "registros",
     f"{coleta['fontes']} fontes: NASA (API), CRESESB (Selenium), ABSOLAR e Canal Solar (Beautiful Soup)"),
    ("2 · Processamento", str(proc["documentos_corpus"]), "documentos",
     f"corpus com {ui.brl(proc['palavras_corpus'])} palavras; {ui.brl(proc['ausentes_irradiacao_pct'], 1)}% de ausentes tratados"),
    ("3 · Análise", str(etapas["Análise"]["indicadores"]["ferramentas_do_agente"]), "ferramentas",
     "dimensionamento, payback com Fio B, sazonalidade, CO₂ e benchmark"),
    ("4 · IA local", str(ia["noticias_classificadas"]), "notícias lidas",
     f"{ia['modelo']}: notícias, agente consultor e chat com memória"),
    ("5 · Apresentação", str(etapas["Apresentação"]["indicadores"]["rotas_api"]), "rotas na API",
     "API FastAPI consumida por este painel"),
]
for col, (etapa, valor, unidade, detalhe) in zip(st.columns(5), CARTOES):
    with col.container(border=True, height="stretch"):
        st.caption(f"**{etapa.upper()}**")
        st.markdown(f"<span style='font-size:1.7rem;font-weight:700'>{valor}</span> {unidade}", unsafe_allow_html=True)
        st.caption(detalhe)

aba_consultor, aba_capitais, aba_setor, aba_ciclo = st.tabs(
    [":material/psychology: Consultor com IA", ":material/map: Panorama das capitais",
     ":material/newspaper: Setor solar com IA", ":material/account_tree: Como funciona"]
)


# ---------------------------------------------------------------------------
# Consultor com IA (agente + chat com memória)
# ---------------------------------------------------------------------------
def grafico_fluxo(fin: dict) -> alt.Chart:
    df = pd.DataFrame(fin["fluxo_caixa"])
    df["situação"] = df["acumulado_rs"].map(lambda v: "recuperando o investimento" if v < 0 else "lucro acumulado")
    barras = alt.Chart(df).mark_bar().encode(
        x=alt.X("ano:O", title="ano"),
        y=alt.Y("acumulado_rs:Q", title="saldo acumulado (R$)"),
        color=alt.Color("situação:N", scale=alt.Scale(domain=["recuperando o investimento", "lucro acumulado"], range=["#f87171", "#22c55e"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("ano:O"), alt.Tooltip("acumulado_rs:Q", title="saldo (R$)", format=",.0f"),
                 alt.Tooltip("economia_rs:Q", title="economia no ano (R$)", format=",.0f")],
    )
    return (barras + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#78716c").encode(y="y:Q")).properties(height=280)


def grafico_mensal(rec: dict) -> alt.Chart:
    pot, dim = rec["resultados"]["potencial_solar"], rec["resultados"]["dimensionar_sistema"]
    mensal = climate.geracao_mensal([pot["mensal"][m] for m in climate.MESES], dim["potencia_kwp"], dim["desempenho"], 0)
    mensal["mês"] = pd.Categorical(mensal["mes"], categories=climate.MESES, ordered=True)
    base = alt.Chart(mensal).encode(x=alt.X("mês:O", sort=climate.MESES, title=None))
    barras = base.mark_bar(color="#f59e0b").encode(
        y=alt.Y("geracao_kwh:Q", title="kWh no mês"), tooltip=[alt.Tooltip("mes:N", title="mês"), alt.Tooltip("geracao_kwh:Q", title="geração (kWh)")]
    )
    consumo = alt.Chart(pd.DataFrame({"y": [dim["consumo_kwh_mes"]]})).mark_rule(color="#2563eb", strokeDash=[6, 4], size=2).encode(y="y:Q")
    return (barras + consumo).properties(height=280)


def grafico_benchmark(bench: dict, destaque: str, campo: str = "payback_anos", titulo: str = "payback (anos)") -> alt.Chart:
    df = pd.DataFrame(bench["capitais"] if isinstance(bench, dict) else bench)
    df["destaque"] = df["cidade"] == destaque
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("cidade:N", sort=alt.EncodingSortField(campo, order="ascending"), title=None),
            x=alt.X(f"{campo}:Q", title=titulo),
            color=alt.condition("datum.destaque", alt.value("#f59e0b"), alt.value("#cbd5e1")),
            tooltip=["cidade", alt.Tooltip(f"{campo}:Q", title=titulo), alt.Tooltip("investimento_rs:Q", title="investimento (R$)", format=",.0f"),
                     "classificacao"],
        )
        .properties(height=300)
    )


def mostrar_passos(rec: dict) -> None:
    st.markdown("**Como o agente decidiu**")
    st.caption("A cada passo, o modelo local escolhe a próxima análise pela probabilidade que dá a cada opção. Os números vêm das ferramentas.")
    for p in rec["passos"]:
        if p["decisao"]["modo"] == "regra":
            quem = "regra (única opção)"
        else:
            quem = f"modelo, {p['decisao']['probabilidades'][p['acao']]:.0%}"
        with st.expander(f"{p['n']}. {p['titulo']} · {quem}", icon=":material/task_alt:" if p["acao"] != "finalizar" else ":material/flag:"):
            if p["resumo"]:
                st.markdown(ui.sem_latex(p["resumo"]))
            if p["decisao"]["modo"] == "modelo":
                probs = pd.DataFrame(list(p["decisao"]["probabilidades"].items()), columns=["ação", "probabilidade"])
                st.dataframe(probs, hide_index=True, width="stretch",
                             column_config={"probabilidade": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
            st.caption(f"{p['tempo_s']:.2f} s")


def mostrar_recomendacao(rec: dict) -> None:
    res = rec["resultados"]
    dim, fin = res["dimensionar_sistema"], res["analise_financeira"]
    classe = rec["classificacao"]
    cidade = rec["perfil"]["cidade"]
    st.markdown(
        f"<span style='background:{CORES_CLASSE[classe]};color:white;padding:4px 12px;border-radius:12px;font-weight:600'>"
        f"Viabilidade {classe}</span> &nbsp; {cidade} · {ui.brl(rec['perfil']['consumo_kwh_mes'])} kWh/mês · "
        f"R\\$ {ui.brl(rec['perfil']['tarifa'], 2)}/kWh",
        unsafe_allow_html=True,
    )
    if "impacto_ambiental" in res:
        ultimo = ("CO₂ evitado", f"{ui.brl(res['impacto_ambiental']['co2_evitado_t_25_anos'], 1)} t", "em 25 anos")
    else:
        ultimo = ("Economia em 25 anos", moeda(fin["economia_25_anos_rs"]), f"{ui.brl(fin['retorno_sobre_investimento'], 1)} vezes o investimento")
    indicadores = [
        ("Sistema", f"{ui.brl(dim['potencia_kwp'], 2)} kWp", f"{dim['modulos']} painéis, cerca de {ui.brl(dim['area_m2'], 0)} m²"),
        ("Investimento", moeda(fin["investimento_rs"]), f"R$ {ui.brl(fin['custo_kwp'])} por kWp".replace("$", "\\$")),
        ("Economia por mês", moeda(fin["economia_mes_rs"]), f"{ui.brl(dim['cobertura_consumo_pct'], 0)}% do consumo coberto"),
        ("Retorno (payback)", f"{ui.brl(fin['payback_anos'], 1)} anos" if fin["payback_anos"] else "> 25 anos",
         f"TIR de {ui.brl(fin['tir_pct'], 1)}% ao ano" if fin["tir_pct"] is not None else "não se paga"),
        ultimo,
    ]
    for col, (rotulo, valor, detalhe) in zip(st.columns(5), indicadores):
        col.metric(rotulo, valor, border=True)
        col.caption(detalhe)

    esq, dir_ = st.columns([3, 2], gap="large")
    with esq:
        with st.container(border=True):
            st.markdown(ui.sem_latex(titulos_menores(rec["recomendacao"])))
        v = rec["verificacao"]
        if v["confiavel"]:
            st.success(f"Os {len(v['numeros_citados'])} números do texto conferem com os cálculos das ferramentas.", icon=":material/verified:")
        else:
            fora = ", ".join(ui.brl(n, 2) for n in v["nao_encontrados"])
            st.warning(f"{len(v['nao_encontrados'])} número(s) do texto não aparecem nos cálculos ({fora}). Confie nos indicadores acima.",
                       icon=":material/report:")
        st.caption(
            f"{rec['modelo']['nome']} ({rec['modelo']['dispositivo']}) · {rec['uso']['tempo_s']:.1f} s · "
            f"{rec['uso']['tokens_saida']} tokens gerados" + (" · do cache" if rec["uso"]["em_cache"] else "")
        )
    with dir_:
        mostrar_passos(rec)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Saldo acumulado em 25 anos**")
        st.altair_chart(grafico_fluxo(fin), use_container_width=True)
        st.caption("Investimento no ano 0; economia anual já descontados o Fio B (Lei 14.300), a manutenção e a degradação dos painéis.")
    with g2:
        st.markdown("**Geração mês a mês x consumo**")
        st.altair_chart(grafico_mensal(rec), use_container_width=True)
        st.caption("Barras: geração no ano típico (CRESESB). Linha tracejada: seu consumo. Excedentes viram créditos para os meses fracos.")

    if "comparar_capitais" in res:
        st.markdown(f"**{cidade} entre as capitais** (mesmo consumo e tarifa)")
        st.altair_chart(grafico_benchmark(res["comparar_capitais"], cidade), use_container_width=True)
        st.caption("Na comparação, cada capital usa a potência exata para o consumo (sem arredondar para painéis inteiros), "
                   "então a diferença vem só do sol de cada lugar. Por isso o payback da sua cidade aqui pode ser um pouco menor.")
    if "noticias_setor" in res and res["noticias_setor"]["destaques"]:
        st.markdown("**Notícias do setor que afetam a decisão** (resumidas pela IA)")
        for n in res["noticias_setor"]["destaques"]:
            cor = {"positivo": "green", "neutro": "gray", "negativo": "red"}[n["sentimento"]]
            st.markdown(ui.sem_latex(f":{cor}-badge[{n['sentimento']}] [{n['titulo']}]({n['url']}) · {n['fonte']}, {n['data']}"))
            st.caption(ui.sem_latex(n["resumo_ia"]))
    with st.expander("Alertas e premissas do cálculo", icon=":material/info:"):
        for a in rec["alertas"]:
            st.markdown(f"- {ui.sem_latex(a)}")
        st.dataframe(
            pd.DataFrame([{"premissa": k.replace("_", " "), "valor": str(p["valor"]), "unidade": p["unidade"], "fonte": p["fonte"]}
                          for k, p in rec["premissas"].items()]),
            hide_index=True, width="stretch",
        )


def chat_recomendacao(rec: dict) -> None:
    st.subheader("Converse sobre o resultado")
    st.caption(
        "O assistente lembra da recomendação e das perguntas anteriores: dá para perguntar \"e em Fortaleza?\" "
        "ou \"por que isso?\". Cada resposta cita as fontes."
    )
    conv_id = st.session_state.get("conversa_painel")
    mensagens = st.session_state.setdefault("chat_painel", [])
    for msg in mensagens:
        with st.chat_message(msg["papel"]):
            if msg.get("erro"):
                ui.aviso_llm(msg["erro"])
                continue
            st.markdown(ui.sem_latex(msg["texto"]))
            if msg.get("reescrita"):
                st.caption(f":material/psychology_alt: Entendi como: *{ui.sem_latex(msg['reescrita'])}*")
            if msg.get("fontes"):
                citadas = [f for f in msg["fontes"] if f["citada"]]
                if citadas:
                    st.caption("Fontes: " + "; ".join(f"[{f['n']}] {f['titulo']}" for f in citadas))
    pendente = None
    if not mensagens:
        cols = st.columns(len(SUGESTOES_CHAT))
        for i, s in enumerate(SUGESTOES_CHAT):
            if cols[i].button(s, key=f"sug_painel_{i}", width="stretch", disabled=not llm_pronto):
                pendente = s
    pergunta = st.chat_input("Pergunte sobre a recomendação", key="chat_painel_input", disabled=not llm_pronto) or pendente
    if pergunta and conv_id:
        mensagens.append({"papel": "user", "texto": pergunta})
        try:
            with st.spinner("Consultando a memória da conversa e os dados..."):
                r = api_client.post(f"/conversas/{conv_id}/mensagens", {"pergunta": pergunta})
        except api_client.ErroAPI as exc:
            mensagens.append({"papel": "assistant", "erro": exc})
        else:
            mensagens.append({
                "papel": "assistant",
                "texto": r["resposta"],
                "fontes": r["fontes"],
                "reescrita": r["pergunta_reescrita"] if r["pergunta_reescrita"] != r["pergunta"] else None,
            })
            st.session_state.memoria_painel = r["memoria"]
        st.rerun()
    mem = st.session_state.get("memoria_painel")
    if mem:
        with st.expander("O que o assistente lembra desta conversa", icon=":material/memory:"):
            st.json(mem["fatos"], expanded=True)
            st.markdown(f"**Resumo das trocas antigas:** {ui.sem_latex(mem['resumo']) or '(ainda não precisou resumir)'}")
            st.caption(
                f"{mem['total_turnos']} troca(s) · {mem['turnos_na_janela']} enviada(s) na íntegra ao modelo "
                f"(janela de {mem['tamanho_janela']}) · {mem['turnos_resumidos']} condensada(s) no resumo"
            )


with aba_consultor:
    capitais = sorted(c["cidade"] for c in painel["capitais"])
    with st.form("form_agente", border=True):
        f1, f2, f3 = st.columns([2, 1, 1])
        cidade = f1.selectbox("Cidade", capitais, index=capitais.index("Recife (PE)") if "Recife (PE)" in capitais else 0)
        consumo = f2.number_input("Consumo médio (kWh/mês)", 50, 20000, 300, step=10, help="Está na sua conta de luz, no histórico de consumo.")
        tarifa = f3.number_input("Tarifa (R$/kWh)", 0.30, 3.00, 0.95, step=0.05, help="Valor total da conta dividido pelos kWh consumidos.")
        objetivo = st.selectbox(
            "O que você quer saber? (opcional, orienta o agente)", OBJETIVOS, accept_new_options=True,
            format_func=lambda x: x or "Avaliação completa", help="Escolha uma pergunta ou escreva a sua.",
        )
        with st.expander("Mais opções"):
            o1, o2, o3, o4 = st.columns(4)
            ligacao = o1.selectbox("Ligação", list(LIGACOES), index=1, format_func=LIGACOES.get)
            custo_kwp = o2.number_input("Custo (R$/kWp)", 1000, 15000, 3000, step=100, help="Média de mercado em 2026: cerca de R$ 3.000 por kWp.")
            potencia = o3.number_input("Potência (kWp)", 0.0, 1000.0, 0.0, step=0.5, help="0 = o agente dimensiona pelo consumo.")
            reajuste = o4.number_input("Reajuste da tarifa (% a.a.)", 0.0, 20.0, 0.0, step=0.5)
        enviar = st.form_submit_button("Analisar com o agente", type="primary", icon=":material/auto_awesome:", disabled=not llm_pronto)
    if enviar:
        pedido = {
            "cidade": cidade, "consumo_kwh_mes": consumo, "tarifa": tarifa, "ligacao": ligacao, "custo_kwp": custo_kwp,
            "potencia_kwp": potencia or None, "reajuste_tarifa": reajuste / 100, "objetivo": objetivo or "",
        }
        try:
            with st.spinner("O agente está escolhendo as análises e escrevendo a recomendação (10 a 20 segundos)..."):
                rec = api_client.post("/agente/recomendar", pedido)
                conversa = api_client.post("/conversas", {"recomendacao_id": rec["id"]})
        except api_client.ErroAPI as exc:
            ui.aviso_llm(exc)
        else:
            st.session_state.recomendacao = rec
            st.session_state.conversa_painel = conversa["id"]
            st.session_state.chat_painel = []
            st.session_state.memoria_painel = conversa["memoria"]
    rec = st.session_state.get("recomendacao")
    if rec:
        mostrar_recomendacao(rec)
        st.divider()
        chat_recomendacao(rec)
    else:
        st.info(
            "Preencha cidade, consumo e tarifa e clique em **Analisar com o agente**. O resultado traz o tamanho do "
            "sistema, o investimento, o tempo de retorno, o CO₂ evitado e uma recomendação explicada, além de um chat "
            "para tirar dúvidas sobre ela.",
            icon=":material/lightbulb:",
        )


# ---------------------------------------------------------------------------
# Panorama das capitais
# ---------------------------------------------------------------------------
with aba_capitais:
    st.markdown("**Onde a energia solar se paga mais rápido?** O mesmo perfil de consumo simulado nas 10 capitais.")
    p1, p2 = st.columns(2)
    consumo_b = p1.slider("Consumo (kWh/mês)", 100, 1500, 300, step=50, key="consumo_bench")
    tarifa_b = p2.slider("Tarifa (R$/kWh)", 0.50, 1.50, 0.95, step=0.05, key="tarifa_bench")
    try:
        bench = dados_painel(float(consumo_b), float(tarifa_b))["capitais"]
    except api_client.ErroAPI as exc:
        ui.erro_api(exc)
    df_b = pd.DataFrame(bench)
    melhor, pior = df_b.iloc[0], df_b.iloc[-1]
    st.info(
        f"**Insight:** com {consumo_b} kWh/mês e R\\$ {ui.brl(tarifa_b, 2)}/kWh, o investimento se paga em "
        f"**{ui.brl(melhor['payback_anos'], 1)} anos em {melhor['cidade']}** e em **{ui.brl(pior['payback_anos'], 1)} anos em "
        f"{pior['cidade']}**. Todas as capitais ficam abaixo de {ui.brl(df_b['payback_anos'].max() + 0.05, 1)} anos, "
        "para um sistema que dura 25: a energia solar é viável em todo o país, e o que muda é a velocidade do retorno.",
        icon=":material/insights:",
    )
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Tempo de retorno por capital**")
        st.altair_chart(grafico_benchmark(bench, melhor["cidade"]), use_container_width=True)
    with g2:
        st.markdown("**Irradiação x tempo de retorno**")
        disp = (
            alt.Chart(df_b)
            .mark_circle(size=160)
            .encode(
                x=alt.X("irradiacao_media:Q", title="irradiação média (kWh/m²/dia)", scale=alt.Scale(zero=False)),
                y=alt.Y("payback_anos:Q", title="payback (anos)", scale=alt.Scale(zero=False)),
                color=alt.Color("classificacao:N", scale=alt.Scale(domain=list(CORES_CLASSE), range=list(CORES_CLASSE.values())),
                                legend=alt.Legend(orient="bottom", title="viabilidade")),
                tooltip=["cidade", "irradiacao_media", "payback_anos", "potencia_kwp"],
            )
        )
        rotulos = disp.mark_text(align="left", dx=9, fontSize=11).encode(text=alt.Text("cidade:N"), color=alt.value("#57534e"))
        st.altair_chart((disp + rotulos).properties(height=300), use_container_width=True)
    st.markdown("**Irradiação ao longo do ano (CRESESB, painel inclinado)**")
    try:
        lp = ui.longo_prazo("Maior média anual")
    except api_client.ErroAPI as exc:
        ui.erro_api(exc)
    calor = lp.melt(id_vars="cidade", value_vars=climate.MESES, var_name="mês", value_name="irradiação")
    st.altair_chart(
        alt.Chart(calor)
        .mark_rect()
        .encode(
            x=alt.X("mês:O", sort=climate.MESES, title=None),
            y=alt.Y("cidade:N", sort=alt.EncodingSortField("irradiação", op="mean", order="descending"), title=None),
            color=alt.Color("irradiação:Q", scale=alt.Scale(scheme="goldorange"), legend=alt.Legend(title="kWh/m²/dia")),
            tooltip=["cidade", "mês", alt.Tooltip("irradiação:Q", format=".2f")],
        )
        .properties(height=320),
        use_container_width=True,
    )
    st.caption("Cidades do Sul e Sudeste têm inverno bem mais fraco; no Nordeste a geração é alta e estável o ano todo.")
    tabela = df_b.rename(columns={
        "cidade": "Capital", "irradiacao_media": "Irradiação (kWh/m²/dia)", "potencia_kwp": "Sistema (kWp)",
        "investimento_rs": "Investimento (R$)", "payback_anos": "Payback (anos)", "economia_25_anos_rs": "Economia 25 anos (R$)",
        "co2_evitado_kg_ano": "CO₂ evitado (kg/ano)", "classificacao": "Viabilidade",
    })
    st.dataframe(tabela, hide_index=True, width="stretch",
                 column_config={"Investimento (R$)": st.column_config.NumberColumn(format="localized"),
                                "Economia 25 anos (R$)": st.column_config.NumberColumn(format="localized")})


# ---------------------------------------------------------------------------
# Setor solar com IA
# ---------------------------------------------------------------------------
with aba_setor:
    try:
        nia = noticias_ia()
    except api_client.ErroAPI as exc:
        ui.erro_api(exc)
    if nia.empty:
        st.warning("As notícias ainda não foram processadas pelo modelo. Rode `python -m src.llm.enriquecer_noticias`.")
    else:
        nia["data"] = pd.to_datetime(nia["data"], errors="coerce")
        cont = nia["sentimento"].value_counts()
        k = st.columns(4)
        k[0].metric("Notícias lidas pela IA", len(nia), border=True)
        for i, s in enumerate(["positivo", "neutro", "negativo"], start=1):
            k[i].metric(f"Sentimento {s}", f"{int(cont.get(s, 0))} ({cont.get(s, 0) / len(nia):.0%})", border=True)
        st.info(
            f"**Insight:** para quem quer instalar, o noticiário do setor é majoritariamente favorável "
            f"({cont.get('positivo', 0)} positivas contra {cont.get('negativo', 0)} negativa(s)); as neutras tratam de "
            "regulação, leilões e debates ainda sem efeito definido. Sentimento classificado pelo LLM local do ponto de "
            "vista de quem quer instalar energia solar.",
            icon=":material/insights:",
        )
        s1, s2 = st.columns(2)
        cor_sent = alt.Scale(domain=list(CORES_SENTIMENTO), range=list(CORES_SENTIMENTO.values()))
        with s1:
            st.markdown("**Sentimento por fonte**")
            st.altair_chart(
                alt.Chart(nia).mark_bar().encode(
                    y=alt.Y("fonte:N", title=None), x=alt.X("count():Q", title="notícias", stack="normalize", axis=alt.Axis(format="%")),
                    color=alt.Color("sentimento:N", scale=cor_sent, legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["fonte", "sentimento", alt.Tooltip("count():Q", title="notícias")],
                ).properties(height=160),
                use_container_width=True,
            )
        with s2:
            st.markdown("**Notícias por semana e sentimento**")
            semanal = nia.dropna(subset=["data"]).assign(semana=lambda d: d["data"].dt.to_period("W").dt.start_time)
            st.altair_chart(
                alt.Chart(semanal).mark_bar().encode(
                    x=alt.X("semana:T", title=None), y=alt.Y("count():Q", title="notícias"),
                    color=alt.Color("sentimento:N", scale=cor_sent, legend=None),
                    tooltip=[alt.Tooltip("semana:T", title="semana de"), "sentimento", alt.Tooltip("count():Q", title="notícias")],
                ).properties(height=160),
                use_container_width=True,
            )
        filtro = st.radio("Mostrar", ["todas", "positivo", "neutro", "negativo"], horizontal=True, key="filtro_sent")
        lista = nia if filtro in (None, "todas") else nia[nia["sentimento"] == filtro]
        for r in lista.sort_values("data", ascending=False).head(15).itertuples():
            cor = {"positivo": "green", "neutro": "gray", "negativo": "red"}[r.sentimento]
            with st.container(border=True):
                st.markdown(ui.sem_latex(f":{cor}-badge[{r.sentimento} · {r.confianca:.0%}] **[{r.titulo}]({r.url})**"))
                st.markdown(ui.sem_latex(r.resumo_ia))
                st.caption(f"{r.fonte} · {r.data:%d/%m/%Y}" + (f" · {ui.sem_latex(r.justificativa)}" if r.justificativa else ""))
        st.caption(f"Resumos e sentimentos gerados em lote pelo modelo {painel['modelo_noticias']} (`python -m src.llm.enriquecer_noticias`).")


# ---------------------------------------------------------------------------
# Como funciona (ciclo de ciência de dados)
# ---------------------------------------------------------------------------
with aba_ciclo:
    st.markdown("**Do dado bruto à recomendação:** cada etapa do ciclo de ciência de dados (TDSP) e o que ela entrega ao usuário.")
    st.graphviz_chart(
        """
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fillcolor="#FEF3C7", color="#F59E0B"];
  edge [color="#78716C", fontname="Helvetica", fontsize=9];
  subgraph cluster_c { label="1. Coleta"; style=dashed; color="#a8a29e"; fontname="Helvetica";
    nasa [label="NASA POWER\\nAPI REST"]; cresesb [label="CRESESB SunData\\nSelenium"]; news [label="ABSOLAR e Canal Solar\\nBeautiful Soup"]; }
  proc [label="2. Processamento\\nlimpeza, médias mensais,\\ncorpus em trechos, fichas"];
  anal [label="3. Análise\\ndimensionamento, payback\\ncom Fio B, CO2, benchmark"];
  llm [label="4. IA (Qwen3-1.7B local)\\nnotícias: resumo e sentimento\\nagente decide as análises\\nchat com memória", fillcolor="#DBEAFE", color="#3B82F6"];
  api [label="API FastAPI", fillcolor="#FDE68A"];
  app [label="5. Apresentação\\npainel Streamlit", fillcolor="#FDE68A"];
  user [label="Usuário", shape=ellipse, fillcolor="white"];
  nasa -> proc; cresesb -> proc; news -> proc; proc -> anal; proc -> llm; anal -> llm [label="fatos"];
  anal -> api; llm -> api; api -> app; user -> app [label="cidade, consumo,\\nperguntas"];
}
"""
    )
    st.dataframe(
        pd.DataFrame([{"etapa": e["etapa"], "o que acontece": e["descricao"],
                       "indicadores": " · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in e["indicadores"].items())} for e in painel["ciclo"]]),
        hide_index=True, width="stretch",
    )
    st.markdown(
        """
**Por que o modelo não faz as contas.** Modelos pequenos erram aritmética. Por isso as ferramentas calculam tudo
sobre os dados coletados, o modelo decide *quais* análises fazem sentido para o objetivo do usuário e escreve a
explicação, e cada número do texto é conferido com os cálculos antes de aparecer aqui.

**Por que um modelo local.** Os dados do usuário (consumo, conta de luz) não saem da máquina, não há custo por
pergunta e o painel funciona sem internet depois do download do modelo. O Qwen3-1.7B foi escolhido no TP4
comparando 5 modelos nas tarefas do projeto.

**Eficiência.** As notícias são resumidas e classificadas uma vez, em lote, e não a cada visita; respostas
repetidas vêm do cache; quando só há uma ação possível o agente nem consulta o modelo.
"""
    )
    st.page_link("paginas/inicio.py", label="Problema de negócio, ODS e inspirações", icon=":material/info:")
