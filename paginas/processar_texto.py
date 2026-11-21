"""Resumo e sentimento: envia um texto ao LLM local pela rota `POST /processar_texto`.

O texto pode ser digitado/colado ou vir de uma notícia coletada (`GET /noticias`). A página
mostra o resultado estruturado (Response Model da API) e o JSON bruto da resposta.
"""

import pandas as pd
import streamlit as st

from src import api_client, ui

st.title("Resumo e sentimento")
st.write(
    "Envie um texto para o **modelo de linguagem local** resumir ou classificar o sentimento. "
    "A página usa a rota `POST /processar_texto` da API, que devolve sempre o mesmo formato de resposta."
)
status = ui.painel_llm()

try:
    noticias = ui.noticias()
except api_client.ErroAPI as exc:
    ui.erro_api(exc)

origem = st.radio("Texto", ["Uma notícia coletada", "Escrever ou colar"], horizontal=True, key="pt_origem")
if origem == "Uma notícia coletada":
    com_texto = noticias[noticias["n_palavras"] > 50].reset_index(drop=True)
    rotulos = [f"{r.titulo} ({r.fonte})" for r in com_texto.itertuples()]
    escolha = st.selectbox("Notícia", range(len(rotulos)), format_func=lambda i: rotulos[i], key="pt_noticia")
    noticia = com_texto.iloc[escolha]
    texto = f"{noticia['titulo']}. {noticia['texto']}"
    data = f" · {noticia['data']:%d/%m/%Y}" if pd.notna(noticia["data"]) else ""
    st.caption(f"[Abrir a matéria original]({noticia['url']}) · {noticia['n_palavras']} palavras{data}")
    with st.expander("Texto que será enviado"):
        st.write(ui.sem_latex(texto))
else:
    texto = st.text_area(
        "Texto (20 a 20.000 caracteres)",
        height=200,
        key="pt_texto",
        placeholder="Cole aqui uma notícia, um orçamento de instalador, um comentário...",
    )

c1, c2, c3 = st.columns(3)
tarefa = c1.radio("Tarefa", ["resumo", "sentimento"], horizontal=True, key="pt_tarefa")
if tarefa == "resumo":
    max_palavras = c2.slider("Tamanho máximo (palavras)", 20, 200, 60, step=10, key="pt_max")
    pedido = {"texto": texto, "tarefa": "resumo", "max_palavras": max_palavras}
else:
    ponto = c2.radio(
        "Ponto de vista",
        ["setor_solar", "geral"],
        format_func={"setor_solar": "quem quer instalar solar", "geral": "tom geral do texto"}.get,
        key="pt_ponto",
    )
    justificar = c3.toggle("Pedir justificativa", True, key="pt_justificar")
    pedido = {"texto": texto, "tarefa": "sentimento", "ponto_de_vista": ponto, "justificar": justificar}

if st.button("Processar com o modelo local", type="primary", icon=":material/smart_toy:", disabled=not (status and status["pronto"])):
    try:
        with st.spinner("O modelo está processando o texto..."):
            st.session_state.pt_resposta = api_client.post("/processar_texto", pedido)
            st.session_state.pt_erro = None
    except api_client.ErroAPI as exc:
        st.session_state.pt_resposta, st.session_state.pt_erro = None, exc

if st.session_state.get("pt_erro"):
    ui.aviso_llm(st.session_state.pt_erro)

resposta = st.session_state.get("pt_resposta")
if resposta:
    res = resposta["resultado"]
    st.subheader("Resultado")
    if res["tipo"] == "resumo":
        st.success(ui.sem_latex(res["resumo"]), icon=":material/summarize:")
        m1, m2, m3 = st.columns(3)
        m1.metric("Palavras no texto", resposta["entrada"]["n_palavras"])
        m2.metric("Palavras no resumo", res["n_palavras_resumo"])
        m3.metric("Compressão", f"{res['taxa_compressao']:.0%}".replace(".", ","))
    else:
        cores = {"positivo": "green", "neutro": "gray", "negativo": "red"}
        st.markdown(f"### :{cores[res['sentimento']]}[{res['sentimento'].capitalize()}] · confiança {res['confianca']:.0%}")
        if res["justificativa"]:
            st.write(ui.sem_latex(res["justificativa"]))
        probs = pd.DataFrame({"probabilidade": res["probabilidades"]}).reindex(["positivo", "neutro", "negativo"])
        st.bar_chart(probs, horizontal=True, x_label="probabilidade", height=180)
    uso, modelo = resposta["uso"], resposta["modelo"]
    st.caption(
        f"{modelo['nome']} ({modelo['dispositivo']}, {modelo['dtype']}) · {resposta['entrada']['n_tokens']} tokens de entrada · "
        f"{uso['tokens_saida']} gerados · {uso['tempo_s']:.1f} s" + (" · resposta do cache" if uso["em_cache"] else "")
    )
    with st.expander("JSON da resposta (Response Model `RespostaProcessamento`)"):
        st.json(resposta)
