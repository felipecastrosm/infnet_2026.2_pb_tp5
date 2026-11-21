"""Pergunte ao Solar Viável: chat com memória sobre as notícias e os dados das capitais.

Desde o TP5 a conversa tem memória (`/conversas` da API): perguntas de acompanhamento como
"e em Manaus?" ou "qual delas tem mais sol?" são reescritas como perguntas completas antes
da busca de contexto, e o modelo vê as últimas trocas, um resumo das anteriores e os fatos
lembrados (cidade, potência, consumo). As respostas citam os trechos usados ([1], [2]...).
Nada sai da máquina: o modelo roda localmente.
"""

import streamlit as st

from src import api_client, ui

st.title("Pergunte ao Solar Viável")
st.write(
    "Tire dúvidas sobre energia solar em linguagem simples. As respostas são geradas por um **modelo de "
    "linguagem que roda localmente** e usam só as notícias do setor e os dados das 10 capitais coletados "
    "pelo projeto, com as fontes citadas. **O assistente lembra da conversa**: pergunte sobre uma cidade e "
    "depois só \"e em Manaus?\"."
)
status = ui.painel_llm()

st.session_state.setdefault("chat", [])
EXEMPLOS = [
    "Quanto um sistema de 5 kWp gera por mês em Curitiba?",
    "Qual capital tem a maior irradiação solar?",
    "O que é o leilão de baterias?",
    "Quanto da expansão da matriz elétrica em 2026 veio da energia solar?",
]


def conversa_id() -> str:
    """Cria a conversa na API na primeira pergunta (e de novo se ela tiver expirado)."""
    if "conversa_chat" not in st.session_state:
        st.session_state.conversa_chat = api_client.post("/conversas")["id"]
    return st.session_state.conversa_chat


def mostrar_fontes(fontes: list) -> None:
    citadas = [f for f in fontes if f["citada"]] or fontes
    if not citadas:
        return
    with st.expander(f"Fontes ({len([f for f in fontes if f['citada']])} citada(s) de {len(fontes)} consultada(s))"):
        for f in fontes:
            titulo = f"[{f['titulo']}]({f['url']})" if f["url"] else f["titulo"]
            marca = ":material/format_quote:" if f["citada"] else ""
            st.markdown(ui.sem_latex(f"**[{f['n']}]** {titulo} {marca}"))
            st.caption(f"{f['fonte']} · {f['tipo'].replace('_', ' ')}{' · ' + f['data'] if f['data'] else ''}")


for msg in st.session_state.chat:
    with st.chat_message(msg["papel"]):
        if msg.get("erro"):
            ui.aviso_llm(msg["erro"])
        else:
            st.markdown(ui.sem_latex(msg["texto"]))
        if msg.get("reescrita"):
            st.caption(f":material/psychology_alt: Entendi como: *{ui.sem_latex(msg['reescrita'])}*")
        if msg.get("fontes"):
            mostrar_fontes(msg["fontes"])
        if msg.get("rodape"):
            st.caption(msg["rodape"])

if not st.session_state.chat and "pergunta_pendente" not in st.session_state:
    st.caption("Sugestões:")
    colunas = st.columns(2)
    for i, exemplo in enumerate(EXEMPLOS):
        if colunas[i % 2].button(exemplo, key=f"exemplo_chat_{i}", width="stretch"):
            st.session_state.pergunta_pendente = exemplo
            st.rerun()

pergunta = st.chat_input("Escreva sua pergunta", disabled=not (status and status["pronto"])) or st.session_state.pop(
    "pergunta_pendente", None
)
if pergunta:
    st.session_state.chat.append({"papel": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(ui.sem_latex(pergunta))
    with st.chat_message("assistant"):
        pedido = {"pergunta": pergunta, "k": st.session_state.get("chat_k", 4)}
        try:
            with st.spinner("Lembrando da conversa, buscando nos dados e gerando a resposta..."):
                try:
                    r = api_client.post(f"/conversas/{conversa_id()}/mensagens", pedido)
                except api_client.ErroAPI as exc:
                    if exc.erro != "conversa_nao_encontrada":
                        raise
                    st.session_state.pop("conversa_chat", None)  # expirou: começa outra
                    r = api_client.post(f"/conversas/{conversa_id()}/mensagens", pedido)
        except api_client.ErroAPI as exc:
            ui.aviso_llm(exc)
            st.session_state.chat.append({"papel": "assistant", "erro": exc})
        else:
            reescrita = r["pergunta_reescrita"] if r["pergunta_reescrita"] != r["pergunta"] else None
            st.markdown(ui.sem_latex(r["resposta"]))
            if reescrita:
                st.caption(f":material/psychology_alt: Entendi como: *{ui.sem_latex(reescrita)}*")
            mostrar_fontes(r["fontes"])
            uso = r["uso"]
            rodape = f"{r['modelo']['nome']} · {uso['tempo_s']:.1f} s · {uso['tokens_saida']} tokens"
            st.caption(rodape)
            st.session_state.chat.append(
                {"papel": "assistant", "texto": r["resposta"], "fontes": r["fontes"], "rodape": rodape, "reescrita": reescrita}
            )
            st.session_state.memoria_chat = r["memoria"]

with st.sidebar:
    st.subheader("Assistente")
    st.slider("Trechos consultados por pergunta", 1, 8, 4, key="chat_k", help="Mais trechos dão mais contexto, mas a resposta demora mais.")
    if st.button("Nova conversa", icon=":material/delete_sweep:", disabled=not st.session_state.chat):
        if "conversa_chat" in st.session_state:
            try:
                api_client.delete(f"/conversas/{st.session_state.pop('conversa_chat')}")
            except api_client.ErroAPI:
                pass
        st.session_state.chat = []
        st.session_state.pop("memoria_chat", None)
        st.rerun()
    mem = st.session_state.get("memoria_chat")
    if mem:
        st.markdown("**Memória da conversa**")
        fatos = {k: v for k, v in mem["fatos"].items() if v}
        st.json(fatos or {}, expanded=True)
        if mem["resumo"]:
            st.caption(f"Resumo: {ui.sem_latex(mem['resumo'])}")
        st.caption(f"{mem['total_turnos']} troca(s); as últimas {mem['turnos_na_janela']} vão na íntegra ao modelo.")
    st.caption(
        "O modelo pode errar. Os números exatos de geração e economia estão no Painel e nas páginas de análise "
        "(calculados pela API)."
    )
