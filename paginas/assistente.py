"""Pergunte ao Solar Viável: chat que responde com base nas notícias e nos dados das capitais.

Cada pergunta vai para `POST /llm/perguntar`: a API recupera os trechos mais relevantes do
corpus, o LLM local responde citando os trechos ([1], [2]...) e a resposta volta com a
lista de fontes. Nada sai da máquina: o modelo roda localmente.
"""

import streamlit as st

from src import api_client, ui

st.title("Pergunte ao Solar Viável")
st.write(
    "Tire dúvidas sobre energia solar em linguagem simples. As respostas são geradas por um **modelo de "
    "linguagem que roda localmente** e usam só as notícias do setor e os dados das 10 capitais coletados "
    "pelo projeto, com as fontes citadas."
)
status = ui.painel_llm()

st.session_state.setdefault("chat", [])
EXEMPLOS = [
    "Qual capital tem a maior irradiação solar?",
    "Quanto um sistema de 5 kWp gera por mês em Curitiba?",
    "O que é o leilão de baterias?",
    "Quanto da expansão da matriz elétrica em 2026 veio da energia solar?",
]


def mostrar_fontes(fontes: list) -> None:
    citadas = [f for f in fontes if f["citada"]] or fontes
    if not citadas:
        return
    with st.expander(f"Fontes ({len(citadas)} citada(s) de {len(fontes)} consultada(s))"):
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
        try:
            with st.spinner("Buscando nos dados e gerando a resposta..."):
                r = api_client.post("/llm/perguntar", {"pergunta": pergunta, "k": st.session_state.get("chat_k", 4)})
        except api_client.ErroAPI as exc:
            ui.aviso_llm(exc)
            st.session_state.chat.append({"papel": "assistant", "erro": exc})
        else:
            st.markdown(ui.sem_latex(r["resposta"]))
            mostrar_fontes(r["fontes"])
            uso = r["uso"]
            rodape = f"{r['modelo']['nome']} · {uso['tempo_s']:.1f} s · {uso['tokens_saida']} tokens" + (" · do cache" if uso["em_cache"] else "")
            st.caption(rodape)
            st.session_state.chat.append({"papel": "assistant", "texto": r["resposta"], "fontes": r["fontes"], "rodape": rodape})

with st.sidebar:
    st.subheader("Assistente")
    st.slider("Trechos consultados por pergunta", 1, 8, 4, key="chat_k", help="Mais trechos dão mais contexto, mas a resposta demora mais.")
    if st.button("Limpar conversa", icon=":material/delete_sweep:", disabled=not st.session_state.chat):
        st.session_state.chat = []
        st.rerun()
    st.caption(
        "O modelo pode errar. Os números exatos de geração e economia estão nas páginas de análise "
        "(calculados pela API). Cada pergunta é respondida sozinha, sem memória das anteriores."
    )
