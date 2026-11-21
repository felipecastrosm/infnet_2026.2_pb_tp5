"""Modelo local: estado do LLM, como ele foi integrado à API e como os dados chegam até ele.

Mostra o estado do modelo (GET /llm/status), o corpus usado nas respostas (GET /llm/corpus) e
a etapa de recuperação de contexto (GET /llm/contexto), que decide quais trechos o modelo lê
antes de responder no assistente.
"""

import pandas as pd
import streamlit as st

from src import api_client, ui

st.title("Modelo local (LLM)")
st.write(
    "O Solar Viável usa um **modelo de linguagem da HuggingFace executado na própria máquina**, com a "
    "biblioteca Transformers e o PyTorch, sem serviços na nuvem. O modelo fica dentro da API FastAPI; "
    "o app só envia textos e recebe respostas estruturadas. A escolha do modelo e o passo a passo da "
    "instalação estão em `docs/llm_escolha_modelo.md` e `docs/llm_integracao_local.md`."
)

st.header("1. Estado do modelo")
status = ui.painel_llm()
if status:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estado", status["estado"].replace("_", " "))
    c2.metric("Hardware", (status["dispositivo"] or "n/d").upper())
    c3.metric("Memória do modelo", f"{status['memoria_modelo_mb'] / 1024:.1f} GB".replace(".", ",") if status["memoria_modelo_mb"] else "n/d")
    c4.metric("Requisições atendidas", status["requisicoes_atendidas"])
    limites = status["limites"]
    st.caption(
        f"Limites configurados: prazo de {limites['timeout_s']:.0f} s por geração (depois disso, HTTP 504), "
        f"espera de até {limites['espera_fila_s']:.0f} s na fila (HTTP 503) e até {limites['max_tokens_entrada']} tokens "
        f"de entrada (HTTP 413). Carregado em {status['tempo_carregamento_s'] or 'n/d'} s."
    )

st.header("2. Como a integração funciona")
st.markdown(
    """
| Etapa | Onde | O que acontece |
|---|---|---|
| Download (uma vez) | `hf download` → `~/.cache/huggingface` | os pesos do modelo vão para o disco; depois disso nada é baixado |
| Carga | `src/llm/servico.py` | ao subir a API, o modelo é carregado em segundo plano na GPU (CUDA ou Apple MPS) ou na CPU |
| Requisição | `POST /processar_texto`, `/llm/perguntar`, `/llm/relatorio` | o Pydantic valida a entrada; o texto vira tokens e passa pelo modelo |
| Inferência | `src/llm/tarefas.py` | resumo e respostas por geração gulosa (mesma entrada, mesma saída); sentimento pelas probabilidades de cada rótulo |
| Resposta | Response Models (`src/api/schemas.py`) | JSON com o resultado, o modelo usado e o custo (tokens e tempo) |
| Erros | handlers em `src/api/main.py` | cada falha vira um código HTTP (413, 422, 502, 503, 504...) com mensagem e sugestão |
"""
)
st.caption(
    "Princípio de projeto: **os cálculos continuam na API** (determinísticos e testados). O modelo explica, "
    "resume e conversa; no relatório de viabilidade, cada número do texto gerado é conferido com a API."
)

st.header("3. Corpus usado nas respostas")
try:
    stats = api_client.get("/llm/corpus")
except api_client.ErroAPI as exc:
    ui.erro_api(exc)
c1, c2, c3 = st.columns(3)
c1.metric("Documentos (trechos)", stats["documentos"])
c2.metric("Palavras", ui.brl(stats["palavras"]))
c3.metric("Fichas de local", stats["por_tipo"].get("ficha_local", 0))
st.write(
    "As matérias viram trechos de até ~220 palavras, com o título em cada trecho; os números de clima viram "
    "**fichas de local** e um **ranking das capitais** em texto, já que um modelo pequeno compara melhor uma "
    "lista já ordenada do que dez fichas soltas. Cada documento leva fonte, link e data, para a resposta citar de onde veio."
)
st.dataframe(pd.DataFrame({"documentos": stats["por_fonte"]}).rename_axis("fonte"), width="content")

st.header("4. Recuperação de contexto")
st.write(
    "Antes de responder, o assistente escolhe os trechos que o modelo vai ler: busca por palavras-chave "
    "(TF-IDF) mais regras de metadados (a ficha da cidade citada entra sempre; perguntas comparativas "
    "recebem o ranking). Veja abaixo o que seria enviado ao modelo (`GET /llm/contexto`)."
)
exemplos = [
    "Qual capital tem a maior irradiação solar?",
    "O que é o leilão de baterias?",
    "Como está o custo da energia solar no Brasil?",
    "Quanto um sistema de 5 kWp gera em Curitiba?",
]
pergunta = st.selectbox("Pergunta de exemplo (ou escreva a sua abaixo)", exemplos, key="ia_exemplo")
pergunta = st.text_input("Pergunta", pergunta, key=f"ia_pergunta_{pergunta}")
k = st.slider("Quantos trechos recuperar", 1, 8, 4, key="ia_k")
try:
    contexto = api_client.get("/llm/contexto", pergunta=pergunta, k=k)
except api_client.ErroAPI as exc:
    st.warning(str(exc))
    st.stop()

if not contexto["trechos"]:
    st.info("Nenhum trecho do corpus contém as palavras da pergunta.")
for i, t in enumerate(contexto["trechos"], start=1):
    with st.container(border=True):
        titulo = f"[{t['titulo']}]({t['url']})" if t["url"] else t["titulo"]
        st.markdown(ui.sem_latex(f"**[{i}] {titulo}**"))
        st.caption(f"{t['fonte']} · {t['tipo']} · {t['data']} · relevância {t['pontuacao']:.2f}")
        st.write(ui.sem_latex(t["texto"][:600] + ("..." if len(t["texto"]) > 600 else "")))
with st.expander("Prompt montado com esses trechos"):
    st.code(contexto["prompt"], language="markdown")
