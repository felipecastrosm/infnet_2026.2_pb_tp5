"""Testes do app Streamlit com múltiplas páginas (Streamlit AppTest).

O app sobe a API embutida (porta definida em conftest.py) e consome os dados por HTTP.
"""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data_acquisition import nasa_power_client
from src.data_processing import climate

PAGINAS = [
    "paginas/inicio.py",
    "paginas/clima.py",
    "paginas/longo_prazo.py",
    "paginas/noticias.py",
    "paginas/meus_dados.py",
    "paginas/api.py",
    "paginas/assistente.py",
    "paginas/processar_texto.py",
    "paginas/ia.py",
]


def abrir(pagina: str = None) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=60).run()
    if pagina:
        at.switch_page(pagina).run()
    return at


@pytest.fixture
def nasa_falsa(monkeypatch):
    def api_falsa(lat, lon, start, end, *args, **kwargs):
        return pd.DataFrame({v: 5.0 for v in climate.VARIAVEIS}, index=pd.date_range(start, end, name="date"))

    monkeypatch.setattr(nasa_power_client, "get_power_data", api_falsa)


@pytest.mark.parametrize("pagina", PAGINAS)
def test_todas_as_paginas_abrem_sem_erro(pagina):
    at = abrir(pagina)
    assert not at.exception
    assert not at.error
    assert at.title  # cada página tem seu título


def test_menu_de_navegacao_tem_todas_as_paginas():
    from pathlib import Path

    app = Path("app.py").read_text()
    assert "st.navigation(" in app
    assert all(p in app for p in PAGINAS)


def test_parametros_persistem_entre_paginas():
    at = abrir("paginas/clima.py")
    at.number_input(key="potencia").set_value(10.0).run()
    at.switch_page("paginas/noticias.py").run()  # página sem esses widgets
    at.switch_page("paginas/longo_prazo.py").run()
    assert not at.exception
    assert at.number_input(key="potencia").value == 10.0


def test_clima_interativo():
    at = abrir("paginas/clima.py")
    at.checkbox(key="suavizar").uncheck().run()
    at.selectbox(key="variavel").select("T2M").run()
    assert not at.exception
    assert at.session_state["suavizar"] is False


def test_consulta_ao_vivo_preserva_estado_e_selecao(nasa_falsa):
    """Regressões do TP2: a consulta não pode resetar widgets nem a seleção de locais."""
    at = abrir("paginas/clima.py")
    at.checkbox(key="suavizar").uncheck().run()
    at.multiselect(key="cidades").set_value(["Recife (PE)", "Manaus (AM)"]).run()

    [b for b in at.button if "NASA" in b.label][0].click().run()

    assert not at.exception
    assert at.session_state["suavizar"] is False
    selecionadas = at.multiselect(key="cidades").value
    assert "Recife (PE)" in selecionadas and "Manaus (AM)" in selecionadas
    assert any("ao vivo" in c for c in selecionadas)  # o novo local entra selecionado


def test_medicao_manual_e_enviada_para_a_api():
    from src import api_client

    at = abrir("paginas/meus_dados.py")
    form = at.get("form")[0]
    form.text_input[0].set_value("Telhado de teste")
    form.number_input[0].set_value(6.5)
    [b for b in at.button if b.label == "Enviar medição"][0].click().run()
    assert not at.exception

    salvas = api_client.get("/medicoes", cidade="Telhado de teste")
    assert [m["ALLSKY_SFC_SW_DWN"] for m in salvas] == [6.5]
    api_client.delete("/medicoes/Telhado de teste")


def test_pagina_ia_recupera_contexto():
    at = abrir("paginas/ia.py")
    assert not at.exception
    assert any("Prompt" in e.label for e in at.expander)


# ---------------------------------------------------------------------------
# Páginas com o LLM local (backend falso de conftest.py)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pagina", ["paginas/assistente.py", "paginas/processar_texto.py", "paginas/ia.py"])
def test_paginas_de_ia_com_modelo_nao_carregado_oferecem_carga(pagina):
    from src.llm.servico import servico

    servico().usar_backend(None)
    at = abrir(pagina)
    assert not at.exception
    assert any("não foi carregado" in w.value for w in at.warning)
    assert any(b.label == "Carregar o modelo" for b in at.button)


def test_status_do_modelo_na_pagina_ia(llm):
    at = abrir("paginas/ia.py")
    assert not at.exception
    assert any("modelo-falso/teste" in c.value for c in at.caption)
    assert at.metric[0].value == "pronto"


def test_chat_responde_com_fontes(llm):
    _, backend = llm
    backend.resposta = "Recife (PE) tem a maior irradiação solar, com 6,06 kWh/m²/dia [1]."
    at = abrir("paginas/assistente.py")
    at.chat_input[0].set_value("Qual capital tem a maior irradiação solar?").run()
    assert not at.exception
    assert any("Recife (PE) tem a maior" in m.value for m in at.markdown)
    assert any(e.label.startswith("Fontes (1 citada") for e in at.expander)
    assert len(at.session_state["chat"]) == 2


def test_chat_mostra_erro_amigavel(llm):
    from src.llm import erros

    llm[1].erro = erros.TempoEsgotado("A geração passou do prazo de 90 s e foi interrompida.")
    at = abrir("paginas/assistente.py")
    at.chat_input[0].set_value("O que é o leilão de baterias?").run()
    assert not at.exception
    assert any("passou do prazo" in w.value for w in at.warning)
    assert any("HTTP 504" in c.value for c in at.caption)


def test_resumo_de_noticia_pela_pagina(llm):
    llm[1].resposta = "O setor solar superou as projeções de 2026 com três meses de antecedência."
    at = abrir("paginas/processar_texto.py")
    [b for b in at.button if "Processar" in b.label][0].click().run()
    assert not at.exception
    resultado = at.session_state["pt_resposta"]["resultado"]
    assert resultado["tipo"] == "resumo" and "superou as projeções" in resultado["resumo"]
    assert any(m.label == "Palavras no resumo" for m in at.metric)


def test_relatorio_na_pagina_de_longo_prazo(llm):
    llm[1].resposta = "## Potencial solar do local\nPeça 3 orçamentos e confira as regras da ANEEL para geração distribuída na sua cidade."
    at = abrir("paginas/longo_prazo.py")
    [b for b in at.button if b.label == "Gerar relatório"][0].click().run()
    assert not at.exception
    assert any("conferem com os dados da API" in s.value for s in at.success)
    assert at.session_state["relatorio_lp"]["verificacao"]["confiavel"]


def test_noticia_resumida_e_avaliada(llm):
    llm[1].resposta = "Resumo curto da matéria gerado pelo modelo local."
    at = abrir("paginas/noticias.py")
    [b for b in at.button if "Resumir e avaliar" in b.label][0].click().run()
    assert not at.exception
    assert len(at.session_state["analises_ia"]) == 1
    assert any("positivo para quem quer instalar" in m.value for m in at.markdown)
