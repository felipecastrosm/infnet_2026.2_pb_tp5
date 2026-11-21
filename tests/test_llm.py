"""Testes da integração com o LLM: rotas, Response Models, tratamento de erros e tarefas.

Usam o backend falso de conftest.py (rápido e determinístico). O último teste usa o modelo
real e só roda com SOLAR_TESTAR_LLM=1 (o modelo precisa estar no cache local).
"""

import os
import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import repositorio
from src.api import schemas
from src.api.main import app
from src.data_processing import llm_prep
from src.llm import erros, tarefas
from src.llm.config import ConfigLLM
from src.llm.servico import Estado, ServicoLLM, servico

client = TestClient(app)

NOTICIA = (
    "A fonte solar respondeu por 73% da expansão da matriz elétrica brasileira entre janeiro e agosto de 2026, "
    "com 3,5 GW adicionados. A capacidade total de geração centralizada do país já supera 220 GW."
)


def campos_de_erro(resp, status, erro):
    corpo = resp.json()
    assert resp.status_code == status, corpo
    assert corpo["status"] == status and corpo["erro"] == erro
    schemas.RespostaErro.model_validate(corpo)  # todo erro segue o mesmo modelo
    assert corpo["mensagem"]
    return corpo


# ---------------------------------------------------------------------------
# Rotas com o modelo pronto (backend falso)
# ---------------------------------------------------------------------------
def test_status_mostra_modelo_pronto(llm):
    st = client.get("/llm/status").json()
    schemas.StatusLLM.model_validate(st)
    assert st["estado"] == "pronto" and st["pronto"]
    assert st["modelo"] == "modelo-falso/teste" and st["dispositivo"] == "cpu"
    assert "POST /processar_texto" in st["tarefas"]
    assert client.get("/saude").json() == {"status": "ok", "llm": "pronto"}


def test_processar_texto_resumo(llm):
    _, backend = llm
    backend.resposta = "Resumo: A energia solar liderou a expansão da matriz elétrica em 2026."
    resp = client.post("/processar_texto", json={"texto": NOTICIA, "tarefa": "resumo", "max_palavras": 30})
    assert resp.status_code == 200
    corpo = schemas.RespostaProcessamento.model_validate(resp.json())
    assert corpo.tarefa == "resumo" and corpo.resultado.tipo == "resumo"
    assert corpo.resultado.resumo.startswith("A energia solar")  # o rótulo "Resumo:" é removido
    assert corpo.resultado.n_palavras_resumo == 11
    assert 0 < corpo.resultado.taxa_compressao < 1
    assert corpo.entrada.n_palavras == len(NOTICIA.split())
    assert corpo.modelo.nome == "modelo-falso/teste"
    assert corpo.uso.em_cache is False
    assert "no máximo 30 palavras" in backend.chamadas[0][1][1]["content"]


def test_mesma_entrada_vem_do_cache(llm):
    _, backend = llm
    pedido = {"texto": NOTICIA, "tarefa": "resumo"}
    primeira = client.post("/processar_texto", json=pedido).json()
    segunda = client.post("/processar_texto", json=pedido).json()
    assert segunda["uso"]["em_cache"] is True and segunda["uso"]["tempo_s"] == 0
    assert segunda["resultado"] == primeira["resultado"]
    assert len(backend.chamadas) == 1


def test_resumo_respeita_limite_e_remove_repeticoes(llm):
    _, backend = llm
    frase = "A energia solar cresceu muito no Brasil em 2026."
    backend.resposta = " ".join([frase] * 3 + ["Outra frase com informações adicionais sobre o setor elétrico brasileiro."] * 4)
    resultado = client.post("/processar_texto", json={"texto": NOTICIA, "max_palavras": 20}).json()["resultado"]
    assert resultado["resumo"].count(frase) == 1
    assert resultado["n_palavras_resumo"] <= 23


def test_processar_texto_sentimento(llm):
    _, backend = llm
    backend.probabilidades = {"positivo": 0.1, "neutro": 0.15, "negativo": 0.75}
    backend.resposta = "O texto relata prejuízos para as usinas solares."
    resp = client.post("/processar_texto", json={"texto": NOTICIA, "tarefa": "sentimento", "ponto_de_vista": "setor_solar"})
    corpo = schemas.RespostaProcessamento.model_validate(resp.json())
    res = corpo.resultado
    assert res.tipo == "sentimento" and res.sentimento == "negativo"
    assert res.confianca == pytest.approx(0.75)
    assert sum(res.probabilidades.model_dump().values()) == pytest.approx(1.0)
    assert res.justificativa == "O texto relata prejuízos para as usinas solares."
    assert "setor de energia solar" in backend.chamadas[0][1][0]["content"]  # o ponto de vista entra no prompt


def test_sentimento_sem_justificativa_nao_gera_texto(llm):
    _, backend = llm
    res = client.post("/processar_texto", json={"texto": NOTICIA, "tarefa": "sentimento", "justificar": False}).json()["resultado"]
    assert res["justificativa"] is None
    assert [c[0] for c in backend.chamadas] == ["pontuar"]


def test_perguntar_cita_fontes(llm):
    _, backend = llm
    backend.resposta = "Um sistema de 5 kWp gera cerca de 534 kWh por mês em Curitiba [1]. [1]"
    resp = client.post("/llm/perguntar", json={"pergunta": "Quanto um sistema de 5 kWp gera em Curitiba?", "k": 3})
    corpo = schemas.RespostaPergunta.model_validate(resp.json())
    assert corpo.resposta.endswith("[1].")  # "[1]. [1]" repetido vira uma citação só
    assert corpo.fontes[0].titulo == "Ficha de potencial solar: Curitiba (PR)"
    assert corpo.fontes[0].citada and not any(f.citada for f in corpo.fontes[1:])
    prompt = backend.chamadas[0][1][1]["content"]
    assert "[1] Ficha de potencial solar: Curitiba (PR)" in prompt


def test_perguntar_sem_contexto_nao_chama_o_modelo(llm):
    _, backend = llm
    corpo = client.post("/llm/perguntar", json={"pergunta": "xyzw qwerty asdfg zxcvb"}).json()
    assert corpo["sem_contexto"] is True and corpo["fontes"] == []
    assert "Não encontrei" in corpo["resposta"]
    assert backend.chamadas == []


def test_relatorio_com_numeros_da_api(llm):
    _, backend = llm
    est = client.get("/estimativa", params={"cidade": "Recife (PE)", "potencia_kwp": 4, "tarifa": 0.95}).json()

    def relatorio_fiel(mensagens):
        assert "Use SOMENTE os números" in mensagens[0]["content"]
        assert "Um sistema de 5,0 kWp" not in mensagens[1]["content"]  # a frase da ficha (5 kWp) não entra
        return (
            f"## Potencial solar do local\nRecife tem {str(est['irradiacao_media']).replace('.', ',')} kWh/m² por dia.\n"
            f"## Geração e economia esperadas\nO sistema de 4 kWp gera {str(est['geracao_mes_kwh']).replace('.', ',')} kWh por mês "
            f"e economiza R$ {str(est['economia_mes_rs']).replace('.', ',')} por mês.\n## Próximos passos\nPeça 3 orçamentos."
        )

    backend.resposta = relatorio_fiel
    corpo = client.post("/llm/relatorio", json={"cidade": "Recife (PE)", "potencia_kwp": 4, "tarifa": 0.95}).json()
    schemas.RespostaRelatorio.model_validate(corpo)
    assert corpo["verificacao"]["confiavel"] is True
    assert est["geracao_mes_kwh"] in corpo["verificacao"]["numeros_citados"]
    assert corpo["estimativa"] == est


def test_relatorio_aponta_numero_inventado(llm):
    _, backend = llm
    backend.resposta = (
        "## Geração e economia esperadas\nO investimento se paga em 7 anos e custa R$ 25.000,00 com tarifa de R$ 0,95 por kWh. "
        "Isso é tudo o que precisamos dizer sobre este sistema fotovoltaico de exemplo."
    )
    v = client.post("/llm/relatorio", json={"cidade": "Recife (PE)", "potencia_kwp": 4, "tarifa": 0.95}).json()["verificacao"]
    assert v["confiavel"] is False
    assert 25000.0 in v["nao_encontrados"] and 7.0 in v["nao_encontrados"]  # "7 anos" é um dado inventado
    assert 0.95 not in v["nao_encontrados"]


# ---------------------------------------------------------------------------
# Erros: cada cenário com o código HTTP adequado e o mesmo formato
# ---------------------------------------------------------------------------
def test_modelo_nao_carregado_503():
    servico().usar_backend(None)
    corpo = campos_de_erro(client.post("/processar_texto", json={"texto": NOTICIA}), 503, "modelo_nao_carregado")
    assert "POST /llm/carregar" in corpo["sugestao"]
    assert corpo["caminho"] == "/processar_texto"


def test_modelo_carregando_503_com_retry_after(llm):
    s, _ = llm
    s.estado = Estado.CARREGANDO
    resp = client.post("/llm/perguntar", json={"pergunta": "O que é o leilão de baterias?"})
    campos_de_erro(resp, 503, "modelo_carregando")
    assert resp.headers["Retry-After"] == "15"


def test_falha_ao_carregar_modelo_503():
    def fabrica_quebrada(config):
        raise OSError("modelo inexistente/xyz não está no HuggingFace Hub")

    s = ServicoLLM(ConfigLLM(modelo="inexistente/xyz"), fabrica=fabrica_quebrada)
    with pytest.raises(erros.FalhaAoCarregarModelo):
        s.carregar()
    assert s.estado == Estado.ERRO and "inexistente/xyz" in s.erro
    with pytest.raises(erros.FalhaAoCarregarModelo):  # as rotas passam a responder 503 com o motivo
        s.info_modelo()


def test_bibliotecas_ausentes_desativam_o_llm():
    def sem_torch(config):
        raise ImportError("No module named 'torch'")

    s = ServicoLLM(ConfigLLM(), fabrica=sem_torch)
    with pytest.raises(erros.ModeloDesativado):
        s.carregar()


def test_carga_em_segundo_plano():
    s = ServicoLLM(ConfigLLM(), fabrica=lambda config: __import__("conftest").BackendFalso())
    assert s.carregar_em_segundo_plano() is True
    for _ in range(100):
        if s.estado == Estado.PRONTO:
            break
        threading.Event().wait(0.02)
    assert s.estado == Estado.PRONTO and s.tempo_carregamento_s is not None
    assert s.carregar_em_segundo_plano() is False  # já está pronto


def test_llm_desativado_503(llm):
    s, _ = llm
    s.estado = Estado.DESATIVADO
    campos_de_erro(client.post("/processar_texto", json={"texto": NOTICIA}), 503, "modelo_desativado")
    campos_de_erro(client.post("/llm/carregar"), 503, "modelo_desativado")


def test_carregar_inicia_a_carga(monkeypatch):
    s = servico()
    s.usar_backend(None)
    chamadas = []
    monkeypatch.setattr(s, "carregar_em_segundo_plano", lambda: chamadas.append(1) or True)
    resp = client.post("/llm/carregar")
    assert resp.status_code == 202 and chamadas == [1]


@pytest.mark.parametrize(
    "pedido, campo",
    [
        ({"texto": "curto"}, "body.texto"),
        ({"texto": NOTICIA, "tarefa": "traduzir"}, "body.tarefa"),
        ({"texto": NOTICIA, "max_palavras": 5}, "body.max_palavras"),
        ({"texto": "1234567890 1234567890 1234567890"}, "body.texto"),  # sem palavras
        ({"texto": NOTICIA, "ponto_de_vista": "investidor"}, "body.ponto_de_vista"),
        ({}, "body.texto"),
    ],
)
def test_entrada_invalida_422(llm, pedido, campo):
    corpo = campos_de_erro(client.post("/processar_texto", json=pedido), 422, "dados_invalidos")
    assert corpo["detalhes"][0]["campo"] == campo
    assert corpo["detalhes"][0]["problema"]  # explicação em português
    assert llm[1].chamadas == []  # o modelo nem é chamado


def test_json_malformado_422(llm):
    resp = client.post("/processar_texto", content=b"{texto: sem aspas", headers={"Content-Type": "application/json"})
    corpo = campos_de_erro(resp, 422, "dados_invalidos")
    assert corpo["detalhes"][0]["problema"] == "JSON malformado"


def test_texto_longo_demais_413(llm, configurar_llm):
    configurar_llm(max_tokens_entrada=10)
    corpo = campos_de_erro(client.post("/processar_texto", json={"texto": NOTICIA}), 413, "texto_longo_demais")
    assert "limite é 10" in corpo["mensagem"]


def test_tempo_esgotado_504(llm):
    llm[1].erro = erros.TempoEsgotado("A geração passou do prazo de 90 s e foi interrompida.")
    campos_de_erro(client.post("/processar_texto", json={"texto": NOTICIA}), 504, "tempo_esgotado")


def test_modelo_ocupado_503(llm, configurar_llm):
    s, _ = llm
    configurar_llm(espera_fila_s=0.05)
    s._fila.acquire()  # outra requisição ocupando o modelo
    try:
        resp = client.post("/processar_texto", json={"texto": NOTICIA})
    finally:
        s._fila.release()
    campos_de_erro(resp, 503, "modelo_ocupado")
    assert resp.headers["Retry-After"] == "10"


def test_falta_de_memoria_503(llm):
    llm[1].erro = RuntimeError("MPS backend out of memory (MPS allocated: 17.00 GB)")
    campos_de_erro(client.post("/processar_texto", json={"texto": NOTICIA}), 503, "memoria_insuficiente")


def test_falha_na_inferencia_500(llm):
    llm[1].erro = ValueError("tensor com formato inesperado")
    corpo = campos_de_erro(client.post("/processar_texto", json={"texto": NOTICIA}), 500, "falha_na_inferencia")
    assert "ValueError" in corpo["mensagem"]
    assert llm[0]._fila.acquire(blocking=False)  # a fila foi liberada mesmo com o erro
    llm[0]._fila.release()


def test_saida_vazia_do_modelo_502(llm):
    llm[1].resposta = "   "
    campos_de_erro(client.post("/processar_texto", json={"texto": NOTICIA}), 502, "saida_invalida_do_modelo")


def test_erro_inesperado_500(llm, monkeypatch):
    def quebrar(*args, **kwargs):
        raise KeyError("campo")

    monkeypatch.setattr(tarefas, "resumir", quebrar)
    resp = TestClient(app, raise_server_exceptions=False).post("/processar_texto", json={"texto": NOTICIA})
    campos_de_erro(resp, 500, "erro_interno")


def test_relatorio_local_inexistente_404(llm):
    campos_de_erro(client.post("/llm/relatorio", json={"cidade": "Atlântida"}), 404, "nao_encontrado")


def test_rota_e_metodo_inexistentes(llm):
    campos_de_erro(client.get("/nao-existe"), 404, "nao_encontrado")
    campos_de_erro(client.get("/processar_texto"), 405, "metodo_nao_permitido")


def test_openapi_documenta_modelos_e_erros():
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/processar_texto"]["post"]
    assert op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("RespostaProcessamento")
    for codigo in ("413", "422", "500", "502", "503", "504"):
        assert op["responses"][codigo]["content"]["application/json"]["schema"]["$ref"].endswith("RespostaErro")
    # toda rota declara um modelo de resposta
    for caminho, ops in spec["paths"].items():
        for metodo, operacao in ops.items():
            sucesso = next(c for c in operacao["responses"] if c.startswith("2"))
            assert "schema" in operacao["responses"][sucesso]["content"]["application/json"], f"{metodo} {caminho}"


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def test_extrair_numeros_formatos():
    assert tarefas.extrair_numeros("R$ 1.234,56 e 4,31 kWh; 6412 kWh; 4.31") == [1234.56, 4.31, 6412.0, 4.31]
    assert tarefas.extrair_numeros("em 12/2025 foram 534 kWh") == [534.0]


def test_verificacao_ignora_contagens_mas_confere_unidades():
    dados = {"geracao_mes_kwh": 534.2, "economia_mes_rs": 480.78}
    ok = tarefas.verificar_numeros("Peça 3 orçamentos em 2026. Gera 534 kWh e economiza R$ 480,78.", dados)
    assert ok["confiavel"] and ok["numeros_citados"] == [480.78, 534.0]
    ruim = tarefas.verificar_numeros("Retorno em 4 anos e 15% de desconto.", dados)
    assert ruim["nao_encontrados"] == [4.0, 15.0]


def test_recuperacao_usa_metadados():
    corpus = repositorio.carregar_corpus_llm()
    curitiba = llm_prep.buscar_contexto(corpus, "Quanto gera um sistema em curitiba?", 3)
    assert curitiba.iloc[0]["cidade"] == "Curitiba (PR)" and len(curitiba) == 3
    comparativa = llm_prep.buscar_contexto(corpus, "Qual capital tem a maior irradiação solar?", 4)
    assert comparativa.iloc[0]["tipo"] == "ranking"
    assert "Recife (PE)" in comparativa.iloc[0]["texto"].split("1º ")[1][:20]


def test_corpus_tem_ranking_ordenado():
    corpus = repositorio.carregar_corpus_llm()
    [ranking] = corpus.loc[corpus["tipo"] == "ranking", "texto"]
    assert ranking.index("1º") < ranking.index("10º")


# ---------------------------------------------------------------------------
# Modelo real (opcional)
# ---------------------------------------------------------------------------
@pytest.mark.llm
@pytest.mark.skipif(os.environ.get("SOLAR_TESTAR_LLM") != "1", reason="defina SOLAR_TESTAR_LLM=1 para usar o modelo real")
def test_modelo_real_de_ponta_a_ponta():
    """Carrega o modelo de SOLAR_LLM_MODELO (do cache local) e usa as rotas de verdade."""
    s = servico()
    s.config = ConfigLLM.do_ambiente()
    s.usar_backend(None)
    tarefas.limpar_cache()
    try:
        s.carregar()
        resumo = client.post("/processar_texto", json={"texto": NOTICIA, "max_palavras": 30}).json()
        assert resumo["resultado"]["n_palavras_resumo"] > 3
        sent = client.post("/processar_texto", json={"texto": NOTICIA, "tarefa": "sentimento", "ponto_de_vista": "setor_solar"}).json()
        assert sent["resultado"]["sentimento"] == "positivo"
        resposta = client.post("/llm/perguntar", json={"pergunta": "Qual capital tem a maior irradiação solar?"}).json()
        assert "Recife" in resposta["resposta"]
    finally:
        s.usar_backend(None)
        tarefas.limpar_cache()
