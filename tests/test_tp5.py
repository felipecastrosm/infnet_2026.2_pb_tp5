"""Testes do TP5: análise de viabilidade, agente consultor, conversa com memória e painel.

Os cálculos (viabilidade.py) são testados com os dados reais coletados. O agente e a
memória usam o backend falso de conftest.py, que permite controlar cada decisão do modelo
(as probabilidades das ações) e cada texto gerado.
"""

import pytest
from fastapi.testclient import TestClient

from src import repositorio
from src.api import schemas
from src.api.main import app
from src.data_processing import viabilidade as v
from src.llm import agente, memoria

client = TestClient(app)
SUNDATA = repositorio.carregar_sundata()
CLIMA = repositorio.carregar_clima_base()
CORPUS = repositorio.carregar_corpus_llm()


# ---------------------------------------------------------------------------
# Análise de viabilidade (ferramentas do agente)
# ---------------------------------------------------------------------------
def test_dimensionamento_cobre_o_consumo_descontando_o_minimo():
    d = v.dimensionar_sistema(SUNDATA, "Recife (PE)", 350, "bifasica")
    assert d["compensavel_kwh_mes"] == 300  # 350 - 50 kWh de custo de disponibilidade
    assert d["potencia_kwp"] == pytest.approx(d["modulos"] * v.POTENCIA_MODULO_KWP)
    assert d["potencia_kwp"] >= d["potencia_necessaria_kwp"] > d["potencia_kwp"] - v.POTENCIA_MODULO_KWP
    assert d["geracao_kwh_mes"] >= d["compensavel_kwh_mes"]
    menor = v.dimensionar_sistema(SUNDATA, "Recife (PE)", 350, "trifasica")
    assert menor["potencia_necessaria_kwp"] < d["potencia_necessaria_kwp"]


def test_dimensionamento_com_potencia_informada():
    d = v.dimensionar_sistema(SUNDATA, "Curitiba (PR)", 300, potencia_kwp=4.0)
    assert d["potencia_kwp"] == 4.0 and d["origem_potencia"] == "informada pelo usuário"


def test_financeira_payback_fio_b_e_classificacao():
    d = v.dimensionar_sistema(SUNDATA, "Fortaleza (CE)", 400)
    f = v.analise_financeira(d, 0.95)
    fluxo = f["fluxo_caixa"]
    assert fluxo[0]["acumulado_rs"] == -f["investimento_rs"] and len(fluxo) == v.VIDA_UTIL_ANOS + 1
    ano_virada = next(x["ano"] for x in fluxo if x["acumulado_rs"] >= 0)
    assert ano_virada - 1 < f["payback_anos"] <= ano_virada
    # o Fio B reduz o valor do kWh injetado e cresce até 100% em 2029
    assert v.valor_kwh_compensado(1.0, 2027) > v.valor_kwh_compensado(1.0, 2029) == pytest.approx(1 - 0.7 * v.PARCELA_FIO_B)
    assert f["classificacao"] == v.classificar(f["payback_anos"])
    assert v.classificar(3) == "excelente" and v.classificar(4.5) == "boa" and v.classificar(7) == "moderada"
    assert v.classificar(12) == "baixa" and v.classificar(None) == "baixa"
    caro = v.analise_financeira(d, 0.95, custo_kwp=9000)
    assert caro["payback_anos"] > f["payback_anos"]


def test_sazonalidade_e_impacto():
    d = v.dimensionar_sistema(SUNDATA, "Porto Alegre (RS)", 300)
    s = v.sazonalidade(SUNDATA, "Porto Alegre (RS)", d)
    assert len(s["meses"]) == 12 and "jun" in s["meses_com_deficit"]
    assert s["maior_geracao_kwh"] > s["menor_geracao_kwh"]
    i = v.impacto_ambiental(d)
    assert i["co2_evitado_kg_ano"] == pytest.approx(d["geracao_kwh_ano"] * v.FATOR_EMISSAO_T_MWH, abs=1)


def test_comparar_capitais_ordena_pelo_payback():
    b = v.comparar_capitais(SUNDATA, 300, 0.95, cidade="Curitiba (PR)")
    paybacks = [c["payback_anos"] for c in b["capitais"]]
    assert len(paybacks) == 10 and paybacks == sorted(paybacks)
    assert b["melhor"] == b["capitais"][0]["cidade"] and b["pior"] == b["capitais"][-1]["cidade"]
    assert 1 <= b["posicao_payback"] <= 10


# ---------------------------------------------------------------------------
# Agente consultor
# ---------------------------------------------------------------------------
PEDIDO = {"cidade": "Recife (PE)", "consumo_kwh_mes": 350, "tarifa": 0.95, "objetivo": "Quero saber o impacto ambiental"}


TEXTO_RECOMENDACAO = (
    "## Veredito\nViabilidade excelente: vale a pena instalar energia solar em Recife, porque o sistema se paga rápido.\n"
    "## Ações recomendadas\n1. Peça três orçamentos a instaladores.\n2. Confira o telhado.\n3. Acompanhe as regras da distribuidora."
)


def configurar_agente(backend, probabilidades, texto=TEXTO_RECOMENDACAO):
    backend.probabilidades = probabilidades
    backend.resposta = texto


def test_agente_segue_as_decisoes_do_modelo(llm):
    _, backend = llm
    configurar_agente(
        backend,
        {"dimensionar_sistema": 0.8, "impacto_ambiental": 0.9, "analise_financeira": 0.7, "finalizar": 0.5},
    )
    resp = client.post("/agente/recomendar", json=PEDIDO)
    assert resp.status_code == 200, resp.json()
    r = schemas.RespostaAgente.model_validate(resp.json())
    assert [p.acao for p in r.passos] == ["potencial_solar", "dimensionar_sistema", "impacto_ambiental", "analise_financeira", "finalizar"]
    assert r.passos[0].decisao.modo == "regra"  # só havia uma ação possível
    assert all(p.decisao.modo == "modelo" for p in r.passos[1:])
    assert set(r.resultados) == {"potencial_solar", "dimensionar_sistema", "impacto_ambiental", "analise_financeira"}
    assert r.recomendacao.startswith("## Veredito") and r.classificacao in ("excelente", "boa")
    # o objetivo do usuário e as observações entram no prompt de decisão
    decisoes = [c for c in backend.chamadas if c[0] == "pontuar"]
    assert "Quero saber o impacto ambiental" in decisoes[0][1][1]["content"]
    assert "Recife (PE) recebe em média" in decisoes[1][1][1]["content"]
    # a síntese recebe os fatos calculados e pede a seção de ODS (porque o impacto foi analisado)
    sintese = backend.chamadas[-1][1][1]["content"]
    assert "Fatos calculados" in sintese and "## Impacto ESG e ODS" in sintese


def test_agente_so_oferece_finalizar_depois_das_obrigatorias(llm):
    _, backend = llm
    configurar_agente(backend, {"finalizar": 0.99, "noticias_setor": 0.5, "dimensionar_sistema": 0.4, "analise_financeira": 0.3})
    r = client.post("/agente/recomendar", json=PEDIDO).json()
    acoes = [p["acao"] for p in r["passos"]]
    assert acoes[-1] == "finalizar"
    for obrigatoria in agente.OBRIGATORIAS:
        assert acoes.index(obrigatoria) < acoes.index("finalizar")
    primeira_decisao = [c for c in backend.chamadas if c[0] == "pontuar"][0][1][1]["content"]
    assert "- finalizar:" not in primeira_decisao  # ainda não estava disponível


def test_agente_confere_os_numeros_da_recomendacao(llm):
    _, backend = llm
    configurar_agente(backend, {"finalizar": 0.9}, TEXTO_RECOMENDACAO + "\nO investimento se paga em 99,9 anos e custa R$ 123.456,00.")
    r = client.post("/agente/recomendar", json=PEDIDO).json()
    assert r["verificacao"]["confiavel"] is False
    assert 99.9 in r["verificacao"]["nao_encontrados"]


def test_agente_cache_e_consulta_por_id(llm):
    _, backend = llm
    configurar_agente(backend, {"finalizar": 0.9})
    primeira = client.post("/agente/recomendar", json=PEDIDO).json()
    n_chamadas = len(backend.chamadas)
    segunda = client.post("/agente/recomendar", json=PEDIDO).json()
    assert segunda["uso"]["em_cache"] and len(backend.chamadas) == n_chamadas
    assert segunda["id"] != primeira["id"]
    assert client.get(f"/agente/recomendacoes/{primeira['id']}").json()["recomendacao"] == primeira["recomendacao"]


def test_agente_erros(llm):
    resp = client.post("/agente/recomendar", json={**PEDIDO, "cidade": "Gotham (XX)"})
    assert resp.status_code == 404 and resp.json()["erro"] == "nao_encontrado"
    resp = client.post("/agente/recomendar", json={**PEDIDO, "consumo_kwh_mes": -5})
    assert resp.status_code == 422
    assert client.get("/agente/recomendacoes/naoexiste").status_code == 404


def test_agente_sem_modelo_responde_503():
    from src.llm.servico import servico

    servico().usar_backend(None)
    resp = client.post("/agente/recomendar", json=PEDIDO)
    assert resp.status_code == 503 and resp.json()["erro"] == "modelo_nao_carregado"


def test_ferramentas_publicas():
    nomes = [f["nome"] for f in client.get("/agente/ferramentas").json()]
    assert nomes[-1] == "finalizar" and "comparar_capitais" in nomes


# ---------------------------------------------------------------------------
# Memória conversacional
# ---------------------------------------------------------------------------
def test_extrair_fatos():
    f = memoria.extrair_fatos(CORPUS, "Moro em Recife, consumo 350 kWh por mês, tarifa de R$ 0,95/kWh e quero 3,5 kWp")
    assert f == {"cidade": "Recife (PE)", "consumo_kwh_mes": 350.0, "tarifa": 0.95, "potencia_kwp": 3.5}
    assert "consumo_kwh_mes" not in memoria.extrair_fatos(CORPUS, "Manaus tem 4,5 kWh/m² por dia?")


def resposta_por_tarefa(reescrita="Quanto um sistema de 5 kWp gera por mês em Manaus?"):
    """Imita o modelo: reescreve quando recebe o prompt de reescrita, responde nos outros casos."""

    def _responder(mensagens):
        sistema = mensagens[0]["content"]
        if sistema.startswith("Você reescreve"):
            return reescrita
        if sistema.startswith("Você mantém um resumo"):
            return "O usuário comparou a geração de 5 kWp em Curitiba e Manaus."
        return "Gera cerca de 500 kWh por mês [1]."

    return _responder


def test_conversa_entende_pergunta_de_acompanhamento(llm):
    _, backend = llm
    backend.resposta = resposta_por_tarefa()
    conversa = client.post("/conversas").json()
    cid = conversa["id"]
    r1 = client.post(f"/conversas/{cid}/mensagens", json={"pergunta": "Quanto um sistema de 5 kWp gera por mês em Curitiba?"}).json()
    assert r1["metodo_reescrita"] == "original"
    assert r1["memoria"]["fatos"] == {"cidade": "Curitiba (PR)", "potencia_kwp": 5.0, "cidades_citadas": ["Curitiba (PR)"]}
    r2 = client.post(f"/conversas/{cid}/mensagens", json={"pergunta": "E em Manaus?"})
    corpo = schemas.RespostaMensagem.model_validate(r2.json())
    assert corpo.metodo_reescrita == "modelo"
    assert corpo.pergunta_reescrita == "Quanto um sistema de 5 kWp gera por mês em Manaus?"
    assert corpo.fontes[0].titulo == "Ficha de potencial solar: Manaus (AM)"  # a busca usou a pergunta reescrita
    assert corpo.memoria.fatos["cidade"] == "Manaus (AM)" and corpo.memoria.fatos["potencia_kwp"] == 5.0
    assert corpo.memoria.fatos["cidades_citadas"] == ["Curitiba (PR)", "Manaus (AM)"]
    # a troca anterior vai na íntegra para o modelo (janela) e os fatos entram no prompt de sistema
    resposta = [c for c in backend.chamadas if c[0] == "gerar"][-1][1]
    assert resposta[1] == {"role": "user", "content": "Quanto um sistema de 5 kWp gera por mês em Curitiba?"}
    assert "Fatos: cidade em foco: Manaus (AM); cidades já citadas: Curitiba (PR), Manaus (AM); potência do sistema: 5 kWp" in resposta[0]["content"]
    historico = client.get(f"/conversas/{cid}").json()
    assert [t["pergunta"] for t in historico["turnos"]] == ["Quanto um sistema de 5 kWp gera por mês em Curitiba?", "E em Manaus?"]


def test_reescrita_que_so_repete_a_pergunta_usa_regra(llm):
    _, backend = llm
    backend.resposta = resposta_por_tarefa(reescrita="E quanto economiza?")
    cid = client.post("/conversas").json()["id"]
    client.post(f"/conversas/{cid}/mensagens", json={"pergunta": "Quanto gera um sistema de 3 kWp em Salvador?"})
    r = client.post(f"/conversas/{cid}/mensagens", json={"pergunta": "E quanto economiza?"}).json()
    assert r["metodo_reescrita"] == "regra" and "Salvador (BA)" in r["pergunta_reescrita"]
    # o prompt de reescrita traz exemplos (few-shot) antes da pergunta real
    reescrita = [c for c in backend.chamadas if c[0] == "gerar" and c[1][0]["content"].startswith("Você reescreve")][-1][1]
    assert len(reescrita) == 2 + 2 * len(memoria.EXEMPLOS_REESCRITA)


def test_mudanca_de_assunto_nao_herda_a_cidade(llm):
    _, backend = llm
    backend.resposta = resposta_por_tarefa(reescrita="O que é o leilão de baterias?")
    cid = client.post("/conversas").json()["id"]
    client.post(f"/conversas/{cid}/mensagens", json={"pergunta": "Quanto gera um sistema de 3 kWp em Salvador?"})
    r = client.post(f"/conversas/{cid}/mensagens", json={"pergunta": "O que é o leilão de baterias?"}).json()
    assert r["pergunta_reescrita"] == "O que é o leilão de baterias?" and r["metodo_reescrita"] == "modelo"
    assert memoria.parece_acompanhamento("E em Manaus?") and memoria.parece_acompanhamento("Quanto gera lá?")
    assert not memoria.parece_acompanhamento("O que é o leilão de baterias?")


def test_reescrita_invalida_usa_regra(llm):
    _, backend = llm
    backend.resposta = resposta_por_tarefa(reescrita="")
    cid = client.post("/conversas").json()["id"]
    client.post(f"/conversas/{cid}/mensagens", json={"pergunta": "Quanto gera um sistema de 4 kWp em Fortaleza?"})
    r = client.post(f"/conversas/{cid}/mensagens", json={"pergunta": "E a economia do sistema?"}).json()
    assert r["metodo_reescrita"] == "regra"
    assert "Fortaleza (CE)" in r["pergunta_reescrita"] and "4 kWp" in r["pergunta_reescrita"]


def test_memoria_resume_o_que_sai_da_janela_e_persiste(llm):
    _, backend = llm
    backend.resposta = resposta_por_tarefa()
    cid = client.post("/conversas").json()["id"]
    perguntas = ["Qual capital tem mais sol?", "E a pior?", "Quanto gera em Curitiba?", "E em Recife?"]
    for p in perguntas:
        r = client.post(f"/conversas/{cid}/mensagens", json={"pergunta": p}).json()
    assert r["resumo_atualizado"] is True
    m = r["memoria"]
    assert m["total_turnos"] == 4 and m["turnos_resumidos"] == 1 and m["turnos_na_janela"] == memoria.JANELA
    assert m["resumo"].startswith("O usuário comparou")
    # outra instância do repositório (ex.: API reiniciada) lê a conversa do disco
    recarregada = memoria.RepositorioConversas().obter(cid)
    assert len(recarregada.turnos) == 4 and recarregada.resumo == m["resumo"]
    assert client.delete(f"/conversas/{cid}").json() == {"id": cid, "apagada": True}
    resp = client.get(f"/conversas/{cid}")
    assert resp.status_code == 404 and resp.json()["erro"] == "conversa_nao_encontrada"


def test_conversa_sobre_a_recomendacao_do_agente(llm):
    _, backend = llm
    configurar_agente(backend, {"finalizar": 0.9})
    rec = client.post("/agente/recomendar", json=PEDIDO).json()
    backend.resposta = resposta_por_tarefa(reescrita="Por que o payback da instalação em Recife é esse?")
    conversa = client.post("/conversas", json={"recomendacao_id": rec["id"]}).json()
    assert conversa["memoria"]["fatos"]["cidade"] == "Recife (PE)"
    r = client.post(f"/conversas/{conversa['id']}/mensagens", json={"pergunta": "Por que o payback é esse?"}).json()
    assert r["fontes"][0]["tipo"] == "recomendacao"
    assert client.post("/conversas", json={"recomendacao_id": "naoexiste"}).status_code == 404


def test_mensagem_valida_entrada(llm):
    cid = client.post("/conversas").json()["id"]
    assert client.post(f"/conversas/{cid}/mensagens", json={"pergunta": "?? 12"}).status_code == 422
    assert client.post("/conversas/naoexiste/mensagens", json={"pergunta": "E em Manaus?"}).status_code == 404


# ---------------------------------------------------------------------------
# Painel
# ---------------------------------------------------------------------------
def test_painel():
    p = schemas.Painel.model_validate(client.get("/painel", params={"consumo_kwh_mes": 250, "tarifa": 1.0}).json())
    assert [e.etapa for e in p.ciclo] == ["Coleta", "Processamento", "Análise", "IA", "Apresentação"]
    assert p.ciclo[0].indicadores["capitais"] == 10 and p.ciclo[0].indicadores["noticias"] > 0
    assert len(p.capitais) == 10 and p.perfil["consumo_kwh_mes"] == 250
    assert "custo_kwp" in p.premissas
    if p.noticias_processadas:
        assert sum(p.sentimento_noticias.values()) == p.noticias_processadas
    for n in client.get("/noticias/ia", params={"limite": 5}).json():
        schemas.NoticiaIA.model_validate(n)
