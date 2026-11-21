"""Registra chamadas reais à API com o LLM carregado (evidência pedida no feedback do TP4).

Os testes automatizados usam um modelo falso (rápidos e determinísticos). Este script faz
o contrário: chama a API **rodando de verdade, com o Qwen3-1.7B carregado**, e salva cada
requisição e resposta, com horário, tempo de resposta e uso de tokens, em:

- docs/evidencias/chamadas_reais_api.md   (relatório legível)
- docs/evidencias/chamadas_reais_api.json (requisições e respostas completas)

Uso (com a API no ar em outro terminal: `uvicorn src.api.main:app`):
    python scripts/registrar_chamadas_reais.py
    SOLAR_API_URL=http://127.0.0.1:8020 python scripts/registrar_chamadas_reais.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

URL = os.environ.get("SOLAR_API_URL", "http://127.0.0.1:8000").rstrip("/")
SAIDA = Path(__file__).resolve().parents[1] / "docs" / "evidencias"
NOTICIA = (
    "A fonte solar respondeu por 73% da expansão da matriz elétrica brasileira entre janeiro e agosto de 2026, "
    "com 3,5 GW adicionados. A capacidade total de geração centralizada do país já supera 220 GW, segundo a ANEEL. "
    "Para a ABSOLAR, o resultado mostra a competitividade da fonte, mas o setor alerta para os cortes de geração "
    "(curtailment) que reduzem a receita das usinas no Nordeste."
)

registros: list[dict] = []


def chamar(titulo: str, metodo: str, caminho: str, corpo: dict | None = None, params: dict | None = None) -> dict:
    inicio = time.perf_counter()
    quando = datetime.now().isoformat(timespec="seconds")
    r = requests.request(metodo, URL + caminho, json=corpo, params=params, timeout=600)
    tempo = time.perf_counter() - inicio
    resposta = r.json()
    registros.append(
        {
            "titulo": titulo,
            "quando": quando,
            "requisicao": {"metodo": metodo, "url": URL + caminho, "params": params, "corpo": corpo},
            "status": r.status_code,
            "tempo_total_s": round(tempo, 2),
            "resposta": resposta,
        }
    )
    print(f"{quando}  {metodo} {caminho} -> HTTP {r.status_code} em {tempo:.1f} s  ({titulo})")
    return resposta


def esperar_modelo(limite_s: int = 300) -> dict:
    inicio = time.time()
    while time.time() - inicio < limite_s:
        st = requests.get(URL + "/llm/status", timeout=10).json()
        if st["estado"] == "pronto":
            return st
        if st["estado"] in ("erro", "desativado"):
            sys.exit(f"O modelo não está disponível: {st['estado']} ({st.get('erro')})")
        if st["estado"] == "nao_carregado":
            requests.post(URL + "/llm/carregar", timeout=10)
        print(f"Aguardando o modelo carregar ({st['estado']})...")
        time.sleep(3)
    sys.exit("O modelo não carregou a tempo.")


def bloco_json(dados, limite: int = 2500) -> str:
    texto = json.dumps(dados, ensure_ascii=False, indent=2)
    if len(texto) > limite:
        texto = texto[:limite] + "\n... (resposta completa em chamadas_reais_api.json)"
    return f"```json\n{texto}\n```"


def resumo_uso(resp: dict) -> str:
    uso, modelo = resp.get("uso"), resp.get("modelo")
    if not uso or not modelo:
        return ""
    return (
        f"Modelo **{modelo['nome']}** em **{modelo['dispositivo']}** ({modelo['dtype']}), "
        f"{uso['tokens_entrada']} tokens de entrada, {uso['tokens_saida']} de saída, "
        f"{uso['tempo_s']} s de processamento{' (cache)' if uso.get('em_cache') else ''}."
    )


def escrever(status: dict) -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "chamadas_reais_api.json").write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")
    linhas = [
        "# Chamadas reais à API com o modelo carregado",
        "",
        "Gerado por `scripts/registrar_chamadas_reais.py` em "
        f"{datetime.now():%d/%m/%Y %H:%M}, chamando a API **em execução** (`{URL}`) com o LLM local carregado. "
        "Diferente dos testes automatizados (que usam um modelo falso), aqui todas as respostas vieram do "
        "modelo real. As respostas completas estão em [chamadas_reais_api.json](chamadas_reais_api.json).",
        "",
        "## Ambiente",
        "",
        f"- Máquina: {platform.system()} {platform.release()} ({platform.machine()}), Python {platform.python_version()}",
        f"- Modelo: **{status['modelo']}**, {status['parametros_milhoes']} milhões de parâmetros, "
        f"dispositivo `{status['dispositivo']}`, `{status['dtype']}`",
        f"- Carregado em {status['tempo_carregamento_s']} s; memória do modelo: {status['memoria_modelo_mb']} MB",
        "",
        "## Resumo das chamadas",
        "",
        "| # | Chamada | Rota | HTTP | Tempo total |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(registros, start=1):
        caminho = r["requisicao"]["url"].replace(URL, "")
        linhas.append(f"| {i} | {r['titulo']} | `{r['requisicao']['metodo']} {caminho}` | {r['status']} | {r['tempo_total_s']} s |")
    for i, r in enumerate(registros, start=1):
        req = r["requisicao"]
        caminho = req["url"].replace(URL, "")
        linhas += ["", f"## {i}. {r['titulo']}", "", f"`{req['metodo']} {caminho}` em {r['quando']}, HTTP {r['status']}, {r['tempo_total_s']} s."]
        if req["corpo"]:
            linhas += ["", "Requisição:", "", bloco_json(req["corpo"])]
        if req["params"]:
            linhas += ["", "Parâmetros:", "", bloco_json(req["params"])]
        resp = r["resposta"]
        uso = resumo_uso(resp) if isinstance(resp, dict) else ""
        if uso:
            linhas += ["", uso]
        destaque = r.get("destaque")
        if destaque:
            linhas += ["", destaque]
        linhas += ["", "Resposta:", "", bloco_json(resp)]
    (SAIDA / "chamadas_reais_api.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"Evidências salvas em {SAIDA}")


def main() -> None:
    status = esperar_modelo()
    chamar("Estado do modelo", "GET", "/llm/status")

    r = chamar("Resumo de uma notícia", "POST", "/processar_texto", {"texto": NOTICIA, "tarefa": "resumo", "max_palavras": 40})
    registros[-1]["destaque"] = f"**Resumo gerado:** {r['resultado']['resumo']}"
    r = chamar("Sentimento para quem quer instalar", "POST", "/processar_texto",
               {"texto": NOTICIA, "tarefa": "sentimento", "ponto_de_vista": "setor_solar"})
    registros[-1]["destaque"] = f"**Sentimento:** {r['resultado']['sentimento']} ({r['resultado']['confianca']:.0%}). {r['resultado']['justificativa']}"

    rec = chamar(
        "Agente consultor (Curitiba, preocupação com o inverno)",
        "POST",
        "/agente/recomendar",
        {"cidade": "Curitiba (PR)", "consumo_kwh_mes": 250, "tarifa": 0.92, "ligacao": "bifasica",
         "objetivo": "Me preocupa o inverno: a geração cai muito? Vale a pena mesmo assim?"},
    )
    passos = ", depois ".join(p["acao"] for p in rec["passos"])
    registros[-1]["destaque"] = (
        f"**Passos escolhidos pelo agente:** {passos}. **Viabilidade:** {rec['classificacao']}. "
        f"**Números conferidos:** {'todos' if rec['verificacao']['confiavel'] else 'não: ' + str(rec['verificacao']['nao_encontrados'])}.\n\n"
        f"**Recomendação escrita pelo modelo:**\n\n{rec['recomendacao']}"
    )

    conversa = chamar("Nova conversa sobre a recomendação", "POST", "/conversas", {"recomendacao_id": rec["id"]})
    for pergunta in ["Quanto o sistema gera no pior mês?", "E em Fortaleza, como seria?", "Qual das duas cidades tem mais sol?"]:
        m = chamar(f"Mensagem com memória: \"{pergunta}\"", "POST", f"/conversas/{conversa['id']}/mensagens", {"pergunta": pergunta})
        registros[-1]["destaque"] = (
            f"**Pergunta entendida como:** {m['pergunta_reescrita']} ({m['metodo_reescrita']}). "
            f"**Fatos lembrados:** {json.dumps(m['memoria']['fatos'], ensure_ascii=False)}.\n\n**Resposta:** {m['resposta']}"
        )
    chamar("Histórico e memória da conversa", "GET", f"/conversas/{conversa['id']}")
    chamar("Painel (ciclo de ciência de dados)", "GET", "/painel", params={"consumo_kwh_mes": 300, "tarifa": 0.95})
    escrever(status)


if __name__ == "__main__":
    main()
