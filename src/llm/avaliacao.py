"""Comparação dos modelos candidatos, executados localmente, nas tarefas reais do projeto.

Mede, para cada modelo, o custo computacional (tempo de carga, memória, velocidade de
geração) e o desempenho em cada tarefa, usando o mesmo código das rotas da API
(`src/llm/tarefas.py`):

- **sentimento**: 34 notícias rotuladas à mão (data/avaliacao/sentimento_noticias.csv),
  acurácia e F1 macro, do ponto de vista do setor solar;
- **resumo**: 8 matérias; respeito ao limite de palavras, saída em português e
  fidelidade numérica (os números do resumo existem no texto original?);
- **perguntas e respostas**: 12 perguntas (data/avaliacao/perguntas.csv) com a resposta
  esperada, incluindo 2 fora do escopo (o modelo deve dizer que não sabe); mede
  acerto, uso de citações [n] e recusa correta;
- **relatório**: 3 capitais; proporção de relatórios cujos números conferem com a API.

Também avalia um classificador de sentimento do tipo BERT (XLM-RoBERTa) como linha de
base para a tarefa de classificação.

Uso (na raiz do projeto; cada modelo roda em um processo separado para medir a memória):
    python -m src.llm.avaliacao                      # todos os candidatos
    python -m src.llm.avaliacao --modelos Qwen/Qwen2.5-0.5B-Instruct
Resultados: docs/avaliacao/resultados_modelos.json e docs/avaliacao/resultados_modelos.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "data" / "avaliacao"
SAIDA = RAIZ / "docs" / "avaliacao"

CANDIDATOS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3-4B-Instruct-2507",
]
BASELINE_SENTIMENTO = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

NOTICIAS_RESUMO = [0, 7, 12, 19, 24, 40, 47, 52]  # posições em repositorio.carregar_noticias()
CIDADES_RELATORIO = ["Curitiba (PR)", "Recife (PE)", "Manaus (AM)"]
MAX_PALAVRAS_TEXTO = 400  # notícias cortadas para a avaliação caber em tempo razoável

PALAVRAS_PT = {"de", "que", "o", "a", "do", "da", "em", "para", "com", "não", "uma", "os", "no", "na", "por", "mais", "as", "dos", "é", "ao"}
PALAVRAS_EN = {"the", "of", "and", "to", "in", "is", "that", "for", "with", "on", "are", "this", "by", "from"}
RECUSA = re.compile(r"não (encontr\w+|há|tenho|sei|consta|possuo|existe|está|consegui)|nenhuma informação|não (é|foi) possível", re.I)


def _f1_macro(reais: list[str], previstos: list[str]) -> float:
    f1s = []
    for rotulo in sorted(set(reais)):
        vp = sum(r == p == rotulo for r, p in zip(reais, previstos))
        fp = sum(p == rotulo != r for r, p in zip(reais, previstos))
        fn = sum(r == rotulo != p for r, p in zip(reais, previstos))
        precisao = vp / (vp + fp) if vp + fp else 0.0
        revocacao = vp / (vp + fn) if vp + fn else 0.0
        f1s.append(2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0)
    return sum(f1s) / len(f1s)


def _em_portugues(texto: str) -> bool:
    palavras = re.findall(r"\w+", texto.lower())
    return sum(p in PALAVRAS_PT for p in palavras) > 2 * sum(p in PALAVRAS_EN for p in palavras)


def _texto_noticia(row) -> str:
    corpo = " ".join(str(row.texto or row.resumo).split()[:MAX_PALAVRAS_TEXTO])
    return f"{row.titulo}. {corpo}"


def _memoria_gb(servico) -> float | None:
    mb = servico.backend.memoria_mb() if hasattr(servico.backend, "memoria_mb") else None
    if mb is None:
        import psutil

        mb = psutil.Process(os.getpid()).memory_info().rss / 2**20
    return round(mb / 1024, 2)


def avaliar_llm(modelo: str) -> dict:
    from fastapi.testclient import TestClient

    from src import repositorio
    from src.api.main import app
    from src.llm import tarefas
    from src.llm.config import ConfigLLM
    from src.llm.servico import ServicoLLM

    servico = ServicoLLM(ConfigLLM(modelo=modelo, permitir_download=False, timeout_s=300, max_tokens_entrada=6000))
    inicio = time.perf_counter()
    servico.carregar()
    carga_s = time.perf_counter() - inicio
    status = servico.status()
    noticias = repositorio.carregar_noticias()
    corpus = repositorio.carregar_corpus_llm()
    exemplos = {}

    # Sentimento -------------------------------------------------------------
    rotulos = pd.read_csv(DADOS / "sentimento_noticias.csv")
    por_url = noticias.set_index("url")
    previstos, tempos = [], []
    for r in rotulos.itertuples():
        t = time.perf_counter()
        res = tarefas.classificar_sentimento(servico, _texto_noticia(por_url.loc[r.url]), "setor_solar", justificar=False)
        tempos.append(time.perf_counter() - t)
        previstos.append(res["resultado"]["sentimento"])
    sentimento = {
        "acuracia": round(sum(p == r for p, r in zip(previstos, rotulos["rotulo"])) / len(rotulos), 3),
        "f1_macro": round(_f1_macro(list(rotulos["rotulo"]), previstos), 3),
        "tempo_medio_s": round(sum(tempos) / len(tempos), 2),
        "erros": [
            {"titulo": t, "esperado": r, "previsto": p}
            for t, r, p in zip(rotulos["titulo"], rotulos["rotulo"], previstos)
            if p != r
        ],
    }

    # Resumo -------------------------------------------------------------------
    linhas = []
    for i in NOTICIAS_RESUMO:
        row = noticias.iloc[i]
        texto = _texto_noticia(row)
        res = tarefas.resumir(servico, texto, max_palavras=60)
        resumo = res["resultado"]["resumo"]
        numeros = tarefas.extrair_numeros(resumo)
        fieis = [n for n in numeros if tarefas._confere(n, set(tarefas.extrair_numeros(texto)))]
        linhas.append(
            {
                "palavras": res["resultado"]["n_palavras_resumo"],
                "no_limite": res["resultado"]["n_palavras_resumo"] <= 69,
                "portugues": _em_portugues(resumo),
                "numeros_fieis": len(fieis) / len(numeros) if numeros else 1.0,
                "tempo_s": res["uso"]["tempo_s"],
                "tokens_por_s": res["uso"]["tokens_por_s"] or 0,
            }
        )
        if i == NOTICIAS_RESUMO[1]:
            exemplos["resumo"] = {"titulo": row.titulo, "resumo": resumo}
    df = pd.DataFrame(linhas)
    resumo_m = {
        "palavras_media": round(df["palavras"].mean(), 1),
        "no_limite": round(df["no_limite"].mean(), 3),
        "em_portugues": round(df["portugues"].mean(), 3),
        "numeros_fieis": round(df["numeros_fieis"].mean(), 3),
        "tempo_medio_s": round(df["tempo_s"].mean(), 2),
        "tokens_por_s": round(df["tokens_por_s"].mean(), 1),
    }

    # Perguntas e respostas -------------------------------------------------------------
    perguntas = pd.read_csv(DADOS / "perguntas.csv")
    linhas = []
    for p in perguntas.itertuples():
        res = tarefas.responder_pergunta(servico, corpus, p.pergunta, k=4)
        resposta = res["resposta"]
        if p.esperado == "RECUSA":
            # recusa correta: diz que não sabe e não traz conteúdo do corpus como se fosse resposta
            acerto = bool(RECUSA.search(resposta)) and len(resposta.split()) <= 40
        else:
            acerto = any(e.lower() in resposta.lower() for e in p.esperado.split("|"))
        linhas.append(
            {
                "pergunta": p.pergunta,
                "tipo": p.tipo,
                "acerto": acerto,
                "citou": bool(re.search(r"\[\d+\]", resposta)),
                "portugues": _em_portugues(resposta),
                "tempo_s": res["uso"]["tempo_s"],
                "resposta": resposta,
            }
        )
    df = pd.DataFrame(linhas)
    dentro = df[df["tipo"] != "fora_do_escopo"]
    qa = {
        "acerto": round(df["acerto"].mean(), 3),
        "acerto_no_escopo": round(dentro["acerto"].mean(), 3),
        "recusa_correta": round(df.loc[df["tipo"] == "fora_do_escopo", "acerto"].mean(), 3),
        "citou_fonte": round(dentro["citou"].mean(), 3),
        "em_portugues": round(df["portugues"].mean(), 3),
        "tempo_medio_s": round(df["tempo_s"].mean(), 2),
        "respostas": df[["pergunta", "acerto", "resposta"]].to_dict("records"),
    }

    # Relatório -------------------------------------------------------------------------------
    client = TestClient(app)
    linhas = []
    for cidade in CIDADES_RELATORIO:
        est = client.get("/estimativa", params={"cidade": cidade, "potencia_kwp": 4, "tarifa": 0.95}).json()
        [lp] = client.get("/irradiacao-longo-prazo", params={"cidade": cidade, "plano": "Maior média anual"}).json()
        ficha = corpus.loc[(corpus["tipo"] == "ficha_local") & (corpus["cidade"] == cidade), "texto"].iloc[0]
        res = tarefas.gerar_relatorio(servico, est, lp["mensal"], ficha)
        v = res["verificacao"]
        linhas.append(
            {
                "confiavel": v["confiavel"],
                "pct_numeros_ok": 1 - len(v["nao_encontrados"]) / len(v["numeros_citados"]) if v["numeros_citados"] else 1.0,
                "secoes": res["relatorio"].count("## "),
                "tempo_s": res["uso"]["tempo_s"],
            }
        )
        if cidade == CIDADES_RELATORIO[0]:
            exemplos["relatorio"] = {"cidade": cidade, "relatorio": res["relatorio"], "verificacao": v}
    df = pd.DataFrame(linhas)
    relatorio = {
        "relatorios_confiaveis": round(df["confiavel"].mean(), 3),
        "numeros_ok": round(df["pct_numeros_ok"].mean(), 3),
        "secoes_media": round(df["secoes"].mean(), 1),
        "tempo_medio_s": round(df["tempo_s"].mean(), 2),
    }

    return {
        "modelo": modelo,
        "parametros_milhoes": status["parametros_milhoes"],
        "dispositivo": status["dispositivo"],
        "dtype": status["dtype"],
        "carga_s": round(carga_s, 1),
        "memoria_gb": _memoria_gb(servico),
        "sentimento": sentimento,
        "resumo": resumo_m,
        "qa": qa,
        "relatorio": relatorio,
        "exemplos": exemplos,
    }


def avaliar_baseline_sentimento(modelo: str = BASELINE_SENTIMENTO) -> dict:
    """Classificador do tipo BERT (XLM-RoBERTa) treinado para sentimento (tom geral, não do setor)."""
    import torch
    from transformers import AutoTokenizer, pipeline

    from src import repositorio

    from src.llm.servico import escolher_dispositivo

    inicio = time.perf_counter()
    # o repositório do modelo só traz o tokenizador "lento" (sentencepiece), que o Transformers 5
    # não converte; o vocabulário é o mesmo do XLM-RoBERTa base, então usamos o dele.
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    clf = pipeline("text-classification", model=modelo, tokenizer=tokenizer, device=escolher_dispositivo(), truncation=True, max_length=512)
    carga_s = time.perf_counter() - inicio
    mapa = {"positive": "positivo", "neutral": "neutro", "negative": "negativo"}
    noticias = repositorio.carregar_noticias().set_index("url")
    rotulos = pd.read_csv(DADOS / "sentimento_noticias.csv")
    t = time.perf_counter()
    previstos = [mapa[clf(_texto_noticia(noticias.loc[u]))[0]["label"].lower()] for u in rotulos["url"]]
    tempo = (time.perf_counter() - t) / len(rotulos)
    parametros = sum(p.numel() for p in clf.model.parameters())
    del clf
    gc.collect()
    return {
        "modelo": modelo,
        "parametros_milhoes": round(parametros / 1e6, 1),
        "carga_s": round(carga_s, 1),
        "acuracia": round(sum(p == r for p, r in zip(previstos, rotulos["rotulo"])) / len(rotulos), 3),
        "f1_macro": round(_f1_macro(list(rotulos["rotulo"]), previstos), 3),
        "tempo_medio_s": round(tempo, 3),
        "torch": torch.__version__,
    }


def _tabela_markdown(resultados: list[dict], baseline: dict | None) -> str:
    linhas = [
        "| Modelo | Parâm. (M) | Carga (s) | Memória (GB) | Tokens/s | Sentimento acc / F1 | Resumo no limite / PT / nº fiéis | Q&A acerto / recusa / cita | Relatório números OK |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in resultados:
        s, m, q, rel = r["sentimento"], r["resumo"], r["qa"], r["relatorio"]
        linhas.append(
            f"| `{r['modelo']}` | {r['parametros_milhoes']:,.0f} | {r['carga_s']} | {r['memoria_gb']} | {m['tokens_por_s']} | "
            f"{s['acuracia']:.0%} / {s['f1_macro']:.2f} | {m['no_limite']:.0%} / {m['em_portugues']:.0%} / {m['numeros_fieis']:.0%} | "
            f"{q['acerto_no_escopo']:.0%} / {q['recusa_correta']:.0%} / {q['citou_fonte']:.0%} | {rel['numeros_ok']:.0%} |"
        )
    if baseline:
        linhas.append(
            f"| `{baseline['modelo']}` (só sentimento) | {baseline['parametros_milhoes']:,.0f} | {baseline['carga_s']} | n/d | n/d | "
            f"{baseline['acuracia']:.0%} / {baseline['f1_macro']:.2f} | n/a | n/a | n/a |"
        )
    return "\n".join(linhas).replace(",", ".")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelos", nargs="*", default=CANDIDATOS)
    parser.add_argument("--um", help=argparse.SUPPRESS)  # uso interno: avalia um modelo e imprime JSON
    parser.add_argument("--baseline", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.um:
        print("@@JSON@@" + json.dumps(avaliar_llm(args.um), ensure_ascii=False, default=str))
        return
    if args.baseline:
        print("@@JSON@@" + json.dumps(avaliar_baseline_sentimento(), ensure_ascii=False))
        return

    SAIDA.mkdir(parents=True, exist_ok=True)
    arquivo = SAIDA / "resultados_modelos.json"
    anteriores = json.loads(arquivo.read_text()) if arquivo.exists() else {"modelos": [], "baseline": None}
    resultados = {r["modelo"]: r for r in anteriores["modelos"]}
    ambiente = {**os.environ, "HF_HUB_OFFLINE": "1", "SOLAR_LLM_CARREGAR": "0"}

    def rodar(*extra) -> dict:
        proc = subprocess.run([sys.executable, "-m", "src.llm.avaliacao", *extra], cwd=RAIZ, env=ambiente, capture_output=True, text=True)
        saida = [linha for linha in proc.stdout.splitlines() if linha.startswith("@@JSON@@")]
        if proc.returncode or not saida:
            raise RuntimeError(f"Falha ao avaliar {extra}:\n{proc.stderr[-3000:]}")
        return json.loads(saida[-1][len("@@JSON@@"):])

    for modelo in args.modelos:
        print(f"Avaliando {modelo}...", flush=True)
        resultados[modelo] = rodar("--um", modelo)
        print(json.dumps({k: v for k, v in resultados[modelo].items() if k != "exemplos"}, ensure_ascii=False)[:600], flush=True)
    baseline = anteriores.get("baseline")
    if baseline is None or BASELINE_SENTIMENTO in args.modelos or args.modelos == CANDIDATOS:
        print(f"Avaliando a linha de base {BASELINE_SENTIMENTO}...", flush=True)
        baseline = rodar("--baseline")

    ordem = [m for m in CANDIDATOS if m in resultados] + [m for m in resultados if m not in CANDIDATOS]
    final = {"gerado_em": time.strftime("%Y-%m-%d %H:%M"), "modelos": [resultados[m] for m in ordem], "baseline": baseline}
    arquivo.write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str))
    (SAIDA / "resultados_modelos.md").write_text(_tabela_markdown(final["modelos"], baseline) + "\n")
    print(_tabela_markdown(final["modelos"], baseline))


if __name__ == "__main__":
    main()
