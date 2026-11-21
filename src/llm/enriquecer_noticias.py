"""Enriquecimento das notícias coletadas com o LLM local, em lote (TP5).

Para cada notícia da ABSOLAR e do Canal Solar (coletadas com Beautiful Soup), o modelo
local gera um resumo curto e classifica o sentimento do ponto de vista de quem quer
instalar energia solar, com a probabilidade de cada rótulo e uma justificativa.

Por que em lote, e não a cada acesso ao painel: as notícias mudam só quando os scrapers
rodam de novo, então processá-las uma vez e guardar o resultado evita repetir o mesmo
trabalho do modelo a cada visita (o painel abre na hora e o modelo fica livre para o
agente e o chat). O resultado alimenta o painel (sentimento do setor) e a ferramenta
`noticias_setor` do agente consultor.

Uso (na raiz do projeto, com o modelo no cache local):
    python -m src.llm.enriquecer_noticias              # processa só as notícias novas
    python -m src.llm.enriquecer_noticias --refazer    # processa tudo de novo

Saída: data/processed/noticias_ia.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

import pandas as pd

from src import repositorio
from src.llm import erros, tarefas
from src.llm.servico import ServicoLLM

SAIDA = repositorio.PROCESSED_DIR / "noticias_ia.csv"
MAX_PALAVRAS_ENTRADA = 900  # cabe com folga no limite de tokens de entrada do modelo
COLUNAS = [
    "url", "fonte", "titulo", "data", "n_palavras", "resumo_ia", "sentimento", "confianca",
    "p_positivo", "p_neutro", "p_negativo", "justificativa", "modelo", "tempo_s", "processado_em",
]

log = logging.getLogger("solar_viavel.enriquecer")


def texto_da_noticia(row) -> str:
    corpo = row.texto if isinstance(row.texto, str) and row.texto.strip() else row.resumo
    palavras = f"{row.titulo}. {corpo}".split()
    return " ".join(palavras[:MAX_PALAVRAS_ENTRADA])


def enriquecer(servico: ServicoLLM, noticias: pd.DataFrame, anteriores: pd.DataFrame | None = None) -> pd.DataFrame:
    feitas = set(anteriores["url"]) if anteriores is not None and not anteriores.empty else set()
    linhas = []
    pendentes = noticias[~noticias["url"].isin(feitas)]
    for i, row in enumerate(pendentes.itertuples(), start=1):
        texto = texto_da_noticia(row)
        inicio = time.perf_counter()
        try:
            resumo = tarefas.resumir(servico, texto, max_palavras=60)["resultado"]["resumo"]
            sent = tarefas.classificar_sentimento(servico, texto, "setor_solar", justificar=True)["resultado"]
        except erros.ErroLLM as exc:
            log.warning("Pulando '%s': %s", row.titulo, exc.mensagem)
            continue
        linhas.append(
            {
                "url": row.url,
                "fonte": row.fonte,
                "titulo": row.titulo,
                "data": row.data.date().isoformat() if pd.notna(row.data) else "",
                "n_palavras": len(texto.split()),
                "resumo_ia": resumo,
                "sentimento": sent["sentimento"],
                "confianca": sent["confianca"],
                "p_positivo": sent["probabilidades"]["positivo"],
                "p_neutro": sent["probabilidades"]["neutro"],
                "p_negativo": sent["probabilidades"]["negativo"],
                "justificativa": sent["justificativa"],
                "modelo": servico.config.modelo,
                "tempo_s": round(time.perf_counter() - inicio, 2),
                "processado_em": datetime.now().isoformat(timespec="seconds"),
            }
        )
        log.info("[%d/%d] %s -> %s (%.0f%%)", i, len(pendentes), row.titulo[:60], sent["sentimento"], sent["confianca"] * 100)
    novas = pd.DataFrame(linhas, columns=COLUNAS)
    if anteriores is not None and not anteriores.empty:
        novas = pd.concat([anteriores, novas], ignore_index=True)
    return novas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refazer", action="store_true", help="processa de novo todas as notícias")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    servico = ServicoLLM()
    servico.carregar()
    anteriores = None if args.refazer or not SAIDA.exists() else pd.read_csv(SAIDA, keep_default_na=False)
    inicio = time.perf_counter()
    resultado = enriquecer(servico, repositorio.carregar_noticias(), anteriores)
    resultado.to_csv(SAIDA, index=False)
    log.info("%d notícias em %s (%.0f s)", len(resultado), SAIDA, time.perf_counter() - inicio)
    log.info("Sentimento: %s", resultado["sentimento"].value_counts().to_dict())


if __name__ == "__main__":
    main()
