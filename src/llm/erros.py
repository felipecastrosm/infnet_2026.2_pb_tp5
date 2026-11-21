"""Exceções da integração com o LLM.

Cada exceção carrega o código HTTP adequado, um código curto (`erro`) e uma sugestão
para o usuário. A API (src/api/main.py) converte qualquer `ErroLLM` em uma resposta
JSON padronizada (`schemas.RespostaErro`), de modo que o app e outros clientes sempre
recebem o mesmo formato de erro.
"""

from __future__ import annotations

from typing import Optional


class ErroLLM(Exception):
    status_code = 500
    erro = "erro_llm"
    sugestao: Optional[str] = None

    def __init__(self, mensagem: str, *, sugestao: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(mensagem)
        self.mensagem = mensagem
        if sugestao is not None:
            self.sugestao = sugestao
        self.retry_after = retry_after  # segundos; vira o cabeçalho HTTP Retry-After


class ModeloDesativado(ErroLLM):
    """O LLM foi desligado por configuração (SOLAR_LLM_DESATIVADO=1) ou faltam bibliotecas."""

    status_code = 503
    erro = "modelo_desativado"
    sugestao = "Instale torch e transformers (requirements.txt) e não defina SOLAR_LLM_DESATIVADO."


class ModeloNaoCarregado(ErroLLM):
    status_code = 503
    erro = "modelo_nao_carregado"
    sugestao = "Carregue o modelo com POST /llm/carregar e acompanhe em GET /llm/status."


class ModeloCarregando(ErroLLM):
    status_code = 503
    erro = "modelo_carregando"
    sugestao = "O modelo está sendo carregado na memória. Tente de novo em alguns segundos."


class FalhaAoCarregarModelo(ErroLLM):
    status_code = 503
    erro = "falha_ao_carregar_modelo"
    sugestao = (
        "Confira o nome do modelo (SOLAR_LLM_MODELO), se ele já foi baixado "
        "(`hf download <modelo>`) e se há memória livre. Detalhes em GET /llm/status."
    )


class ModeloOcupado(ErroLLM):
    """A fila de inferência está cheia (o modelo atende uma requisição por vez)."""

    status_code = 503
    erro = "modelo_ocupado"
    sugestao = "O modelo está atendendo outras requisições. Tente de novo em instantes."


class MemoriaInsuficiente(ErroLLM):
    status_code = 503
    erro = "memoria_insuficiente"
    sugestao = "Reduza o texto, feche outros programas ou use um modelo menor (SOLAR_LLM_MODELO)."


class TextoLongoDemais(ErroLLM):
    """O texto passa do limite de tokens configurado para a entrada do modelo."""

    status_code = 413
    erro = "texto_longo_demais"
    sugestao = "Envie um texto menor ou divida-o em partes."


class TempoEsgotado(ErroLLM):
    status_code = 504
    erro = "tempo_esgotado"
    sugestao = "Tente um texto menor, peça uma resposta mais curta ou aumente SOLAR_LLM_TIMEOUT_S."


class SaidaInvalida(ErroLLM):
    """O modelo respondeu, mas a saída não pôde ser aproveitada (vazia, cortada etc.)."""

    status_code = 502
    erro = "saida_invalida_do_modelo"
    sugestao = "Tente novamente reformulando o texto. Se persistir, use outro modelo."


class FalhaNaInferencia(ErroLLM):
    status_code = 500
    erro = "falha_na_inferencia"
    sugestao = "Erro inesperado ao executar o modelo. Veja o log da API."
