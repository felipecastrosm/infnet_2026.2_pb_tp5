"""Configuração do LLM local, lida de variáveis de ambiente.

| Variável                     | Padrão                       | Para que serve |
|------------------------------|------------------------------|----------------|
| SOLAR_LLM_MODELO             | Qwen/Qwen3-1.7B              | modelo da HuggingFace (nome no Hub ou pasta local) |
| SOLAR_LLM_DISPOSITIVO        | auto                         | auto, cuda, mps (GPU da Apple) ou cpu |
| SOLAR_LLM_DTYPE              | auto                         | auto, float32, float16 ou bfloat16 |
| SOLAR_LLM_TIMEOUT_S          | 90                           | prazo máximo de uma geração (vira HTTP 504) |
| SOLAR_LLM_ESPERA_FILA_S      | 30                           | quanto uma requisição espera o modelo ficar livre (vira HTTP 503) |
| SOLAR_LLM_MAX_TOKENS_ENTRADA | 3000                         | limite de tokens do texto de entrada (vira HTTP 413) |
| SOLAR_LLM_CARREGAR           | 1                            | 1 = carrega o modelo em segundo plano quando a API sobe |
| SOLAR_LLM_BAIXAR             | 1                            | 1 = baixa o modelo do Hub se ainda não estiver no cache local |
| SOLAR_LLM_DESATIVADO         | 0                            | 1 = desliga o LLM (as rotas respondem 503) |

Com o modelo já baixado, `HF_HUB_OFFLINE=1` garante que nada é consultado na internet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

MODELO_PADRAO = "Qwen/Qwen3-1.7B"


def _bool(nome: str, padrao: bool) -> bool:
    valor = os.environ.get(nome)
    if valor is None:
        return padrao
    return valor.strip().lower() in {"1", "true", "sim", "yes", "on"}


@dataclass(frozen=True)
class ConfigLLM:
    modelo: str = MODELO_PADRAO
    dispositivo: str = "auto"
    dtype: str = "auto"
    timeout_s: float = 90.0
    espera_fila_s: float = 30.0
    max_tokens_entrada: int = 3000
    carregar_na_inicializacao: bool = True
    permitir_download: bool = True
    desativado: bool = False

    @classmethod
    def do_ambiente(cls) -> "ConfigLLM":
        return cls(
            modelo=os.environ.get("SOLAR_LLM_MODELO", MODELO_PADRAO),
            dispositivo=os.environ.get("SOLAR_LLM_DISPOSITIVO", "auto").lower(),
            dtype=os.environ.get("SOLAR_LLM_DTYPE", "auto").lower(),
            timeout_s=float(os.environ.get("SOLAR_LLM_TIMEOUT_S", 90)),
            espera_fila_s=float(os.environ.get("SOLAR_LLM_ESPERA_FILA_S", 30)),
            max_tokens_entrada=int(os.environ.get("SOLAR_LLM_MAX_TOKENS_ENTRADA", 3000)),
            carregar_na_inicializacao=_bool("SOLAR_LLM_CARREGAR", True),
            permitir_download=_bool("SOLAR_LLM_BAIXAR", True) and not _bool("HF_HUB_OFFLINE", False),
            desativado=_bool("SOLAR_LLM_DESATIVADO", False),
        )
