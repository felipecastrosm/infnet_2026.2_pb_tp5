"""Cliente HTTP da API FastAPI, usado pelo app Streamlit.

O app não lê os arquivos de data/ diretamente: toda consulta e todo envio de dados
passa pela API. O endereço vem da variável de ambiente SOLAR_API_URL (padrão
http://127.0.0.1:8000).

- Em desenvolvimento, rode a API em um terminal (`uvicorn src.api.main:app --reload`)
  e o app em outro (`streamlit run app.py`).
- Se a API não estiver no ar e o endereço for local, `garantir_api()` sobe a própria
  API em uma thread do processo do Streamlit (útil no deploy do Streamlit Community
  Cloud, que roda um único processo). Mesmo assim o app continua falando com ela por
  HTTP, exatamente como falaria com uma API externa.
"""

from __future__ import annotations

import os
import threading
import time
from urllib.parse import urlparse

import requests

TIMEOUT_S = 60
TIMEOUT_LLM_S = 300  # rotas do LLM local podem levar mais (a própria API corta em SOLAR_LLM_TIMEOUT_S)


class ErroAPI(Exception):
    """Erro devolvido pela API (mensagem já pronta para mostrar ao usuário).

    Traz também os campos da `RespostaErro` da API: `status` (código HTTP), `erro`
    (código curto, ex.: "modelo_carregando"), `sugestao` e `retry_after` (segundos).
    `status` é None quando a API nem respondeu (rede, tempo esgotado).
    """

    def __init__(self, mensagem: str, status=None, erro=None, sugestao=None, retry_after=None):
        super().__init__(mensagem)
        self.mensagem = mensagem
        self.status = status
        self.erro = erro
        self.sugestao = sugestao
        self.retry_after = retry_after


def url_base() -> str:
    return os.environ.get("SOLAR_API_URL", "http://127.0.0.1:8000").rstrip("/")


def api_no_ar() -> bool:
    try:
        return requests.get(f"{url_base()}/saude", timeout=2).ok
    except requests.RequestException:
        return False


def garantir_api() -> str:
    """Garante que a API responde; devolve 'externa' ou 'embutida' (iniciada aqui)."""
    if api_no_ar():
        return "externa"
    destino = urlparse(url_base())
    if destino.hostname not in ("127.0.0.1", "localhost"):
        raise ErroAPI(f"A API em {url_base()} não respondeu.")

    import uvicorn

    from src.api.main import app

    config = uvicorn.Config(app, host=destino.hostname, port=destino.port or 80, log_level="warning")
    servidor = uvicorn.Server(config)
    servidor.install_signal_handlers = lambda: None  # thread secundária não pode tratar sinais
    threading.Thread(target=servidor.run, daemon=True, name="solar-viavel-api").start()
    for _ in range(100):
        if api_no_ar():
            return "embutida"
        time.sleep(0.1)
    raise ErroAPI("Não foi possível iniciar a API local.")


def _tratar(resposta: requests.Response):
    if resposta.ok:
        return resposta.json()
    try:
        corpo = resposta.json()
    except ValueError:
        corpo = {"mensagem": resposta.text}
    mensagem = corpo.get("mensagem") or corpo.get("detail") or resposta.reason
    retry = resposta.headers.get("Retry-After")
    raise ErroAPI(
        f"{resposta.status_code}: {mensagem}",
        status=resposta.status_code,
        erro=corpo.get("erro"),
        sugestao=corpo.get("sugestao"),
        retry_after=int(retry) if retry and retry.isdigit() else None,
    )


def _requisitar(metodo: str, caminho: str, timeout: float, **kwargs):
    try:
        return _tratar(requests.request(metodo, f"{url_base()}{caminho}", timeout=timeout, **kwargs))
    except requests.Timeout as exc:
        raise ErroAPI(f"A API não respondeu em {timeout:.0f} s.", erro="tempo_esgotado_cliente") from exc
    except requests.RequestException as exc:
        raise ErroAPI(f"API indisponível ({exc}).", erro="api_indisponivel") from exc


def get(caminho: str, **params):
    return _requisitar("GET", caminho, TIMEOUT_S, params=params)


def post(caminho: str, dados: dict = None, timeout: float = None):
    padrao = TIMEOUT_LLM_S if caminho.startswith(("/llm", "/processar_texto")) else TIMEOUT_S
    return _requisitar("POST", caminho, timeout or padrao, json=dados)


def delete(caminho: str):
    return _requisitar("DELETE", caminho, TIMEOUT_S)
