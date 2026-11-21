"""Configuração comum dos testes.

As medições enviadas nos testes vão para um CSV temporário (não sujam data/processed/),
e o app usa uma porta própria para a API embutida.

O LLM não é carregado automaticamente nos testes (SOLAR_LLM_CARREGAR=0): os testes
injetam um backend falso (`BackendFalso`) com respostas controladas, o que permite testar
rotas, Response Models e cada cenário de erro sem baixar nem executar o modelo. O teste
com o modelo real (marcado com `llm`) só roda com SOLAR_TESTAR_LLM=1.
"""

import dataclasses
import os
import socket
import tempfile
from pathlib import Path

import pytest

_tmp = Path(tempfile.mkdtemp(prefix="solar_viavel_testes_"))
os.environ["SOLAR_MEDICOES_CSV"] = str(_tmp / "medicoes.csv")
os.environ["SOLAR_LLM_CARREGAR"] = "0"
os.environ.pop("SOLAR_LLM_DESATIVADO", None)

with socket.socket() as _s:
    _s.bind(("127.0.0.1", 0))
    os.environ["SOLAR_API_URL"] = f"http://127.0.0.1:{_s.getsockname()[1]}"


class BackendFalso:
    """Imita o backend do Transformers: devolve respostas prontas e registra as chamadas."""

    nome = "modelo-falso/teste"
    dispositivo = "cpu"
    dtype = "float32"
    parametros = 1_000_000

    def __init__(self, resposta="Resposta gerada pelo modelo falso. [1]", probabilidades=None, erro=None):
        self.resposta = resposta
        self.probabilidades = probabilidades or {"positivo": 0.7, "neutro": 0.2, "negativo": 0.1}
        self.erro = erro
        self.chamadas = []

    def contar_tokens(self, texto: str) -> int:
        return len(texto.split())

    def gerar(self, mensagens, max_novos_tokens, prazo_s):
        from src.llm.servico import Geracao

        self.chamadas.append(("gerar", mensagens))
        if self.erro:
            raise self.erro
        texto = self.resposta(mensagens) if callable(self.resposta) else self.resposta
        return Geracao(texto, sum(len(m["content"].split()) for m in mensagens), len(texto.split()), 0.01)

    def pontuar(self, mensagens, opcoes):
        self.chamadas.append(("pontuar", mensagens))
        if self.erro:
            raise self.erro
        return {r: self.probabilidades[r] for r in opcoes}


@pytest.fixture
def llm():
    """Coloca o serviço de LLM em estado "pronto" com um backend falso; devolve (serviço, backend)."""
    from src.llm import tarefas
    from src.llm.servico import servico

    s = servico()
    config_original = s.config
    backend = BackendFalso()
    s.usar_backend(backend)
    tarefas.limpar_cache()
    yield s, backend
    s.config = config_original
    s.usar_backend(None)
    tarefas.limpar_cache()


@pytest.fixture
def configurar_llm(llm):
    """Troca limites do serviço no teste, ex.: configurar_llm(max_tokens_entrada=10)."""
    s, _ = llm

    def _trocar(**valores):
        s.config = dataclasses.replace(s.config, **valores)

    return _trocar
