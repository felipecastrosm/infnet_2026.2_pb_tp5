"""Execução local do LLM com HuggingFace Transformers + PyTorch.

O modelo é baixado uma única vez do HuggingFace Hub para o cache local
(~/.cache/huggingface) e, a partir daí, carregado e executado na própria máquina, sem
nenhum serviço externo. Este módulo cuida de:

- **escolher o hardware**: GPU NVIDIA (cuda), GPU da Apple (mps) ou CPU, e o tipo
  numérico (float16 na GPU, float32 na CPU) para caber na memória disponível;
- **carregar em segundo plano**: a API sobe na hora e o modelo fica pronto alguns
  segundos depois; enquanto isso as rotas de LLM respondem 503 com `Retry-After`;
- **executar com segurança**: uma inferência por vez (fila com tempo de espera),
  prazo máximo por geração (504), limite de tokens de entrada (413) e conversão de
  falta de memória e de outras falhas em exceções com código HTTP (ver `erros.py`);
- **dar respostas consistentes**: decodificação gulosa (sem amostragem), então o mesmo
  texto de entrada gera sempre a mesma saída.

Os testes trocam o backend real por um falso (`ServicoLLM.usar_backend`), para não
precisar baixar nem executar o modelo.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional, Protocol

from src.llm import erros
from src.llm.config import ConfigLLM

log = logging.getLogger("solar_viavel.llm")


class Estado(str, Enum):
    DESATIVADO = "desativado"
    NAO_CARREGADO = "nao_carregado"
    CARREGANDO = "carregando"
    PRONTO = "pronto"
    ERRO = "erro"


@dataclass
class Geracao:
    texto: str
    tokens_entrada: int
    tokens_saida: int
    tempo_s: float
    cortada: bool = False  # parou por atingir o limite de tokens, não por terminar a frase


class Backend(Protocol):
    """O que o serviço precisa de um modelo carregado (real ou falso, nos testes)."""

    nome: str
    dispositivo: str
    dtype: str
    parametros: int

    def contar_tokens(self, texto: str) -> int: ...

    def gerar(self, mensagens: list[dict], max_novos_tokens: int, prazo_s: float) -> Geracao: ...

    def pontuar(self, mensagens: list[dict], opcoes: dict[str, list[str]]) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Backend real: HuggingFace Transformers
# ---------------------------------------------------------------------------
def escolher_dispositivo(preferencia: str = "auto") -> str:
    import torch

    if preferencia != "auto":
        if preferencia == "cuda" and not torch.cuda.is_available():
            raise erros.FalhaAoCarregarModelo("SOLAR_LLM_DISPOSITIVO=cuda, mas não há GPU NVIDIA disponível.")
        if preferencia == "mps" and not torch.backends.mps.is_available():
            raise erros.FalhaAoCarregarModelo("SOLAR_LLM_DISPOSITIVO=mps, mas a GPU da Apple (MPS) não está disponível.")
        return preferencia
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def escolher_dtype(dispositivo: str, preferencia: str = "auto"):
    """float16 em GPU (metade da memória); float32 em CPU (float16 é lento ou instável na CPU)."""
    import torch

    tipos = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if preferencia != "auto":
        return tipos[preferencia]
    if dispositivo == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dispositivo == "mps":
        return torch.float16
    return torch.float32


class _Prazo:
    """Critério de parada da geração: interrompe quando o prazo acaba (vira HTTP 504)."""

    def __init__(self, segundos: float):
        self.limite = time.monotonic() + segundos
        self.estourou = False

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        if time.monotonic() > self.limite:
            self.estourou = True
        return torch.full((input_ids.shape[0],), self.estourou, dtype=torch.bool, device=input_ids.device)


class BackendTransformers:
    def __init__(self, config: ConfigLLM):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.nome = config.modelo
        self.dispositivo = escolher_dispositivo(config.dispositivo)
        dtype = escolher_dtype(self.dispositivo, config.dtype)
        self.dtype = str(dtype).replace("torch.", "")
        somente_local = not config.permitir_download
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.nome, local_files_only=somente_local)
            self.modelo = AutoModelForCausalLM.from_pretrained(self.nome, dtype=dtype, local_files_only=somente_local)
        except OSError as exc:
            dica = " O download está desligado (SOLAR_LLM_BAIXAR=0 ou HF_HUB_OFFLINE=1); rode `hf download " + self.nome + "` antes." if somente_local else ""
            raise erros.FalhaAoCarregarModelo(f"Não foi possível obter o modelo '{self.nome}': {exc}.{dica}") from exc
        self.modelo.to(self.dispositivo)
        self.modelo.eval()
        self.parametros = sum(p.numel() for p in self.modelo.parameters())
        self._torch = torch

    # -- utilidades -----------------------------------------------------------
    def _prompt(self, mensagens: list[dict]) -> str:
        # enable_thinking=False desliga o "modo raciocínio" de modelos como Qwen3/SmolLM3
        # (respostas mais rápidas); modelos sem esse recurso ignoram o parâmetro.
        return self.tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def contar_tokens(self, texto: str) -> int:
        return len(self.tokenizer(texto, add_special_tokens=False).input_ids)

    # -- geração de texto ------------------------------------------------------
    def gerar(self, mensagens: list[dict], max_novos_tokens: int, prazo_s: float) -> Geracao:
        torch = self._torch
        inicio = time.perf_counter()
        entrada = self.tokenizer(self._prompt(mensagens), return_tensors="pt", add_special_tokens=False).to(self.dispositivo)
        n_entrada = entrada.input_ids.shape[1]
        prazo = _Prazo(prazo_s)
        with torch.inference_mode():
            saida = self.modelo.generate(
                **entrada,
                max_new_tokens=max_novos_tokens,
                do_sample=False,  # decodificação gulosa: mesma entrada, mesma saída
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.1,
                stopping_criteria=[prazo],
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        if prazo.estourou:
            raise erros.TempoEsgotado(f"A geração passou do prazo de {prazo_s:.0f} s e foi interrompida.")
        novos = saida[0, n_entrada:]
        eos = self.modelo.generation_config.eos_token_id
        fim = {self.tokenizer.eos_token_id, *(eos if isinstance(eos, list) else [eos])}
        cortada = len(novos) >= max_novos_tokens and int(novos[-1]) not in fim
        texto = self.tokenizer.decode(novos, skip_special_tokens=True)
        return Geracao(texto, n_entrada, len(novos), time.perf_counter() - inicio, cortada)

    # -- classificação por pontuação de rótulos ---------------------------------
    def pontuar(self, mensagens: list[dict], opcoes: dict[str, list[str]]) -> dict[str, float]:
        """Probabilidade de cada rótulo ser a resposta do modelo.

        Em vez de pedir um JSON (que modelos pequenos às vezes quebram), mede a
        log-probabilidade que o modelo dá a cada rótulo como início da resposta e
        normaliza com softmax. A saída é sempre um dos rótulos válidos, com confiança.
        `opcoes` = {rótulo: [variantes de escrita]}, ex.: {"positivo": ["positivo", "Positivo"]}.
        """
        torch = self._torch
        base = self.tokenizer(self._prompt(mensagens), add_special_tokens=False).input_ids
        pontos = {}
        with torch.inference_mode():
            for rotulo, variantes in opcoes.items():
                logps = []
                for variante in variantes:
                    ids_op = self.tokenizer(variante, add_special_tokens=False).input_ids
                    ids = torch.tensor([base + ids_op], device=self.dispositivo)
                    logits = self.modelo(ids).logits[0, len(base) - 1 : -1].float()
                    alvo = torch.tensor(ids_op, device=self.dispositivo)[:, None]
                    logps.append(torch.log_softmax(logits, dim=-1).gather(1, alvo).sum().item())
                pontos[rotulo] = max(logps) + math.log(sum(math.exp(lp - max(logps)) for lp in logps))
        maior = max(pontos.values())
        exp = {r: math.exp(p - maior) for r, p in pontos.items()}
        total = sum(exp.values())
        return {r: v / total for r, v in exp.items()}

    def memoria_mb(self) -> Optional[float]:
        torch = self._torch
        if self.dispositivo == "cuda":
            return torch.cuda.memory_allocated() / 2**20
        if self.dispositivo == "mps":
            return torch.mps.current_allocated_memory() / 2**20
        return None


# ---------------------------------------------------------------------------
# Serviço (estado, fila, prazos e tradução de falhas)
# ---------------------------------------------------------------------------
def _falta_de_memoria(exc: BaseException) -> bool:
    texto = str(exc).lower()
    return "out of memory" in texto or type(exc).__name__ == "OutOfMemoryError"


class ServicoLLM:
    def __init__(self, config: Optional[ConfigLLM] = None, fabrica: Optional[Callable[[ConfigLLM], Backend]] = None):
        self.config = config or ConfigLLM.do_ambiente()
        self._fabrica = fabrica or BackendTransformers
        self.backend: Optional[Backend] = None
        self.estado = Estado.DESATIVADO if self.config.desativado else Estado.NAO_CARREGADO
        self.erro: Optional[str] = None
        self.carregado_em: Optional[datetime] = None
        self.tempo_carregamento_s: Optional[float] = None
        self._trava_carga = threading.Lock()
        self._fila = threading.Semaphore(1)  # uma inferência por vez
        self.requisicoes = 0

    # -- ciclo de vida -------------------------------------------------------------
    def carregar(self) -> None:
        """Carrega o modelo (bloqueante). Seguro para chamar mais de uma vez."""
        if self.estado == Estado.DESATIVADO:
            raise erros.ModeloDesativado("O LLM está desativado nesta instalação (SOLAR_LLM_DESATIVADO=1).")
        with self._trava_carga:
            if self.estado == Estado.PRONTO:
                return
            self.estado, self.erro = Estado.CARREGANDO, None
            inicio = time.perf_counter()
            log.info("Carregando o modelo %s...", self.config.modelo)
            try:
                backend = self._fabrica(self.config)
            except ImportError as exc:
                self.estado, self.erro = Estado.ERRO, f"Bibliotecas ausentes: {exc}. Instale torch e transformers."
                raise erros.ModeloDesativado(self.erro) from exc
            except erros.ErroLLM as exc:
                self.estado, self.erro = Estado.ERRO, exc.mensagem
                raise
            except Exception as exc:  # noqa: BLE001 - qualquer falha de carga vira 503 explicativo
                motivo = "memória insuficiente para carregar o modelo" if _falta_de_memoria(exc) else f"{type(exc).__name__}: {exc}"
                self.estado, self.erro = Estado.ERRO, f"Falha ao carregar '{self.config.modelo}': {motivo}"
                log.exception("Falha ao carregar o modelo")
                raise erros.FalhaAoCarregarModelo(self.erro) from exc
            self.backend = backend
            self.tempo_carregamento_s = round(time.perf_counter() - inicio, 1)
            self.carregado_em = datetime.now()
            self.estado = Estado.PRONTO
            log.info("Modelo pronto em %.1f s (%s, %s)", self.tempo_carregamento_s, backend.dispositivo, backend.dtype)

    def carregar_em_segundo_plano(self) -> bool:
        """Dispara a carga numa thread; devolve False se não havia o que carregar."""
        if self.estado in (Estado.PRONTO, Estado.CARREGANDO, Estado.DESATIVADO):
            return False
        self.estado = Estado.CARREGANDO

        def _rodar():
            try:
                self.carregar()
            except erros.ErroLLM:
                pass  # o erro fica registrado em self.erro e aparece em GET /llm/status

        threading.Thread(target=_rodar, daemon=True, name="carga-llm").start()
        return True

    def usar_backend(self, backend: Optional[Backend]) -> None:
        """Injeta um backend já pronto (usado nos testes com um modelo falso)."""
        self.backend = backend
        self.estado = Estado.PRONTO if backend else Estado.NAO_CARREGADO
        self.erro = None
        self.carregado_em = datetime.now() if backend else None

    # -- estado ------------------------------------------------------------------------
    def status(self) -> dict:
        b = self.backend
        memoria = b.memoria_mb() if b is not None and hasattr(b, "memoria_mb") else None
        try:
            import psutil

            memoria_processo = psutil.Process(os.getpid()).memory_info().rss / 2**20
        except ImportError:  # pragma: no cover
            memoria_processo = None
        return {
            "estado": self.estado.value,
            "pronto": self.estado == Estado.PRONTO,
            "modelo": b.nome if b else self.config.modelo,
            "dispositivo": b.dispositivo if b else None,
            "dtype": b.dtype if b else None,
            "parametros_milhoes": round(b.parametros / 1e6, 1) if b else None,
            "carregado_em": self.carregado_em,
            "tempo_carregamento_s": self.tempo_carregamento_s,
            "memoria_modelo_mb": round(memoria, 1) if memoria is not None else None,
            "memoria_processo_mb": round(memoria_processo, 1) if memoria_processo is not None else None,
            "requisicoes_atendidas": self.requisicoes,
            "limites": {
                "timeout_s": self.config.timeout_s,
                "espera_fila_s": self.config.espera_fila_s,
                "max_tokens_entrada": self.config.max_tokens_entrada,
            },
            "erro": self.erro,
        }

    def info_modelo(self) -> dict:
        b = self._exigir_pronto()
        return {"nome": b.nome, "dispositivo": b.dispositivo, "dtype": b.dtype, "parametros_milhoes": round(b.parametros / 1e6, 1)}

    def _exigir_pronto(self) -> Backend:
        if self.estado == Estado.PRONTO and self.backend is not None:
            return self.backend
        if self.estado == Estado.DESATIVADO:
            raise erros.ModeloDesativado("O LLM está desativado nesta instalação (SOLAR_LLM_DESATIVADO=1).")
        if self.estado == Estado.CARREGANDO:
            raise erros.ModeloCarregando(f"O modelo '{self.config.modelo}' ainda está sendo carregado.", retry_after=15)
        if self.estado == Estado.ERRO:
            raise erros.FalhaAoCarregarModelo(self.erro or "O modelo não pôde ser carregado.")
        raise erros.ModeloNaoCarregado(f"O modelo '{self.config.modelo}' não está carregado.")

    # -- inferência ----------------------------------------------------------------------
    def contar_tokens(self, texto: str) -> int:
        return self._exigir_pronto().contar_tokens(texto)

    def exigir_tamanho(self, texto: str, rotulo: str = "O texto") -> int:
        """Conta os tokens e recusa (413) o que não cabe no limite configurado."""
        n = self.contar_tokens(texto)
        if n > self.config.max_tokens_entrada:
            raise erros.TextoLongoDemais(
                f"{rotulo} tem {n} tokens; o limite é {self.config.max_tokens_entrada} "
                f"(cerca de {int(self.config.max_tokens_entrada * 0.6)} palavras)."
            )
        return n

    def _executar(self, funcao: Callable, *args):
        backend = self._exigir_pronto()
        if not self._fila.acquire(timeout=self.config.espera_fila_s):
            raise erros.ModeloOcupado(
                f"O modelo ficou ocupado por mais de {self.config.espera_fila_s:.0f} s com outras requisições.", retry_after=10
            )
        try:
            resultado = funcao(backend, *args)
            self.requisicoes += 1
            return resultado
        except erros.ErroLLM:
            raise
        except Exception as exc:  # noqa: BLE001 - traduz falhas do PyTorch em erros HTTP claros
            if _falta_de_memoria(exc):
                raise erros.MemoriaInsuficiente("O modelo ficou sem memória ao processar este texto.") from exc
            log.exception("Falha na inferência")
            raise erros.FalhaNaInferencia(f"Falha ao executar o modelo ({type(exc).__name__}).") from exc
        finally:
            self._fila.release()

    def gerar(self, mensagens: list[dict], max_novos_tokens: int = 256) -> Geracao:
        return self._executar(lambda b: b.gerar(mensagens, max_novos_tokens, self.config.timeout_s))

    def pontuar(self, mensagens: list[dict], opcoes: dict[str, list[str]]) -> dict[str, float]:
        return self._executar(lambda b: b.pontuar(mensagens, opcoes))


# Instância única usada pela API.
_servico: Optional[ServicoLLM] = None


def servico() -> ServicoLLM:
    global _servico
    if _servico is None:
        _servico = ServicoLLM()
    return _servico
