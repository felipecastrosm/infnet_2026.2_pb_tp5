# Integração local do LLM com a API FastAPI

Este documento descreve **como o modelo de linguagem foi integrado ao Solar Viável rodando
inteiramente na máquina local**: preparação do ambiente, bibliotecas, download do modelo,
ajustes para o hardware, arquitetura do código, rotas, Response Models, tratamento de
erros e testes. A justificativa da escolha do modelo está em
[llm_escolha_modelo.md](llm_escolha_modelo.md).

## 1. Visão geral

```
                         download único (hf download)
HuggingFace Hub  ───────────────────────────────────────►  ~/.cache/huggingface/hub
                                                                   │
                                                                   ▼  carregado na memória (GPU ou CPU)
App Streamlit ──HTTP──► API FastAPI ──► src/llm/servico.py ──► Transformers + PyTorch
   (paginas/)            (src/api/)       fila, prazo, erros        (modelo local)
        ▲                    │
        └── JSON validado ◄──┘  Response Models (src/api/schemas.py)
```

- O modelo é **baixado uma vez** do HuggingFace Hub para o cache local. Depois disso, com
  `HF_HUB_OFFLINE=1`, nenhuma requisição sai para a internet: carga e inferência
  acontecem no processo da API.
- O modelo fica **dentro da API**, não no app. O app Streamlit só envia textos e recebe
  JSON por HTTP (como em todo o projeto desde o TP3), então qualquer outro cliente pode
  usar as mesmas rotas.
- Os cálculos de geração e economia continuam determinísticos na API. O modelo explica,
  resume, classifica e responde; no relatório, cada número do texto é conferido contra a API.

## 2. Ambiente usado no desenvolvimento

| Item | Valor |
|---|---|
| Máquina | Apple M5 Pro, 18 núcleos, 48 GB de memória unificada |
| Sistema | macOS (Darwin 25.6) |
| Python | 3.12.14 (ambiente virtual em `.venv`) |
| Aceleração | GPU da Apple via **MPS** (Metal Performance Shaders) do PyTorch |
| PyTorch | 2.15.0 |
| Transformers | 5.23.0 |
| huggingface-hub | 2.4.0 (traz o comando `hf`) |
| FastAPI / Uvicorn / Pydantic | 0.128.8 / 0.39.0 / 2.14 |

> **Por que Python 3.12 e não o 3.9 do TP3?** O PyTorch recente não publica mais pacotes
> para o Python 3.9, e o Transformers 5 exige Python ≥ 3.10. O 3.12 é a versão mais nova
> com pacotes do PyTorch para Mac, Linux e Windows (CPU e CUDA). As demais dependências
> do projeto funcionaram sem mudança.

## 3. Passo a passo da instalação

### 3.1 Ambiente virtual e dependências

```bash
python3.12 -m venv .venv
source .venv/bin/activate              # Windows (git bash): source .venv/Scripts/activate
pip install -r requirements-dev.txt    # inclui requirements.txt
```

As bibliotecas novas do TP4 (em `requirements.txt`):

| Biblioteca | Para quê |
|---|---|
| `torch` | executa a rede neural (tensores na CPU ou GPU) |
| `transformers` | baixa e carrega o tokenizador e o modelo (`AutoTokenizer`, `AutoModelForCausalLM`), aplica o *chat template* e gera texto (`generate`) |
| `huggingface-hub` | (dependência do Transformers) cache local e o comando `hf download` |
| `psutil` | mede a memória do processo (mostrada em `GET /llm/status`) |

**GPU NVIDIA (Windows/Linux):** o `pip install torch` padrão no Windows instala a versão
só para CPU. Para usar a GPU, instale o PyTorch com CUDA seguindo o seletor em
<https://pytorch.org/get-started/locally/> (ex.: `pip install torch --index-url
https://download.pytorch.org/whl/cu128`) antes do `requirements.txt`. No Mac com Apple
Silicon, o pacote padrão já inclui o MPS.

Conferir o hardware que o PyTorch enxerga:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())"
```

### 3.2 Download do modelo (uma vez)

```bash
hf download Qwen/Qwen3-1.7B            # ~4,1 GB em ~/.cache/huggingface/hub
```

Se esse passo for pulado, a API baixa o modelo sozinha na primeira carga
(`SOLAR_LLM_BAIXAR=1`, padrão). Para garantir que nada seja consultado na internet depois
do download:

```bash
export HF_HUB_OFFLINE=1                # Windows (PowerShell): $env:HF_HUB_OFFLINE="1"
```

### 3.3 Rodar

```bash
uvicorn src.api.main:app --reload       # API em http://127.0.0.1:8000 (o modelo carrega em segundo plano)
streamlit run app.py                    # app em http://localhost:8501
```

A API responde na hora; o modelo leva alguns segundos para ficar pronto (~3 s neste
Mac). Enquanto isso, as rotas de LLM respondem **503** com `Retry-After`, e o app mostra
"o modelo está sendo carregado". O estado aparece em `GET /llm/status` e na página
*Modelo local (LLM)*.

Teste rápido pelo terminal:

```bash
curl http://127.0.0.1:8000/llm/status
curl -X POST http://127.0.0.1:8000/processar_texto -H "Content-Type: application/json" \
  -d '{"texto": "A fonte solar respondeu por 73% da expansão da matriz elétrica brasileira em 2026.", "tarefa": "resumo", "max_palavras": 20}'
```

## 4. Configuração (variáveis de ambiente)

| Variável | Padrão | Efeito |
|---|---|---|
| `SOLAR_LLM_MODELO` | `Qwen/Qwen3-1.7B` | modelo do Hub (ou uma pasta local com os arquivos do modelo) |
| `SOLAR_LLM_DISPOSITIVO` | `auto` | `auto` escolhe CUDA → MPS → CPU; ou force `cuda`, `mps`, `cpu` |
| `SOLAR_LLM_DTYPE` | `auto` | `auto` = float16 na GPU (bfloat16 em CUDA que suporta), float32 na CPU |
| `SOLAR_LLM_TIMEOUT_S` | 90 | prazo de uma geração; ao estourar, a geração para e a rota responde 504 |
| `SOLAR_LLM_ESPERA_FILA_S` | 30 | tempo máximo esperando o modelo ficar livre; depois, 503 `modelo_ocupado` |
| `SOLAR_LLM_MAX_TOKENS_ENTRADA` | 3000 | limite de tokens do texto enviado; acima disso, 413 |
| `SOLAR_LLM_CARREGAR` | 1 | carrega o modelo ao subir a API (0 = só com `POST /llm/carregar`) |
| `SOLAR_LLM_BAIXAR` | 1 | permite baixar o modelo se não estiver no cache (0 = só arquivos locais) |
| `SOLAR_LLM_DESATIVADO` | 0 | 1 = desliga o LLM (as rotas respondem 503 `modelo_desativado`) |
| `HF_HUB_OFFLINE` | (não definida) | 1 = Transformers não acessa a internet de jeito nenhum |

Exemplos: rodar um modelo menor numa máquina modesta
(`SOLAR_LLM_MODELO=Qwen/Qwen2.5-0.5B-Instruct`), ou forçar a CPU para comparar
(`SOLAR_LLM_DISPOSITIVO=cpu`).

## 5. Ajustes para o hardware local

| Ajuste | Onde | Por quê |
|---|---|---|
| **Escolha automática do dispositivo** (CUDA → MPS → CPU) | `servico.escolher_dispositivo` | o mesmo código roda em Mac, PC com GPU NVIDIA ou só CPU |
| **float16 na GPU, float32 na CPU** | `servico.escolher_dtype` | float16 usa metade da memória (1,7 B parâmetros ≈ 3,4 GB em vez de 6,8 GB) e é rápido na GPU; na CPU, float16 é lento ou sem suporte, então usa float32 |
| **Carga em segundo plano** | `ServicoLLM.carregar_em_segundo_plano` + `lifespan` da API | a API sobe na hora; a carga (alguns segundos) não trava o servidor nem o app |
| **Uma inferência por vez** (semáforo) | `ServicoLLM._executar` | duas gerações simultâneas no mesmo modelo dobrariam a memória e disputariam a GPU (no MPS, podem até falhar); as requisições esperam na fila até `SOLAR_LLM_ESPERA_FILA_S` |
| **Prazo dentro da geração** | `_Prazo` (critério de parada do `generate`) | interrompe a geração no meio quando o tempo acaba, liberando a GPU de verdade (um timeout só no cliente deixaria o modelo trabalhando) |
| **Limite de tokens de entrada** | `ServicoLLM.exigir_tamanho` | textos grandes aumentam memória e tempo de forma não linear; o limite protege a máquina e devolve 413 com explicação |
| **Modo "raciocínio" desligado** | `apply_chat_template(..., enable_thinking=False)` | Qwen3 e SmolLM3 "pensam em voz alta" antes de responder; desligar reduz muito o tempo e os tokens, sem perda relevante nestas tarefas |
| **Decodificação gulosa** (`do_sample=False`) e `repetition_penalty=1.1` | `BackendTransformers.gerar` | mesma entrada → mesma saída (respostas consistentes e testáveis) e menos laços de repetição, comuns em modelos pequenos |
| **`torch.inference_mode()`** | geração e pontuação | não guarda gradientes: menos memória e mais velocidade |
| **Cache de respostas** (256 entradas) | `tarefas._CACHE` | a mesma pergunta/texto não executa o modelo de novo (`uso.em_cache=true`) |
| **Falta de memória vira 503** | `ServicoLLM._executar` | `OutOfMemoryError` (CUDA) ou "MPS backend out of memory" viram `memoria_insuficiente`, com sugestão de usar texto menor ou modelo menor |

Desempenho medido neste Mac (Qwen3-1.7B, MPS, float16): carga em ~3 s, ~3,2 GB de memória
do modelo e ~60 tokens/s na geração. Um resumo de notícia leva ~3 s, uma resposta do
assistente ~2 s e um relatório completo ~7 s. Na CPU (float32) o mesmo modelo ocupa ~7,3 GB
de RAM e gera ~12 tokens/s (resumo em ~15 s; ver a seção de custo computacional em
[llm_escolha_modelo.md](llm_escolha_modelo.md)); para máquinas sem GPU e com pouca memória,
o `Qwen/Qwen2.5-0.5B-Instruct` (1 GB) é a alternativa.

## 6. Código

```
src/llm/
├── config.py      # ConfigLLM: lê as variáveis de ambiente
├── erros.py       # ErroLLM e subclasses, cada uma com status HTTP, código e sugestão
├── servico.py     # BackendTransformers (carga, geração, pontuação) e ServicoLLM (estado, fila, prazo, erros)
├── tarefas.py     # prompts e pós-processamento: resumo, sentimento, perguntas (RAG), relatório
└── avaliacao.py   # comparação dos modelos candidatos (python -m src.llm.avaliacao)
src/api/main.py    # rotas, handlers de erro e carga do modelo no lifespan
src/api/schemas.py # Response Models (Pydantic)
```

### 6.1 Carga do modelo (`servico.py`)

```python
tokenizer = AutoTokenizer.from_pretrained(nome, local_files_only=somente_local)
modelo = AutoModelForCausalLM.from_pretrained(nome, dtype=dtype, local_files_only=somente_local)
modelo.to(dispositivo)      # "cuda", "mps" ou "cpu"
modelo.eval()
```

### 6.2 Duas formas de usar o modelo

1. **Geração de texto** (resumo, justificativa, resposta, relatório): o pedido vira uma
   conversa (`system` + `user`), o *chat template* do modelo a transforma em tokens e o
   `generate` produz a continuação, com decodificação gulosa, limite de tokens e prazo.
2. **Pontuação de rótulos** (sentimento): em vez de pedir um JSON (que modelos pequenos às
   vezes quebram), a API mede a probabilidade que o modelo dá a cada rótulo (`positivo`,
   `neutro`, `negativo`, com e sem maiúscula) como início da resposta e normaliza. A
   saída é **sempre** um rótulo válido, e as probabilidades viram o campo `confianca`.

### 6.3 Tarefas (`tarefas.py`)

| Tarefa | Rota | Prompt e pós-processamento |
|---|---|---|
| Resumo | `POST /processar_texto` (`tarefa=resumo`) | "resuma em um parágrafo de no máximo N palavras, só com informações do texto"; remove rótulos ("Resumo:"), frases repetidas e frase final incompleta; corta por frases se passar de N |
| Sentimento | `POST /processar_texto` (`tarefa=sentimento`) | pontuação dos rótulos com a definição de cada classe; ponto de vista `geral` ou `setor_solar`; justificativa opcional em uma frase |
| Perguntas | `POST /llm/perguntar` | recupera trechos (TF-IDF + ficha da cidade citada + ranking para comparações), numera-os e pede resposta curta só com o contexto, citando [n] e com frase fixa quando não souber; marca as fontes citadas |
| Relatório | `POST /llm/relatorio` | dados de `/estimativa` + irradiação mensal + ficha do local; proíbe criar números; depois confere cada número do texto contra a entrada (`verificacao`) |

## 7. Response Models

Toda rota da API declara `response_model`; o FastAPI valida a saída antes de enviá-la e o
`/docs` mostra o formato. Os principais, em `src/api/schemas.py`:

| Modelo | Campos principais |
|---|---|
| `PedidoProcessarTexto` (entrada) | `texto` (20 a 20.000 caracteres, ≥ 3 palavras; espaços normalizados), `tarefa` (`resumo`/`sentimento`), `max_palavras` (20 a 250), `ponto_de_vista`, `justificar` |
| `RespostaProcessamento` | `tarefa`, `entrada` (caracteres, palavras, tokens), `resultado`, `modelo` (`InfoModelo`), `uso` (`UsoModelo`), `processado_em` |
| `ResultadoResumo` / `ResultadoSentimento` | união discriminada pelo campo `tipo`; o sentimento só aceita os três rótulos e probabilidades entre 0 e 1 |
| `RespostaPergunta` | `resposta`, `fontes` (`FonteCitada`: n, título, fonte, link, data, `citada`), `sem_contexto`, `modelo`, `uso` |
| `RespostaRelatorio` | `relatorio` (Markdown), `verificacao` (`numeros_citados`, `nao_encontrados`, `confiavel`), `estimativa`, `modelo`, `uso` |
| `StatusLLM` | `estado`, `dispositivo`, `dtype`, `parametros_milhoes`, memória, `limites`, `erro` |
| `RespostaErro` | `status`, `erro`, `mensagem`, `sugestao`, `detalhes`, `caminho` (todo erro de toda rota) |

Exemplos completos de requisição e resposta: [api.md](api.md).

## 8. Tratamento de erros

Cada falha possível vira uma exceção com código HTTP próprio (`src/llm/erros.py`), e
*exception handlers* da API (`src/api/main.py`) convertem tudo para `RespostaErro`:

| Cenário | Exceção | HTTP |
|---|---|---|
| Texto vazio, curto, sem palavras, tarefa desconhecida, JSON malformado | `RequestValidationError` (Pydantic), mensagens traduzidas | 422 `dados_invalidos` |
| Texto com mais tokens que o limite | `TextoLongoDemais` | 413 |
| Modelo ainda carregando | `ModeloCarregando` (+ `Retry-After: 15`) | 503 |
| Modelo não carregado (`SOLAR_LLM_CARREGAR=0`) | `ModeloNaoCarregado` | 503 |
| Falha ao carregar (nome errado, não baixado em modo offline, sem memória) | `FalhaAoCarregarModelo` | 503 |
| PyTorch/Transformers ausentes ou LLM desligado | `ModeloDesativado` | 503 |
| Fila cheia (outra geração em andamento por muito tempo) | `ModeloOcupado` (+ `Retry-After: 10`) | 503 |
| Falta de memória durante a inferência | `MemoriaInsuficiente` | 503 |
| Geração passou do prazo | `TempoEsgotado` | 504 |
| Saída vazia ou inaproveitável | `SaidaInvalida` | 502 |
| Outro erro do PyTorch na inferência | `FalhaNaInferencia` | 500 |
| Qualquer outra exceção | handler genérico (detalhes só no log) | 500 `erro_interno` |
| Local inexistente no relatório | `HTTPException` | 404 |

No app, `ui.aviso_llm` mostra cada caso de um jeito: espera (azul) para modelo
carregando/ocupado, aviso (amarelo) para entrada inválida, texto longo ou prazo, erro
(vermelho) para falhas, sempre com a sugestão da API e o código HTTP. Quando o modelo não
está carregado, as páginas de IA oferecem o botão **Carregar o modelo**.

## 9. Testes

```bash
pytest                                   # 93 testes, ~4 s, sem baixar nem executar o modelo
SOLAR_TESTAR_LLM=1 pytest -m llm         # teste de ponta a ponta com o modelo real (do cache local)
```

- `tests/test_llm.py`: rotas com um **backend falso** (respostas controladas), validação de
  cada Response Model, **cada cenário de erro** da tabela acima (413, 422, 500, 502, 503,
  504, 404, 405), cache, citações, verificação de números, recuperação de contexto e
  documentação OpenAPI (toda rota com modelo de resposta e erros documentados).
- `tests/test_app.py`: as páginas de IA abrem sem erro, oferecem a carga do modelo, o chat
  responde com fontes e mostra erro amigável, o resumo, a análise de notícia e o relatório
  funcionam pela interface.

## 10. Limitações conhecidas

- Modelos de 1 a 2 bilhões de parâmetros erram mais que modelos grandes: o assistente
  pode simplificar demais ou, raramente, misturar trechos. Por isso as respostas citam as
  fontes e os números exatos continuam nas páginas de análise.
- A verificação do relatório confere se cada número **existe** nos dados de entrada, não
  se foi usado no sentido certo (ex.: um valor diário apresentado como mensal).
- O sentimento depende bastante do prompt (ver [llm_escolha_modelo.md](llm_escolha_modelo.md)),
  e o conjunto de avaliação é pequeno (34 notícias rotuladas à mão).
- No Streamlit Community Cloud (1 GB de RAM, sem GPU) o modelo não cabe: lá, defina
  `SOLAR_LLM_DESATIVADO=1` (as páginas de IA mostram o aviso e o restante do app funciona)
  ou aponte `SOLAR_API_URL` para uma API rodando numa máquina com o modelo.
