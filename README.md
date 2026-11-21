# Solar Viável

App e API de avaliação de potencial de geração solar fotovoltaica para residências e
pequenos negócios no Brasil. Combina dados climáticos da
[NASA POWER API](https://power.larc.nasa.gov/), a irradiação de longo prazo do
[SunData (CRESESB/CEPEL)](https://cresesb.cepel.br/index.php?section=sundata), coletada com
Selenium, notícias do setor extraídas com Beautiful Soup (ABSOLAR e Canal Solar), dados
enviados pelo próprio usuário e, desde o TP4, um **modelo de linguagem (LLM) da HuggingFace
executado localmente**, que resume e classifica notícias, responde dúvidas citando as fontes
e escreve relatórios de viabilidade. Projeto acadêmico (Projeto de Bloco, Infnet), alinhado ao
ODS 7 (Energia limpa e acessível) e ao ODS 13 (Ação contra a mudança global do clima).

Iniciativas que inspiraram o projeto: Atlas Brasileiro de Energia Solar (INPE), SunData (CRESESB), Global Solar Atlas, PVWatts, PVGIS, ANEEL e ABSOLAR (links no app e no Project Charter, seção 9.1).

Documentação: [Project Charter](docs/project/project_charter.md) ·
[Data Summary Report](docs/data_report/data_summary_report.md) ·
[API](docs/api.md) ·
[Escolha do modelo LLM](docs/llm_escolha_modelo.md) ·
[Integração local do LLM](docs/llm_integracao_local.md) ·
[Planejamento de IA (TP3)](docs/llm_planejamento.md) ·
[Metodologia (CRISP-DM x TDSP)](docs/methodology.md)

## Novidades do TP4

| Requisito | Como foi atendido | Onde |
|---|---|---|
| Escolha do modelo LLM local (desempenho, custo computacional, acessibilidade) | 5 modelos da HuggingFace (0,5 B a 4 B parâmetros) e um classificador BERT comparados **nas tarefas do projeto**, rodando nesta máquina: memória, velocidade, acurácia de sentimento, resumos, perguntas e relatórios. Escolhido: **Qwen3-1.7B** | [docs/llm_escolha_modelo.md](docs/llm_escolha_modelo.md), `src/llm/avaliacao.py`, `docs/avaliacao/` |
| Integração com FastAPI no ambiente local | Modelo baixado uma vez (`hf download`) e executado com **Transformers + PyTorch** (GPU CUDA/Apple MPS ou CPU), carregado em segundo plano quando a API sobe; funciona offline (`HF_HUB_OFFLINE=1`) | `src/llm/servico.py`, [docs/llm_integracao_local.md](docs/llm_integracao_local.md) |
| Rota `POST /processar_texto` | **Resumo** ou **sentimento** (com probabilidades e justificativa) de um texto; mais `POST /llm/perguntar` (perguntas com fontes), `POST /llm/relatorio` (relatório com checagem dos números), `GET /llm/status` e `POST /llm/carregar` | `src/api/main.py`, `src/llm/tarefas.py` |
| Response Models | Toda rota declara `response_model`; respostas do LLM com resultado, modelo usado e custo (tokens, tempo, cache); união discriminada resumo/sentimento | `src/api/schemas.py` |
| Tratamento robusto de erros | Formato único `RespostaErro` (status, código, mensagem, sugestão, detalhes) para **toda** a API; 413, 422 (mensagens em português), 502, 503 (carregando, ocupado, sem memória, falha na carga, desativado, com `Retry-After`), 504 (prazo) e 500 | `src/llm/erros.py`, handlers em `src/api/main.py`, `ui.aviso_llm` |
| Aplicação usa o LLM | Páginas *Pergunte ao Solar Viável* (chat) e *Resumo e sentimento*; botão *Resumir e avaliar com IA* nas notícias; *Gerar relatório* no potencial de longo prazo; página *Modelo local* com o estado do modelo | `paginas/` |
| Testes | 93 testes (49 novos): cada cenário de erro, Response Models, tarefas e páginas, com um modelo falso; teste opcional com o modelo real | `tests/test_llm.py`, `tests/test_app.py` |

## Novidades do TP3

| Requisito | Como foi atendido | Onde |
|---|---|---|
| Múltiplas páginas com menu de navegação | 7 páginas em 4 seções (`st.navigation`); parâmetros da barra lateral preservados ao trocar de página | `app.py`, `paginas/`, `src/ui.py` |
| API com FastAPI (GET e POST), documentada | 15 rotas: locais, clima, consulta ao vivo, longo prazo, estimativa, notícias, medições (POST/GET/DELETE) e LLM; Swagger em `/docs` | `src/api/`, [docs/api.md](docs/api.md) |
| Aplicação integrada à API | O app não lê `data/` diretamente: todas as páginas consultam e enviam dados por HTTP; a página *Explorar a API* testa qualquer rota | `src/api_client.py`, `paginas/` |
| Web scraping de página dinâmica (Selenium) | Preenche o formulário do SunData (envio por JavaScript) para as 10 capitais | `src/data_acquisition/cresesb_sundata.py` → `data/raw/cresesb_sundata_capitais.csv` |
| Mais dados com Beautiful Soup | Texto completo, data e tags das matérias do Canal Solar; coletas acumuladas (54 notícias, 51 com texto) | `src/data_acquisition/canalsolar_scraper.py`, `absolar_scraper.py` |
| Documentação atualizada | Charter (problema reavaliado, metas, escopo, riscos) e Data Summary Report (novas fontes) | `docs/` |
| Preparação para LLM | Corpus em trechos + fichas de local, recuperação de contexto e prompt; plano dos casos de uso | `src/data_processing/llm_prep.py`, `paginas/ia.py`, [docs/llm_planejamento.md](docs/llm_planejamento.md) |

## Páginas do app

| Seção | Página | O que faz | Rotas da API |
|---|---|---|---|
| Projeto | Início | Problema, objetivo, arquitetura e inspirações | n/a |
| Análises | Clima e geração | Compara locais e períodos; estima geração e economia; consulta ao vivo | `/locais`, `/clima`, `/clima/ao-vivo` |
| Análises | Potencial de longo prazo | Ano típico mês a mês, efeito da inclinação, NASA x CRESESB e **relatório de viabilidade gerado pelo LLM** | `/irradiacao-longo-prazo`, `/estimativa`, `/clima/resumo`, `/llm/relatorio` |
| Análises | Notícias do setor | Nuvem de palavras, frequências, publicações por semana, manchetes, textos e **resumo + sentimento com o LLM** | `/noticias`, `/processar_texto` |
| Dados | Meus dados | Envio de CSV ou medição avulsa, gestão das medições e downloads | `POST/GET/DELETE /medicoes` e outras |
| Dados | Explorar a API | Lista as rotas (OpenAPI) e envia requisições | todas |
| IA | Pergunte ao Solar Viável | Chat com o LLM local: respostas com base nas notícias e nos dados das capitais, com fontes citadas | `/llm/status`, `/llm/perguntar` |
| IA | Resumo e sentimento | Resumo ou sentimento de uma notícia coletada ou de um texto qualquer; mostra o JSON da resposta | `/llm/status`, `/noticias`, `/processar_texto` |
| IA | Modelo local (LLM) | Estado do modelo (hardware, memória, limites), como a integração funciona, corpus e recuperação de contexto | `/llm/status`, `/llm/carregar`, `/llm/corpus`, `/llm/contexto` |

## Estrutura do projeto

```
.
├── app.py                                   # entrada do Streamlit: menu de navegação e conexão com a API
├── paginas/                                 # uma página por funcionalidade
├── requirements.txt / requirements-dev.txt
├── .streamlit/config.toml
├── docs/
│   ├── api.md                               # rotas da API e formato dos erros
│   ├── llm_escolha_modelo.md                # justificativa da escolha do modelo (TP4)
│   ├── llm_integracao_local.md              # ambiente, bibliotecas, ajustes de hardware, erros (TP4)
│   ├── llm_planejamento.md                  # plano de uso de LLMs (escrito no TP3)
│   ├── avaliacao/                           # resultados da comparação de modelos
│   ├── methodology.md                       # CRISP-DM x TDSP
│   ├── project/project_charter.md
│   └── data_report/data_summary_report.md
├── data/
│   ├── raw/                                 # dados coletados (NASA, CRESESB, notícias)
│   ├── processed/                           # corpus para LLM (e medições enviadas, fora do Git)
│   └── avaliacao/                           # notícias rotuladas e perguntas para avaliar os modelos
├── src/
│   ├── api/                                 # FastAPI: main.py (rotas e erros), schemas.py (Response Models)
│   ├── llm/                                 # LLM local: config, erros, servico (Transformers), tarefas, avaliacao
│   ├── api_client.py                        # cliente HTTP usado pelo app
│   ├── repositorio.py                       # leitura/gravação dos arquivos de dados (usado pela API)
│   ├── ui.py                                # funções comuns às páginas (cache, barra lateral)
│   ├── locations.py                         # capitais usadas nas coletas
│   ├── data_acquisition/                    # nasa_power_client, cresesb_sundata (Selenium), absolar_scraper, canalsolar_scraper
│   └── data_processing/                     # climate.py, text.py, llm_prep.py
└── tests/                                   # API, LLM (modelo falso), parsers (fixtures HTML) e páginas do app
```

## Ambiente de desenvolvimento

Python **3.12** (o PyTorch e o Transformers atuais não suportam mais o 3.9 usado até o TP3).

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows (git bash): source .venv/Scripts/activate
pip install -r requirements-dev.txt
hf download Qwen/Qwen3-1.7B      # baixa o modelo uma vez (~4 GB) para ~/.cache/huggingface
```

Com GPU NVIDIA, instale antes o PyTorch com CUDA (<https://pytorch.org/get-started/locally/>);
no Mac com Apple Silicon a GPU (MPS) já funciona com o pacote padrão. Sem GPU, o modelo roda
na CPU, mais devagar. Passo a passo, variáveis de ambiente (`SOLAR_LLM_*`) e ajustes de
hardware em [docs/llm_integracao_local.md](docs/llm_integracao_local.md).

O scraper com Selenium precisa do Google Chrome instalado. O chromedriver é baixado
automaticamente pelo Selenium Manager.

## Como rodar

API e app em dois terminais:

```bash
uvicorn src.api.main:app --reload            # API em http://127.0.0.1:8000 (Swagger em /docs)
streamlit run app.py                         # app em http://localhost:8501
```

Rodar só o `streamlit run app.py` também funciona: se não encontrar a API em
`SOLAR_API_URL` (padrão `http://127.0.0.1:8000`), o app a inicia em uma thread.

O LLM é carregado em segundo plano quando a API sobe (alguns segundos; acompanhe em
`GET /llm/status` ou na página *Modelo local*). Depois do download, `HF_HUB_OFFLINE=1`
garante que nada seja consultado na internet. Para usar outro modelo:
`SOLAR_LLM_MODELO=Qwen/Qwen2.5-0.5B-Instruct uvicorn src.api.main:app`.

Coleta de dados (scripts independentes do app; rode a partir da raiz do projeto):

```bash
python src/data_acquisition/nasa_power_client.py             # 10 capitais, últimos 365 dias
python -m src.data_acquisition.cresesb_sundata               # Selenium: longo prazo (use --mostrar para ver o Chrome)
python -m src.data_acquisition.absolar_scraper               # notícias + texto completo (acumula)
python -m src.data_acquisition.canalsolar_scraper            # notícias + texto completo (acumula)
python -m src.data_processing.llm_prep                       # regera o corpus para LLM
python -m src.llm.avaliacao                                  # compara os modelos candidatos (precisa deles no cache)
```

Testes:

```bash
pytest                                       # 93 testes, sem precisar do modelo (usa um modelo falso)
SOLAR_TESTAR_LLM=1 pytest -m llm             # teste de ponta a ponta com o modelo real
```

## Deploy (Streamlit Community Cloud)

Aponte o [share.streamlit.io](https://share.streamlit.io) para este repositório, branch
`main` e arquivo principal `app.py`. Como o Community Cloud roda um único processo, o app
sobe a API FastAPI internamente e continua consumindo-a por HTTP. Para usar uma API
hospedada à parte, defina a variável de ambiente `SOLAR_API_URL`.

O Community Cloud não tem GPU nem memória para o LLM: lá, defina `SOLAR_LLM_DESATIVADO=1`
(as páginas de IA avisam que o modelo está desligado e o restante do app funciona) ou
aponte `SOLAR_API_URL` para uma API rodando numa máquina com o modelo.

## Fluxo de Git

Branch `main` estável, com commits pequenos e descritivos, um por parte da entrega. No
TP4: base do TP3 → ambiente (Python 3.12, PyTorch, Transformers) → corpus com ranking →
núcleo do LLM local → tarefas → rotas, Response Models e erros da API → páginas do app →
comparação de modelos → documentação (`git log --oneline`).
