# Solar Viável

App e API de avaliação de potencial de geração solar fotovoltaica para residências e
pequenos negócios no Brasil. Combina dados climáticos da
[NASA POWER API](https://power.larc.nasa.gov/), a irradiação de longo prazo do
[SunData (CRESESB/CEPEL)](https://cresesb.cepel.br/index.php?section=sundata), coletada com
Selenium, notícias do setor extraídas com Beautiful Soup (ABSOLAR e Canal Solar), dados
enviados pelo próprio usuário e um **modelo de linguagem (LLM) da HuggingFace executado
localmente**. Na versão final (TP5), um **painel** reúne o ciclo completo: o usuário informa
cidade, consumo e tarifa, um **agente de IA** escolhe as análises, calcula sobre os dados e
recomenda se vale a pena instalar, e um **chat com memória** responde as dúvidas seguintes.
Projeto acadêmico (Projeto de Bloco, Infnet), alinhado ao ODS 7 (Energia limpa e acessível) e
ao ODS 13 (Ação contra a mudança global do clima).

Iniciativas que inspiraram o projeto: Atlas Brasileiro de Energia Solar (INPE), SunData (CRESESB), Global Solar Atlas, PVWatts, PVGIS, ANEEL e ABSOLAR (links no app e no Project Charter, seção 9.1).

Documentação: [Apresentação da solução (TP5)](docs/apresentacao_solucao.md) ·
[Agente e memória (TP5)](docs/agente_e_memoria.md) ·
[Chamadas reais com o modelo carregado](docs/evidencias/chamadas_reais_api.md) ·
[Project Charter](docs/project/project_charter.md) ·
[Data Summary Report](docs/data_report/data_summary_report.md) ·
[API](docs/api.md) ·
[Escolha do modelo LLM](docs/llm_escolha_modelo.md) ·
[Integração local do LLM](docs/llm_integracao_local.md) ·
[Planejamento de IA (TP3)](docs/llm_planejamento.md) ·
[Metodologia (CRISP-DM x TDSP)](docs/methodology.md)

## Novidades do TP5 (entrega final)

| Requisito | Como foi atendido | Onde |
|---|---|---|
| Artefatos do TDSP completos | Charter: como dados, IA e engenharia de prompts resolvem o problema, técnicas de prompt, metas do TP5, resultados e impacto em ESG/ODS. Data Summary Report: notícias com IA, premissas com fonte e a **linhagem dos dados** da coleta ao painel | [Charter](docs/project/project_charter.md) (2.3, 2.4, 12), [DSR](docs/data_report/data_summary_report.md) (7.2 a 7.4, 11), [Metodologia](docs/methodology.md) |
| Caminho escolhido: **agente inteligente** (Opção 3) + **memória conversacional** (Opção 1) | Agente: a cada passo o LLM escolhe (por decodificação restrita) uma de 7 ferramentas que calculam sobre os dados (dimensionamento, payback com Fio B, sazonalidade, CO₂, benchmark, notícias); depois escreve a recomendação, com os números conferidos. Memória: janela de 3 trocas, resumo pelo LLM, fatos lembrados e reescrita da pergunta com exemplos | `src/llm/agente.py`, `src/data_processing/viabilidade.py`, `src/llm/memoria.py`, [docs](docs/agente_e_memoria.md) |
| Uso dos dados coletados pela IA | As 54 notícias resumidas e classificadas em lote pelo LLM; irradiação do CRESESB e da NASA em todas as ferramentas do agente; simulações calculadas na hora para o chat | `src/llm/enriquecer_noticias.py`, `data/processed/noticias_ia.csv` |
| **Dashboard final** com IA | Painel (página inicial): as 5 etapas do ciclo com números; aba *Consultor com IA* (formulário de 3 campos, indicadores, recomendação, passos do agente, saldo em 25 anos, geração x consumo, benchmark, notícias e chat com memória); *Panorama das capitais*; *Setor solar com IA*; *Como funciona* | `paginas/painel.py`, telas em [docs/evidencias/telas](docs/evidencias/telas/) |
| API | 9 rotas novas: `POST /agente/recomendar`, `/conversas` (criar, consultar, mensagem, apagar), `GET /painel`, `GET /noticias/ia` e outras; todas com Response Model | `src/api/main.py`, `src/api/schemas.py`, [docs/api.md](docs/api.md) |
| Apresentação e argumentação | Problema, solução, ciclo, insights com dados reais, uso da IA, ESG/ODS e limitações | [docs/apresentacao_solucao.md](docs/apresentacao_solucao.md) |
| **Feedback do TP4: chamada real com o modelo carregado** | Script que chama a API em execução com o Qwen3-1.7B carregado e salva requisições, respostas, tempos e tokens; telas do painel capturadas com o modelo real | `scripts/registrar_chamadas_reais.py`, [docs/evidencias/chamadas_reais_api.md](docs/evidencias/chamadas_reais_api.md), `scripts/capturar_telas.py` |
| Testes | 116 testes (23 novos: ferramentas, decisões do agente, guardrails, verificação de números, memória, reescrita, persistência, painel) | `tests/test_tp5.py` |

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
| Web scraping de página dinâmica (Selenium) | Preenche o formulário do SunData (envio por JavaScript) para as 10 capitais | `src/data_acquisition/cresesb_sundata.py`, salvo em `data/raw/cresesb_sundata_capitais.csv` |
| Mais dados com Beautiful Soup | Texto completo, data e tags das matérias do Canal Solar; coletas acumuladas (54 notícias, 51 com texto) | `src/data_acquisition/canalsolar_scraper.py`, `absolar_scraper.py` |
| Documentação atualizada | Charter (problema reavaliado, metas, escopo, riscos) e Data Summary Report (novas fontes) | `docs/` |
| Preparação para LLM | Corpus em trechos + fichas de local, recuperação de contexto e prompt; plano dos casos de uso | `src/data_processing/llm_prep.py`, `paginas/ia.py`, [docs/llm_planejamento.md](docs/llm_planejamento.md) |

## Páginas do app

| Seção | Página | O que faz | Rotas da API |
|---|---|---|---|
| (início) | **Painel (TP5)** | Ciclo de ciência de dados, consultor com IA (agente + chat com memória), panorama das capitais e setor solar com IA | `/painel`, `/agente/recomendar`, `/conversas`, `/noticias/ia`, `/irradiacao-longo-prazo` |
| Dados e projeto | Sobre o projeto | Problema, objetivo, arquitetura e inspirações | n/a |
| Análises | Clima e geração | Compara locais e períodos; estima geração e economia; consulta ao vivo | `/locais`, `/clima`, `/clima/ao-vivo` |
| Análises | Potencial de longo prazo | Ano típico mês a mês, efeito da inclinação, NASA x CRESESB e **relatório de viabilidade gerado pelo LLM** | `/irradiacao-longo-prazo`, `/estimativa`, `/clima/resumo`, `/llm/relatorio` |
| Análises | Notícias do setor | Nuvem de palavras, frequências, publicações por semana, manchetes, textos e **resumo + sentimento com o LLM** | `/noticias`, `/processar_texto` |
| Dados | Meus dados | Envio de CSV ou medição avulsa, gestão das medições e downloads | `POST/GET/DELETE /medicoes` e outras |
| Dados | Explorar a API | Lista as rotas (OpenAPI) e envia requisições | todas |
| IA | Pergunte ao Solar Viável | Chat **com memória** (TP5): entende "e em Manaus?", mostra a pergunta reescrita e o que lembra; fontes citadas | `/llm/status`, `/conversas` |
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
│   ├── apresentacao_solucao.md              # apresentação e argumentação da solução (TP5)
│   ├── agente_e_memoria.md                  # agente consultor e memória conversacional (TP5)
│   ├── evidencias/                          # chamadas reais com o modelo carregado e telas do painel (TP5)
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
│   ├── llm/                                 # LLM local: servico, tarefas, agente e memoria (TP5), enriquecer_noticias (TP5)
│   ├── api_client.py                        # cliente HTTP usado pelo app
│   ├── repositorio.py                       # leitura/gravação dos arquivos de dados (usado pela API)
│   ├── ui.py                                # funções comuns às páginas (cache, barra lateral)
│   ├── locations.py                         # capitais usadas nas coletas
│   ├── data_acquisition/                    # nasa_power_client, cresesb_sundata (Selenium), absolar_scraper, canalsolar_scraper
│   └── data_processing/                     # climate.py, text.py, llm_prep.py, viabilidade.py (TP5)
├── scripts/                                 # registrar_chamadas_reais.py e capturar_telas.py (TP5)
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
python -m src.llm.enriquecer_noticias                        # TP5: resumo e sentimento das notícias novas, em lote
python scripts/registrar_chamadas_reais.py                   # TP5: chamadas reais à API com o modelo (API no ar)
python scripts/capturar_telas.py                             # TP5: telas do painel (API e app no ar)
```

Testes:

```bash
pytest                                       # 116 testes, sem precisar do modelo (usa um modelo falso)
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
TP5: base do TP4, análise de viabilidade, notícias com IA em lote, agente consultor, memória
conversacional, rotas da API e testes, painel, ajustes de memória, evidências com o modelo
real (chamadas e telas) e documentação (`git log --oneline`).
