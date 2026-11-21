# Metodologia: CRISP-DM e TDSP aplicados ao Solar Viável

O projeto segue o **TDSP** (Team Data Science Process) como processo de gestão do
projeto e usa as práticas analíticas do **CRISP-DM** (Cross-Industry Standard Process for
Data Mining) dentro de cada fase. Os dois métodos têm fases correspondentes, em
granularidades diferentes: o TDSP acrescenta, em torno das mesmas fases analíticas do
CRISP-DM, preocupações de gestão de equipe e de infraestrutura.

## Tabela de correspondência

| CRISP-DM | TDSP | Diretório no repositório | Status no TP5 (final) |
|---|---|---|---|
| Business Understanding | Business Understanding | `docs/project/project_charter.md`, `docs/methodology.md` | Concluído: como dados, IA e engenharia de prompts resolvem o problema (Charter, seções 2.3, 2.4 e 12) |
| Data Understanding | Data Acquisition and Understanding | `src/data_acquisition/`, `data/raw/`, `docs/data_report/data_summary_report.md` | Concluído: linhagem dos dados até o painel (DSR, seção 11) |
| Data Preparation | Data Acquisition and Understanding | `src/data_processing/`, `src/repositorio.py`, `data/processed/` | Notícias enriquecidas pelo LLM em lote; simulações calculadas para o chat |
| Modeling | Modeling | `src/data_processing/viabilidade.py`, `src/llm/` | Agente de decisão (7 ferramentas) e memória conversacional sobre o LLM local |
| Evaluation | Modeling / Model Evaluation | `tests/`, `docs/evidencias/`, `docs/avaliacao/` | 116 testes com modelo falso + chamadas reais com o modelo carregado registradas |
| Deployment | Deployment | `app.py`, `paginas/painel.py`, `src/api/`, `requirements.txt` | Painel final (página inicial) consumindo a API com 29 rotas |
| (sem equivalente) | Customer Acceptance | `docs/apresentacao_solucao.md`, `docs/evidencias/` | Apresentação e argumentação da solução; validação com usuários reais fica como próximo passo |

## Como cada fase se aplica a este projeto

**Business Understanding.** Definição do problema (falta de acesso a diagnóstico de
viabilidade solar para residências e pequenos negócios), dos ODS 7 e 13, do público-alvo e
das metas. No TP3 o problema foi reavaliado à luz das novas ferramentas: estimar com um
único ano e com o painel deitado podia enganar o usuário, e os dados ficavam presos à
interface. As metas foram ajustadas (seção 2.1 do [Project Charter](project/project_charter.md)).

**Data Acquisition and Understanding.** NASA POWER (requests), ABSOLAR e Canal Solar
(Beautiful Soup, agora com o texto completo das matérias e coletas acumuladas) e, no
TP3, o **SunData do CRESESB com Selenium**. Essa página exige preencher um formulário
cujo envio depende de JavaScript. As coletas rodam fora do app e salvam CSV/TXT em
`data/raw/`. Fontes, dicionários, estatísticas e limitações estão no
[Data Summary Report](data_report/data_summary_report.md).

**Data Preparation.** Conversão de `-999` em `NaN`; validação de medições do usuário no app
e de novo na API (Pydantic); tokenização e stopwords em português; e, no TP3, a preparação
para LLM: divisão das matérias em trechos com sobreposição e geração de fichas de local em
texto a partir dos números (`src/data_processing/llm_prep.py`).

**Modeling.** A estimativa de geração (irradiância x kWp x desempenho) passou a usar, por
padrão, o **ano típico do CRESESB com a inclinação de maior média anual**, somando a
geração mês a mês (`climate.geracao_mensal`). Os recursos com LLM (perguntas e respostas,
resumo, sentimento e relatório) foram implementados no TP4 com um **LLM executado
localmente** (Qwen3-1.7B via HuggingFace Transformers), escolhido por desempenho, custo
computacional e acessibilidade ([Escolha do modelo](llm_escolha_modelo.md)). O sentimento
usa as probabilidades que o próprio modelo dá a cada rótulo, não texto livre.

**Evaluation.** O app e o relatório comparam o último ano (NASA) com o longo prazo
(CRESESB): as diferenças chegaram a +11% (Recife, Manaus) e −5% (Rio), o que justificou a
mudança da estimativa padrão. No TP4, os modelos candidatos foram avaliados em 34
notícias rotuladas à mão, 12 perguntas com resposta esperada e 3 relatórios cujos números
são conferidos com a API (`python -m src.llm.avaliacao`). Os 93 testes automatizados cobrem
a API (incluindo cada cenário de erro do LLM, com um modelo falso), os parsers dos
scrapers, a preparação para LLM e as páginas do app.

**Deployment.** Arquitetura em duas camadas: a **API FastAPI** (`src/api/`) serve dados e
cálculos e executa o LLM local, e o **app Streamlit com 9 páginas** (`app.py` + `paginas/`, com menu
`st.navigation`) os consome por HTTP. No deploy de processo único (Streamlit Community
Cloud), o app sobe a própria API em uma thread. Dependências fixadas em
`requirements.txt` e versionamento com Git.

**Modeling e Deployment no TP5.** O LLM deixou de ser só uma ferramenta de texto e virou o
"cérebro" de um **agente de decisão**: a cada passo ele escolhe, por decodificação restrita, a
próxima análise entre 7 ferramentas que calculam sobre os dados coletados
(`src/data_processing/viabilidade.py`), e no fim escreve a recomendação. Um **chat com memória**
(janela, resumo e fatos) responde as perguntas seguintes. Tudo aparece em um **painel** que
mostra as 5 etapas do ciclo ([Agente e memória](agente_e_memoria.md)).

**Customer Acceptance.** A entrega final traz a apresentação da solução
([apresentacao_solucao.md](apresentacao_solucao.md)), chamadas reais registradas e telas do
painel com o modelo carregado ([evidencias/](evidencias/chamadas_reais_api.md)). A validação
com usuários reais do público-alvo (residencial e pequenos negócios) fica como próximo passo.
