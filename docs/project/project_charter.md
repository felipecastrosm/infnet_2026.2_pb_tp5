# Project Charter: Solar Viável

**Versão:** TP4 (revisão do TP3) · **Autor:** Felipe Castro Sotto Mayor · **Disciplina:** Projeto de Bloco, Data Science e IA (Infnet)

> **O que mudou no TP4:** integração de um LLM executado localmente (seção 2.2), objetivos
> 10 a 12, novas metas na seção 4 (coluna TP4), escopo, riscos e cronograma atualizados.
>
> **O que mudou no TP3:** reavaliação do problema com as novas ferramentas (seção 2.1),
> novos objetivos 6 a 9, novas metas (seção 4), escopo, stakeholders, fontes de dados e
> riscos atualizados, e cronograma com o TP3 concluído e o TP4 (LLM) planejado.

## 1. Nome do projeto

**Solar Viável**, um app de avaliação de potencial de geração solar fotovoltaica para
residências e pequenos negócios no Brasil.

## 2. Contexto e problema de negócio

Decidir se vale a pena instalar painéis solares em um imóvel exige avaliar variáveis
climáticas (irradiância solar, temperatura, nuvens) que não são triviais de encontrar
e interpretar para quem não é da área técnica. Hoje, essa avaliação normalmente
depende de uma consultoria especializada (paga) ou de simuladores comerciais que
priorizam a venda de um sistema específico, não uma análise neutra de viabilidade.

Isso cria uma barreira de acesso à informação para o público residencial e para
pequenos negócios, que muitas vezes desistem de considerar a energia solar por falta
de um primeiro diagnóstico rápido, gratuito e confiável.

### 2.1 Reavaliação do problema no TP3 (FastAPI, Selenium e preparação para LLM)

O problema continua o mesmo, mas ao usar o app do TP2 três fraquezas ficaram claras.
As ferramentas do TP3 resolvem cada uma delas:

| Fraqueza observada no TP2 | Por que importa para o usuário | Resposta no TP3 |
|---|---|---|
| A estimativa usava **um único ano** de dados da NASA, com o painel **deitado** (plano horizontal) | Um ano atípico distorce a decisão, e painéis reais são inclinados. Em São Paulo, Curitiba ou Porto Alegre a inclinação muda a geração em cerca de 5% | **Selenium** coleta do SunData (CRESESB/CEPEL) a irradiação de **longo prazo** mês a mês, no plano horizontal e inclinado. Esses dados alimentam a nova página *Potencial de longo prazo* e a estimativa padrão da API |
| Os dados e cálculos ficavam **presos ao app** Streamlit, e os uploads se perdiam ao fechar a sessão | Instaladores, outros apps ou um futuro assistente não tinham como reutilizar os cálculos. O usuário precisava reenviar os dados a cada visita | **API FastAPI** própria: consulta (GET) de locais, clima, longo prazo, estimativas e notícias, e envio (POST/DELETE) de medições, que ficam guardadas. O app passa a ser um cliente dessa API |
| As notícias tinham só manchetes do Canal Solar, e o app tinha **uma página só** (abas) | Pouco texto para análise. Navegação confusa conforme o app cresce | Coleta do **texto completo** das matérias com Beautiful Soup, coletas acumuladas e **app com 7 páginas** e menu de navegação |

O próximo passo natural é a linguagem: o público-alvo é leigo, e números e gráficos
ainda exigem interpretação. Por isso, o TP3 já deixa os dados prontos para **LLMs**
(corpus em trechos, fichas de local em texto e recuperação de contexto), que no TP4
vão responder perguntas, resumir e classificar notícias e redigir um relatório de
viabilidade. Detalhes em [Planejamento de IA](../llm_planejamento.md).

**Metas ajustadas:** a estimativa padrão passa a usar o **ano típico com painel
inclinado** (CRESESB), e o ano mais recente medido pela NASA vira um comparativo. A meta de
"dashboard" passa a ser "app + API", ou seja, os dados ficam disponíveis para outros
clientes, e não só para a interface.

### 2.2 O que o LLM local resolve (TP4)

Mesmo com gráficos e estimativas, o público leigo ainda precisava **interpretar** números
e ler notícias longas e técnicas. No TP4, um modelo de linguagem da HuggingFace, executado
na própria máquina (sem enviar dados a serviços externos e sem custo por uso), passou a:
responder dúvidas em português com base nas notícias e nos dados das capitais, citando as
fontes; resumir e classificar o sentimento das notícias do ponto de vista de quem quer
instalar solar; e redigir um relatório de viabilidade em linguagem simples, cujos números
são conferidos com a API. A escolha do modelo foi feita medindo candidatos nas tarefas do
próprio projeto ([Escolha do modelo](../llm_escolha_modelo.md)).

## 3. Objetivos

**Objetivo geral.** Oferecer um app e uma API que traduzam dados de irradiância e clima
em uma estimativa de geração e economia com energia solar fotovoltaica, em linguagem
acessível e com contexto sobre o setor.

**Objetivos específicos.**

1. Coletar dados climáticos históricos de múltiplas localidades brasileiras (NASA POWER).
2. Extrair conteúdo da web (notícias do setor solar) com Beautiful Soup e alimentar a interface com nuvem de palavras e estatísticas.
3. Permitir que o usuário simule um sistema (potência, desempenho, tarifa) e compare locais e períodos.
4. Permitir que o usuário complemente o sistema com seus próprios dados e exporte os dados consolidados.
5. Documentar o projeto (Project Charter, Data Summary Report, metodologia) e versioná-lo com Git, pronto para deploy.
6. **(TP3)** Coletar com **Selenium** dados de uma página dinâmica (formulário do SunData/CRESESB) com a irradiação de longo prazo, inclusive no plano inclinado.
7. **(TP3)** Disponibilizar os dados e cálculos em uma **API FastAPI** documentada, com rotas de consulta (GET) e envio (POST), e fazer o app consumir essa API.
8. **(TP3)** Organizar o app em **múltiplas páginas** com menu de navegação, uma por funcionalidade.
9. **(TP3)** Preparar os dados para **LLMs** (corpus, fichas de local, recuperação de contexto) e planejar os casos de uso do TP4.
10. **(TP4)** Escolher, com critérios de desempenho, custo computacional e acessibilidade, um **LLM que rode localmente** e integrá-lo à API FastAPI.
11. **(TP4)** Oferecer no app perguntas e respostas com fontes, resumo e sentimento de notícias e relatório de viabilidade, com respostas estruturadas (Response Models).
12. **(TP4)** Tratar os erros da API de forma robusta e informativa (modelo carregando, entrada inválida, texto longo, prazo, falta de memória).

## 4. Metas e indicadores de sucesso (KPIs)

| Meta | Indicador | Meta | TP2 | TP3 | TP4 |
|---|---|---|---|---|---|
| Cobertura de dados | Locais com série climática completa | ≥ 5 capitais | 10 capitais, 366 dias | 10 capitais, 366 dias (série atualizada até 17/10/2026) | igual ao TP3 |
| Dados de longo prazo | Capitais com irradiação mensal típica e plano inclinado | ≥ 5 | n/a | 10 capitais x 4 planos de inclinação (Selenium) | igual ao TP3 |
| Qualidade dos dados | Ausentes em `ALLSKY_SFC_SW_DWN` | < 5% | 1,9% | 1,9% | 1,9% |
| Conteúdo da web | Fontes raspadas e salvas em CSV/TXT | ≥ 2 fontes | 2 fontes, 30 notícias | 3 fontes; 54 notícias, 51 com texto completo (~24,6 mil palavras) | igual ao TP3 |
| Páginas e navegação | Páginas acessíveis pelo menu | ≥ 4 | 1 (abas) | 7 páginas em 4 seções | 9 páginas em 4 seções (3 de IA) |
| API | Rotas documentadas (Swagger) | ≥ 2 (GET e POST) | n/a | 15 rotas (11 GET, 1 POST, 1 DELETE e 2 utilitárias) | 20 rotas, todas com Response Model |
| Integração app + API | Páginas que leem/enviam dados pela API | todas as que usam dados | n/a | todas (o app não lê `data/` diretamente) | todas, inclusive as de IA |
| Robustez | Entradas inválidas rejeitadas com mensagem clara | Sim/Não | Sim (upload) | Sim (upload e API: 404/409/422) | Sim: formato único de erro; 404, 405, 409, 413, 422, 500, 502, 503 e 504 |
| Preparação para LLM | Corpus em trechos com fonte e data | Sim/Não | n/a | 163 documentos (153 trechos de notícias + 10 fichas de local) | 164 documentos (+ ranking das capitais) |
| **(TP4)** LLM local | Modelo executado na máquina, sem nuvem | Sim/Não | n/a | n/a | Sim: Qwen3-1.7B (Transformers + PyTorch), 3,2 GB na GPU |
| **(TP4)** Escolha do modelo | Candidatos medidos nas tarefas do projeto | ≥ 3 | n/a | n/a | 5 LLMs + 1 classificador BERT |
| **(TP4)** Qualidade do LLM | Sentimento (34 notícias rotuladas) / perguntas com fonte / relatórios sem número inventado | ≥ 70% / ≥ 80% / 100% | n/a | n/a | 79% / 100% (10 de 10, com citação) / 100% (3 de 3) |
| **(TP4)** Tempo de resposta | Resposta do assistente (GPU) | ≤ 10 s | n/a | n/a | ~2 s (resumo ~3 s, relatório ~7 s) |
| Qualidade de código | Testes automatizados passando | Sim/Não | 10 testes | 44 testes (API, parsers, LLM, app) | 93 testes (+ 1 opcional com o modelo real) |

## 5. Escopo

**Dentro do escopo (TP1 a TP3):**

- Definição do problema, ODS, público-alvo e metodologia (TDSP/CRISP-DM).
- Coleta da NASA POWER para 10 capitais e consulta ao vivo para coordenadas livres.
- Web scraping com Beautiful Soup (ABSOLAR e Canal Solar, com texto completo e coletas acumuladas).
- **(TP3)** Web scraping de página dinâmica com Selenium (CRESESB/SunData).
- **(TP3)** API FastAPI com consulta, estimativa, notícias, envio/remoção de medições e recuperação de contexto para LLM.
- **(TP3)** App Streamlit com 7 páginas e menu de navegação, consumindo a API.
- **(TP3)** Preparação dos dados para LLM e planejamento dos casos de uso.
- **(TP4)** LLM local (HuggingFace Transformers) integrado à API: resumo, sentimento, perguntas e respostas com fontes e relatório de viabilidade; tratamento de erros padronizado.
- Repositório Git, testes automatizados e configuração para deploy.

**Fora do escopo (etapas futuras):**

- Busca semântica com embeddings e ajuste fino (*fine-tuning*) do modelo com dados do setor.
- Modelo de geração detalhado (orientação do telhado diferente do norte, sombreamento, degradação).
- Retorno financeiro completo (payback, financiamento, regras de compensação da ANEEL).
- Geocodificação de endereço e mapa interativo.
- Autenticação de usuários e controle de acesso na API (hoje qualquer cliente pode enviar e remover medições).

## 6. Stakeholders

| Stakeholder | Papel | Interesse / expectativa |
|---|---|---|
| Usuário residencial / pequeno negócio | Usuário final | Um primeiro diagnóstico rápido, gratuito e neutro sobre solar |
| Felipe Castro Sotto Mayor | Autor, analista de dados e desenvolvedor | Entregar o projeto conforme os critérios da disciplina |
| Professor/avaliador (Infnet) | Patrocinador acadêmico | Aderência a CRISP-DM/TDSP, uso de dados da web, qualidade técnica e ODS |
| Instaladores e integradores solares | Beneficiário indireto; **(TP3)** possível cliente da API | Clientes mais informados; estimativas via API para usar em orçamentos |
| Desenvolvedores de outros apps | **(TP3)** Clientes da API | Rotas estáveis e documentadas (Swagger em `/docs`) |
| Setor solar (ex.: ABSOLAR) e distribuidoras | Fonte de informação / partes interessadas | Divulgação da fonte solar; as notícias do setor alimentam o app |
| NASA POWER, CRESESB/CEPEL, ABSOLAR, Canal Solar | Provedores de dados | Uso responsável e citação das fontes (sem sobrecarga nos servidores) |
| Sociedade e meio ambiente | Beneficiário final | Maior adoção de energia limpa e menor emissão de gases de efeito estufa |

## 7. Público-alvo

Pessoas físicas e pequenos negócios no Brasil que estão avaliando a instalação de
painéis solares e precisam de um primeiro diagnóstico de viabilidade, sem
conhecimento técnico prévio em energia ou meteorologia. A partir do TP3, a API também
atende, de forma secundária, quem quer reutilizar os dados em outras aplicações.

## 8. ODS e justificativa

- **ODS 7, Energia limpa e acessível (principal):** reduz a barreira de informação para a
  adoção de energia solar (metas 7.1 e 7.2).
- **ODS 13, Ação contra a mudança global do clima (secundário):** incentiva a
  substituição de geração fóssil por solar, reduzindo emissões.

## 9. Fontes de dados

NASA POWER API, **CRESESB/SunData (Selenium, TP3)**, notícias da ABSOLAR e do Canal
Solar (Beautiful Soup) e medições enviadas pelo usuário (agora pela API, guardadas em
`data/processed/`). Detalhes no [Data Summary Report](../data_report/data_summary_report.md).

## 9.1 Iniciativas e projetos relacionados (inspiração)

| Iniciativa | Instituição | O que inspirou no Solar Viável |
|---|---|---|
| [Atlas Brasileiro de Energia Solar](https://www.gov.br/pt-br/servicos/obter-dados-do-atlas-brasileiro-de-energia-solar-atlas-solar) | INPE/LABREN | Comparação do potencial solar entre regiões do Brasil (17 anos de imagens de satélite) |
| [SunData](https://cresesb.cepel.br/index.php?section=sundata) | CRESESB/CEPEL | Irradiação média mensal e no plano inclinado; virou fonte de dados no TP3 |
| [Global Solar Atlas](https://globalsolaratlas.info/) | Banco Mundial/ESMAP e Solargis | Diagnóstico rápido de pré-viabilidade por localização |
| [PVWatts](https://pvwatts.nlr.gov/pvwatts.php) | NREL/NLR (EUA) | Estimativa de geração a partir de potência (kWp) e taxa de desempenho; também oferece API pública, referência para a API do projeto |
| [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/) | Comissão Europeia (JRC) | Estimativa fotovoltaica gratuita com premissas explícitas |
| [Micro e Minigeração Distribuída](https://www.gov.br/aneel/pt-br/assuntos/geracao-distribuida) | ANEEL | Contexto regulatório da geração solar própria |
| [Infográfico do mercado](https://www.absolar.org.br/mercado/infografico/) | ABSOLAR | Panorama do setor solar brasileiro |

**Posicionamento:** essas ferramentas são, em sua maioria, voltadas a profissionais e/ou
em inglês. O Solar Viável propõe um primeiro diagnóstico em português, neutro e acessível,
que une clima, estimativa de geração, notícias do setor e dados do próprio usuário.

## 10. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Dados ausentes (`-999`), especialmente `CLOUD_AMT` (21,6%) | Conversão para `NaN`; estimativa usa `ALLSKY_SFC_SW_DWN` (1,9% de ausentes) |
| API da NASA fora do ar | Dados salvos em `data/raw/`; a consulta ao vivo (via API) devolve erro 502 tratado, sem derrubar o app |
| Sites mudarem o HTML e quebrarem os scrapers | Scrapers isolados do app; dados já salvos continuam disponíveis; testes dos parsers com HTML salvo (`tests/fixtures/`) acusam a quebra |
| **(TP3)** Selenium depende de navegador e é mais lento | Coleta roda só quando necessário (dados de longo prazo mudam pouco), fora do app; o Selenium Manager baixa o driver compatível |
| Scraping abusivo ou contra os termos do site | Respeito ao `robots.txt`, `User-Agent` identificado e pausa entre requisições e consultas |
| **(TP3)** Envio de dados inválidos ou abusivos à API | Validação com Pydantic (faixas físicas plausíveis, datas, tamanho do lote), nomes das capitais protegidos (409); autenticação fica para depois |
| **(TP3)** API fora do ar no deploy | O app sobe a própria API em uma thread quando não encontra uma no endereço configurado |
| **(TP4)** LLM inventar números ou fatos | Números sempre calculados pela API; o LLM só explica e resume com base no contexto recuperado, citando as fontes; no relatório, cada número do texto é conferido com a API e o app avisa se algum não confere |
| **(TP4)** Máquina sem GPU ou com pouca memória | Modelo de 1,7 B parâmetros (3,2 GB em float16); escolha automática de GPU/CPU; modelo trocável por variável de ambiente (0,5 B ocupa 1 GB); erros 503 explicativos se faltar memória |
| **(TP4)** Modelo lento travar a API | Carga em segundo plano, uma inferência por vez com fila limitada (503) e prazo por geração (504) |
| Usuário confundir a estimativa com projeto de engenharia | Premissas e limitações explicitadas na interface e na documentação |
| Escopo crescer demais | Priorizar residencial/pequeno negócio e funcionalidades do enunciado |

## 11. Cronograma macro (por fase TDSP)

| Fase TDSP | Entrega | Status |
|---|---|---|
| Business Understanding | Project Charter, metodologia | Concluído (TP1), revisado no TP2 e no TP3 |
| Data Acquisition and Understanding | NASA POWER, scrapers BS4, Selenium (CRESESB), Data Summary Report | Concluído no TP2, ampliado no TP3 |
| Deployment | App com múltiplas páginas + API FastAPI, Git, testes e config de deploy | TP3, ampliado no TP4 (LLM local na API) |
| Modeling | Estimativa com longo prazo e inclinação (TP3); escolha e avaliação do LLM local; Q&A, resumo, sentimento e relatório | Concluído no TP4 |
| Customer Acceptance | Validação com usuários-alvo | Entrega final |
