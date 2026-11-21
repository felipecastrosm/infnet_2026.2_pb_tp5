# API do Solar Viável (FastAPI)

A API expõe os dados e cálculos do projeto por HTTP. O app Streamlit é o principal
cliente (todas as páginas consultam e enviam dados por ela), mas qualquer outro
programa, como uma planilha ou outro app, pode usá-la. Desde o TP4 ela também executa um
**LLM local** (HuggingFace Transformers): ver [Integração local do LLM](llm_integracao_local.md).

- Código: [`src/api/main.py`](../src/api/main.py) (rotas) e [`src/api/schemas.py`](../src/api/schemas.py) (modelos Pydantic)
- Acesso aos arquivos: [`src/repositorio.py`](../src/repositorio.py)
- Cliente usado pelo app: [`src/api_client.py`](../src/api_client.py)
- LLM: [`src/llm/`](../src/llm/) (serviço, tarefas e exceções)
- Testes: [`tests/test_api.py`](../tests/test_api.py) e [`tests/test_llm.py`](../tests/test_llm.py)

## Como rodar

```bash
uvicorn src.api.main:app --reload            # http://127.0.0.1:8000
```

- Documentação interativa (Swagger, permite testar as rotas): http://127.0.0.1:8000/docs
- Documentação alternativa (ReDoc): http://127.0.0.1:8000/redoc
- Especificação OpenAPI (JSON): http://127.0.0.1:8000/openapi.json

O app procura a API em `SOLAR_API_URL` (padrão `http://127.0.0.1:8000`). Se ela não estiver
no ar e o endereço for local, o próprio app a inicia em uma thread (útil no Streamlit
Community Cloud). A página **Explorar a API** do app lista as rotas a partir do
`/openapi.json` e permite enviar requisições.

## Rotas

| Método | Rota | Grupo | Para que serve |
|---|---|---|---|
| GET | `/` | Geral | Boas-vindas e links |
| GET | `/saude` | Geral | Verifica se a API está no ar (usada pelo app) |
| GET | `/locais` | Locais e clima | Lista os locais (capitais + enviados por usuários) com período e irradiância média |
| GET | `/clima` | Locais e clima | Série diária de clima, filtrável por local(is) e período |
| GET | `/clima/resumo` | Locais e clima | Média, mínimo e máximo de cada variável por local |
| GET | `/clima/ao-vivo` | Locais e clima | Consulta a NASA POWER para qualquer coordenada (não salva) |
| GET | `/irradiacao-longo-prazo` | Longo prazo | Irradiação mensal típica do CRESESB, por plano de inclinação |
| GET | `/estimativa` | Estimativa | Geração (kWh) e economia (R$) de um sistema |
| GET | `/noticias` | Notícias | Notícias do setor, com busca e filtro por fonte |
| GET | `/noticias/palavras` | Notícias | Palavras mais frequentes (sem stopwords) |
| **POST** | `/medicoes` | Medições do usuário | **Envia** medições de um local |
| GET | `/medicoes` | Medições do usuário | Lista as medições enviadas |
| **DELETE** | `/medicoes/{cidade}` | Medições do usuário | Remove as medições de um local |
| **POST** | `/processar_texto` | Processamento de texto (LLM) | **Resumo** ou **sentimento** de um texto, pelo LLM local |
| GET | `/llm/status` | LLM | Estado do modelo (carregando, pronto...), hardware, memória e limites |
| **POST** | `/llm/carregar` | LLM | Inicia a carga do modelo em segundo plano (202) |
| **POST** | `/llm/perguntar` | LLM | Perguntas e respostas sobre notícias e capitais (RAG), com fontes |
| **POST** | `/llm/relatorio` | LLM | Relatório de viabilidade em linguagem simples, com checagem dos números |
| GET | `/llm/corpus` | LLM | Estatísticas do corpus usado pelo LLM |
| GET | `/llm/contexto` | LLM | Mostra os trechos que seriam enviados ao modelo para uma pergunta |

Toda rota declara um **Response Model** (Pydantic), então o FastAPI valida cada resposta
antes de enviá-la e o `/docs` mostra o formato exato.

## Erros

Todo erro, de qualquer rota, sai no mesmo formato (`RespostaErro`):

```json
{
  "status": 503,
  "erro": "modelo_carregando",
  "mensagem": "O modelo 'Qwen/Qwen3-1.7B' ainda está sendo carregado.",
  "sugestao": "O modelo está sendo carregado na memória. Tente de novo em alguns segundos.",
  "detalhes": null,
  "caminho": "/processar_texto"
}
```

| HTTP | `erro` | Quando acontece |
|---|---|---|
| 404 | `nao_encontrado` | local inexistente, rota inexistente, nenhuma medição para apagar |
| 405 | `metodo_nao_permitido` | método errado (ex.: GET em `/processar_texto`) |
| 409 | `conflito` | enviar medições com o nome de uma capital da base |
| 413 | `texto_longo_demais` | o texto passa do limite de tokens do modelo (`SOLAR_LLM_MAX_TOKENS_ENTRADA`) |
| 422 | `dados_invalidos` | corpo ou parâmetro inválido (texto curto, tarefa desconhecida, JSON malformado...); `detalhes` diz o campo e o problema, em português |
| 500 | `falha_na_inferencia` | erro inesperado do PyTorch ao executar o modelo |
| 500 | `erro_interno` | qualquer outra exceção não tratada (a mensagem não expõe detalhes internos; o log da API registra) |
| 502 | `saida_invalida_do_modelo` | o modelo respondeu, mas a saída não é aproveitável (vazia, curta demais) |
| 502 | `falha_em_servico_externo` | a NASA POWER falhou na consulta ao vivo |
| 503 | `modelo_carregando` | o modelo ainda está sendo carregado (cabeçalho `Retry-After: 15`) |
| 503 | `modelo_nao_carregado` | o modelo não foi carregado (`SOLAR_LLM_CARREGAR=0`); use `POST /llm/carregar` |
| 503 | `modelo_ocupado` | a fila de inferência ficou cheia por mais de `SOLAR_LLM_ESPERA_FILA_S` (`Retry-After: 10`) |
| 503 | `memoria_insuficiente` | falta de memória na GPU/CPU durante a inferência |
| 503 | `falha_ao_carregar_modelo` | modelo inexistente, não baixado (modo offline) ou sem memória para carregar |
| 503 | `modelo_desativado` | LLM desligado (`SOLAR_LLM_DESATIVADO=1`) ou PyTorch/Transformers não instalados |
| 504 | `tempo_esgotado` | a geração passou de `SOLAR_LLM_TIMEOUT_S` e foi interrompida |

## Detalhes e exemplos

### `GET /locais`

Parâmetro: `incluir_medicoes` (bool, padrão `true`). Resposta (lista, ordenada pela irradiância):

```json
[{"cidade": "Recife (PE)", "latitude": -8.0476, "longitude": -34.877, "origem": "nasa",
  "dias": 366, "inicio": "2025-10-17", "fim": "2026-10-17", "irradiacao_media": 6.057}]
```

### `GET /clima`

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `cidade` | texto, repetível | um ou mais locais; vazio = todos |
| `inicio`, `fim` | data (AAAA-MM-DD) | período (opcional) |
| `incluir_medicoes` | bool | inclui medições de usuários (padrão `true`) |

```bash
curl "http://127.0.0.1:8000/clima?cidade=Recife%20(PE)&cidade=Manaus%20(AM)&inicio=2026-01-01&fim=2026-01-31"
```

Cada item traz `date`, `cidade`, `latitude`, `longitude`, `ALLSKY_SFC_SW_DWN`, `T2M`,
`RH2M`, `WS10M`, `CLOUD_AMT` e `origem` (`nasa` ou `usuario`). Ausentes vêm como `null`.

### `GET /clima/ao-vivo`

`latitude` (−35 a 6), `longitude` (−75 a −30), `dias` (7 a 365, padrão 180), `nome`.
Usada pelo formulário *Consultar outro local* da página *Clima e geração*.

### `GET /irradiacao-longo-prazo`

`cidade` (repetível) e `plano` (`Plano Horizontal`, `Ângulo igual a latitude`,
`Maior média anual` ou `Maior mínimo mensal`).

```json
[{"cidade": "Curitiba (PR)", "estacao": "Curitiba", "municipio": "Curitiba", "uf": "PR",
  "distancia_km": 3.9, "plano": "Maior média anual", "inclinacao_graus": 21,
  "mensal": {"jan": 4.93, "fev": 5.01, "...": "...", "dez": 5.08}, "media_anual": 4.39, "delta": 1.56}]
```

### `GET /estimativa`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `cidade` | (obrigatório) | local |
| `potencia_kwp` | 5,0 | potência instalada (0 a 1000) |
| `desempenho` | 0,80 | taxa de desempenho (0,5 a 0,95) |
| `tarifa` | 0,90 | R$/kWh |
| `fonte` | `cresesb` | `cresesb`: ano típico, painel na inclinação de maior média anual, mês a mês; `nasa`: média do último ano no plano horizontal (funciona também para locais enviados por usuários) |

```bash
curl "http://127.0.0.1:8000/estimativa?cidade=Fortaleza%20(CE)&potencia_kwp=4"
```

```json
{"cidade": "Fortaleza (CE)", "fonte": "cresesb", "irradiacao_media": 5.7, "potencia_kwp": 4.0,
 "desempenho": 0.8, "tarifa": 0.9, "geracao_dia_kwh": 18.24, "geracao_mes_kwh": 554.7,
 "geracao_ano_kwh": 6656.0, "economia_mes_rs": 499.23, "economia_ano_rs": 5990.76,
 "observacao": "Estação Fortaleza (8.1 km), painéis a 3° voltados para o norte."}
```

### `GET /noticias` e `GET /noticias/palavras`

`/noticias`: `fonte` (repetível: `ABSOLAR`, `Canal Solar`), `busca` (termo no título,
resumo ou texto), `limite` (1 a 500, padrão 50) e `incluir_texto` (padrão `false`).
`/noticias/palavras`: `fonte` e `top` (1 a 100).

### `POST /medicoes`

Corpo (JSON):

```json
{
  "cidade": "Meu telhado - Niterói",
  "latitude": -22.88,
  "longitude": -43.10,
  "medicoes": [
    {"date": "2026-09-01", "ALLSKY_SFC_SW_DWN": 5.2, "T2M": 24.1},
    {"date": "2026-09-02", "ALLSKY_SFC_SW_DWN": 3.8}
  ]
}
```

```bash
curl -X POST http://127.0.0.1:8000/medicoes -H "Content-Type: application/json" \
  -d '{"cidade": "Meu telhado", "medicoes": [{"date": "2026-09-01", "ALLSKY_SFC_SW_DWN": 5.2}]}'
```

Resposta **201**: `{"cidade": "Meu telhado", "recebidas": 1, "mensagem": "1 medição(ões) salva(s)."}`.
As regras de validação estão no [Data Summary Report, seção 6](data_report/data_summary_report.md#6-medições-do-usuário-fonte-5-agora-via-api).
O local passa a aparecer em `/locais`, `/clima` e `/estimativa?fonte=nasa`.

### `GET /medicoes` e `DELETE /medicoes/{cidade}`

`GET /medicoes?cidade=...` lista o que foi enviado. `DELETE /medicoes/Meu%20telhado` remove
o local (`{"cidade": ..., "removidas": n}`) ou devolve 404 se não houver medições.

### `POST /processar_texto`

| Campo | Padrão | Descrição |
|---|---|---|
| `texto` | (obrigatório) | 20 a 20.000 caracteres, com pelo menos 3 palavras |
| `tarefa` | `resumo` | `resumo` ou `sentimento` |
| `max_palavras` | 80 | resumo: 20 a 250 palavras |
| `ponto_de_vista` | `geral` | sentimento: `geral` (tom do texto) ou `setor_solar` (para quem quer instalar solar) |
| `justificar` | `true` | sentimento: pede uma frase de justificativa ao modelo |

```bash
curl -X POST http://127.0.0.1:8000/processar_texto -H "Content-Type: application/json" \
  -d '{"texto": "Os cortes de geração já somam 73,4 milhões de MWh desde 2021 e podem causar prejuízo de até R$ 13,6 bilhões às usinas renováveis.", "tarefa": "sentimento", "ponto_de_vista": "setor_solar"}'
```

Resposta (`RespostaProcessamento`; o campo `resultado` muda conforme `tarefa`):

```json
{
  "tarefa": "sentimento",
  "entrada": {"n_caracteres": 143, "n_palavras": 25, "n_tokens": 54, "inicio": "Os cortes de geração..."},
  "resultado": {
    "tipo": "sentimento", "sentimento": "negativo", "confianca": 1.0,
    "probabilidades": {"positivo": 0.0, "neutro": 0.0, "negativo": 1.0},
    "ponto_de_vista": "setor_solar",
    "justificativa": "Os cortes de geração podem causar prejuízos significativos às usinas renováveis, prejudicando sua operação e rendimento."
  },
  "modelo": {"nome": "Qwen/Qwen3-1.7B", "dispositivo": "mps", "dtype": "float16", "parametros_milhoes": 1720.6},
  "uso": {"tokens_entrada": 362, "tokens_saida": 32, "tempo_s": 1.01, "tokens_por_s": 31.8, "em_cache": false},
  "processado_em": "2026-11-16T14:14:49"
}
```

Com `tarefa=resumo`, `resultado` traz `resumo`, `n_palavras_resumo`, `taxa_compressao` e
`cortado`. A mesma entrada devolve sempre a mesma saída (decodificação gulosa) e, a partir
da segunda vez, vem do cache (`em_cache: true`).

### `POST /llm/perguntar`

Corpo: `{"pergunta": "Quanto um sistema de 5 kWp gera por mês em Curitiba?", "k": 4}`
(`k` = trechos do corpus enviados ao modelo, 1 a 8). Resposta (`RespostaPergunta`):
`resposta` (com citações [n]), `fontes` (título, fonte, link, data e se foi `citada`),
`sem_contexto` (true quando nenhum trecho foi encontrado; o modelo nem é chamado),
`modelo` e `uso`.

### `POST /llm/relatorio`

Corpo: `{"cidade": "Curitiba (PR)", "potencia_kwp": 4, "desempenho": 0.8, "tarifa": 0.95}`.
A API calcula a estimativa (mesma lógica de `GET /estimativa`, fonte CRESESB), junta a
irradiação mês a mês e a ficha do local e pede ao modelo um relatório em Markdown.
Resposta (`RespostaRelatorio`): `relatorio`, `estimativa` (os números de origem) e
`verificacao` (`numeros_citados`, `nao_encontrados` e `confiavel`): cada número do texto
é procurado nos dados de entrada, com tolerância de arredondamento.

### `GET /llm/status` e `POST /llm/carregar`

`/llm/status` devolve `estado` (`nao_carregado`, `carregando`, `pronto`, `erro`,
`desativado`), `modelo`, `dispositivo` (`cuda`, `mps` ou `cpu`), `dtype`,
`parametros_milhoes`, `tempo_carregamento_s`, memória do modelo e do processo,
`requisicoes_atendidas`, `limites` e o `erro` da carga, se houver. `/llm/carregar`
responde **202** e começa a carregar o modelo em segundo plano.

### `GET /llm/corpus` e `GET /llm/contexto`

`/llm/corpus` devolve o total de documentos e palavras, por tipo e por fonte.
`/llm/contexto?pergunta=...&k=4` mostra os trechos que `POST /llm/perguntar` enviaria ao
modelo (TF-IDF + regras de metadados: ficha da cidade citada e ranking para perguntas
comparativas) e o prompt montado.

## Como o app usa a API

| Página do app | Rotas |
|---|---|
| Clima e geração | `GET /locais`, `GET /clima`, `GET /clima/ao-vivo` |
| Potencial de longo prazo | `GET /irradiacao-longo-prazo`, `GET /estimativa`, `GET /clima/resumo`, `POST /llm/relatorio` |
| Notícias do setor | `GET /noticias`, `POST /processar_texto` (botão *Resumir e avaliar com IA*) |
| Meus dados | `POST /medicoes`, `GET /medicoes`, `DELETE /medicoes/{cidade}`, `GET /clima`, `GET /noticias`, `GET /irradiacao-longo-prazo` |
| Explorar a API | `GET /openapi.json` e qualquer rota escolhida pelo usuário |
| Pergunte ao Solar Viável | `GET /llm/status`, `POST /llm/perguntar` |
| Resumo e sentimento | `GET /llm/status`, `GET /noticias`, `POST /processar_texto` |
| Modelo local (LLM) | `GET /llm/status`, `POST /llm/carregar`, `GET /llm/corpus`, `GET /llm/contexto` |
