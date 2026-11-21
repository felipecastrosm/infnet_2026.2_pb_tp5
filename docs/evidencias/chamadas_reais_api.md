# Chamadas reais à API com o modelo carregado

Gerado por `scripts/registrar_chamadas_reais.py` em 21/11/2026 10:20, chamando a API **em execução** (`http://127.0.0.1:8020`) com o LLM local carregado. Diferente dos testes automatizados (que usam um modelo falso), aqui todas as respostas vieram do modelo real. As respostas completas estão em [chamadas_reais_api.json](chamadas_reais_api.json).

## Ambiente

- Máquina: Darwin 25.6.0 (arm64), Python 3.12.14
- Modelo: **Qwen/Qwen3-1.7B**, 1720.6 milhões de parâmetros, dispositivo `mps`, `float16`
- Carregado em 2.5 s; memória do modelo: 3296.2 MB

## Resumo das chamadas

| # | Chamada | Rota | HTTP | Tempo total |
|---|---|---|---|---|
| 1 | Estado do modelo | `GET /llm/status` | 200 | 0.0 s |
| 2 | Resumo de uma notícia | `POST /processar_texto` | 200 | 1.66 s |
| 3 | Sentimento para quem quer instalar | `POST /processar_texto` | 200 | 1.27 s |
| 4 | Agente consultor (Curitiba, preocupação com o inverno) | `POST /agente/recomendar` | 200 | 11.82 s |
| 5 | Nova conversa sobre a recomendação | `POST /conversas` | 201 | 0.0 s |
| 6 | Mensagem com memória: "Quanto o sistema gera no pior mês?" | `POST /conversas/85004508432e/mensagens` | 200 | 2.26 s |
| 7 | Mensagem com memória: "E em Fortaleza, como seria?" | `POST /conversas/85004508432e/mensagens` | 200 | 2.28 s |
| 8 | Mensagem com memória: "Qual das duas cidades tem mais sol?" | `POST /conversas/85004508432e/mensagens` | 200 | 2.02 s |
| 9 | Histórico e memória da conversa | `GET /conversas/85004508432e` | 200 | 0.0 s |
| 10 | Painel (ciclo de ciência de dados) | `GET /painel` | 200 | 0.03 s |

## 1. Estado do modelo

`GET /llm/status` em 2026-11-21T10:19:56, HTTP 200, 0.0 s.

Resposta:

```json
{
  "estado": "pronto",
  "pronto": true,
  "modelo": "Qwen/Qwen3-1.7B",
  "dispositivo": "mps",
  "dtype": "float16",
  "parametros_milhoes": 1720.6,
  "carregado_em": "2026-11-21T10:19:53.207181",
  "tempo_carregamento_s": 2.5,
  "memoria_modelo_mb": 3296.2,
  "memoria_processo_mb": 1385.9,
  "requisicoes_atendidas": 0,
  "limites": {
    "timeout_s": 90.0,
    "espera_fila_s": 30.0,
    "max_tokens_entrada": 3000
  },
  "erro": null,
  "tarefas": [
    "POST /processar_texto",
    "POST /llm/perguntar",
    "POST /llm/relatorio",
    "POST /agente/recomendar",
    "POST /conversas/{id}/mensagens"
  ]
}
```

## 2. Resumo de uma notícia

`POST /processar_texto` em 2026-11-21T10:19:56, HTTP 200, 1.66 s.

Requisição:

```json
{
  "texto": "A fonte solar respondeu por 73% da expansão da matriz elétrica brasileira entre janeiro e agosto de 2026, com 3,5 GW adicionados. A capacidade total de geração centralizada do país já supera 220 GW, segundo a ANEEL. Para a ABSOLAR, o resultado mostra a competitividade da fonte, mas o setor alerta para os cortes de geração (curtailment) que reduzem a receita das usinas no Nordeste.",
  "tarefa": "resumo",
  "max_palavras": 40
}
```

Modelo **Qwen/Qwen3-1.7B** em **mps** (float16), 195 tokens de entrada, 100 de saída, 1.65 s de processamento.

**Resumo gerado:** A fonte solar contribuiu com 73% da expansão da matriz elétrica brasileira entre janeiro e agosto de 2026, adicionando 3,5 GW, enquanto a capacidade centralizada atingiu 220 GW.

Resposta:

```json
{
  "tarefa": "resumo",
  "entrada": {
    "n_caracteres": 383,
    "n_palavras": 65,
    "n_tokens": 195,
    "inicio": "A fonte solar respondeu por 73% da expansão da matriz elétrica brasileira entre janeiro e agosto de 2026, com 3,5 GW adi"
  },
  "resultado": {
    "tipo": "resumo",
    "resumo": "A fonte solar contribuiu com 73% da expansão da matriz elétrica brasileira entre janeiro e agosto de 2026, adicionando 3,5 GW, enquanto a capacidade centralizada atingiu 220 GW.",
    "n_palavras_resumo": 28,
    "taxa_compressao": 0.431,
    "cortado": false
  },
  "modelo": {
    "nome": "Qwen/Qwen3-1.7B",
    "dispositivo": "mps",
    "dtype": "float16",
    "parametros_milhoes": 1720.6
  },
  "uso": {
    "tokens_entrada": 195,
    "tokens_saida": 100,
    "tempo_s": 1.65,
    "tokens_por_s": 60.5,
    "em_cache": false
  },
  "processado_em": "2026-11-21T10:19:58.148447"
}
```

## 3. Sentimento para quem quer instalar

`POST /processar_texto` em 2026-11-21T10:19:58, HTTP 200, 1.27 s.

Requisição:

```json
{
  "texto": "A fonte solar respondeu por 73% da expansão da matriz elétrica brasileira entre janeiro e agosto de 2026, com 3,5 GW adicionados. A capacidade total de geração centralizada do país já supera 220 GW, segundo a ANEEL. Para a ABSOLAR, o resultado mostra a competitividade da fonte, mas o setor alerta para os cortes de geração (curtailment) que reduzem a receita das usinas no Nordeste.",
  "tarefa": "sentimento",
  "ponto_de_vista": "setor_solar"
}
```

Modelo **Qwen/Qwen3-1.7B** em **mps** (float16), 491 tokens de entrada, 52 de saída, 1.27 s de processamento.

**Sentimento:** neutro (100%). O texto apresenta informações objetivas sobre o desempenho da energia solar no Brasil, incluindo crescimento e dados técnicos, mas também menciona alertas sobre cortes de geração, mantendo um tom equilibrado.

Resposta:

```json
{
  "tarefa": "sentimento",
  "entrada": {
    "n_caracteres": 383,
    "n_palavras": 65,
    "n_tokens": 118,
    "inicio": "A fonte solar respondeu por 73% da expansão da matriz elétrica brasileira entre janeiro e agosto de 2026, com 3,5 GW adi"
  },
  "resultado": {
    "tipo": "sentimento",
    "sentimento": "neutro",
    "confianca": 0.9967,
    "probabilidades": {
      "positivo": 0.0033,
      "neutro": 0.9967,
      "negativo": 0.0
    },
    "ponto_de_vista": "setor_solar",
    "justificativa": "O texto apresenta informações objetivas sobre o desempenho da energia solar no Brasil, incluindo crescimento e dados técnicos, mas também menciona alertas sobre cortes de geração, mantendo um tom equilibrado."
  },
  "modelo": {
    "nome": "Qwen/Qwen3-1.7B",
    "dispositivo": "mps",
    "dtype": "float16",
    "parametros_milhoes": 1720.6
  },
  "uso": {
    "tokens_entrada": 491,
    "tokens_saida": 52,
    "tempo_s": 1.27,
    "tokens_por_s": 41.0,
    "em_cache": false
  },
  "processado_em": "2026-11-21T10:19:59.420614"
}
```

## 4. Agente consultor (Curitiba, preocupação com o inverno)

`POST /agente/recomendar` em 2026-11-21T10:19:59, HTTP 200, 11.82 s.

Requisição:

```json
{
  "cidade": "Curitiba (PR)",
  "consumo_kwh_mes": 250,
  "tarifa": 0.92,
  "ligacao": "bifasica",
  "objetivo": "Me preocupa o inverno: a geração cai muito? Vale a pena mesmo assim?"
}
```

Modelo **Qwen/Qwen3-1.7B** em **mps** (float16), 1188 tokens de entrada, 597 de saída, 11.8 s de processamento.

**Passos escolhidos pelo agente:** potencial_solar, depois comparar_capitais, depois noticias_setor, depois dimensionar_sistema, depois analise_financeira, depois sazonalidade, depois impacto_ambiental, depois finalizar. **Viabilidade:** boa. **Números conferidos:** todos.

**Recomendação escrita pelo modelo:**

## Veredito  
A viabilidade da energia solar é **boa** para Curitiba (PR), considerando o potencial solar moderado, o custo de instalação e a economia financeira esperada. A geração varia significativamente entre meses, mas há oportunidades de aproveitar o excesso de energia em meses de alta produção.

## Por que  
O potencial solar de Curitiba é moderado, com geração média de 4,39 kWh/m²/dia, o que permite um sistema adequado para atender ao consumo de 250 kWh/mês. A economia de energia é real, com 94,1% do consumo atendido pela geração solar, e o retorno do investimento é de 3,8 anos. A geração varia entre 185,9 e 277,2 kWh/mês, o que oferece flexibilidade para ajustes no consumo. Além disso, o impacto ambiental é positivo, reduzindo emissões de CO₂ e contribuindo para a sustentabilidade.

## Ações recomendadas  
1. **Conferir o telhado**: Verifique se há espaço suficiente para instalar os painéis e se o telhado é adequado para a configuração solicitada.  
2. **Pedir orçamentos a instaladores**: Compare diferentes empresas para garantir a menor custo de instalação e melhores condições de serviço.  
3. **Acompanhar as regras da distribuidora**: Confira as normas da empresa de distribuição para evitar multas ou penalidades por irregularidades na conexão.

## Pontos de atenção  
- A Lei 14.300/2022 aumenta o Fio B, que pode afetar a economia. É importante considerar esse fator na análise financeira.  
- A geração está abaixo do consumo em alguns meses, o que pode gerar créditos energéticos, mas também demanda planejamento para evitar sobrecarga.  
- O sistema deve ser dimensionado com segurança, garantindo que a geração nunca ultrapasse o consumo em meses de baixa produtividade.

## Impacto ESG e ODS  
- Reduz 109 kg de CO₂ por ano e 2,6 toneladas em 25 anos, contribuindo para a mitigação de mudanças climáticas.  
- Atende 94,1% do consumo com energia renovável, alinhando-se aos objetivos de sustentabilidade (ODS 7).  
- A matriz elétrica brasileira é já pouco poluente, tornando a energia solar uma opção acessível e socialmente benéfica.

Resposta:

```json
{
  "id": "6f5e4d7ebcac",
  "criado_em": "2026-11-21T10:20:11",
  "perfil": {
    "cidade": "Curitiba (PR)",
    "consumo_kwh_mes": 250.0,
    "tarifa": 0.92,
    "ligacao": "bifasica",
    "custo_kwp": 3000.0,
    "desempenho": 0.8,
    "potencia_kwp": null,
    "reajuste_tarifa": 0.0
  },
  "objetivo": "Me preocupa o inverno: a geração cai muito? Vale a pena mesmo assim?",
  "passos": [
    {
      "n": 1,
      "acao": "potencial_solar",
      "titulo": "Potencial solar do local",
      "decisao": {
        "modo": "regra",
        "motivo": "única ação disponível neste passo",
        "probabilidades": {
          "potencial_solar": 1.0
        }
      },
      "resumo": "Curitiba (PR) recebe em média 4,39 kWh/m²/dia com os painéis inclinados a 21° (CRESESB), 9º lugar entre 10 capitais (potencial entre os mais baixos das capitais). Melhor mês: dez (5,08); pior mês: jun (3,52). No último ano, a NASA mediu 4,31 kWh/m²/dia no plano horizontal (2,9% em relação à média histórica).",
      "tempo_s": 0.0
    },
    {
      "n": 2,
      "acao": "comparar_capitais",
      "titulo": "Comparação com as capitais",
      "decisao": {
        "modo": "modelo",
        "motivo": "o modelo deu 53% de probabilidade a esta ação",
        "probabilidades": {
          "comparar_capitais": 0.5253,
          "dimensionar_sistema": 0.4637,
          "noticias_setor": 0.0109
        }
      },
      "resumo": "Com o mesmo consumo e tarifa, Curitiba (PR) fica em 9º lugar em tempo de retorno entre 10 capitais (1º = retorno mais rápido; retorno entre os mais lentos). Melhor: Fortaleza (CE); pior: Manaus (AM); payback mediano das capitais: 2,7 anos.",
      "tempo_s": 0.25
    },
    {
      "n": 3,
      "acao": "noticias_setor",
      "titulo": "Notícias do setor",
      "decisao": {
        "modo": "modelo",
        "motivo": "o modelo deu 73% de probabilidade a esta ação",
        "probabilidades": {
          "noticias_setor": 0.7292,
          "dimensionar_sistema": 0.2708
        }
      },
      "resumo": "Das 54 notícias do setor, 24 tratam de tarifa, regras ou custos (sentimento: 12 positivo, 11 neutro, 1 negativo). Destaques: \"ANEEL aprova reajuste nas tarifas da Equatorial Pará; veja os índices por classe\" (neutro); \"Projeto quer impedir novas cobranças sobre consumidores com energia solar\" (positivo); \"Reforma tributária: ANEEL propõe consultas sobre PIS/Cofins e créditos no setor elétrico\" (neutro).",
      "tempo_s": 0.18
    },
    {
      "n": 4,
      "
... (resposta completa em chamadas_reais_api.json)
```

## 5. Nova conversa sobre a recomendação

`POST /conversas` em 2026-11-21T10:20:11, HTTP 201, 0.0 s.

Requisição:

```json
{
  "recomendacao_id": "6f5e4d7ebcac"
}
```

Resposta:

```json
{
  "id": "85004508432e",
  "criada_em": "2026-11-21T10:20:11",
  "atualizada_em": "2026-11-21T10:20:11",
  "turnos": [],
  "memoria": {
    "fatos": {
      "cidade": "Curitiba (PR)",
      "cidade_recomendacao": "Curitiba (PR)",
      "cidades_citadas": [
        "Curitiba (PR)"
      ],
      "consumo_kwh_mes": 250.0,
      "tarifa": 0.92,
      "potencia_kwp": 2.2
    },
    "resumo": "",
    "total_turnos": 0,
    "turnos_na_janela": 0,
    "turnos_resumidos": 0,
    "tamanho_janela": 3,
    "recomendacao_id": "6f5e4d7ebcac"
  }
}
```

## 6. Mensagem com memória: "Quanto o sistema gera no pior mês?"

`POST /conversas/85004508432e/mensagens` em 2026-11-21T10:20:11, HTTP 200, 2.26 s.

Requisição:

```json
{
  "pergunta": "Quanto o sistema gera no pior mês?"
}
```

Modelo **Qwen/Qwen3-1.7B** em **mps** (float16), 2741 tokens de entrada, 102 de saída, 2.25 s de processamento.

**Pergunta entendida como:** Quantos kWh o sistema gera no pior mês? (modelo). **Fatos lembrados:** {"cidade": "Curitiba (PR)", "cidade_recomendacao": "Curitiba (PR)", "cidades_citadas": ["Curitiba (PR)"], "consumo_kwh_mes": 250.0, "tarifa": 0.92, "potencia_kwp": 2.2}.

**Resposta:** O sistema gera 185,9 kWh no pior mês [2]. O pior mês é junho, com 185,9 kWh [2]. No pior mês, o sistema gera aproximadamente 185,9 kWh [2]. A geração varia 49,1% entre o melhor e o pior mês [2].

Resposta:

```json
{
  "conversa_id": "85004508432e",
  "pergunta": "Quanto o sistema gera no pior mês?",
  "pergunta_reescrita": "Quantos kWh o sistema gera no pior mês?",
  "metodo_reescrita": "modelo",
  "resposta": "O sistema gera 185,9 kWh no pior mês [2]. O pior mês é junho, com 185,9 kWh [2]. No pior mês, o sistema gera aproximadamente 185,9 kWh [2]. A geração varia 49,1% entre o melhor e o pior mês [2].",
  "fontes": [
    {
      "n": 1,
      "titulo": "Recomendação do agente consultor",
      "fonte": "Agente consultor do Solar Viável",
      "tipo": "recomendacao",
      "url": "",
      "data": "2026-11-21",
      "citada": false
    },
    {
      "n": 2,
      "titulo": "Simulação para Curitiba (PR) com 2,20 kWp",
      "fonte": "Cálculo do Solar Viável (CRESESB)",
      "tipo": "simulacao",
      "url": "",
      "data": "2026-11-21",
      "citada": true
    },
    {
      "n": 3,
      "titulo": "EPE prepara chamada de projetos de usinas reversíveis para futura contratação no Brasil",
      "fonte": "Canal Solar",
      "tipo": "noticia",
      "url": "https://canalsolar.com.br/epe-prepara-chamada-projetos-usinas-reversiveis/",
      "data": "2026-09-16",
      "citada": false
    },
    {
      "n": 4,
      "titulo": "Agentes pedem neutralidade no edital do leilão de baterias e revisão do risco de inadimplência",
      "fonte": "ABSOLAR",
      "tipo": "noticia",
      "url": "https://www.absolar.org.br/home/agentes-pedem-neutralidade-no-edital-do-leilao-de-baterias-e-revisao-do-risco-de-inadimplencia/",
      "data": "2026-09-03",
      "citada": false
    }
  ],
  "memoria": {
    "fatos": {
      "cidade": "Curitiba (PR)",
      "cidade_recomendacao": "Curitiba (PR)",
      "cidades_citadas": [
        "Curitiba (PR)"
      ],
      "consumo_kwh_mes": 250.0,
      "tarifa": 0.92,
      "potencia_kwp": 2.2
    },
    "resumo": "",
    "total_turnos": 1,
    "turnos_na_janela": 1,
    "turnos_resumidos": 0,
    "tamanho_janela": 3,
    "recomendacao_id": "6f5e4d7ebcac"
  },
  "resumo_atualizado": false,
  "modelo": {
    "nome": "Qwen/Qwen3-1.7B",
    "dispositivo": "mps",
    "dtype": "float16",
    "parametros_milhoes": 1720.6
  },
  "uso": {
    "tokens_entrada": 2741,
    "tokens_saida": 102,
    "tempo_s": 2.25,
    "tokens_por_s": 45.3,
    "em_cache": false
  }
}
```

## 7. Mensagem com memória: "E em Fortaleza, como seria?"

`POST /conversas/85004508432e/mensagens` em 2026-11-21T10:20:13, HTTP 200, 2.28 s.

Requisição:

```json
{
  "pergunta": "E em Fortaleza, como seria?"
}
```

Modelo **Qwen/Qwen3-1.7B** em **mps** (float16), 3300 tokens de entrada, 96 de saída, 2.28 s de processamento.

**Pergunta entendida como:** E em Fortaleza, quanto o sistema de 2,2 kWp gera por mês? (modelo). **Fatos lembrados:** {"cidade": "Fortaleza (CE)", "cidade_recomendacao": "Curitiba (PR)", "cidades_citadas": ["Curitiba (PR)", "Fortaleza (CE)"], "consumo_kwh_mes": 250.0, "tarifa": 0.92, "potencia_kwp": 2.2}.

**Resposta:** Em Fortaleza, o sistema de 2,2 kWp gera média de 305,1 kWh por mês [1]. O pior mês é abril, com 258,7 kWh [1]. A geração varia 49,1% entre o melhor e o pior mês [1].

Resposta:

```json
{
  "conversa_id": "85004508432e",
  "pergunta": "E em Fortaleza, como seria?",
  "pergunta_reescrita": "E em Fortaleza, quanto o sistema de 2,2 kWp gera por mês?",
  "metodo_reescrita": "modelo",
  "resposta": "Em Fortaleza, o sistema de 2,2 kWp gera média de 305,1 kWh por mês [1]. O pior mês é abril, com 258,7 kWh [1]. A geração varia 49,1% entre o melhor e o pior mês [1].",
  "fontes": [
    {
      "n": 1,
      "titulo": "Simulação para Fortaleza (CE) com 2,20 kWp",
      "fonte": "Cálculo do Solar Viável (CRESESB)",
      "tipo": "simulacao",
      "url": "",
      "data": "2026-11-21",
      "citada": true
    },
    {
      "n": 2,
      "titulo": "Ranking das capitais por potencial solar",
      "fonte": "NASA POWER + CRESESB",
      "tipo": "ranking",
      "url": "",
      "data": "2026-10-17",
      "citada": false
    },
    {
      "n": 3,
      "titulo": "BESS reduz conta de energia de hortifrúti em Curitiba (PR)",
      "fonte": "Canal Solar",
      "tipo": "noticia",
      "url": "https://canalsolar.com.br/bess-reduz-conta-energia-hortifruti-curitiba/",
      "data": "2026-09-17",
      "citada": false
    },
    {
      "n": 4,
      "titulo": "Recomendação do agente consultor",
      "fonte": "Agente consultor do Solar Viável",
      "tipo": "recomendacao",
      "url": "",
      "data": "2026-11-21",
      "citada": false
    }
  ],
  "memoria": {
    "fatos": {
      "cidade": "Fortaleza (CE)",
      "cidade_recomendacao": "Curitiba (PR)",
      "cidades_citadas": [
        "Curitiba (PR)",
        "Fortaleza (CE)"
      ],
      "consumo_kwh_mes": 250.0,
      "tarifa": 0.92,
      "potencia_kwp": 2.2
    },
    "resumo": "",
    "total_turnos": 2,
    "turnos_na_janela": 2,
    "turnos_resumidos": 0,
    "tamanho_janela": 3,
    "recomendacao_id": "6f5e4d7ebcac"
  },
  "resumo_atualizado": false,
  "modelo": {
    "nome": "Qwen/Qwen3-1.7B",
    "dispositivo": "mps",
    "dtype": "float16",
    "parametros_milhoes": 1720.6
  },
  "uso": {
    "tokens_entrada": 3300,
    "tokens_saida": 96,
    "tempo_s": 2.28,
    "tokens_por_s": 42.1,
    "em_cache": false
  }
}
```

## 8. Mensagem com memória: "Qual das duas cidades tem mais sol?"

`POST /conversas/85004508432e/mensagens` em 2026-11-21T10:20:15, HTTP 200, 2.02 s.

Requisição:

```json
{
  "pergunta": "Qual das duas cidades tem mais sol?"
}
```

Modelo **Qwen/Qwen3-1.7B** em **mps** (float16), 2726 tokens de entrada, 90 de saída, 2.01 s de processamento.

**Pergunta entendida como:** Qual cidade tem mais sol entre Fortaleza e Curitiba? (modelo). **Fatos lembrados:** {"cidade": "Fortaleza (CE)", "cidade_recomendacao": "Curitiba (PR)", "cidades_citadas": ["Curitiba (PR)", "Fortaleza (CE)"], "consumo_kwh_mes": 250.0, "tarifa": 0.92, "potencia_kwp": 2.2}.

**Resposta:** Fortaleza tem mais sol que Curitiba [2]. A irradiação média em Fortaleza é de 5,70 kWh/m²/dia, enquanto em Curitiba é de 4,39 kWh/m²/dia [2]. Fortaleza está no 1º lugar entre as cidades no índice de irradiação [2].

Resposta:

```json
{
  "conversa_id": "85004508432e",
  "pergunta": "Qual das duas cidades tem mais sol?",
  "pergunta_reescrita": "Qual cidade tem mais sol entre Fortaleza e Curitiba?",
  "metodo_reescrita": "modelo",
  "resposta": "Fortaleza tem mais sol que Curitiba [2]. A irradiação média em Fortaleza é de 5,70 kWh/m²/dia, enquanto em Curitiba é de 4,39 kWh/m²/dia [2]. Fortaleza está no 1º lugar entre as cidades no índice de irradiação [2].",
  "fontes": [
    {
      "n": 1,
      "titulo": "Recomendação do agente consultor",
      "fonte": "Agente consultor do Solar Viável",
      "tipo": "recomendacao",
      "url": "",
      "data": "2026-11-21",
      "citada": false
    },
    {
      "n": 2,
      "titulo": "Comparação entre as cidades da conversa",
      "fonte": "Cálculo do Solar Viável (CRESESB)",
      "tipo": "simulacao",
      "url": "",
      "data": "2026-11-21",
      "citada": true
    },
    {
      "n": 3,
      "titulo": "Simulação para Curitiba (PR) com 2,20 kWp",
      "fonte": "Cálculo do Solar Viável (CRESESB)",
      "tipo": "simulacao",
      "url": "",
      "data": "2026-11-21",
      "citada": false
    },
    {
      "n": 4,
      "titulo": "Simulação para Fortaleza (CE) com 2,20 kWp",
      "fonte": "Cálculo do Solar Viável (CRESESB)",
      "tipo": "simulacao",
      "url": "",
      "data": "2026-11-21",
      "citada": false
    }
  ],
  "memoria": {
    "fatos": {
      "cidade": "Fortaleza (CE)",
      "cidade_recomendacao": "Curitiba (PR)",
      "cidades_citadas": [
        "Curitiba (PR)",
        "Fortaleza (CE)"
      ],
      "consumo_kwh_mes": 250.0,
      "tarifa": 0.92,
      "potencia_kwp": 2.2
    },
    "resumo": "",
    "total_turnos": 3,
    "turnos_na_janela": 3,
    "turnos_resumidos": 0,
    "tamanho_janela": 3,
    "recomendacao_id": "6f5e4d7ebcac"
  },
  "resumo_atualizado": false,
  "modelo": {
    "nome": "Qwen/Qwen3-1.7B",
    "dispositivo": "mps",
    "dtype": "float16",
    "parametros_milhoes": 1720.6
  },
  "uso": {
    "tokens_entrada": 2726,
    "tokens_saida": 90,
    "tempo_s": 2.01,
    "tokens_por_s": 44.7,
    "em_cache": false
  }
}
```

## 9. Histórico e memória da conversa

`GET /conversas/85004508432e` em 2026-11-21T10:20:17, HTTP 200, 0.0 s.

Resposta:

```json
{
  "id": "85004508432e",
  "criada_em": "2026-11-21T10:20:11",
  "atualizada_em": "2026-11-21T10:20:17",
  "turnos": [
    {
      "pergunta": "Quanto o sistema gera no pior mês?",
      "resposta": "O sistema gera 185,9 kWh no pior mês [2]. O pior mês é junho, com 185,9 kWh [2]. No pior mês, o sistema gera aproximadamente 185,9 kWh [2]. A geração varia 49,1% entre o melhor e o pior mês [2].",
      "pergunta_reescrita": "Quantos kWh o sistema gera no pior mês?",
      "fontes": [
        {
          "n": 1,
          "titulo": "Recomendação do agente consultor",
          "fonte": "Agente consultor do Solar Viável",
          "tipo": "recomendacao",
          "url": "",
          "data": "2026-11-21",
          "citada": false
        },
        {
          "n": 2,
          "titulo": "Simulação para Curitiba (PR) com 2,20 kWp",
          "fonte": "Cálculo do Solar Viável (CRESESB)",
          "tipo": "simulacao",
          "url": "",
          "data": "2026-11-21",
          "citada": true
        },
        {
          "n": 3,
          "titulo": "EPE prepara chamada de projetos de usinas reversíveis para futura contratação no Brasil",
          "fonte": "Canal Solar",
          "tipo": "noticia",
          "url": "https://canalsolar.com.br/epe-prepara-chamada-projetos-usinas-reversiveis/",
          "data": "2026-09-16",
          "citada": false
        },
        {
          "n": 4,
          "titulo": "Agentes pedem neutralidade no edital do leilão de baterias e revisão do risco de inadimplência",
          "fonte": "ABSOLAR",
          "tipo": "noticia",
          "url": "https://www.absolar.org.br/home/agentes-pedem-neutralidade-no-edital-do-leilao-de-baterias-e-revisao-do-risco-de-inadimplencia/",
          "data": "2026-09-03",
          "citada": false
        }
      ],
      "criado_em": "2026-11-21T10:20:13"
    },
    {
      "pergunta": "E em Fortaleza, como seria?",
      "resposta": "Em Fortaleza, o sistema de 2,2 kWp gera média de 305,1 kWh por mês [1]. O pior mês é abril, com 258,7 kWh [1]. A geração varia 49,1% entre o melhor e o pior mês [1].",
      "pergunta_reescrita": "E em Fortaleza, quanto o sistema de 2,2 kWp gera por mês?",
      "fontes": [
        {
          "n": 1,
          "titulo": "Simulação para Fortaleza (CE) com 2,20 kWp",
          "fonte": "Cálculo do Solar Viável (CRESESB)",
          "tipo": "simulacao",
          "url": "",
          "data": "2026-11-21",
          "citada": true
        },
        {
        
... (resposta completa em chamadas_reais_api.json)
```

## 10. Painel (ciclo de ciência de dados)

`GET /painel` em 2026-11-21T10:20:17, HTTP 200, 0.03 s.

Parâmetros:

```json
{
  "consumo_kwh_mes": 300,
  "tarifa": 0.95
}
```

Resposta:

```json
{
  "ciclo": [
    {
      "etapa": "Coleta",
      "descricao": "API REST da NASA POWER, Selenium no SunData (CRESESB) e Beautiful Soup na ABSOLAR e no Canal Solar",
      "indicadores": {
        "fontes": 4,
        "dias_de_clima": 366,
        "registros_nasa": 3660,
        "capitais": 10,
        "series_cresesb": 40,
        "noticias": 54
      }
    },
    {
      "etapa": "Processamento",
      "descricao": "limpeza (-999 vira ausente), padronização, trechos de texto, fichas das capitais e ranking",
      "indicadores": {
        "ausentes_irradiacao_pct": 1.9,
        "documentos_corpus": 164,
        "palavras_corpus": 32192
      }
    },
    {
      "etapa": "Análise",
      "descricao": "geração mês a mês, dimensionamento, payback com Fio B, CO2 evitado e benchmark das capitais",
      "indicadores": {
        "capitais_comparadas": 10,
        "ferramentas_do_agente": 7
      }
    },
    {
      "etapa": "IA",
      "descricao": "LLM local: resumo e sentimento das notícias, agente que decide as análises e chat com memória",
      "indicadores": {
        "noticias_classificadas": 54,
        "modelo": "Qwen/Qwen3-1.7B",
        "estado_modelo": "pronto"
      }
    },
    {
      "etapa": "Apresentação",
      "descricao": "painel Streamlit consumindo a API FastAPI, com recomendações, gráficos e conversa",
      "indicadores": {
        "rotas_api": 29
      }
    }
  ],
  "capitais": [
    {
      "cidade": "Fortaleza (CE)",
      "irradiacao_media": 5.7,
      "potencia_kwp": 1.8,
      "investimento_rs": 5400.0,
      "payback_anos": 2.3,
      "economia_25_anos_rs": 52749.11,
      "co2_evitado_kg_ano": 115.0,
      "classificacao": "excelente"
    },
    {
      "cidade": "Brasília (DF)",
      "irradiacao_media": 5.47,
      "potencia_kwp": 1.88,
      "investimento_rs": 5640.0,
      "payback_anos": 2.4,
      "economia_25_anos_rs": 52813.08,
      "co2_evitado_kg_ano": 116.0,
      "classificacao": "excelente"
    },
    {
      "cidade": "Recife (PE)",
      "irradiacao_media": 5.48,
      "potencia_kwp": 1.88,
      "investimento_rs": 5640.0,
      "payback_anos": 2.4,
      "economia_25_anos_rs": 52847.57,
      "co2_evitado_kg_ano": 116.0,
      "classificacao": "excelente"
    },
    {
      "cidade": "Belo Horizonte (MG)",
      "irradiacao_media": 5.36,
      "potencia_kwp": 1.92,
      "investimento_rs": 5760.0,
      "payback_anos": 2.5,
      "economia_25_anos_rs": 52783.08,
      "co2_evitado_kg_ano": 116.0,
      "clas
... (resposta completa em chamadas_reais_api.json)
```
