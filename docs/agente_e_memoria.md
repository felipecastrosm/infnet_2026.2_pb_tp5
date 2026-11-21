# Agente consultor e memória conversacional (TP5)

Como as duas funcionalidades de IA do TP5 funcionam, por que foram desenhadas assim e como
verificar. O caminho escolhido na Tarefa 2 foi a **Opção 3 (agente inteligente para tomada
de decisão)**, complementada pela **Opção 1 (memória conversacional)**, para o usuário
continuar a conversa sobre a recomendação.

## 1. Agente consultor

Código: `src/llm/agente.py` (ciclo e prompts) e `src/data_processing/viabilidade.py`
(ferramentas). Rota: `POST /agente/recomendar`.

### Ciclo pensar, agir, observar

```
perfil do usuário (cidade, consumo, tarifa, objetivo)
  |
  v
[decidir]  o LLM pontua cada ação disponível; a mais provável é escolhida
  |
  v
[agir]     a ferramenta calcula sobre os dados coletados (NASA, CRESESB, notícias com IA)
  |
  v
[observar] o resultado vira uma frase, que entra no prompt da próxima decisão
  |
  (repete até o modelo escolher "finalizar")
  v
[sintetizar] o LLM escreve a recomendação só com os fatos; os números são conferidos
```

### Ferramentas

| Ferramenta | Depende de | O que calcula | Dados |
|---|---|---|---|
| `potencial_solar` | (primeira) | irradiação no ano típico, inclinação, posição entre capitais, último ano x histórico | CRESESB, NASA |
| `dimensionar_sistema` | potencial | kWp, nº de módulos, área, geração, % do consumo (descontado o custo de disponibilidade) | CRESESB |
| `analise_financeira` | dimensionamento | investimento, economia mensal, payback, TIR, fluxo de caixa em 25 anos com Fio B, degradação e manutenção | premissas com fonte |
| `sazonalidade` | dimensionamento | geração x consumo mês a mês, meses de déficit, créditos | CRESESB |
| `impacto_ambiental` | dimensionamento | CO₂ evitado por ano e em 25 anos, energia limpa, ODS | fator de emissão do MCTI |
| `comparar_capitais` | potencial | o mesmo perfil nas 10 capitais (benchmark de payback) | CRESESB |
| `noticias_setor` | potencial | notícias sobre tarifa, regras e custos, com resumo e sentimento do LLM | `noticias_ia.csv` |

### Decisão por decodificação restrita

Em vez de pedir ao modelo um JSON com a próxima ação (que um modelo de 1,7 B às vezes
quebra), o agente mede a **log-probabilidade** que o modelo dá ao nome de cada ação como
resposta ao prompt de decisão e normaliza (softmax). Assim a escolha é sempre válida e vem
com a confiança, que o painel mostra em cada passo. É a mesma técnica usada no sentimento
desde o TP4.

### Guardrails

- Só aparecem ações cujas dependências já rodaram.
- `finalizar` só aparece depois das três obrigatórias (potencial, dimensionamento, finanças).
- Com uma única opção possível, ela é escolhida por regra, sem chamar o modelo.
- Limite de 9 passos; se for atingido, as obrigatórias rodam antes da síntese.
- Alertas por regras (payback acima de 8 anos, consumo baixo, área grande, ano atípico, Fio B).
- Rótulos qualitativos calculados nos fatos ("entre os mais baixos das capitais").
- Verificação dos números da recomendação contra os resultados das ferramentas.

### O objetivo muda o plano

Registro real (`docs/evidencias/chamadas_reais_api.md` e testes com o modelo real):

| Caso | Objetivo | Ações escolhidas pelo modelo depois do potencial |
|---|---|---|
| Recife, 350 kWh | "em quanto tempo se paga e qual o impacto ambiental" | dimensionamento, finanças, **impacto**, comparação, notícias (sem sazonalidade) |
| Curitiba, 180 kWh | "me preocupa o inverno" | dimensionamento, finanças, **sazonalidade**, comparação, impacto (sem notícias) |
| Manaus, 800 kWh | avaliação completa | dimensionamento, finanças, comparação, impacto, notícias (sem sazonalidade) |

Tempo típico: **10 a 12 s** no Apple M (MPS), sendo ~5 decisões de ~0,3 s e uma síntese de
~600 tokens. Recomendações repetidas vêm do cache.

## 2. Memória conversacional

Código: `src/llm/memoria.py`. Rotas: `/conversas`.

| Camada | Conteúdo | Atualização |
|---|---|---|
| Janela | as últimas 3 trocas, na íntegra, como mensagens de chat | a cada resposta |
| Resumo | o que saiu da janela, em até 70 palavras | pelo LLM, quando a janela enche |
| Fatos | cidade em foco, cidades citadas, consumo, potência, tarifa, cidade da recomendação | por regras (regex e lista de capitais), sem LLM |
| Recomendação | o texto dos passos e alertas do agente | quando a conversa nasce de uma recomendação |

A cada pergunta:

1. **Extrai fatos** da pergunta e atualiza a memória.
2. **Reescreve** a pergunta como pergunta completa (prompt com 3 exemplos, few-shot). Se a
   saída for inválida, ou se o modelo só repetir uma pergunta de acompanhamento, uma regra
   acrescenta os fatos ("(em Fortaleza (CE), com um sistema de 2,2 kWp)"). Em mudança de
   assunto ("o que é o leilão de baterias?") nada é herdado.
3. **Busca contexto** com a pergunta reescrita: simulação calculada na hora para a cidade e a
   potência lembradas, comparação pronta quando há duas ou mais cidades, recomendação do
   agente e trechos do corpus (notícias, fichas, ranking).
4. **Responde** com o prompt de sistema + fatos + resumo, a janela de mensagens e o contexto
   numerado; as citações [n] são marcadas nas fontes.
5. **Resume** as trocas que saíram da janela, se for o caso, e salva a conversa em disco.

Exemplo real (Curitiba, depois Fortaleza):

| Usuário | Entendida como | Resposta (trecho) |
|---|---|---|
| Quanto o sistema gera no pior mês? | Quantos kWh o sistema gera no pior mês? | 185,9 kWh, em junho |
| E em Fortaleza, como seria? | E em Fortaleza, quanto o sistema de 2,2 kWp gera por mês? | média de 305,1 kWh por mês |
| Qual das duas cidades tem mais sol? | Qual cidade tem mais sol entre Fortaleza e Curitiba? | Fortaleza, 5,70 contra 4,39 kWh/m²/dia |

## 3. Eficiência no uso do LLM

- Um único modelo local (Qwen3-1.7B, 3,3 GB em float16) atende agente, chat e notícias.
- As 54 notícias são processadas **uma vez, em lote** (4 min), não a cada visita.
- Decisões do agente custam uma passada do modelo por opção, sem gerar texto; ações únicas
  não chamam o modelo.
- Cache de respostas para entradas repetidas; fila de uma inferência por vez com prazo.
- O modelo nunca faz contas: cálculo em Python, linguagem no LLM.

## 4. Como verificar

```bash
pytest tests/test_tp5.py                                   # agente, memória e painel com modelo falso
uvicorn src.api.main:app                                   # em outro terminal, com o modelo carregado:
python scripts/registrar_chamadas_reais.py                 # chamadas reais, salvas em docs/evidencias/
python scripts/capturar_telas.py                           # telas do painel (precisa do app no ar)
```
