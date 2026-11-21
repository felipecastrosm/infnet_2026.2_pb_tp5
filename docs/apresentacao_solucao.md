# Solar Viável: apresentação da solução final (TP5)

**Autor:** Felipe Castro Sotto Mayor · **Disciplina:** Projeto de Bloco, Data Science e IA (Infnet)

## 1. O problema de negócio

Quem pensa em instalar energia solar em casa ou num pequeno negócio precisa responder a uma
pergunta financeira com dados técnicos que não domina: *vale a pena para mim, de que tamanho
e em quanto tempo se paga?* Hoje essa resposta vem de consultorias pagas ou de quem vende o
sistema. Sem um primeiro diagnóstico neutro, muita gente desiste antes de pedir um
orçamento. Isso atrasa a adoção de energia limpa (**ODS 7**) e a redução de emissões
(**ODS 13**).

## 2. A solução em uma tela

O usuário abre o **Painel**, escolhe a cidade, informa o consumo e a tarifa (e, se quiser,
o que mais o preocupa) e clica em *Analisar com o agente*. Em cerca de 12 segundos recebe:

- o **veredito** (viabilidade excelente, boa, moderada ou baixa);
- os **indicadores**: tamanho do sistema, investimento, economia por mês, payback e CO₂ evitado;
- uma **recomendação em português simples**, com ações práticas e pontos de atenção;
- **como o agente decidiu**: cada análise escolhida, com a probabilidade dada pelo modelo;
- **gráficos**: saldo acumulado em 25 anos, geração mês a mês x consumo, comparação com as
  outras capitais e as notícias do setor que afetam a decisão;
- um **chat com memória** para continuar: "e em Fortaleza?", "por que isso?".

Outras abas mostram o **panorama das 10 capitais**, o **sentimento do setor** lido pela IA e
**como funciona** o ciclo de dados. Telas em [evidencias/telas/](evidencias/telas/).

## 3. O ciclo de ciência de dados, etapa por etapa

| Etapa | O que foi feito | Por que importa para a decisão |
|---|---|---|
| **1. Coleta** | NASA POWER por API REST (3.660 dias-cidade); CRESESB/SunData por **Selenium** (120 séries mensais, painel inclinado); ABSOLAR e Canal Solar por **Beautiful Soup** (54 notícias com texto completo) | Dados reais, públicos e neutros sobre o sol de cada lugar e sobre o mercado |
| **2. Processamento** | Ausentes (`-999`) tratados; ano típico por mês e inclinação; estação de referência por capital; corpus de 164 documentos (trechos, fichas, ranking) | Sem isso, um ano atípico (+11% em Recife) distorceria a decisão |
| **3. Análise** | Dimensionamento, payback com **Fio B da Lei 14.300**, degradação e manutenção, sazonalidade, CO₂ evitado, benchmark das capitais | São os números da decisão, calculados de forma auditável |
| **4. IA** | LLM local (Qwen3-1.7B): **agente** que escolhe as análises e escreve a recomendação; **memória** para perguntas seguintes; **resumo e sentimento** das 54 notícias em lote | Transforma números em uma resposta compreensível para leigos, adaptada ao objetivo de cada pessoa |
| **5. Apresentação** | API FastAPI (29 rotas) + painel Streamlit | Uma tela, três campos, resposta em segundos |

## 4. Insights gerados com dados reais

1. **A energia solar se paga em todas as capitais analisadas.** Com 300 kWh/mês e
   R$ 0,95/kWh, já descontado o Fio B, o retorno vai de **2,3 anos em Fortaleza** a **3,1 anos
   em Curitiba e Manaus**, para um sistema que dura 25 anos. A questão não é *se*, e sim
   *quanto instalar*.
2. **O inverno pesa no Sul, mas não muda a conclusão.** Em Porto Alegre o melhor mês gera 2,1
   vezes o pior; os créditos dos meses de sol cobrem os meses fracos, então o
   dimensionamento olha o ano, não o mês.
3. **Um ano de dados engana.** O último ano medido pela NASA ficou até 11% acima da média
   histórica em Recife e Manaus; a solução usa o ano típico do CRESESB e avisa quando o ano
   recente foi atípico.
4. **O impacto climático é real, mas o maior ganho é o acesso à energia.** Com a matriz
   brasileira já limpa (38,5 kg de CO₂ por MWh), um sistema residencial evita de 110 a 170 kg
   de CO₂ por ano. O benefício principal é a conta de luz menor (ODS 7.1) e mais energia
   renovável no consumo (ODS 7.2); a solução mostra isso com transparência, sem exagerar o ODS 13.
5. **O setor está otimista.** Das 54 notícias lidas pelo modelo, 52% são positivas para quem
   quer instalar e 2% negativas. Ressalva: as fontes são do próprio setor e o modelo tende a
   classificar notícias negativas como neutras (79% de acerto no conjunto rotulado).

## 5. Por que a IA foi usada assim

- **O modelo não faz contas.** Ferramentas em Python calculam sobre os dados; o LLM decide
  quais análises importam e explica. Cada número do texto é conferido com os cálculos
  (nas chamadas reais registradas, 100% conferiram).
- **Decisões transparentes.** O painel mostra cada passo do agente e a probabilidade de cada
  opção: o usuário vê *por que* a recomendação inclui, por exemplo, a análise do inverno.
- **Engenharia de prompts para um modelo pequeno.** Decodificação restrita, fatos e rótulos
  pré-calculados, exemplos (few-shot) na reescrita, memória em camadas e contexto calculado
  na hora (Charter, seção 2.4).
- **Local e eficiente.** Os dados do usuário não saem da máquina, não há custo por pergunta,
  as notícias são processadas uma vez em lote e repetições vêm do cache.

## 6. Alinhamento com ESG e ODS

| Dimensão | Como a solução contribui |
|---|---|
| **E** (ambiental) | Incentiva a geração renovável; mede CO₂ evitado e energia limpa em 25 anos (ODS 13.2, 7.2) |
| **S** (social) | Diagnóstico gratuito, neutro e em português para quem não tem acesso a consultoria; foco em reduzir a conta de luz (ODS 7.1) |
| **G** (governança) | Premissas com fonte, números conferidos, fontes citadas, decisões do agente rastreáveis, dados do usuário locais |

## 7. Limitações e próximos passos

- Modelo de geração simplificado (sem sombreamento ou orientação diferente do norte); custo e
  Fio B são médias nacionais.
- O modelo de 1,7 B às vezes repete números do histórico ou erra o tom de notícias negativas;
  as mitigações estão documentadas, e os indicadores do painel vêm sempre dos cálculos.
- Próximos passos: validar com usuários reais, tarifa por distribuidora, financiamento e
  geocodificação de endereço.

## 8. Evidências

- Chamadas reais à API com o modelo carregado: [evidencias/chamadas_reais_api.md](evidencias/chamadas_reais_api.md)
- Telas do painel: [evidencias/telas/](evidencias/telas/)
- Documentação técnica: [agente_e_memoria.md](agente_e_memoria.md), [Project Charter](project/project_charter.md),
  [Data Summary Report](data_report/data_summary_report.md)
