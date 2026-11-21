# Escolha do modelo LLM (execução local)

**Modelo escolhido: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)**, executado
localmente com HuggingFace Transformers + PyTorch, sem modo "raciocínio" e com
decodificação gulosa.

A escolha não foi feita só pela ficha técnica: cinco modelos da HuggingFace e um
classificador do tipo BERT foram **baixados e executados nesta máquina, nas tarefas reais
do Solar Viável e com o mesmo código das rotas da API** (`python -m src.llm.avaliacao`).
Este documento mostra os critérios, os candidatos, as medições e como cada fator pesou na
decisão. Como a integração foi feita está em [llm_integracao_local.md](llm_integracao_local.md).

## 1. O que o projeto precisa de um modelo

| Necessidade | Consequência para a escolha |
|---|---|
| Textos em **português** (notícias, dúvidas de leigos) | o modelo precisa ter sido treinado com português; modelos só em inglês ficam de fora |
| Quatro tarefas diferentes: **resumo**, **sentimento**, **perguntas com contexto (RAG)** e **relatório** | um modelo **generativo ajustado para instruções** (família GPT) cobre as quatro; um classificador (família BERT) só cobre sentimento |
| Seguir instruções rígidas: usar só o contexto, citar [n], não inventar números, admitir quando não sabe | exige modelo *instruct*/chat, não um modelo base |
| Contexto de ~4 a 10 trechos (até ~2.500 tokens) | janela de contexto ≥ 8 mil tokens |
| Rodar **localmente**, inclusive em notebooks de estudantes | caber em ~4 GB de GPU/memória unificada, ou ~8 GB de RAM na CPU |
| Baixar e usar sem burocracia | pesos abertos, licença permissiva, sem aceite de termos nem login no Hub |
| Respostas em poucos segundos dentro de um app web | ≥ ~30 tokens/s na GPU |

## 2. Critérios de seleção

| Critério | Como foi medido |
|---|---|
| **Desempenho** (qualidade e tipo de tarefa) | arquitetura e treino do modelo (generativo x classificador; *instruct*); acurácia e F1 de sentimento em 34 notícias rotuladas à mão; resumos (limite de palavras, português, números fiéis ao texto); 12 perguntas com resposta esperada, incluindo 2 fora do escopo; 3 relatórios com conferência dos números |
| **Capacidade de resposta** (velocidade) | tokens gerados por segundo e tempo médio de cada tarefa |
| **Custo computacional local** | memória ocupada pelo modelo carregado (GPU), tempo de carga, e teste na CPU sem GPU |
| **Acessibilidade local** | carrega com `AutoModelForCausalLM` do Transformers sem código extra; licença; repositório aberto (sem *gated*); tamanho do download |

## 3. Candidatos

Todos são da HuggingFace Model Hub, com pesos em `safetensors`, carregados pelo
Transformers 5.23. As famílias citadas no enunciado foram consideradas assim:

- **GPT** (decodificadores generativos): os cinco candidatos principais são dessa família, pois
  são os únicos que fazem as quatro tarefas com uma instrução.
- **BERT** (codificadores): fazem classificação, não geram texto. Entrou como **linha de base de
  sentimento** um XLM-RoBERTa (variante multilíngue do BERT) já ajustado para sentimento em 8
  línguas, incluindo português.
- **T5** (codificador-decodificador): o T5/FLAN-T5 é treinado quase só em inglês, e os T5 em
  português (PTT5) são modelos base, que precisariam de ajuste fino para resumir e não fazem
  perguntas e respostas com instrução. Ficou fora da bancada.

| Modelo | Família / arquitetura | Parâmetros | Download | Contexto | Português | Licença |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | GPT (decodificador), *instruct* | 0,49 B | 1,0 GB | 32 mil | sim (29+ línguas) | Apache 2.0 |
| `Qwen/Qwen2.5-1.5B-Instruct` | GPT, *instruct* | 1,54 B | 3,1 GB | 32 mil | sim (29+ línguas) | Apache 2.0 |
| **`Qwen/Qwen3-1.7B`** | GPT, chat (modo raciocínio opcional) | 1,72 B | 4,1 GB | 40 mil | sim (100+ línguas) | Apache 2.0 |
| `HuggingFaceTB/SmolLM3-3B` | GPT, chat (modo raciocínio opcional) | 3,08 B | 6,2 GB | 64 mil | sim (6 línguas, inclui pt) | Apache 2.0 |
| `Qwen/Qwen3-4B-Instruct-2507` | GPT, *instruct* | 4,02 B | 8,0 GB | 256 mil | sim (100+ línguas) | Apache 2.0 |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | BERT (XLM-RoBERTa), classificador | 0,28 B | 1,1 GB | 512 | sim (ajustado em pt) | (sem licença declarada) |

Descartados sem teste, por **acessibilidade** ou **custo**:

| Modelo | Motivo |
|---|---|
| Llama 3.2 1B/3B (Meta), Gemma 3 1B/4B (Google) | repositórios *gated*: exigem conta no Hub, aceite de licença própria e token, o que dificulta a reprodução por outra pessoa (o professor, por exemplo) |
| Modelos de 7 B ou mais (ex.: Qwen2.5-7B-Instruct) | ~15 GB em float16: não cabem na GPU/RAM da maioria dos notebooks |
| GPT-2 | só inglês e sem ajuste para instruções |
| BERTimbau | ótimo codificador em português, mas exigiria rotular dados e treinar um classificador; não gera texto |

## 4. Ambiente da medição

Apple M5 Pro (18 núcleos, 48 GB de memória unificada), PyTorch 2.15 com GPU via **MPS**, pesos
em **float16**, decodificação gulosa, modo raciocínio desligado. Cada modelo rodou em um processo
separado. Dados brutos: [`docs/avaliacao/resultados_modelos.json`](avaliacao/resultados_modelos.json)
(inclui os erros de sentimento e todas as respostas às perguntas).

## 5. Resultados

### 5.1 Custo computacional e velocidade

| Modelo | Memória do modelo (GPU) | Carga | Geração | Resumo (média) | Pergunta (média) | Relatório (média) |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0,94 GB | 2,2 s | 153 tokens/s | 0,6 s | 0,3 s | 2,6 s |
| Qwen2.5-1.5B-Instruct | 2,90 GB | 2,6 s | 67 tokens/s | 2,5 s | 1,8 s | 5,8 s |
| **Qwen3-1.7B** | **3,22 GB** | **3,1 s** | **58 tokens/s** | **2,9 s** | **1,8 s** | **6,8 s** |
| SmolLM3-3B | 5,73 GB | 6,0 s | 39 tokens/s | 4,6 s | 2,8 s | 6,4 s |
| Qwen3-4B-Instruct-2507 | 7,56 GB | 7,6 s | 27 tokens/s | 3,5 s | 4,0 s | 11,7 s |

**Sem GPU** (CPU, float32, mesma máquina): o Qwen3-1.7B ocupa **7,3 GB de RAM** e gera ~12
tokens/s (resumo em ~15 s, pergunta em ~13 s); o Qwen2.5-0.5B ocupa 2,6 GB e gera ~40 tokens/s.
Em float32 a memória dobra (4 bytes por parâmetro): o Qwen3-4B precisaria de ~16 GB de RAM,
fora do alcance de muitos notebooks. Numa CPU mais modesta que a do M5 Pro, os tempos serão maiores.

### 5.2 Desempenho nas tarefas

| Modelo | Sentimento: acurácia / F1 macro | Resumo: no limite / em português / números fiéis | Perguntas: acerto / recusa correta / cita fonte | Relatório: números que conferem / relatórios 100% corretos |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 56% / 0,52 | 100% / 100% / 99% | 100% / 100% / **0%** | 100% / 100% |
| Qwen2.5-1.5B-Instruct | 50% / 0,30 | 100% / 100% / 100% | 100% / 100% / 90% | 94% / **33%** |
| **Qwen3-1.7B** | **79% / 0,65** | 100% / 100% / 100% | 100% / 100% / 100% | 100% / 100% |
| SmolLM3-3B | 38% / 0,24 | 100% / 100% / 98% | 100% / 100% / 100% | 100% / 100% |
| Qwen3-4B-Instruct-2507 | **82% / 0,79** | 100% / 100% / 100% | 100% / 100% / 100% | 100% / 100% |
| XLM-RoBERTa (BERT, só sentimento) | 35% / 0,25 | n/a | n/a | n/a |

Como ler: *sentimento* compara com 34 notícias rotuladas à mão (16 positivas, 11 neutras,
7 negativas) do ponto de vista de quem quer instalar solar; *números fiéis* é a fração dos números
do resumo que existem no texto original; *acerto* é a resposta conter o dado esperado (ex.:
"Recife", "534", "23,2"); *recusa correta* é dizer que não sabe nas 2 perguntas fora do escopo
(Copa do Mundo, receita de bolo); *cita fonte* é usar [n].

### 5.3 Sensibilidade ao prompt de sentimento

O sentimento foi medido com duas versões do prompt. A v1 dava definições curtas entre
parênteses; a v2 (a usada no projeto) define cada classe com exemplos e avisa que "muitas
notícias do setor são neutras". Resultados (acurácia):

| Modelo | Prompt v1 | Prompt v2 (final) |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 53% | 56% |
| Qwen2.5-1.5B-Instruct | 53% | 50% |
| Qwen3-1.7B | 59% | **79%** |
| SmolLM3-3B | **74%** | 38% |
| Qwen3-4B-Instruct-2507 | 65% | **82%** |
| XLM-RoBERTa (não usa prompt) | 35% | 35% |

Duas lições: (1) modelos pequenos são **muito sensíveis ao prompt**, e o mesmo prompt pode
ajudar um modelo e atrapalhar outro (o SmolLM3 passou a marcar quase tudo como neutro); (2) a
família Qwen3 foi a mais estável, com o melhor resultado nos dois prompts entre os modelos que
cabem em 4 GB. Ressalva: o prompt v2 foi ajustado olhando este mesmo conjunto de 34 notícias,
então os números da v2 tendem a ser otimistas; o conjunto é pequeno e os rótulos refletem
um único anotador.

### 5.4 Observações qualitativas

- **Qwen2.5-0.5B**: rápido e leve, mas **nunca citou as fontes** e, na rodada anterior, errou a
  capital com maior irradiação e afirmou que "o Brasil ganhou a Copa de 2022" citando um trecho.
  Inadequado para um assistente que precisa ser confiável.
- **Qwen2.5-1.5B**: bom em perguntas, mas em 2 de 3 relatórios escreveu números que não estão nos
  dados (e, num teste manual, inventou um "retorno em 5 anos" e atribuiu dados ao INPE); viés
  forte para "positivo" no sentimento.
- **Qwen3-1.7B**: respostas curtas, com [n] em todas, números copiados corretamente e recusa
  limpa ("Não encontrei essa informação nos dados do Solar Viável."). No sentimento, erra
  sobretudo notícias negativas que classifica como neutras.
- **SmolLM3-3B**: textos bons, mas sentimento instável (seção 5.3) e, numa rodada, respondeu à
  pergunta do bolo de cenoura falando de reforma tributária.
- **Qwen3-4B**: o melhor em qualidade geral, mas o dobro da memória e metade da velocidade do 1,7B.
- **XLM-RoBERTa** (BERT): carrega em 1 s e classifica em 24 ms, mas foi treinado com tweets e mede o
  **tom** do texto, não o efeito para quem quer instalar solar: classificou quase tudo como neutro.
  Para chegar perto dos LLMs, precisaria de ajuste fino com dados rotulados do setor.

## 6. Decisão

| Critério | Qwen2.5-0.5B | Qwen2.5-1.5B | **Qwen3-1.7B** | SmolLM3-3B | Qwen3-4B | XLM-R (BERT) |
|---|---|---|---|---|---|---|
| Desempenho: tarefas cobertas | 4 | 4 | **4** | 4 | 4 | 1 |
| Desempenho: qualidade | fraca (sem citações) | média (números inventados) | **boa** | boa, instável | **a melhor** | fraca no domínio |
| Capacidade de resposta | excelente | boa | **boa (58 tok/s)** | média | média (27 tok/s) | excelente |
| Custo local (GPU) | 0,9 GB | 2,9 GB | **3,2 GB** | 5,7 GB | 7,6 GB | < 1 GB |
| Custo local (só CPU, float32)¹ | 2,6 GB, rápido | ~6 GB | **7,3 GB, usável** | ~12 GB | ~16 GB | ~1 GB |
| Acessibilidade | aberto, Apache 2.0 | aberto, Apache 2.0 | **aberto, Apache 2.0** | aberto, Apache 2.0 | aberto, Apache 2.0 | aberto, sem licença declarada |

¹ Medido para o Qwen2.5-0.5B e o Qwen3-1.7B; para os demais, estimado em 4 bytes por parâmetro.

**Por que o Qwen3-1.7B:**

1. **Desempenho**: é o menor modelo que acertou todas as perguntas **citando as fontes**, recusou
   as perguntas fora do escopo e escreveu relatórios sem números inventados. No sentimento, ficou
   3 pontos abaixo do Qwen3-4B (79% x 82%) e muito acima dos demais e do classificador BERT.
2. **Custo computacional**: ocupa 3,2 GB em float16 (cabe em GPUs de 4 GB e em Macs de 8 GB) e
   ainda funciona só na CPU com ~7 GB de RAM. O Qwen3-4B entregaria um pouco mais de qualidade
   pelo dobro da memória e metade da velocidade; o ganho não compensa num app que precisa
   rodar na máquina de qualquer pessoa.
3. **Capacidade de resposta**: ~58 tokens/s na GPU: um resumo em ~3 s e uma resposta do
   assistente em ~2 s, compatíveis com uma interface web.
4. **Acessibilidade**: licença Apache 2.0 (uso livre, inclusive comercial), repositório aberto
   (sem login nem aceite de termos), carregado direto por `AutoModelForCausalLM`, 4,1 GB de
   download único. Suporta mais de 100 línguas, inclusive português.

**Alternativas configuráveis** (sem mudar código, via `SOLAR_LLM_MODELO`):

- máquina com pouca memória e sem GPU: `Qwen/Qwen2.5-0.5B-Instruct` (2,6 GB na CPU), sabendo que as
  respostas não citam as fontes;
- máquina com GPU de 8 GB ou mais: `Qwen/Qwen3-4B-Instruct-2507`, para o melhor sentimento.

## 7. Como reproduzir

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct    # repita para cada candidato da seção 3
python -m src.llm.avaliacao                # todos os candidatos (cerca de 20 a 30 min nesta máquina)
python -m src.llm.avaliacao --modelos Qwen/Qwen3-1.7B
```

Os dados de avaliação ficam em `data/avaliacao/` (`sentimento_noticias.csv` e
`perguntas.csv`); os resultados, em `docs/avaliacao/`.
