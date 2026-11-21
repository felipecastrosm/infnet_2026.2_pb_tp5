"""Integração do Solar Viável com um LLM executado localmente (HuggingFace Transformers).

- `config`: parâmetros (modelo, dispositivo, limites), lidos de variáveis de ambiente;
- `erros`: exceções do domínio, cada uma ligada a um código HTTP;
- `servico`: carrega o modelo na máquina local e executa a inferência (geração de
  texto e pontuação de rótulos), com prazo, fila e estado de carregamento;
- `tarefas`: prompts e pós-processamento de cada recurso (resumo, sentimento,
  perguntas e respostas sobre o corpus e relatório de viabilidade);
- `avaliacao`: comparação dos modelos candidatos (usada para justificar a escolha).
"""
