# ROADMAP

## Feito

- Pipeline de OCR com variacoes de preprocessamento
- Exportacao de resultados em .toon
- Arquitetura reorganizada em `src/ocr` com processadores por documento
- POC validada para RG (frente e verso)
- Notebook de validacao executado com sucesso

## Em andamento

- Integracao com LLMs para estruturacao dos dados
- Definicao do schema de saida por documento (RG, CNH, Passaporte)

## Planejado (proximas entregas)

- Benchmark comparativo:
  - Pipeline proprio (preprocess + OCR + .toon + LLM)
  - LLM direto (imagem -> JSON)
- Medicao de custo:
  - Tokens de entrada e saida
  - Tempo total de execucao
  - Custo USD por documento
- Expansao da POC para CNH (frente e verso)
- Expansao da POC para Passaporte (frente)
- Ajustes finos de preprocessing por documento
- Relatorio de qualidade (precisao por campo)

## Observacoes

- O benchmark deve quantificar o trade-off entre tempo de processamento e economia de tokens/custo final.
- A comparacao deve considerar o mesmo objetivo: dados estruturados para os mesmos campos.
