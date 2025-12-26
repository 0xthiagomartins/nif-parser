# OCR de Documentos Brasileiros (RG, CNH, Passaporte)

Projeto para transformar documentos brasileiros em dados estruturados, com foco em custo, tempo e qualidade.

## Motivacao

Este projeto nasceu para reduzir custo e tempo na extracao de dados de documentos brasileiros.
Estavamos comparando pipelines e otimizando cada etapa para diminuir gasto com LLM e evitar dependencia
excessiva de servicos cloud caros, mantendo qualidade de dados estruturados.

## O que fazemos

Pipeline proprio:
- Preprocessamento de imagem (multiplas variacoes)
- OCR local (Tesseract)
- Conversao para .toon
- LLM para gerar JSON estruturado

## Comparacao

- Nosso pipeline (preprocess + OCR + .toon + LLM)
- Amazon [Textract]("https://docs.aws.amazon.com/textract/latest/dg/what-is.html") + LLM (Textract entrega texto nao estruturado)

## Dados reais (26/12/2025)

| Documento | Metodo | Tempo total (s) | Tokens entrada | Tokens saida | Custo LLM (USD) | Custo base (USD) | Custo total (USD) |
|---|---|---:|---:|---:|---:|---:|---:|
| RG | Pipeline proprio (preprocess + OCR + .toon + LLM) | 15.88 | 3199 | 202 | 0.0006 | 0.0000 | 0.0006 |
| RG | Textract + LLM | 10.71 | 1600 | 202 | 0.0004 | 0.0500 | 0.0504 |

**Preco por RG processado:**
- Pipeline proprio: **US$ 0.0006**
- Textract + LLM: **US$ 0.0504**
