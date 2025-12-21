# OCR de Documentos Brasileiros (RG, CNH, Passaporte)

Este projeto transforma documentos brasileiros em dados estruturados, com foco em qualidade, custo e velocidade.

## Motivacao

Empresas precisam extrair dados confiaveis de RG, CNH e passaporte para onboarding, KYC e antifraude. O objetivo aqui e comparar duas abordagens:

1) Pipeline proprio de OCR
- Gera N variacoes de preprocessamento
- Executa OCR
- Consolida em um arquivo .toon para LLM

2) OCR direto via LLM
- Envia a imagem para a LLM
- Converte diretamente em dado estruturado

## Valor de negocio

- Reducao de custo por documento (tokens e tempo)
- Maior controle sobre qualidade do texto extraido
- Flexibilidade para ajustar o preprocessing por tipo de documento

## Benchmark (template)

A tabela abaixo compara custo e tempo das duas abordagens. Preencha com dados reais de cada documento e lote de testes.

| Documento | Metodo | Tempo total (s) | Tokens entrada | Tokens saida | Custo (USD) | Observacoes |
|---|---|---:|---:|---:|---:|---|
| RG | Pipeline proprio (preprocess + OCR + .toon + LLM) |  |  |  |  |  |
| RG | LLM direto (imagem -> JSON) |  |  |  |  |  |
| CNH | Pipeline proprio (preprocess + OCR + .toon + LLM) |  |  |  |  |  |
| CNH | LLM direto (imagem -> JSON) |  |  |  |  |  |
| Passaporte | Pipeline proprio (preprocess + OCR + .toon + LLM) |  |  |  |  |  |
| Passaporte | LLM direto (imagem -> JSON) |  |  |  |  |  |

## Criterios de comparacao

- Qualidade do dado estruturado (precisao dos campos)
- Tempo total de execucao
- Custo em tokens (entrada e saida)
- Custo final em USD por documento

## Proximos passos

- Preencher o benchmark com dados reais
- Integrar LLMs e medir custo real por documento
- Definir threshold de qualidade minima por tipo de documento
