# RAG de Artigos Científicos

Sistema de **Retrieval-Augmented Generation (RAG)** para consulta de artigos científicos em PDF via linha de comando. Indexa os artigos localmente com embeddings multilíngues e usa o **Gemini 2.5 Flash** (via Vertex AI) para gerar respostas fundamentadas exclusivamente no conteúdo dos artigos carregados.

## Como funciona

```
PDF → extração de texto → chunking → embeddings locais (sentence-transformers)
                                              ↓
pergunta → busca vetorial (ChromaDB) → trechos relevantes → Gemini → resposta
```

- **Embeddings**: `intfloat/multilingual-e5-large` — rodando localmente, sem custo, alta qualidade multilíngue
- **Banco vetorial**: ChromaDB persistido em `.chromadb/`
- **LLM**: Google Gemini 2.5 Flash via Vertex AI (requer projeto GCP)

## Pré-requisitos

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- Projeto no Google Cloud com Vertex AI habilitado

## Instalação

```bash
# Clonar / entrar na pasta do projeto
cd "AV FC1"

# Instalar dependências
uv sync

# Criar o arquivo de variáveis de ambiente
cp .env.example .env
# Editar .env com seu projeto GCP
```

### Variáveis de ambiente (`.env`)

```env
GCP_PROJECT=seu-projeto-gcp
GCP_LOCATION=us-central1   # opcional, padrão: us-central1
```

## Uso

### Adicionar artigos

```bash
# Adiciona um PDF
uv run python rag.py add artigos/meu_artigo.pdf

# Adiciona todos os PDFs de uma pasta
uv run python rag.py add artigos/
```

### Listar artigos indexados

```bash
uv run python rag.py list
```

### Fazer uma pergunta direta

```bash
uv run python rag.py ask "Qual é a metodologia usada no artigo sobre difração cônica?"
```

### Chat interativo (com histórico)

```bash
uv run python rag.py chat
```

Comandos dentro do chat:

| Comando  | Ação                          |
|----------|-------------------------------|
| `sair`   | Encerra o chat                |
| `limpar` | Reinicia o histórico          |
| `fontes` | Lista os artigos no banco     |

### Remover um artigo

```bash
uv run python rag.py remove nome_do_arquivo.pdf
```

### Apagar o banco inteiro

```bash
uv run python rag.py reset
```

## Pipeline de Avaliação

O projeto inclui um pipeline completo de avaliação com **LLM-as-Judge** (Gemini 2.5 Flash Lite via Vertex AI).

### 1. Gerar dataset de perguntas

```bash
uv run python gerar_dataset.py
```

Cria `dataset.csv` com perguntas e gabaritos fixos gerados a partir dos artigos indexados.

### 2. Gerar respostas do RAG

```bash
uv run python gerar_respostas_rag.py
# Opções:
#   --resume   continua de onde parou (padrão: sobrescreve o arquivo)
```

Lê o `dataset.csv`, consulta o RAG para cada pergunta e salva em `respostas_rag.csv`.

### 3. Avaliar com LLM-as-Judge

```bash
uv run python avaliar.py
# Opções:
#   --resume   continua de onde parou (padrão: sobrescreve o arquivo)
```

Avalia cada resposta com o Gemini e salva resultados detalhados em `avaliacoes.csv` e o histórico de métricas em `metricas.csv`.

### (Opcional) Executar varredura automática de TOP_K

```bash
uv run python run_experimentos.py
# Opções:
#   --top-k 1 3 5 10   valores customizados de TOP_K a testar
#   --skip-existing     pula valores de TOP_K já presentes em metricas.csv
```

Automatiza a sequência gerar → avaliar para múltiplos valores de `TOP_K`, acumulando resultados em `metricas.csv`.

## Avaliação

Métricas calculadas com **LLM-as-Judge** (Gemini 2.5 Flash Lite via Vertex AI) sobre amostras do dataset de perguntas gerado a partir dos artigos indexados.

### Histórico de execuções

| Data | TOP_K | Amostras | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|------|-------|----------|--------------|-----------------|-------------------|----------------|
| 2026-06-24 | 15 | 15 | 0.7933 | 0.6800 | 0.8367 | 0.5000 |
| 2026-06-24 | 15 | 15 | 0.7733 | 0.7333 | 0.8600 | 0.6800 |
| 2026-06-25 | 14 | 14 | 0.8786 | 0.9000 | 0.9286 | 0.8857 |
| 2026-06-26 | 15 | 15 | 0.9200 | 0.8533 | 0.9600 | 0.7533 |

### Experimento TOP_K — Série 1 (2026-06-26, até 18:52)

Avaliação sistemática do impacto do parâmetro `TOP_K` (chunks recuperados por pergunta) nas métricas. O banco indexado tem **136 chunks no total** distribuídos entre os 3 artigos:

| Artigo | Chunks |
|--------|--------|
| Two-Photon Emissive Dyes | 63 |
| PCA | 44 |
| Técnica Difração Cônica | 29 |

| TOP_K | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Média |
|-------|--------------|-----------------|-------------------|----------------|-------|
| 1  | 0.7933 | 0.5467 | 0.7067 | 0.3600 | 0.6017 |
| 3  | 0.9333 | 0.8200 | 0.9733 | 0.6600 | 0.8467 |
| 5  | 0.8467 | 0.7800 | 0.9867 | 0.7200 | 0.8334 |
| 10 | 0.8667 | 0.8533 | 0.9933 | 0.8933 | 0.9016 |
| 15 | 0.9333 | 0.8533 | 0.9400 | 0.8133 | 0.8850 |
| **20** | **0.9667** | 0.9600 | 0.9867 | 0.9133 | 0.9567 |
| 25 | 0.8800 | 0.9180 | 0.9667 | 0.9200 | 0.9212 |
| **30** | 0.9400 | **0.9800** | 0.9733 | **0.9600** | **0.9633** |
| 35 | 0.8933 | 0.9333 | **0.9800** | 0.9200 | 0.9316 |
| 40 | 0.9200 | 0.9733 | **0.9800** | 0.9467 | 0.9550 |
| 50 | 0.9000 | 0.9267 | 0.9933 | 0.9667 | 0.9467 |

**TOP_K=30 apresenta a maior média geral (0.9633).** TOP_K=20 tem o melhor Faithfulness isolado (0.9667).

#### Por que as métricas se estabilizam acima de TOP_K=20

A busca vetorial retorna os chunks mais similares do banco inteiro, não de um artigo específico. Isso cria um **ponto de saturação por artigo**: ao atingir o número de chunks disponíveis no artigo consultado, chunks adicionais vêm de outros artigos e são descartados pelo modelo (que identifica a fonte pelo metadado).

- Para perguntas sobre difração cônica (29 chunks): saturação em TOP_K ≈ 29
- Para perguntas sobre PCA (44 chunks): saturação em TOP_K ≈ 44
- Para perguntas sobre Two-Photon (63 chunks): saturação em TOP_K ≈ 63

O efeito é **assimétrico**: abaixo do ponto de saturação, a ausência de um chunk crítico derruba a resposta completamente (queda abrupta abaixo de TOP_K=10). Acima do ponto de saturação, o ruído extra não prejudica muito porque LLMs são tolerantes a contexto irrelevante.

### Experimento TOP_K — Série 2 (2026-06-26, a partir de 20:47)

Nova varredura com a avaliação de `context_precision` revisada — cada chunk é avaliado individualmente quanto à sua relevância para a resposta, o que reflete o critério original do framework RAGAS de forma mais fiel.

| TOP_K | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Média |
|-------|--------------|-----------------|-------------------|----------------|-------|
| 1  | 0.6696 | 0.5733 | 0.7333 | 0.3811 | 0.5893 |
| 3  | 0.8508 | 0.8600 | 0.5791 | 0.6910 | 0.7452 |
| 5  | 0.8222 | 0.8733 | 0.4000 | 0.6957 | 0.6978 |
| 10 | 0.9704 | 0.8733 | 0.3133 | 0.7817 | 0.7347 |
| 20 | **1.0000** | **0.9733** | 0.1659 | 0.8867 | 0.7565 |
| 25 | 0.9889 | 0.9400 | 0.2039 | 0.9238 | 0.7641 |
| 30 | **1.0000** | 0.9400 | 0.1645 | **0.9554** | 0.7650 |
| 35 | **1.0000** | 0.9667 | 0.1340 | 0.9483 | 0.7623 |
| 40 | **1.0000** | 0.9467 | **0.1073** | 0.9094 | 0.7409 |

#### Análise do trade-off precision × recall

O padrão clássico de **precision vs. recall** ficou explícito nesta série:

- **Faithfulness** e **Answer Relevancy** melhoram consistentemente com TOP_K, chegando a 1.00 e 0.97 para TOP_K ≥ 20 — o modelo não alucina e gera respostas altamente relevantes.
- **Context Recall** sobe de 0.38 (TOP_K=1) para 0.96 (TOP_K=30), confirmando que o retriever está capturando quase todo o conteúdo necessário.
- **Context Precision** colapsa de 0.73 (TOP_K=1) para 0.11 (TOP_K=40) — ao recuperar mais chunks, uma fração crescente é irrelevante para a pergunta específica, indicando ausência de reranking.

Comparação direta com a Série 1 no mesmo TOP_K:

| TOP_K | Precisão — Série 1 | Precisão — Série 2 |
|-------|--------------------|--------------------|
| 3  | 0.9733 | 0.5791 |
| 5  | 0.9867 | 0.4000 |
| 10 | 0.9933 | 0.3133 |
| 20 | 0.9867 | 0.1659 |

A queda revela que a Série 1 subestimava o ruído no contexto (avaliação holística do judge) enquanto a Série 2 detecta com mais precisão quantos chunks são de fato úteis por pergunta.

#### Próximo passo sugerido

Implementar um **reranker (cross-encoder)** após a recuperação vetorial para filtrar chunks irrelevantes antes de enviá-los ao LLM — isso deve recuperar a precisão sem sacrificar o recall e o faithfulness obtidos com TOP_K alto.

## Estrutura do projeto

```
AV FC1/
├── rag.py                   # Script principal (add/list/ask/chat/remove/reset)
├── gerar_dataset.py         # Gera dataset fixo de perguntas e gabaritos
├── gerar_respostas_rag.py   # Gera respostas do RAG para o dataset
├── avaliar.py               # Avalia respostas com LLM-as-Judge
├── eval_utils.py            # Utilitários compartilhados de avaliação
├── run_experimentos.py      # Varredura automática de TOP_K (gerar → avaliar em loop)
├── artigos/                 # PDFs para indexar
├── dataset.csv              # Perguntas e gabaritos de avaliação
├── respostas_rag.csv        # Respostas geradas pelo RAG
├── avaliacoes.csv           # Avaliações detalhadas por amostra
├── metricas.csv             # Histórico de métricas por execução
├── .chromadb/               # Banco vetorial (gerado automaticamente, não versionado)
├── .env                     # Variáveis de ambiente (não versionado)
├── pyproject.toml           # Dependências
└── uv.lock                  # Lock file
```

## Dependências principais

| Pacote | Função |
|--------|--------|
| `chromadb` | Banco vetorial local |
| `sentence-transformers` | Geração de embeddings locais |
| `google-cloud-aiplatform` | Cliente Vertex AI (Gemini) |
| `pdfplumber` | Extração de texto de PDFs |

## Observações

- PDFs escaneados (imagens) não são suportados — o sistema requer PDFs com texto selecionável.
- O modelo responde **somente com base nos artigos indexados**, nunca inventando referências ou dados.
- O histórico do chat mantém as últimas 6 trocas para não estourar o contexto do modelo.
