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
#   --resume   continua de onde parou
#   --reset    apaga e recomeça do zero
```

Lê o `dataset.csv`, consulta o RAG para cada pergunta e salva em `respostas_rag.csv`.

### 3. Avaliar com LLM-as-Judge

```bash
uv run python avaliar.py
# Opções:
#   --resume   continua de onde parou
#   --reset    apaga e recomeça do zero
```

Avalia cada resposta com o Gemini e salva resultados detalhados em `avaliacoes.csv` e o histórico de métricas em `metricas.csv`.

## Avaliação

Métricas calculadas com **LLM-as-Judge** (Gemini 2.5 Flash Lite via Vertex AI) sobre amostras do dataset de perguntas gerado a partir dos artigos indexados.

| Data | Amostras | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|------|----------|--------------|-----------------|-------------------|----------------|
| 2026-06-24 | 15 | 0.7933 | 0.6800 | 0.8367 | 0.5000 |
| 2026-06-24 | 15 | 0.7733 | 0.7333 | 0.8600 | 0.6800 |
| 2026-06-25 | 14 | 0.8786 | 0.9000 | 0.9286 | 0.8857 |
| 2026-06-26 | 15 | **0.9200** | **0.8533** | **0.9600** | **0.7533** |

**Última avaliação (2026-06-26):**

| Métrica | Valor | Descrição |
|---------|-------|-----------|
| Faithfulness | 0.92 | Respostas fiéis ao contexto recuperado |
| Answer Relevancy | 0.85 | Relevância da resposta em relação à pergunta |
| Context Precision | 0.96 | Precisão dos trechos recuperados |
| Context Recall | 0.75 | Cobertura do contexto necessário |

## Estrutura do projeto

```
AV FC1/
├── rag.py                   # Script principal (add/list/ask/chat/remove/reset)
├── gerar_dataset.py         # Gera dataset fixo de perguntas e gabaritos
├── gerar_respostas_rag.py   # Gera respostas do RAG para o dataset
├── avaliar.py               # Avalia respostas com LLM-as-Judge
├── eval_utils.py            # Utilitários compartilhados de avaliação
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
