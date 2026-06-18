# RAG de Artigos Científicos

Sistema de **Retrieval-Augmented Generation (RAG)** para consulta de artigos científicos em PDF via linha de comando. Indexa os artigos localmente com embeddings multilíngues e usa o **Gemini 2.5 Flash** para gerar respostas fundamentadas exclusivamente no conteúdo dos artigos carregados.

## Como funciona

```
PDF → extração de texto → chunking → embeddings locais (sentence-transformers)
                                              ↓
pergunta → busca vetorial (ChromaDB) → trechos relevantes → Gemini → resposta
```

- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` — rodando localmente, sem custo, suporta 50+ idiomas
- **Banco vetorial**: ChromaDB persistido em `.chromadb/`
- **LLM**: Google Gemini 2.5 Flash (requer `GOOGLE_API_KEY`)

## Pré-requisitos

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- Chave de API do Google Gemini

## Instalação

```bash
# Clonar / entrar na pasta do projeto
cd "AV FC1"

# Instalar dependências
uv sync

# Criar o arquivo de variáveis de ambiente
echo "GOOGLE_API_KEY=sua_chave_aqui" > .env
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

## Estrutura do projeto

```
AV FC1/
├── rag.py          # Script principal
├── artigos/        # PDFs para indexar
├── .chromadb/      # Banco vetorial (gerado automaticamente)
├── .env            # Variáveis de ambiente (não versionar)
├── pyproject.toml  # Dependências
└── uv.lock         # Lock file
```

## Dependências principais

| Pacote | Função |
|--------|--------|
| `chromadb` | Banco vetorial local |
| `sentence-transformers` | Geração de embeddings locais |
| `google-generativeai` | Cliente da API Gemini |
| `pdfplumber` | Extração de texto de PDFs |

## Observações

- PDFs escaneados (imagens) não são suportados — o sistema requer PDFs com texto selecionável.
- O modelo responde **somente com base nos artigos indexados**, nunca inventando referências ou dados.
- O histórico do chat mantém as últimas 6 trocas para não estourar o contexto do modelo.
