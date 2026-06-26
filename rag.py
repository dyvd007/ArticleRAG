"""
RAG de Artigos Científicos
===========================
Comandos disponíveis:

  python rag.py add <arquivo.pdf>      Adiciona um artigo ao banco
  python rag.py add <pasta/>           Adiciona todos os PDFs de uma pasta
  python rag.py list                   Lista artigos indexados
  python rag.py remove <nome.pdf>      Remove um artigo do banco
  python rag.py chat                   Inicia conversa com o RAG
  python rag.py ask "<pergunta>"       Faz uma pergunta direta (sem histórico)
  python rag.py reset                  Apaga todo o banco vetorial
"""

import os
import sys
import re
import textwrap
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
load_dotenv()   # carrega as variáveis do .env
DB_DIR          = Path(".chromadb")        # banco vetorial local
COLLECTION_NAME = "artigos"
CHUNK_SIZE      = 1200                     # caracteres por chunk
CHUNK_OVERLAP   = 200                      # sobreposição
TOP_K           = 50                        # chunks recuperados por pergunta
GEMINI_MODEL    = "gemini-2.5-flash"
SENTENCE_MODEL  = "intfloat/multilingual-e5-large"  # multilíngue, cross-lingual superior, local, gratuito

GCP_PROJECT  = os.environ.get("GCP_PROJECT", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

LINHA  = "─" * 62
DLINHA = "═" * 62

# ══════════════════════════════════════════════════════════════════════════════
# CLIENTES
# ══════════════════════════════════════════════════════════════════════════════

def get_collection():
    """Retorna (ou cria) a coleção ChromaDB com embeddings locais (sentence-transformers)."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=SENTENCE_MODEL,
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def get_gemini():
    return GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PARSING DE PDF
# ══════════════════════════════════════════════════════════════════════════════

def _tem_duas_colunas(pag) -> bool:
    """Detecta layout de duas colunas verificando zona vazia no centro da página."""
    words = pag.extract_words()
    if len(words) < 30:
        return False
    meio = pag.width / 2
    margem = pag.width * 0.10
    na_zona_central = sum(
        1 for w in words
        if (meio - margem) < (w["x0"] + w["x1"]) / 2 < (meio + margem)
    )
    return (na_zona_central / len(words)) < 0.05


def extrair_texto(caminho: Path) -> str:
    """Extrai texto de um PDF com pdfplumber, respeitando layout de duas colunas."""
    paginas = []
    with pdfplumber.open(caminho) as pdf:
        for i, pag in enumerate(pdf.pages):
            if _tem_duas_colunas(pag):
                meio = pag.width / 2
                col_esq = pag.crop((0, 0, meio, pag.height)).extract_text(x_tolerance=2, y_tolerance=3) or ""
                col_dir = pag.crop((meio, 0, pag.width, pag.height)).extract_text(x_tolerance=2, y_tolerance=3) or ""
                texto = (col_esq + "\n\n" + col_dir).strip()
            else:
                texto = pag.extract_text(x_tolerance=2, y_tolerance=3) or ""
            if texto.strip():
                paginas.append(f"[p.{i+1}] {texto}")
    return "\n\n".join(paginas)


def limpar(texto: str) -> str:
    """Remove artefatos comuns de PDFs acadêmicos."""
    # Corrige hifenização
    texto = re.sub(r"-\n(\w)", r"\1", texto)
    # Separa palavras coladas de PDFs com colunas duplas (ex: "varianceexplained" → "variance explained")
    texto = re.sub(r"([a-z])([A-Z])", r"\1 \2", texto)
    texto = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", texto)
    texto = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", texto)
    # Colapsa múltiplas linhas em branco
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    # Remove espaços duplos
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    return texto.strip()


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunkar(texto: str, fonte: str) -> list[dict]:
    """
    Divide o texto em chunks com overlap.
    Tenta respeitar quebras de parágrafo quando possível.
    """
    # Divide em parágrafos primeiro
    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]

    chunks = []
    buffer = ""

    for paragrafo in paragrafos:
        # Se o parágrafo sozinho já é enorme, subdivide por frase
        if len(paragrafo) > CHUNK_SIZE * 1.5:
            sentencas = re.split(r"(?<=[.!?])\s+", paragrafo)
            for s in sentencas:
                if len(buffer) + len(s) > CHUNK_SIZE and buffer:
                    chunks.append({"texto": buffer.strip(), "fonte": fonte})
                    buffer = buffer[-CHUNK_OVERLAP:] + " " + s
                else:
                    buffer += (" " if buffer else "") + s
        else:
            if len(buffer) + len(paragrafo) > CHUNK_SIZE and buffer:
                chunks.append({"texto": buffer.strip(), "fonte": fonte})
                buffer = buffer[-CHUNK_OVERLAP:] + "\n\n" + paragrafo
            else:
                buffer += ("\n\n" if buffer else "") + paragrafo

    if buffer.strip():
        chunks.append({"texto": buffer.strip(), "fonte": fonte})

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# METADADOS DO ARTIGO (extrai título, ano, autores da primeira página)
# ══════════════════════════════════════════════════════════════════════════════

_RE_PAGE_PREFIX = re.compile(r"^\[p\.\d+\]\s*")
_RE_NAO_TITULO = re.compile(
    r"^\s*$"
    r"|contents\s+lists?\s+available"
    r"|journal\s+homepage"
    r"|www\."
    r"|https?://"
    r"|\bdoi\b"
    r"|\b(received|accepted|available\s+online)\b"
    r"|©"
    r"|\(\d{4}\)\s+\d{3,}"          # (ano) número-de-artigo ex: (2025) 116620
    r"|\d{3,4}\s*[-–]\s*\d{2,4}"   # range de páginas ex: 175–179
    r"|^\s*\d{1,4}\s*$"             # só número de página
    r"|@\w"                         # e-mails
    r"|universid|institut|depart|facul|school"
    r"|\bpress\b|\bpublish"
    r"|\.\w{2,4}\/",       # URLs sem https:// ex: pubs.acs.org/JPCC
    re.IGNORECASE,
)
# Para antes do abstract (Elsevier usa letras espaçadas: "A B S T R A C T")
_RE_ABSTRACT_HEADER = re.compile(
    r"A\s*B\s*S\s*T\s*R\s*A\s*C\s*T"
    r"|A\s*R\s*T\s*I\s*C\s*L\s*E\s+I\s*N\s*F\s*O"
    r"|\bRESUMO\b",
    re.IGNORECASE,
)


def extrair_metadados(caminho: Path, texto: str) -> dict:
    """Tenta inferir título e ano do texto da primeira página."""
    primeiras_linhas = texto[:2000].splitlines()

    candidatos = []
    for linha in primeiras_linhas[:50]:
        if _RE_ABSTRACT_HEADER.search(linha.strip()):
            break                   # abstract começa aqui, título já passou
        limpa = _RE_PAGE_PREFIX.sub("", linha).strip()
        if (len(limpa) > 20
                and limpa.count(",") < 3
                and not _RE_NAO_TITULO.search(limpa)):
            candidatos.append(limpa)

    # Título é a linha mais longa antes do abstract
    titulo = max(candidatos, key=len) if candidatos else caminho.stem

    # Ano: primeiro número de 4 dígitos entre 1990-2099 (busca na página inteira)
    match_ano = re.search(r"\b(19[9]\d|20[0-3]\d)\b", texto[:5000])
    ano = match_ano.group(1) if match_ano else "desconhecido"

    return {
        "titulo":       titulo[:120],
        "ano":          ano,
        "arquivo":      caminho.name,
        "data_ingestao": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_chars":  str(len(texto)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_add(alvos: list[str]):
    """Adiciona um ou mais PDFs ao banco."""
    col = get_collection()

    # Expande pastas
    pdfs = []
    for alvo in alvos:
        p = Path(alvo)
        if p.is_dir():
            pdfs.extend(p.glob("*.pdf"))
        elif p.suffix.lower() == ".pdf":
            pdfs.append(p)
        else:
            print(f"  ⚠️  Ignorado (não é PDF): {alvo}")

    if not pdfs:
        print("\n❌ Nenhum PDF encontrado.\n")
        return

    for pdf in pdfs:
        print(f"\n{LINHA}")
        print(f"  📄 {pdf.name}")
        print(LINHA)

        # Verifica se já existe
        existente = col.get(where={"arquivo": pdf.name}, limit=1)
        if existente["ids"]:
            print(f"  ⚠️  Já indexado. Use 'remove' primeiro para reindexar.")
            continue

        # Extrai e limpa
        print("  Extraindo texto...", end=" ", flush=True)
        texto = limpar(extrair_texto(pdf))
        print(f"✓  ({len(texto):,} chars)")

        if len(texto) < 200:
            print("  ⚠️  Texto muito curto — PDF pode ser escaneado. Use o parser OCR.")
            continue

        # Metadados
        meta = extrair_metadados(pdf, texto)
        print(f"  Título inferido : {meta['titulo'][:60]}...")
        print(f"  Ano inferido    : {meta['ano']}")

        # Chunking
        print("  Criando chunks...", end=" ", flush=True)
        chunks = chunkar(texto, pdf.name)
        print(f"✓  ({len(chunks)} chunks)")

        # Indexa no ChromaDB
        print("  Indexando embeddings...", end=" ", flush=True)
        ids      = [f"{pdf.stem}__{i}" for i in range(len(chunks))]
        textos   = [c["texto"] for c in chunks]
        metadados = [{**meta, "chunk_index": str(i)} for i in range(len(chunks))]

        # Insere em lotes de 50 (evita timeout de API)
        lote = 50
        for i in range(0, len(chunks), lote):
            col.add(
                ids=ids[i:i+lote],
                documents=textos[i:i+lote],
                metadatas=metadados[i:i+lote],
            )
        print(f"✓")
        print(f"\n  ✅ '{pdf.name}' adicionado com sucesso!")

    print()


def cmd_list():
    """Lista todos os artigos indexados."""
    col = get_collection()
    total = col.count()

    if total == 0:
        print("\n  Banco vazio. Use: python rag.py add <arquivo.pdf>\n")
        return

    # Pega metadados únicos por arquivo
    todos = col.get(include=["metadatas"])
    arquivos = {}
    for meta in todos["metadatas"]:
        arq = meta["arquivo"]
        if arq not in arquivos:
            arquivos[arq] = meta

    print(f"\n{DLINHA}")
    print(f"  📚 Artigos indexados ({len(arquivos)} artigos / {total} chunks)")
    print(DLINHA)
    for i, (arq, meta) in enumerate(sorted(arquivos.items()), 1):
        print(f"\n  [{i}] {arq}")
        print(f"      Título  : {meta.get('titulo','?')[:65]}")
        print(f"      Ano     : {meta.get('ano','?')}")
        print(f"      Ingerido: {meta.get('data_ingestao','?')}")
    print(f"\n{DLINHA}\n")


def cmd_remove(nome: str):
    """Remove todos os chunks de um artigo."""
    col = get_collection()
    resultado = col.get(where={"arquivo": nome})
    ids = resultado["ids"]

    if not ids:
        print(f"\n  ❌ Artigo '{nome}' não encontrado no banco.\n")
        return

    col.delete(ids=ids)
    print(f"\n  🗑️  '{nome}' removido ({len(ids)} chunks apagados).\n")


def cmd_reset():
    """Apaga todo o banco vetorial após confirmação."""
    resp = input("\n  ⚠️  Isso apagará TODOS os artigos. Confirma? (sim/não): ").strip().lower()
    if resp == "sim":
        import shutil
        shutil.rmtree(DB_DIR, ignore_errors=True)
        print("  ✓ Banco apagado.\n")
    else:
        print("  Operação cancelada.\n")


# ══════════════════════════════════════════════════════════════════════════════
# RECUPERAÇÃO + GERAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
Você é um assistente de pesquisa especializado em análise de artigos científicos.
Responda em português de forma clara, precisa e bem estruturada.

Regras:
1. Baseie-se EXCLUSIVAMENTE nos trechos fornecidos como contexto.
2. Cite sempre a fonte (nome do arquivo) quando usar uma informação.
3. Se a informação específica não estiver nos trechos fornecidos, declare "Não encontrei essa informação nos trechos recuperados." Não infira além do que os trechos permitem nem combine informações de artigos diferentes para compor uma resposta que nenhum deles contém individualmente.
4. Para comparações entre artigos, seja explícito sobre qual afirmação vem de qual fonte.
5. Nunca invente referências, dados ou conclusões.
"""

def buscar(col, pergunta: str) -> list[dict]:
    resultado = col.query(
        query_texts=[pergunta],
        n_results=min(TOP_K, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        resultado["documents"][0],
        resultado["metadatas"][0],
        resultado["distances"][0],
    ):
        chunks.append({
            "texto":    doc,
            "arquivo":  meta.get("arquivo", "?"),
            "pagina":   meta.get("chunk_index", "?"),
            "score":    round(1 - dist, 3),   # distância coseno → similaridade
        })
    return chunks


def montar_contexto(chunks: list[dict]) -> str:
    partes = []
    for i, c in enumerate(chunks, 1):
        partes.append(
            f"[Trecho {i} | Fonte: {c['arquivo']} | Relevância: {c['score']}]\n{c['texto']}"
        )
    return "\n\n" + ("─" * 40 + "\n\n").join(partes)


def gerar(modelo, pergunta: str, chunks: list[dict], historico: list[dict]) -> str:
    contexto = montar_contexto(chunks)
    mensagem = (
        f"Contexto recuperado dos artigos:\n{contexto}\n\n"
        f"Pergunta: {pergunta}"
    )
    # Converte formato de histórico para Content/Part do Vertex AI
    gemini_hist = [
        Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[Part.from_text(m["content"])],
        )
        for m in historico
    ]
    chat = modelo.start_chat(history=gemini_hist)
    resp = chat.send_message(mensagem)
    return resp.text


def imprimir_resposta(texto: str):
    print()
    for linha in texto.splitlines():
        if linha.strip():
            print(textwrap.fill(linha, width=70, subsequent_indent="  "))
        else:
            print()


def cmd_ask(pergunta: str):
    """Faz uma pergunta única sem histórico."""
    col = get_collection()
    if col.count() == 0:
        print("\n❌ Banco vazio. Adicione artigos primeiro.\n")
        return

    modelo  = get_gemini()
    chunks  = buscar(col, pergunta)
    fontes  = list(dict.fromkeys(c["arquivo"] for c in chunks))
    print(f"\n  📎 Fontes consultadas: {', '.join(fontes)}")
    resposta = gerar(modelo, pergunta, chunks, [])
    print(f"\n🤖 Resposta:")
    imprimir_resposta(resposta)
    print()


def cmd_chat():
    """Chat interativo com histórico de conversa."""
    col = get_collection()
    if col.count() == 0:
        print("\n❌ Banco vazio. Adicione artigos com: python rag.py add <arquivo.pdf>\n")
        return

    modelo    = get_gemini()
    historico = []

    print(f"\n{DLINHA}")
    print("  🤖 RAG de Artigos Científicos — Chat")
    print(f"{DLINHA}")
    print("  Comandos: 'sair' | 'limpar' (novo histórico) | 'fontes' (lista artigos)")
    print(f"{DLINHA}\n")

    while True:
        try:
            pergunta = input("🎓 Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Até logo!\n")
            break

        if not pergunta:
            continue

        if pergunta.lower() == "sair":
            print("\n  Até logo!\n")
            break

        if pergunta.lower() == "limpar":
            historico = []
            print("  🗑️  Histórico limpo.\n")
            continue

        if pergunta.lower() == "fontes":
            cmd_list()
            continue

        # Recupera e gera
        chunks  = buscar(col, pergunta)
        fontes  = list(dict.fromkeys(c["arquivo"] for c in chunks))
        scores  = [c["score"] for c in chunks]
        print(f"\n  📎 {len(chunks)} trechos de: {', '.join(fontes)}")
        print(f"  📊 Relevâncias: {[f'{s:.2f}' for s in scores]}")
        print("  ⏳ Gerando...\n")

        resposta = gerar(modelo, pergunta, chunks, historico)

        print(f"🤖 Gemini:")
        imprimir_resposta(resposta)
        print(f"\n{LINHA}\n")

        # Atualiza histórico (só a pergunta limpa, sem o contexto)
        historico.append({"role": "user",      "content": pergunta})
        historico.append({"role": "assistant", "content": resposta})
        if len(historico) > 12:           # mantém últimas 6 trocas
            historico = historico[-12:]


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

COMANDOS = {
    "add":    (cmd_add,    "Adiciona PDFs ao banco"),
    "list":   (cmd_list,   "Lista artigos indexados"),
    "remove": (cmd_remove, "Remove um artigo"),
    "chat":   (cmd_chat,   "Chat interativo"),
    "ask":    (cmd_ask,    "Pergunta direta"),
    "reset":  (cmd_reset,  "Apaga o banco"),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMANDOS:
        print(__doc__)
        sys.exit(0)

    if not GCP_PROJECT and sys.argv[1] in ("chat", "ask"):
        print("\n❌ GCP_PROJECT não definido.\n   Adicione GCP_PROJECT=seu-projeto ao .env\n")
        sys.exit(1)

    cmd  = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "add":
        cmd_add(args)
    elif cmd == "list":
        cmd_list()
    elif cmd == "remove":
        cmd_remove(args[0] if args else "")
    elif cmd == "chat":
        cmd_chat()
    elif cmd == "ask":
        cmd_ask(" ".join(args))
    elif cmd == "reset":
        cmd_reset()