"""
Lê o dataset de avaliação, gera as respostas do RAG e salva em CSV.

  python gerar_respostas_rag.py                     # usa dataset.csv
  python gerar_respostas_rag.py --dataset meu.csv   # dataset personalizado
  python gerar_respostas_rag.py --resume             # continua de onde parou
  python gerar_respostas_rag.py --reset              # apaga o CSV e recomeça
"""

import sys
import csv
import time
import argparse
from pathlib import Path

from rag import get_collection, buscar, gerar
from eval_utils import GCP_PROJECT, get_gemini_rag_eval, API_DELAY

DATASET_CSV   = "dataset.csv"
RESPOSTAS_CSV = "respostas_rag.csv"
FIELDNAMES    = [
    "artigo", "secao_artigo", "pergunta",
    "resposta_gabarito", "resposta_rag", "n_chunks_recuperados",
]
DLINHA = "═" * 62


def carregar_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def salvar_linha(path: str, row: dict, escrever_header: bool) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if escrever_header:
            writer.writeheader()
        writer.writerow(row)


def gerar_com_retry(modelo_rag, pergunta: str, chunks: list, max_retries: int = 6) -> str:
    for tentativa in range(1, max_retries + 1):
        time.sleep(API_DELAY)
        try:
            return gerar(modelo_rag, pergunta, chunks, [])
        except Exception as e:
            msg = str(e).lower()
            is_quota = "429" in msg or "quota" in msg or "resource exhausted" in msg
            if not is_quota:
                raise
            if "per day" in msg or "daily" in msg or "per_day" in msg:
                raise RuntimeError(
                    "\n❌ COTA DIÁRIA do Gemini esgotada.\n"
                    "   Tente novamente amanhã ou use outra GOOGLE_API_KEY.\n"
                ) from e
            if tentativa == max_retries:
                raise RuntimeError(f"Cota excedida após {max_retries} tentativas (RAG).") from e
            wait = API_DELAY * (2 ** tentativa)
            print(f"\n  ⏳ Cota excedida - RAG (tentativa {tentativa}/{max_retries}). "
                  f"Aguardando {wait}s...")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Gera respostas do RAG para o dataset")
    parser.add_argument("--dataset", type=str, default=DATASET_CSV,   help="CSV do dataset de entrada")
    parser.add_argument("--output",  type=str, default=RESPOSTAS_CSV, help="CSV de saída")
    parser.add_argument("--resume",  action="store_true",             help="Continua de onde parou")
    parser.add_argument("--reset",   action="store_true",             help="Apaga o CSV e recomeça do zero")
    args = parser.parse_args()

    if args.resume and args.reset:
        print("❌ Use --resume OU --reset, não os dois.\n")
        sys.exit(1)

    if not GCP_PROJECT:
        print("\n❌ GCP_PROJECT não definido.\n   Adicione GCP_PROJECT=seu-projeto ao .env\n")
        sys.exit(1)

    output_path = Path(args.output)

    if args.reset and output_path.exists():
        output_path.unlink()
        print(f"  🔄 Reset: '{args.output}' removido.")
    elif not args.resume and output_path.exists():
        output_path.unlink()

    dataset = carregar_csv(args.dataset)
    if not dataset:
        print(f"\n❌ Dataset '{args.dataset}' vazio ou não encontrado.\n"
              f"   Execute: python gerar_dataset.py\n")
        sys.exit(1)

    existente        = carregar_csv(args.output)
    perguntas_feitas = {r["pergunta"] for r in existente}
    pendentes        = [r for r in dataset if r["pergunta"] not in perguntas_feitas]

    escrever_header = not output_path.exists() or output_path.stat().st_size == 0

    col        = get_collection()
    modelo_rag = get_gemini_rag_eval()

    print(f"\n{DLINHA}")
    print("  GERAR RESPOSTAS RAG")
    print(DLINHA)
    print(f"  Dataset              : {args.dataset} ({len(dataset)} perguntas)")
    print(f"  Já processadas       : {len(perguntas_feitas)}")
    print(f"  A processar          : {len(pendentes)}")
    print(f"  Arquivo de saída     : {args.output}")
    modo = "resume" if args.resume else "reset" if args.reset else "normal"
    print(f"  Modo                 : {modo}")
    print(DLINHA)

    for i, amostra in enumerate(pendentes, 1):
        pergunta = amostra["pergunta"]
        print(f"\n  [{i}/{len(pendentes)}] {pergunta[:60]}...")
        try:
            chunks       = buscar(col, pergunta)
            resposta_rag = gerar_com_retry(modelo_rag, pergunta, chunks)
            row = {
                "artigo":               amostra["artigo"],
                "secao_artigo":         amostra["secao_artigo"],
                "pergunta":             pergunta,
                "resposta_gabarito":    amostra["resposta_gabarito"],
                "resposta_rag":         resposta_rag,
                "n_chunks_recuperados": len(chunks),
            }
            salvar_linha(args.output, row, escrever_header)
            escrever_header = False
            print(f"         ✓  Resposta gerada ({len(chunks)} chunks recuperados)")
        except Exception as e:
            print(f"         ✗  Erro: {e}")

    total = len(existente) + len(pendentes)
    print(f"\n  ✓ {len(pendentes)} respostas geradas. Total no arquivo: {total}")
    print(f"  Respostas salvas em: {args.output}\n")


if __name__ == "__main__":
    main()
