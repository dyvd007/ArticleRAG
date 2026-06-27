"""
Avalia as respostas do RAG com LLM-as-Judge e salva métricas em CSV.

  python avaliar.py                          # usa respostas_rag.csv
  python avaliar.py --respostas meu.csv      # CSV personalizado
  python avaliar.py --resume                 # continua de onde parou
"""

import sys
import csv
import argparse
from pathlib import Path
from statistics import mean
from datetime import datetime

from rag import get_collection, buscar, montar_contexto, TOP_K
from eval_utils import (
    GCP_PROJECT,
    get_gemini_judge,
    chamar_gemini,
    parse_json_llm,
    METRICAS,
    PROMPTS_POR_METRICA,
)

RESPOSTAS_CSV  = "respostas_rag.csv"
AVALIACOES_CSV = "avaliacoes.csv"
METRICAS_CSV   = "metricas.csv"
DLINHA         = "═" * 62
LINHA          = "─" * 62

LABELS = {
    "faithfulness":      "Fidelidade        ",
    "answer_relevancy":  "Relev. Resposta   ",
    "context_precision": "Precisão Contexto ",
    "context_recall":    "Recall Contexto   ",
}

FIELDNAMES_AVAL = (
    ["artigo", "secao_artigo", "pergunta", "resposta_gabarito",
     "resposta_rag", "n_chunks_recuperados"]
    + list(METRICAS)
    + [f"{m}_reasoning" for m in METRICAS]
)

FIELDNAMES_METRICAS = ["timestamp", "top_k"] + list(METRICAS)


def carregar_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def salvar_linha(path: str, row: dict, fieldnames: list, escrever_header: bool) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if escrever_header:
            writer.writeheader()
        writer.writerow(row)


def score_valido(v) -> bool:
    return v not in (None, "", "None")


def avaliar_amostra(judge, col, amostra: dict) -> dict:
    pergunta = amostra["pergunta"]
    gabarito = amostra["resposta_gabarito"]
    resposta = amostra["resposta_rag"]

    # Busca o contexto localmente (ChromaDB, sem custo de API)
    chunks        = buscar(col, pergunta)
    contexto_text = montar_contexto(chunks)
    # Formato sem scores de similaridade para não ancorar o juiz em context_precision
    contexto_numerado = "\n\n".join(
        f"[Chunk {i+1}]\n{c['texto']}" for i, c in enumerate(chunks)
    )

    resultado = dict(amostra)

    for nome, prompt_template in PROMPTS_POR_METRICA.items():
        prompt = prompt_template.format(
            pergunta=pergunta,
            gabarito=gabarito,
            contexto=contexto_text,
            contexto_numerado=contexto_numerado,
            resposta=resposta,
        )
        try:
            texto      = chamar_gemini(judge, prompt)
            julgamento = parse_json_llm(texto)
            score = julgamento.get("score")
            if score is not None:
                score = max(0.0, min(1.0, float(score)))
            resultado[nome]                = score
            resultado[f"{nome}_reasoning"] = julgamento.get("reasoning", "")
        except Exception as e:
            resultado[nome]                = None
            resultado[f"{nome}_reasoning"] = f"Erro: {e}"

    return resultado


def imprimir_relatorio(avaliados: list[dict]) -> dict:
    print(f"\n{DLINHA}")
    print("  RELATÓRIO DE AVALIAÇÃO DO RAG")
    print(DLINHA)
    print(f"  Amostras avaliadas : {len(avaliados)}")
    print()

    resumo = {}
    for nome in METRICAS:
        valores = [float(r[nome]) for r in avaliados if score_valido(r.get(nome))]
        if not valores:
            continue
        media         = mean(valores)
        resumo[nome]  = media
        barra         = "█" * int(media * 20) + "░" * (20 - int(media * 20))
        print(f"  {LABELS[nome]}: {media:.3f}  [{barra}]")

    ruins = [r for r in avaliados if score_valido(r.get("faithfulness")) and float(r["faithfulness"]) < 0.6]
    ruins = sorted(ruins, key=lambda r: float(r["faithfulness"]))

    if ruins:
        print(f"\n  ⚠️  {len(ruins)} amostra(s) com fidelidade < 0.6 (possível alucinação):")
        for r in ruins[:3]:
            print(f"\n  • Pergunta : {r['pergunta'][:70]}")
            print(f"    Fidelidade: {float(r['faithfulness']):.2f} — {r.get('faithfulness_reasoning', '')[:80]}")

    print(f"\n{DLINHA}\n")
    return resumo


def main():
    parser = argparse.ArgumentParser(description="Avalia respostas RAG com LLM-as-Judge")
    parser.add_argument("--respostas", type=str, default=RESPOSTAS_CSV,  help="CSV com respostas RAG")
    parser.add_argument("--output",    type=str, default=AVALIACOES_CSV, help="CSV de avaliações detalhadas")
    parser.add_argument("--metricas",  type=str, default=METRICAS_CSV,   help="CSV de métricas resumidas")
    parser.add_argument("--resume",    action="store_true",              help="Continua de onde parou")
    args = parser.parse_args()

    if not GCP_PROJECT:
        print("\n❌ GCP_PROJECT não definido.\n   Adicione GCP_PROJECT=seu-projeto ao .env\n")
        sys.exit(1)

    output_path = Path(args.output)

    if not args.resume and output_path.exists():
        output_path.unlink()

    respostas = carregar_csv(args.respostas)
    if not respostas:
        print(f"\n❌ Arquivo '{args.respostas}' vazio ou não encontrado.\n"
              f"   Execute: python gerar_respostas_rag.py\n")
        sys.exit(1)

    existente           = carregar_csv(args.output)
    perguntas_avaliadas = {r["pergunta"] for r in existente}
    pendentes           = [r for r in respostas if r["pergunta"] not in perguntas_avaliadas]

    escrever_header = not output_path.exists() or output_path.stat().st_size == 0

    col   = get_collection()
    judge = get_gemini_judge()

    print(f"\n{DLINHA}")
    print("  AVALIAR — LLM-as-Judge")
    print(DLINHA)
    print(f"  Respostas RAG        : {args.respostas} ({len(respostas)} entradas)")
    print(f"  Já avaliadas         : {len(perguntas_avaliadas)}")
    print(f"  A avaliar            : {len(pendentes)}")
    print(f"  Arquivo de saída     : {args.output}")
    modo = "resume" if args.resume else "normal"
    print(f"  Modo                 : {modo}")
    print(DLINHA)

    for i, amostra in enumerate(pendentes, 1):
        print(f"\n  [{i}/{len(pendentes)}] {amostra['pergunta'][:60]}...")
        resultado = avaliar_amostra(judge, col, amostra)

        abrevs = [("F", "faithfulness"), ("AR", "answer_relevancy"),
                  ("CP", "context_precision"), ("CR", "context_recall")]
        partes = [
            f"{k}={float(resultado[v]):.2f}" if score_valido(resultado.get(v)) else f"{k}=ERR"
            for k, v in abrevs
        ]
        print(f"         {' | '.join(partes)}")

        row = {fn: resultado.get(fn, "") for fn in FIELDNAMES_AVAL}
        salvar_linha(args.output, row, FIELDNAMES_AVAL, escrever_header)
        escrever_header = False

    # Relatório final sobre todos os itens (incluindo os já existentes)
    todos_avaliados = carregar_csv(args.output)
    resumo          = imprimir_relatorio(todos_avaliados)

    # Salva métricas resumidas em CSV (acumula histórico de execuções)
    if resumo:
        escrever_header_m = not Path(args.metricas).exists()
        row_m = {
            "timestamp": datetime.now().isoformat(),
            "top_k":     TOP_K,
        }
        row_m.update({m: f"{v:.4f}" for m, v in resumo.items()})
        salvar_linha(args.metricas, row_m, FIELDNAMES_METRICAS, escrever_header_m)
        print(f"  Métricas salvas em: {args.metricas}\n")


if __name__ == "__main__":
    main()
