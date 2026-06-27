"""
Executa gerar_respostas_rag.py e avaliar.py para múltiplos valores de TOP_K.

  python run_experimentos.py
  python run_experimentos.py --top-k 1 3 5 10   # valores customizados
  python run_experimentos.py --skip-existing      # pula TOP_K já presentes em metricas.csv
"""

import os
import sys
import csv
import subprocess
import argparse
from pathlib import Path

TOP_K_PADRAO = [1, 3, 5, 10, 20, 25, 30, 35, 40]
METRICAS_CSV = "metricas.csv"
DLINHA = "═" * 62


def topk_ja_avaliados() -> set[int]:
    p = Path(METRICAS_CSV)
    if not p.exists():
        return set()
    with open(p, newline="", encoding="utf-8") as f:
        return {int(r["top_k"]) for r in csv.DictReader(f) if r.get("top_k")}


def rodar(cmd: list[str], env: dict) -> int:
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Experimentos RAG por TOP_K")
    parser.add_argument("--top-k", nargs="+", type=int, default=TOP_K_PADRAO,
                        help="Valores de TOP_K a testar")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Pula valores de TOP_K já presentes em metricas.csv")
    args = parser.parse_args()

    valores = sorted(set(args.top_k))

    if args.skip_existing:
        ja_feitos = topk_ja_avaliados()
        pendentes = [k for k in valores if k not in ja_feitos]
        pulados = [k for k in valores if k in ja_feitos]
        if pulados:
            print(f"\n  Pulando TOP_K já avaliados: {pulados}")
    else:
        pendentes = valores

    if not pendentes:
        print("\n  Nada a fazer — todos os TOP_K já foram avaliados.\n")
        sys.exit(0)

    print(f"\n{DLINHA}")
    print("  EXPERIMENTOS RAG — TOP_K SWEEP")
    print(DLINHA)
    print(f"  Valores a testar : {pendentes}")
    print(f"  Métricas em      : {METRICAS_CSV}")
    print(DLINHA)

    env_base = {**os.environ}
    python = sys.executable
    erros = []

    for k in pendentes:
        print(f"\n{'═'*62}")
        print(f"  TOP_K = {k}")
        print(f"{'═'*62}")

        env = {**env_base, "TOP_K": str(k)}

        # Gera respostas RAG (sobrescreve respostas_rag.csv a cada rodada)
        rc = rodar(
            [python, "gerar_respostas_rag.py"],
            env,
        )
        if rc != 0:
            print(f"\n  ✗ gerar_respostas_rag.py falhou para TOP_K={k} (código {rc})")
            erros.append(f"TOP_K={k}: gerar_respostas_rag.py falhou")
            continue

        # Avalia com LLM-as-Judge (sobrescreve avaliacoes.csv, acumula em metricas.csv)
        rc = rodar(
            [python, "avaliar.py", "--metricas", METRICAS_CSV],
            env,
        )
        if rc != 0:
            print(f"\n  ✗ avaliar.py falhou para TOP_K={k} (código {rc})")
            erros.append(f"TOP_K={k}: avaliar.py falhou")

    print(f"\n{DLINHA}")
    print("  RESUMO FINAL")
    print(DLINHA)

    if erros:
        print(f"\n  ✗ {len(erros)} erro(s):")
        for e in erros:
            print(f"    • {e}")
    else:
        print(f"\n  ✓ Todos os {len(pendentes)} experimentos concluídos com sucesso.")

    print(f"  Métricas acumuladas em: {METRICAS_CSV}\n")


if __name__ == "__main__":
    main()
