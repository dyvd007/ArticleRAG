"""
Gera o dataset de avaliação (perguntas + gabaritos) com perguntas fixas
criadas pela LLM a partir da leitura direta dos artigos (independente do RAG).

  python gerar_dataset.py          # gera dataset.csv (erro se já existir)
  python gerar_dataset.py --reset  # apaga e recria

As perguntas são fixas e independem de chunking ou qualquer parâmetro do RAG,
garantindo que as métricas de avaliação sejam comparáveis entre experimentos.
"""

import csv
import sys
import argparse
from pathlib import Path

DATASET_CSV = "dataset.csv"
FIELDNAMES  = ["artigo", "secao_artigo", "tipo_pergunta", "pergunta", "resposta_gabarito"]

# 5 perguntas por artigo criadas pela LLM a partir dos PDFs completos.
DATASET_FIXO = [

    # ── tecnica difração cônica.pdf ──────────────────────────────────────────
    {
        "artigo": "tecnica difração cônica.pdf",
        "secao_artigo": "[Seção 2 - Material and methods / Equação 1]",
        "tipo_pergunta": "metodologia",
        "pergunta": (
            "Qual é a equação que relaciona o parâmetro β com as propriedades ópticas e "
            "térmicas do solvente na técnica de difração cônica e o que representa cada termo?"
        ),
        "resposta_gabarito": (
            "O parâmetro β é dado por β = φLeff/(2πKλ) × (dn/dT), onde φ é a eficiência "
            "quântica não radiativa (fração da energia absorvida convertida em calor), "
            "Leff = (1−exp(−αL))/α é o comprimento efetivo, K é a condutividade térmica do "
            "solvente, λ é o comprimento de onda de excitação e dn/dT é o coeficiente "
            "termo-óptico. O número de anéis N segue a relação N = βPe, sendo Pe a potência "
            "do laser."
        ),
    },
    {
        "artigo": "tecnica difração cônica.pdf",
        "secao_artigo": "[Seção 3 - Results and discussion / Tabela 3]",
        "tipo_pergunta": "valor_numerico",
        "pergunta": (
            "De acordo com a Tabela 3 do artigo quais são os valores do rendimento quântico η "
            "e do parâmetro ηa obtidos para o derivado de vitamina B6 em DMSO pela técnica NCD?"
        ),
        "resposta_gabarito": (
            "Para o derivado de vitamina B6 em DMSO os valores obtidos foram η = 0,75 ± 0,02 "
            "e ηa = 0,71."
        ),
    },
    {
        "artigo": "tecnica difração cônica.pdf",
        "secao_artigo": "[Seção 2 - Material and methods / configuração da cubeta]",
        "tipo_pergunta": "causa_efeito",
        "pergunta": (
            "Por que a cubeta deve ser posicionada horizontalmente no setup experimental da "
            "técnica de difração cônica e qual é o efeito observado quando posicionada "
            "verticalmente?"
        ),
        "resposta_gabarito": (
            "A cubeta deve ser posicionada horizontalmente para evitar efeitos de convecção "
            "térmica dentro da amostra. Na posição vertical o fluxo de convecção distorce o "
            "padrão de frente de onda transmitida resultando em anéis não concêntricos no "
            "campo distante. Na posição horizontal a convecção é negligenciável e os anéis "
            "observados são bem concêntricos."
        ),
    },
    {
        "artigo": "tecnica difração cônica.pdf",
        "secao_artigo": "[Seção 3 - Results and discussion / Tabela 2]",
        "tipo_pergunta": "valor_numerico",
        "pergunta": (
            "De acordo com a Tabela 2 do artigo qual é o valor da condutividade térmica K "
            "obtido para a água usando tinta azul-preta como amostra não fluorescente e qual "
            "é o erro percentual?"
        ),
        "resposta_gabarito": (
            "O valor da condutividade térmica K obtido para a água é de 0,589 W/mK com "
            "erro de 2%."
        ),
    },
    {
        "artigo": "tecnica difração cônica.pdf",
        "secao_artigo": "[Seção 4 - Conclusions / técnica NCD]",
        "tipo_pergunta": "metodologia",
        "pergunta": (
            "O que é a técnica NCD (Normalized Conical Diffraction) e qual é sua principal "
            "vantagem em relação à técnica CD convencional?"
        ),
        "resposta_gabarito": (
            "A técnica NCD utiliza uma amostra de referência não fluorescente (φ=1) diluída "
            "no mesmo solvente da amostra de interesse para normalizar as medidas por meio da "
            "Equação 3. Sua principal vantagem é que não é necessário conhecer previamente o "
            "coeficiente termo-óptico (dn/dT) nem a condutividade térmica (K) do solvente "
            "para determinar o rendimento quântico η."
        ),
    },

    # ── PCA.pdf ──────────────────────────────────────────────────────────────
    {
        "artigo": "PCA.pdf",
        "secao_artigo": "[Seção 2.2 - PCA model construction]",
        "tipo_pergunta": "metodologia",
        "pergunta": (
            "Qual software e pacotes foram utilizados para construir o modelo PCA neste "
            "estudo e qual foi o primeiro passo do processo de construção do modelo?"
        ),
        "resposta_gabarito": (
            "O modelo PCA foi construído usando o software R no ambiente de programação "
            "Rstudio com os pacotes FactoMineR e Factoextra para obter a matriz de "
            "covariância e a distribuição de variância. O primeiro passo foi a padronização "
            "dos dados subtraindo a média aritmética e dividindo pelo desvio padrão de cada "
            "variável garantindo que todas as variáveis contribuíssem igualmente para a análise."
        ),
    },
    {
        "artigo": "PCA.pdf",
        "secao_artigo": "[Seção 3 - Results / dois primeiros PCs]",
        "tipo_pergunta": "valor_numerico",
        "pergunta": (
            "Qual é o percentual de variância explicado pelos dois primeiros componentes "
            "principais (PC1 e PC2) no modelo PCA dos derivados de imidazo[4,5-b]piridina?"
        ),
        "resposta_gabarito": (
            "Os dois primeiros componentes principais explicam coletivamente 93% da variância "
            "total com o PC1 respondendo por 80,2% e o PC2 por 13,6% da variância."
        ),
    },
    {
        "artigo": "PCA.pdf",
        "secao_artigo": "[Seção 2.2 - PCA model construction / padronização]",
        "tipo_pergunta": "causa_efeito",
        "pergunta": (
            "Por que a padronização dos dados com média zero e desvio padrão unitário foi "
            "necessária antes de aplicar a PCA neste estudo?"
        ),
        "resposta_gabarito": (
            "A padronização foi necessária porque as variáveis do conjunto de dados possuem "
            "diferentes unidades e escalas. Sem ela a PCA seria dominada pelas variáveis com "
            "maiores magnitudes numéricas prejudicando a identificação correta dos componentes "
            "principais. A padronização garante que todas as variáveis contribuam igualmente "
            "para a análise."
        ),
    },
    {
        "artigo": "PCA.pdf",
        "secao_artigo": "[Seção 2.2 - PCA model construction / Score e Loading plots]",
        "tipo_pergunta": "definicao",
        "pergunta": (
            "O que são o Score plot e o Loading plot na análise PCA e qual informação "
            "cada um fornece?"
        ),
        "resposta_gabarito": (
            "O Score plot mostra a distribuição espacial das amostras (moléculas) no espaço "
            "dos componentes principais indicando distâncias relativas e agrupamentos entre "
            "grupos como A-Imp-A e A-Imp-D. O Loading plot (gráfico de cos²θ) mostra a "
            "correlação entre as variáveis descritoras em termos da relação angular entre "
            "elas no espaço dos componentes permitindo identificar quais variáveis mais "
            "contribuem para cada componente principal."
        ),
    },
    {
        "artigo": "PCA.pdf",
        "secao_artigo": "[Seção 3 - Results / regressão linear PCA e grupos A-Imp-A vs A-Imp-D]",
        "tipo_pergunta": "causa_efeito",
        "pergunta": (
            "De acordo com o modelo de regressão linear PCA qual grupo de compostos "
            "(A-Imp-A ou A-Imp-D) apresenta maior influência do PC1 nos valores de seção "
            "transversal de 2PA e qual é a implicação desse resultado?"
        ),
        "resposta_gabarito": (
            "As moléculas do grupo A-Imp-D apresentam maior influência do PC1 nos valores "
            "de σ2PA com correlação positiva mais forte. Isso indica que variáveis altamente "
            "correlacionadas com o PC1 como o momento de dipolo de transição (μ₀₁) a "
            "mudança de dipolo permanente (Δμ₀₁) e o deslocamento de Stokes normalizado "
            "pelo fator de Onsager (Δν/ΔF) têm maior impacto no σ2PA desse grupo. Para as "
            "moléculas A-Imp-A o PC2 tem maior influência sendo dominado por μ₀₂ e pelo "
            "rendimento quântico de fluorescência (φf)."
        ),
    },

    # ── 2020_Two-Photon Emissive Dyes Based on Push−Pull Purines Derivatives.pdf ──
    {
        "artigo": "2020_Two-Photon Emissive Dyes Based on Push−Pull Purines Derivatives.pdf",
        "secao_artigo": "[Seção II.2 - Linear and Nonlinear Optical Measurements / solventes e Z-scan]",
        "tipo_pergunta": "metodologia",
        "pergunta": (
            "Quais solventes foram utilizados para dissolver os compostos estudados e qual "
            "técnica foi usada para medir a seção transversal de absorção de dois fótons (σ2PA)?"
        ),
        "resposta_gabarito": (
            "Os compostos 3a a 3g foram dissolvidos em diclorometano (DCM) enquanto os "
            "compostos 7a e 7b foram dissolvidos em dimetilsulfóxido (DMSO). A seção "
            "transversal de absorção de dois fótons foi medida usando a técnica Z-scan de "
            "abertura aberta (open-aperture Z-scan) com um laser de femtossegundos sintonizável."
        ),
    },
    {
        "artigo": "2020_Two-Photon Emissive Dyes Based on Push−Pull Purines Derivatives.pdf",
        "secao_artigo": "[Seção II.2 - Linear Optical Measurements / Tabela 1 / φf]",
        "tipo_pergunta": "valor_numerico",
        "pergunta": (
            "Quais são os valores do rendimento quântico de fluorescência (φf) dos compostos "
            "3b e 7b e o que esses valores indicam sobre o caráter emissivo desses compostos?"
        ),
        "resposta_gabarito": (
            "O composto 3b apresenta φf = 81% e o composto 7b apresenta φf = 99%. Esses "
            "altos valores indicam que ambos são altamente emissivos com decaimento "
            "predominantemente radiativo (kr >> knr) sendo candidatos promissores como "
            "biossondas fotoluminescentes ativadas por absorção de dois fótons."
        ),
    },
    {
        "artigo": "2020_Two-Photon Emissive Dyes Based on Push−Pull Purines Derivatives.pdf",
        "secao_artigo": "[Seção III - Conclusions / comparação com 2AP e janela terapêutica]",
        "tipo_pergunta": "causa_efeito",
        "pergunta": (
            "Por que os espectros de 2PA das purinas push-pull estudadas apresentam redshift "
            "em relação à 2-aminopurina (2AP) e qual é a relevância desse deslocamento para "
            "aplicações biomédicas?"
        ),
        "resposta_gabarito": (
            "O redshift ocorre porque a adição de grupos push-pull e do linker estirila à "
            "estrutura da purina aumenta o comprimento de conjugação π e o caráter dipolar "
            "das moléculas reduzindo a energia dos estados excitados. Enquanto as bandas de "
            "2PA da 2AP estão centradas entre 600 e 650 nm as das purinas derivadas estão "
            "em 700–800 nm (1,77–1,55 eV) coincidindo com a janela terapêutica de penetração "
            "óptica em tecidos biológicos o que as torna mais adequadas para bioimagem por "
            "microscopia de dois fótons."
        ),
    },
    {
        "artigo": "2020_Two-Photon Emissive Dyes Based on Push−Pull Purines Derivatives.pdf",
        "secao_artigo": "[Seção II.2 - Brightness / Fig. 4]",
        "tipo_pergunta": "definicao",
        "pergunta": (
            "O que é o parâmetro brilho (brightness) utilizado no artigo como ele foi "
            "calculado e qual composto apresentou o maior valor de brilho a 560 nm?"
        ),
        "resposta_gabarito": (
            "O brilho é definido como o produto do rendimento quântico de fluorescência (φf) "
            "pela seção transversal de absorção de dois fótons (σ2PA) em um comprimento de "
            "onda específico: brilho = φf × σ2PA(λ). No estudo o brilho foi avaliado em "
            "560 nm e 780 nm. O composto 3c apresentou o maior valor de brilho a 560 nm "
            "com 92 GM."
        ),
    },
    {
        "artigo": "2020_Two-Photon Emissive Dyes Based on Push−Pull Purines Derivatives.pdf",
        "secao_artigo": "[Seção II.2 - Linear Optical Measurements / Lippert-Mataga e raio cúbico]",
        "tipo_pergunta": "metodologia",
        "pergunta": (
            "Como o momento de dipolo elétrico permanente (Δμ₀₁) foi estimado nos compostos "
            "estudados e como o raio cúbico molecular (a³) foi determinado?"
        ),
        "resposta_gabarito": (
            "O momento de dipolo permanente Δμ₀₁ foi estimado aplicando a equação de "
            "Lippert-Mataga às medidas de solvatocromismo em seis solventes (acetona etanol "
            "metanol DCM DMSO e tolueno). O raio cúbico molecular a³ foi determinado "
            "assumindo que cada estrutura molecular ocupa um volume esférico (Vol = 4πa³/3) "
            "e aplicando a equação de difusão de Smoluchowski-Einstein; adicionalmente foi "
            "calculado usando o pacote Gaussian 16."
        ),
    },
]


def main():
    parser = argparse.ArgumentParser(description="Gera dataset fixo de avaliação")
    parser.add_argument("--output", type=str, default=DATASET_CSV, help="Arquivo CSV de saída")
    parser.add_argument("--reset",  action="store_true", help="Apaga o CSV existente e recria")
    args = parser.parse_args()

    output_path = Path(args.output)

    if output_path.exists():
        if args.reset:
            output_path.unlink()
            print(f"  Reset: '{args.output}' removido.")
        else:
            print(f"  '{args.output}' já existe. Use --reset para recriar.")
            sys.exit(0)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(DATASET_FIXO)

    artigos = {row["artigo"] for row in DATASET_FIXO}
    print(f"\n  Dataset salvo em: {args.output}")
    print(f"  Total de perguntas: {len(DATASET_FIXO)} ({len(artigos)} artigos)")
    for art in sorted(artigos):
        n = sum(1 for r in DATASET_FIXO if r["artigo"] == art)
        print(f"    {n}x  {art}")
    print()


if __name__ == "__main__":
    main()
