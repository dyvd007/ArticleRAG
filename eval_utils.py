"""Utilitários compartilhados entre os scripts de avaliação do RAG."""

import os
import time
import json

from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel

from rag import SYSTEM_PROMPT

load_dotenv()

GCP_PROJECT  = os.environ.get("GCP_PROJECT", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
API_DELAY    = 10

vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

METRICAS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

PROMPT_FAITHFULNESS = """\
Você é um avaliador rigoroso de sistemas RAG.

## Tarefa
Verifique se cada afirmação da resposta gerada tem suporte EXPLÍCITO no contexto recuperado.
Não avalie se a resposta é útil, correta em termos gerais ou bem escrita — apenas se cada claim pode ser derivado do contexto fornecido.

## Inputs
Pergunta: {pergunta}

Contexto recuperado:
{contexto}

Resposta gerada:
{resposta}

## Definições
- **Suportado**: o contexto afirma explicitamente, ou permite inferir diretamente sem conhecimento externo.
- **Não suportado**: o contexto é omisso, ambíguo ou contradiz a afirmação. Conhecimento geral não conta como suporte.

## Passo a passo
1. Extraia cada afirmação factual da resposta como claims independentes e atômicos.
   Ignore saudações, opiniões sem fato e reformulações da própria pergunta.
2. Para cada claim, localize o trecho do contexto que o sustenta (ou conclua pela ausência).
3. Marque `suportado: true` apenas se o suporte for direto e sem ambiguidade.
4. Calcule: score = claims_suportados / total_claims

Responda SOMENTE com JSON válido:
{{
  "claims": [
    {{"claim": "<afirmação>", "suportado": true, "trecho_contexto": "<cite o trecho ou null>"}},
    ...
  ],
  "claims_suportados": <int>,
  "claims_totais": <int>,
  "score": <float 0-1>,
  "reasoning": "<uma frase explicando o score>"
}}"""

PROMPT_ANSWER_RELEVANCY = """\
Você é um avaliador rigoroso de sistemas RAG.

## Tarefa
Avalie se a resposta gerada responde à pergunta de forma relevante e completa.
NÃO avalie a veracidade da resposta — apenas se ela endereça o que foi perguntado.

## Inputs
Pergunta: {pergunta}

Resposta gerada:
{resposta}

## Instruções
Avalie:
1. A resposta é sobre o mesmo tema da pergunta? (ou desvia para outro assunto?)
2. A resposta aborda diretamente o que foi perguntado?
3. A resposta contém informação suficiente para satisfazer a pergunta?

Penalize respostas que:
- Ignoram a pergunta ou respondem algo diferente
- São excessivamente vagas ou evasivas sem chegar ao ponto
- Contêm repetição ou enrolação em vez de responder ao núcleo da pergunta

Responda SOMENTE com JSON válido:
{{
  "score": <float 0-1>,
  "reasoning": "<uma frase explicando o score>"
}}"""

PROMPT_CONTEXT_PRECISION = """\
Você é um avaliador rigoroso de sistemas RAG.

## Tarefa
Avalie se os trechos recuperados são relevantes para responder à pergunta.
NÃO considere a resposta gerada — avalie apenas a relação entre pergunta, gabarito e contexto.

## Inputs
Pergunta: {pergunta}

Gabarito (resposta correta de referência):
{gabarito}

Contexto recuperado (chunks numerados):
{contexto_numerado}

## Instruções
Para cada chunk, classifique como:
  - relevante: contém informação que contribui para produzir a resposta correta descrita no gabarito
  - irrelevante: não tem relação com a pergunta nem com as informações necessárias para o gabarito
Score = chunks_relevantes / total_chunks

Responda SOMENTE com JSON válido:
{{
  "chunks": [
    {{"chunk_id": 1, "relevante": true, "motivo": "<uma frase>"}},
    ...
  ],
  "chunks_relevantes": <int>,
  "total_chunks": <int>,
  "score": <float 0-1>,
  "reasoning": "<uma frase explicando o score>"
}}"""

PROMPT_CONTEXT_RECALL = """\
Você é um avaliador rigoroso de sistemas RAG.

## Tarefa
Avalie se o contexto recuperado contém as informações necessárias para responder à pergunta conforme o gabarito.
NÃO considere a resposta gerada — avalie apenas a relação entre contexto e gabarito.

## Inputs
Pergunta: {pergunta}

Gabarito (resposta correta de referência):
{gabarito}

Contexto recuperado:
{contexto}

## Instruções
Passo 1 — Liste as afirmações-chave do gabarito (fatos, números, nomes, relações) necessárias para responder à pergunta.
Passo 2 — Para cada afirmação, verifique se o contexto contém suporte EXPLÍCITO ou IMPLÍCITO claro.
          - "presente" = o contexto menciona ou deixa claro esse fato
          - "ausente"  = o contexto não menciona; inferência externa não conta
Passo 3 — Calcule: score = afirmações_presentes / total_afirmações

Responda SOMENTE com JSON válido:
{{
  "afirmacoes": [
    {{"afirmacao": "<fato do gabarito>", "presente": true, "trecho_contexto": "<cite o trecho ou null>"}},
    ...
  ],
  "afirmacoes_presentes": <int>,
  "afirmacoes_totais": <int>,
  "score": <float 0-1>,
  "reasoning": "<uma frase explicando o score>"
}}"""

PROMPTS_POR_METRICA = {
    "faithfulness":      PROMPT_FAITHFULNESS,
    "answer_relevancy":  PROMPT_ANSWER_RELEVANCY,
    "context_precision": PROMPT_CONTEXT_PRECISION,
    "context_recall":    PROMPT_CONTEXT_RECALL,
}


def get_gemini_judge():
    return GenerativeModel(model_name="gemini-2.5-flash-lite")


def get_gemini_rag_eval():
    return GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        system_instruction=SYSTEM_PROMPT,
    )


def parse_json_llm(texto: str) -> dict:
    """Extrai JSON de uma resposta do LLM (tolera blocos ```json ```)."""
    texto = texto.strip()
    if "```" in texto:
        partes = texto.split("```")
        bloco = partes[1] if len(partes) > 1 else partes[0]
        if bloco.startswith("json"):
            bloco = bloco[4:]
        texto = bloco
    return json.loads(texto.strip())


def chamar_gemini(model, prompt: str, max_retries: int = 6) -> str:
    """Chama generate_content com retry + backoff exponencial em caso de quota (429)."""
    for tentativa in range(1, max_retries + 1):
        time.sleep(API_DELAY)
        try:
            return model.generate_content(prompt).text
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
                raise RuntimeError(
                    f"Cota excedida após {max_retries} tentativas. "
                    "Aumente API_DELAY ou aguarde alguns minutos."
                ) from e

            wait = API_DELAY * (2 ** tentativa)
            print(f"\n  ⏳ Cota excedida (tentativa {tentativa}/{max_retries}). "
                  f"Aguardando {wait}s...")
            time.sleep(wait)
