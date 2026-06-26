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

PROMPT_AVALIAR_TUDO = """\
Avalie as 4 métricas abaixo para a resposta gerada pelo RAG.

Pergunta: {pergunta}
Resposta correta (gabarito): {gabarito}

Contexto recuperado:
{contexto}

Resposta gerada:
{resposta}

Métricas (score de 0.0 a 1.0 cada):
1. faithfulness: a resposta só usa informações presentes no contexto? (1.0=totalmente fiel, 0.0=inventa informações)
2. answer_relevancy: a resposta responde à pergunta? (1.0=completa, 0.0=não responde)
3. context_precision: os trechos recuperados são relevantes para a pergunta? (1.0=todos relevantes, 0.0=nenhum relevante)
4. context_recall: o contexto contém a informação necessária para chegar ao gabarito? (1.0=contém tudo, 0.0=não contém nada)

Responda SOMENTE com JSON válido, sem texto fora do JSON:
{{
  "faithfulness":      {{"score": <float 0-1>, "reasoning": "<uma frase curta>"}},
  "answer_relevancy":  {{"score": <float 0-1>, "reasoning": "<uma frase curta>"}},
  "context_precision": {{"score": <float 0-1>, "reasoning": "<uma frase curta>"}},
  "context_recall":    {{"score": <float 0-1>, "reasoning": "<uma frase curta>"}}
}}"""


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
