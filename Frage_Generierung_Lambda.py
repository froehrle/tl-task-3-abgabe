"""AWS Lambda Entrypoint – QA Generation Workflow (pgvector‑only)

This single file can be deployed as a Lambda behind an API Gateway. It:
1. Maps the German request payload keys (anzahl_fragen, thema …) to internal
   parameters for the QA workflow.
2. Runs the four‑agent workflow (Retriever → Generator → Reviewer).
3. Returns the generated question list as JSON.

Upload environment variables via Lambda console or IaC:
    • OPENAI_API_KEY
    • VECTOR_DB_URL  (postgresql+psycopg2://…?sslmode=require)

The Lambda expects an API‑Gateway proxy event: `event['body']` contains a JSON
string with the structure you showed:
```json
{
  "anzahl_fragen": 2,
  "fragetype": "Verständnisfragen",   // or "Rechenfragen"
  "keywords": "rechnen",
  "schwierigkeitsgrad": "mittel",
  "thema": "Statistik",
  "zielgruppe": "bachelor"
}
```

and responds with:
```json
{
  "questions": [ … ]
}
```
"""

from __future__ import annotations

import json
import base64
import logging
import os
from enum import Enum
from textwrap import dedent
from typing import List, Optional

from pydantic import BaseModel

from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.tools.knowledge import KnowledgeTools
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.pgvector import PgVector, SearchType
from agno.workflow import Workflow

# --------------------------------------------------------------------------- #
# Env & logging
# --------------------------------------------------------------------------- #

VECTOR_DB_URL = os.environ["VECTOR_DB_URL"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info("Vector DB → %s", VECTOR_DB_URL)

# --------------------------------------------------------------------------- #
# Pydantic models for questions
# --------------------------------------------------------------------------- #

class QuestionType(str, Enum):
    TEXT = "text"
    MULTIPLE_CHOICE = "multiple_choice"


class Question(BaseModel):
    question_text: str
    question_type: QuestionType
    multiple_choice_options: Optional[List[str]] = []
    correct_answer: str


class QuestionResponse(BaseModel):
    questions: List[Question]

# --------------------------------------------------------------------------- #
# Knowledge base (pgvector only)
# --------------------------------------------------------------------------- #

vector_only_knowledge_base = PDFUrlKnowledgeBase(
    urls=[],
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url=VECTOR_DB_URL,
        search_type=SearchType.hybrid,
    ),
)

knowledge_tools = KnowledgeTools(
    knowledge=vector_only_knowledge_base,
    think=True,
    search=True,
    analyze=True,
    add_instructions=True,
)

# --------------------------------------------------------------------------- #
# LLM & agent factory helpers
# --------------------------------------------------------------------------- #

def get_llm(model_name: str = "gpt-4o-mini") -> OpenAIChat:
    return OpenAIChat(id=model_name, api_key=os.environ["OPENAI_API_KEY"])

LLM = get_llm()


def build_agent(*, name: str, role: str, description: str, instructions: str, response_model: type | None = None, uses_knowledge: bool = False) -> Agent:
    return Agent(
        name=name,
        role=role,
        description=description,
        instructions=dedent(instructions),
        model=LLM,
        tools=[ReasoningTools(add_instructions=True), knowledge_tools] if uses_knowledge else [ReasoningTools(add_instructions=True)],
        knowledge=vector_only_knowledge_base if uses_knowledge else None,
        search_knowledge=uses_knowledge,
        debug_mode=False,  # switch to True for verbose logs
        response_model=response_model,
    )

# --------------------------------------------------------------------------- #
# Agents definitions
# --------------------------------------------------------------------------- #

RETRIEVER_INSTRUCTIONS = dedent("""
Ich greife direkt auf die pgvector-Datenbank zu, um relevante Chunks zum Thema **{query}** zu finden.
Nutze dazu `search`, `think`, `analyze`. Gib deine Antwort exakt so zurück:

Zusammenfassung:
- <Bullet 1>
- … (max. 10 Bullets)
""")

KNOWLEDGE_RETRIEVER = build_agent(
    name="Knowledge Retriever",
    role="Retriever",
    description="Fragt die Vektor-DB ab und liefert eine knappe Zusammenfassung (max 10 Bullets).",
    instructions=RETRIEVER_INSTRUCTIONS,
    uses_knowledge=True,
)

_BASE_GENERATOR_INSTRUCTIONS = dedent("""
Du erstellst Frage-Antort-Paare für eine Lernplattform. Formuliere sämtliche Fragen, Antworten und Lösungswege etc. ausschließlich auf Deutsch.

Erstelle {num} Aufgaben des Typs **{q_type}** im Schwierigkeitsgrad **{difficulty}** für die Zielgruppe **{audience}**.
Nutze folgenden Kontext:
{context}
""")

COMPREHENSION_QA_GENERATOR = build_agent(
    name="Comprehension QA Generator",
    role="Generator",
    description="Erstellt Verständnis-Fragen & Antworten.",
    instructions=dedent("""\
        Deine Aufgabe ist es, basierend auf deinem Wissen qualitativ hochwertige Verständnisfragen samt zugehöriger Antworten (→ Frage-Antwort-Paare) zu dem angegebenen Thema bzw. Themenbereich zu erstellen.

        Die Fragen sollen:
            - inhaltlich vielfältig sein und unterschiedliche Aspekte des Themas / Themenbereichs abdecken
            - idealerweise verschiedene Fragetechniken abdecken (z. B. Multiple-Choice, Lückentext)
            - die übergebenen Anforderungen bezüglich Schwierigkeisgrad, Anzahl der Fragen, Thema, Fragentyp, Zielgruppe und Keywords berücksichtigen 
                        
        Verwende ausschließlich dein eigenes Wissen zur Erstellung der Fragen.
                        
        Gib nur die generierten Frage-Antwort-Paare zusammen mit dem Fragetyp und gegebenenfalls den Multiple-Choice-Optionen aus – ohne zusätzlichen Text.

        Formuliere die Frage-Antwort-Paare auf Deutsch.
    """),
    response_model=QuestionResponse,
)

CALCULATION_QA_GENERATOR = build_agent(
    name="Calculation QA Generator",
    role="Generator",
    description="Erstellt Rechen-Fragen & Lösungen.",
    instructions=dedent("""\
        Deine Aufgabe ist es, basierend auf deinem Wissen qualitativ hochwertige Rechenfragen samt zugehöriger Lösungen (→ Frage-Antwort-Paare) zu dem angegebenen Thema bzw. Themenbereich zu erstellen.
       
        Die Fragen sollen:
            - inhaltlich vielfältig sein und unterschiedliche Aspekte des Themas / Themenbereichs abdecken
            - die übergebenen Anforderungen bezüglich Schwierigkeisgrad, Anzahl der Fragen, Thema, Fragentyp, Zielgruppe und Keywords berücksichtigen 
                        
        Verwende ausschließlich dein eigenes Wissen zur Erstellung der Fragen.
                        
        Gib nur die generierten Frage-Antwort-Paare zusammen mit dem Fragetyp aus – ohne zusätzlichen Text.

        Formuliere die Frage-Antwort-Paare auf Deutsch.
    """),
    response_model=QuestionResponse,
)

QUESTION_REVIEWER = build_agent(
    name="Question Reviewer",
    role="Reviewer",
    description="Validiert Fragenqualität.",
    instructions=dedent("""\
        Deine Aufgabe ist es, die Qualität von Fragen innerhalb gegebener Frage-Antwort-Paare zu prüfen und zu bewerten.
                        
        Berücksichtige dabei unter anderem die folgenden Qualitätskriterien. Wenn du ein Kriterium nicht eindeutig anwenden kannst, orientiere dich an deinem besten Urteilsvermögen.
            - Klarheit & Eindeutigkeit: Ist die Frage eindeutig, verständlich und präzise formuliert und lässt keine Mehrdeutigkeit zu?
            - Präzision & Umfang: Ist die Frage weder zu allgemein noch zu speziell?
            - Zielorienterung: Ist die Frage klar an einem Lernziel ausgerichtet?
            - Kognitive Anforderung (Taxonomiestufe): Erfordert die Frage ein angemessenes Niveau an Denken gemäß Bloom (z. B. Verstehen, Anwenden, Analysieren)? Ist die Frage anspruchsvoll und erfordert ein solides Verständnis des Themas?
            - Aktivierung / Denkförderung: Fördert die Frage aktives Nachdenken, Transfer oder Problemlösen?         
            - Relevanz & Kontext: Passt die Frage zur Domäne oder zum thematischen Kontext? Stimmt die Frage mit deinem Wissen überein?
            - Antwortbarkeit: Basiert die Frage auf deinem bekannten Wissen bzw. lässt sich die Frage mit deinem bekannten Wissen beantworten?
            - Schwierigkeitsgrad: Ist die Frage vom Niveau / Schwierigkeitsgrad her für die Zielgruppe angemessen/geeignet?
            - Tiefe der Frage: Geht die Frage über reine Faktenabfrage hinaus (z. B. „Warum“ / „Wie“)?
            - Nicht-Redundanz: Vermeidet die Frage Wiederholungen oder Variationen von Inhalten anderer Fragen?
            - Informationsdichte: Liefert die Antwort substanzielle Informationen oder ist sie trivial/flach?
            - Kontextualisierung: Ist die Frage eingebettet in ein sinnvolles Szenario oder Problemfeld?
            - Vernetztes Denken / Zusammenhangserkennung: Zielt die Frage darauf ab, Zusammenhänge zwischen Konzepten / Themen herzustellen / zu erkennen oder Wissen zu vernetzen?      
            - Antortorientierung: Ermöglicht oder provoziert die Frage eine substanzielle und informative Antwort?
            - Führt die Frage zu einer guten Antwort?

        Falls eine Frage die Qualitätskriterien nicht erfüllt, formuliere gezieltes Feedback mit konkreten Verbesserungsvorschlägen.
        Wenn du mit der Qualität der Fragen zufrieden bist und keine Verbesserungsvorschläge hast, gib eindeutig „Review OK“ als Feedback zurück.

        Gib in deiner Rückmeldung ausschließlich Feedback zu den Fragen oder "Review OK" zurück – ohne zusätzlichen Text.
                        
        Greife auf dein eigenes Wissen zur Bewertung zurück.
        
        Formuliere dein Feedback stets auf Deutsch.
    """),
)

# --------------------------------------------------------------------------- #
# Workflow implementation
# --------------------------------------------------------------------------- #

class QAWorkflow(Workflow):
    MAX_ROUNDS = 1

    def __init__(self, question_type: str):
        qt_lower = question_type.lower()
        self.generator = COMPREHENSION_QA_GENERATOR if qt_lower.startswith("ver") else CALCULATION_QA_GENERATOR
        self.q_type_label = "Verständnisfrage" if qt_lower.startswith("ver") else "Rechenfrage"

    def run(self, *, subjects: str, keywords: str, num_qas: int, difficulty: str, audience: str) -> List[dict]:
        search_query = f"{subjects} {keywords}".strip()
        retriever_prompt = KNOWLEDGE_RETRIEVER.instructions.format(query=search_query)
        context = KNOWLEDGE_RETRIEVER.run(retriever_prompt).content

        feedback: str | None = None
        for _ in range(self.MAX_ROUNDS):
            gen_prompt = self.generator.instructions.format(
                num=num_qas,
                q_type=self.q_type_label,
                difficulty=difficulty,
                audience=audience,
                context=context,
            )
            if feedback:
                gen_prompt += f"\nVorheriges Feedback:\n{feedback}\n"

            qa_pairs_resp = self.generator.run(gen_prompt).content
            qa_pairs_json = (
                qa_pairs_resp.model_dump_json()
                if isinstance(qa_pairs_resp, BaseModel)
                else json.dumps(qa_pairs_resp)
            )

            review_prompt = (
                "Bewerte folgende Frage-Antwort-Paare:\n" + qa_pairs_json +
                f"\nSchwierigkeitsgrad: {difficulty}\nZielgruppe: {audience}"
            )
            feedback = QUESTION_REVIEWER.run(review_prompt).content

            if "Review OK" in feedback:
                return json.loads(qa_pairs_json)

        return json.loads(qa_pairs_json)

# --------------------------------------------------------------------------- #
# Request/response mapping helpers
# --------------------------------------------------------------------------- #

def handle_request(payload: dict) -> dict:
    """Translate German payload keys to workflow params and run generation."""

    question_type = payload.get("fragetyp", "Verständnisfragen")
    subjects      = payload.get("thema", "")
    keywords      = payload.get("keywords", "")
    num_qas       = int(payload.get("anzahl_fragen", 3))
    difficulty    = payload.get("schwierigkeitsgrad", "mittel")
    audience      = payload.get("zielgruppe", "")

    wf = QAWorkflow(question_type=question_type)
    questions = wf.run(
        subjects=subjects,
        keywords=keywords,
        num_qas=num_qas,
        difficulty=difficulty,
        audience=audience,
    )
    return {"questions": questions}

# --------------------------------------------------------------------------- #
# AWS Lambda entrypoint
# --------------------------------------------------------------------------- #

def lambda_handler(event, _context):
    """API Gateway proxy-compatible entrypoint."""

    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode()
        payload = json.loads(body)

        result = handle_request(payload)
        return {
            "statusCode": 200,
            'headers': {
                "Content-Type": "application/json",
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            },
            "body": json.dumps(result, ensure_ascii=False),
        }
    except Exception as exc:
        logger.exception("Lambda error")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(exc)}),
        }