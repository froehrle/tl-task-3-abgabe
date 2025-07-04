import base64
import json

from enum import Enum
import os
import asyncio

from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector, SearchType

from agno.tools.calculator import CalculatorTools
from pydantic import BaseModel

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools.reasoning import ReasoningTools

from agno.workflow import Workflow
from typing import Iterator
from agno.agent import RunResponse

import logging

class QuestionTypeEnum(Enum):
    COMPREHENSION_QUESTION="Verständnisfragen"
    CALCULATION_QUESTION="Rechenfragen"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

model=OpenAIChat(
    id="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"]
)

class QuestionTypeEnum(Enum):
    COMPREHENSION_QUESTION="Verständnisfragen"
    CALCULATION_QUESTION="Rechenfragen"

class ValidationResult(BaseModel):
    is_correct: bool
    reason: str

pdf_knowledge_base = PDFKnowledgeBase(
    path="knowledge-documents",
    # Table name: ai.pdf_documents
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url="postgresql+psycopg2://postgres:changeit@db.jlqpiruljvmkvumnbtqd.supabase.co:5432/postgres",
        search_type=SearchType.hybrid,
        # TODO: Parameter wie Embedder, Reranker etc. wichtig?
    ),
    reader=PDFReader(chunk=True),
)

# TODO: Mit Team statt Workflow?
# TODO: Ein Agent soll alle Kapitel extrahieren aus den Skipten extrahieren?
# TODO: Parameter sollen dem Template dynamisch übergeben werden können: Schwierigkeisgrad, Anzahl der Fragen, Thema, Typ etc.
# TODO: Verschiedene Setups bei Agno/CrewAI vergleichen
# TODO: feedback_tool für Question Validator?
# TODO: Fragen beziehen sich zum Teil nicht auf Knowledge Base -> Tool verwenden?
# TODO(Alex): Was zeichnet eine gute Frage aus?
# TODO: Beide Agenten sollen mit verschiedenen Modellen ausgestattet werden.
# TODO: Kriterien bzgl. inhaltlicher Korrektheit von Antworten festlegen (Ausführlich genug etc.)? -> Schwellenwert z. B. hinsichtlich Konfidenzwert festlegen? 
# TODO: PythonTools zwecks Auswertung von Rechenaufgaben?

calculator_tools = CalculatorTools(
  add=True,
  subtract=True,
  multiply=True,
  divide=True,
  exponentiate=True,
  factorial=True,
  is_prime=True,
  square_root=True,
)

###########################

answer_validator1 = Agent(
    name="Answer Evaluator 1",
    role="First Answer Evaluator",
    description=(
      "Wertet Antworten auf Fragen hinsichtlich ihrer inhaltlichen Korrektheit aus."
    ),
    instructions="""
        Basierend auf deinem Wissen sollst du Studentenantworten auf Fragen hinsichtlich ihrer inhaltlichen Korrektheit bewerten.
        Vergleiche dabei die Studentenantwort inhaltlich und sinngemäß mit der übergebenen richtigen Antwort. Ignoriere hierbei nicht relevante Details wie beispielsweise Groß- und Kleinschreibung.
        Wenn die Antwort nicht richtig ist, gibt eine entsprechende aussagekräftige Begründung ab.
        Greife auf dein Wissen zurück, bevor du eine Bewertung der Studentenantwort vornimmst.
    """,
    model=model,
    tools=[ReasoningTools(add_instructions=True)],
    expected_output="Feedback zur inhaltlichen Korrektheit der Antwort, inklusive Konfidenzwert.",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

answer_validator2 = Agent(
    name="Answer Evaluator 2",
    role="Second Answer Evaluator",
    description=(
        "Wertet Antworten auf Fragen hinsichtlich ihrer inhaltlichen Korrektheit aus."
    ),
    instructions="""
        Basierend auf deinem Wissen sollst du Studentenantworten auf Fragen hinsichtlich ihrer inhaltlichen Korrektheit bewerten.
        Vergleiche dabei die Studentenantwort inhaltlich und sinngemäß mit der übergebenen richtigen Antwort.
        Wenn die Antwort nicht richtig ist, gibt eine entsprechende aussagekräftige Begründung ab.
        Greife auf dein Wissen zurück, bevor du eine Bewertung der Studentenantwort vornimmst.
    """,
    model=model,
    tools=[ReasoningTools(add_instructions=True)],
    expected_output="Feedback zur inhaltlichen Korrektheit der Antwort, inklusive Konfidenzwert.",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

master_answer_validator = Agent(
    name="Final Answer Evaluator",
    role="Final Answer Evaluator",
    description=(
        "Wertet bei Dissens der anderen Agenten als 'letzte Instanz' die Studentenantwort auf eine Frage hinsichtlich ihrer inhaltlichen Korrektheit aus."
    ),
    instructions="""
        Basierend auf deinem Wissen sollst du die Antworten auf Fragen hinsichtlich ihrer inhaltlichen Korrektheit bewerten.
        Vergleiche dabei die Studentenantwort inhaltlich und sinngemäß mit der übergebenen richtigen Antwort. Ignoriere hierbei nicht relevante Details wie beispielsweise Groß- und Kleinschreibung.
        Wenn die Antwort nicht richtig ist, gibt eine entsprechende aussagekräftige Begründung ab.
        Greife auf dein Wissen zurück, bevor du eine Bewertung der Studentenantwort vornimmst.
    """,
    model=model,
    tools=[ReasoningTools(add_instructions=True), calculator_tools],
    expected_output="Feedback zur inhaltlichen Korrektheit der Antwort, inklusive Konfidenzwert.",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    response_model=ValidationResult,
)

agent_team = Team(
    name="Evaluation Team",
    description="Ein Team, das Antworten auf Fragen hinsichtlich ihrer inhaltlichen Korrektheit auswertet.",
    mode="collaborate",
    model=model,
    members=[answer_validator1, answer_validator2],
    tools=[ReasoningTools(add_instructions=True)],
    instructions = [
        "Du bist Diskussionsleiter.",
        "Du musst die Diskussion beenden, wenn du der Meinung bist, dass das Team einen Konsens erreicht hat.",
        "Wenn kein Konsens erreicht wurde, antworte mit 'Consensus not reached'.",
        "Entscheide, ob die Frage ausreichend beantwortet wurde und gib den entsprechenden Wahrheitswert zurück."
    ],
    success_criteria="Das Team hat einen Konses erreicht.",
    enable_agentic_context=True,
    show_tool_calls=True,
    markdown=True,
    show_members_responses=True,
    response_model=ValidationResult,
)

###########################

prompt_template="""
    Werte die Antwort des Studenten aus dem gegebenen Frage-Antwort-Paar hinsichtlich ihrer inhaltlichen Korrektheit aus, indem du die Stundentenantwort mit der übergebenen richtigen Antwort inhaltlich und sinngemäß vergleichst.

    Frage:
    {question}

    Studentenantwort:
    {student_answer}

    Richtige Antwort:
    {correct_answer}
"""

def lambda_handler(event, context):
  print('Loading function')

  raw_body = event["body"]
  isBase64Encoded = event.get('isBase64Encoded', False)
  body_decoded = base64.b64decode(raw_body) if isBase64Encoded else raw_body

  try:
    body = json.loads(body_decoded)

    question_text = body.get("question_text", "")
    student_answer = body.get("student_answer", "")
    correct_answer = body.get("correct_answer")
    question_style = body.get("question_style", "")

    prompt = prompt_template.format(question=question_text, student_answer=student_answer, correct_answer=correct_answer)
    result = validate(message=prompt, question_type=question_style)

    return {
      'statusCode': 200,
      'headers': {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST,OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
      },
      'body': result.json()
  }
  except Exception as e:
    logger.error("Fehler beim Verarbeiten des Uploads.", exc_info=True)
    return {
      'statusCode': 500,
      'body': json.dumps(f'Fehler: {str(e)}')
    }

def validate(message, question_type: QuestionTypeEnum) -> Iterator[RunResponse]:
    validation_result = None

    if question_type == QuestionTypeEnum.COMPREHENSION_QUESTION:
        validation_result = agent_team.run(
            message=message,
            stream_intermediate_steps=True,
            show_full_reasoning=True
        ).content

        if "Consensus not reached" not in validation_result:
            print("\nConsensus reached.")
    else:
        print("Utilizing final answer validator...")

        validation_result = master_answer_validator.run(
            message=message,
            stream_intermediate_steps=True,
            show_full_reasoning=True
        ).content

    return validation_result