import os
import unicodedata

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOLS
from langchain.agents import create_agent

MODEL_NAME = "claude-opus-5"

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        default_headers = {}
        workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if workspace_id:
            default_headers["anthropic-workspace-id"] = workspace_id
        model = ChatAnthropic(model=MODEL_NAME, default_headers=default_headers or None)
        _agent = create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    return _agent


def ask(question, history=None):
    """Pose une question a l'agent.

    `history` est une liste de tours precedents, chacun un dict
    {"question": str, "answer": str} (le detail code/resultat n'est pas
    renvoye au modele, seulement le fil de discussion en langage naturel).

    L'agent peut appeler l'outil plusieurs fois dans un meme tour (ex. calcul
    d'un tableau chiffre puis generation d'un graphique) : `steps` remonte
    chaque appel dans l'ordre, pas seulement le dernier.

    Retourne un dict {"answer": str, "steps": [{"code": str, "artifact": dict}, ...],
    "refused": bool}.
    """
    agent = get_agent()

    messages = []
    for turn in history or []:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))
    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})
    result_messages = result["messages"]

    answer = ""
    for msg in reversed(result_messages):
        if isinstance(msg, AIMessage) and msg.text:
            answer = str(msg.text)
            break

    steps = []
    for msg in result_messages:
        if isinstance(msg, ToolMessage) and msg.name == "execute_python_code" and msg.artifact:
            steps.append({"code": msg.artifact.get("code"), "artifact": msg.artifact})

    refused = not steps and _contains_refusal_phrase(answer)

    return {"answer": answer, "steps": steps, "refused": refused}


def _strip_accents(text):
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _contains_refusal_phrase(answer):
    return "impossible avec les donnees disponibles" in _strip_accents(answer).lower()
