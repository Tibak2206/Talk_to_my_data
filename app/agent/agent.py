import os
import unicodedata

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOLS
from langchain.agents import create_agent

MODEL_NAME = "claude-opus-5"
# Effort par defaut d'Opus 5 = "high" (thinking adaptatif profond), pense pour du
# raisonnement complexe. Nos questions sont des lookups pandas simples : "medium"
# reduit fortement le cout (moins de thinking) avec un impact attendu minime sur
# la qualite pour ce type de tache (cf. golden set pour validation).
EFFORT = "medium"
INPUT_COST_PER_MTOK = 5.0
OUTPUT_COST_PER_MTOK = 25.0

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        default_headers = {}
        workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if workspace_id:
            default_headers["anthropic-workspace-id"] = workspace_id
        model = ChatAnthropic(
            model=MODEL_NAME,
            effort=EFFORT,
            default_headers=default_headers or None,
        )
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
    usage = _aggregate_usage(result_messages)

    print(
        f"[usage] input={usage['input_tokens']} output={usage['output_tokens']} "
        f"cout_estime=${usage['cost_usd']:.4f} (nb_appels_llm={usage['n_calls']})"
    )

    return {"answer": answer, "steps": steps, "refused": refused, "usage": usage}


def _aggregate_usage(messages):
    input_tokens = 0
    output_tokens = 0
    n_calls = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.usage_metadata:
            input_tokens += msg.usage_metadata.get("input_tokens") or 0
            output_tokens += msg.usage_metadata.get("output_tokens") or 0
            n_calls += 1
    cost_usd = (input_tokens / 1e6) * INPUT_COST_PER_MTOK + (output_tokens / 1e6) * OUTPUT_COST_PER_MTOK
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "n_calls": n_calls,
        "cost_usd": cost_usd,
    }


def _strip_accents(text):
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _contains_refusal_phrase(answer):
    return "impossible avec les donnees disponibles" in _strip_accents(answer).lower()
