import base64

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.agent import ask

st.set_page_config(page_title="Talk to my Data", page_icon="\U0001F4CA")
st.title("Talk to my Data")
st.caption(
    "Assistant d'analyse du dataset de defaut de paiement carte de credit "
    "(direction Recouvrement & Risque). Repond en francais, code pandas a "
    "l'appui, sans SQL ni acces reseau/disque."
)

if "history" not in st.session_state:
    st.session_state.history = []


def render_answer(turn):
    st.markdown(turn["answer"])

    for step in turn.get("steps", []):
        artifact = step.get("artifact")
        if step.get("code"):
            st.code(step["code"], language="python")

        if artifact:
            if artifact.get("image_b64"):
                st.image(base64.b64decode(artifact["image_b64"]))
            elif artifact.get("result_kind") in ("dataframe", "series") and "result_df" in artifact:
                st.dataframe(artifact["result_df"])
            elif artifact.get("result_kind") == "scalar":
                st.metric(label="Resultat", value=str(artifact.get("result_value")))
            if artifact.get("stdout"):
                with st.expander("Sortie standard"):
                    st.text(artifact["stdout"])


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_answer(turn)

question = st.chat_input("Pose une question sur le dataset...")

if question:
    with st.spinner("Analyse en cours..."):
        try:
            result = ask(question, history=st.session_state.history)
        except Exception as exc:
            st.error(f"Erreur lors de l'appel a l'agent : {exc}")
            result = None

    if result is not None:
        turn = {
            "question": question,
            "answer": result["answer"],
            "steps": result["steps"],
        }
        st.session_state.history.append(turn)
        st.rerun()

with st.sidebar:
    st.subheader("Historique")
    st.caption(f"{len(st.session_state.history)} question(s) posee(s)")
    if st.button("Effacer l'historique"):
        st.session_state.history = []
        st.rerun()

    for i, turn in enumerate(st.session_state.history, start=1):
        with st.expander(f"{i}. {turn['question'][:60]}"):
            render_answer(turn)
