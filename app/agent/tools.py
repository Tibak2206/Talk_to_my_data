"""Outil unique d'execution Python controlee pour l'agent (cf. cahier des charges,
point 12) : pas de SQL, pas d'acces reseau, pas d'ecriture disque. Le sandbox ci-dessous
est une protection "best effort" (AST + builtins restreints) adaptee a un POC interne,
pas une isolation de securite hermetique face a du code adversarial.
"""

import ast
import base64
import builtins
import io
from contextlib import redirect_stdout

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import DECISION_THRESHOLD, LEAKAGE_COLUMN, MODEL_PATH, RAW_DATA_PATH, TARGET_COLUMN
from src.data_prep import clean_categoricals, load_and_prepare_data

SEX_LABELS = {"1": "Homme", "2": "Femme"}
EDUCATION_LABELS = {0: "Non documente", 1: "Etudes superieures", 2: "Universite", 3: "Lycee"}
MARITAL_LABELS = {"1": "Marie", "2": "Celibataire", "3": "Autre"}


def _load_dataset_for_agent():
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.rename(columns={"pay_0": "pay_1"})
    df = clean_categoricals(df)
    df = df.drop(columns=[LEAKAGE_COLUMN], errors="ignore")
    df["sex_label"] = df["sex"].map(SEX_LABELS)
    df["education_level_label"] = df["education_level"].map(EDUCATION_LABELS)
    df["marital_status_label"] = df["marital_status"].map(MARITAL_LABELS)
    return df


def _compute_model_info():
    pipeline = joblib.load(MODEL_PATH)
    _, df_test = load_and_prepare_data()
    X_test = df_test.drop(columns=[TARGET_COLUMN])
    y_test = df_test[TARGET_COLUMN]

    proba = pipeline.predict_proba(X_test)[:, 1]
    label_pred = (proba >= DECISION_THRESHOLD).astype(int)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    test_metrics = {
        "pr_auc": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, label_pred)),
        "precision": float(precision_score(y_test, label_pred)),
        "recall": float(recall_score(y_test, label_pred)),
        "f1": float(f1_score(y_test, label_pred)),
        "decision_threshold": DECISION_THRESHOLD,
        "n_test_samples": int(len(y_test)),
    }

    return {
        "model_type": type(pipeline.named_steps["model"]).__name__,
        "feature_importance": feature_importance,
        "test_metrics": test_metrics,
    }


_DATASET = _load_dataset_for_agent()
_MODEL_INFO = _compute_model_info()

_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "breakpoint", "help", "exit", "quit",
    "memoryview", "os", "sys", "subprocess", "socket", "requests", "urllib",
    "pathlib", "shutil", "importlib",
}

_SAFE_BUILTIN_NAMES = (
    "len", "range", "sum", "min", "max", "sorted", "list", "dict", "set", "tuple",
    "str", "int", "float", "bool", "round", "enumerate", "zip", "map", "filter",
    "print", "abs", "all", "any", "isinstance", "type", "repr", "reversed",
)
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}


def _validate_code(code):
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Code invalide (erreur de syntaxe) : {exc}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Les imports ne sont pas autorises : pandas (pd), numpy (np) et "
                              "matplotlib.pyplot (plt) sont deja disponibles.")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"Utilisation interdite : '{node.id}'.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise ValueError("Acces aux attributs '__dunder__' interdit.")


def _capture_figure():
    if not plt.get_fignums():
        return None
    fig = plt.gcf()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@tool(response_format="content_and_artifact")
def execute_python_code(code: str):
    """Execute du code pandas/numpy/matplotlib controle sur le dataset en memoire.

    Variables disponibles dans l'environnement d'execution :
    - `df` : DataFrame du dataset (clients, colonnes brutes + labels lisibles
      sex_label/education_level_label/marital_status_label).
    - `model_info` : dict avec `model_type`, `feature_importance` (DataFrame) et
      `test_metrics` (dict) decrivant le modele de scoring deja entraine.
    - `pd`, `np`, `plt` : pandas, numpy, matplotlib.pyplot.

    Le code DOIT affecter la reponse finale a une variable nommee `result`
    (DataFrame, Series, scalaire ou dict). Pour un graphique, creer la figure
    avec `plt` (elle est capturee automatiquement, pas besoin de l'assigner a
    `result`). Aucun import, aucun acces disque/reseau n'est autorise.
    """
    try:
        _validate_code(code)
    except ValueError as exc:
        return f"Code refuse par le sandbox : {exc}", {"error": str(exc), "code": code}

    safe_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "df": _DATASET.copy(),
        "model_info": _MODEL_INFO,
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    local_vars = {}
    stdout_buffer = io.StringIO()

    try:
        with redirect_stdout(stdout_buffer):
            exec(compile(ast.parse(code, mode="exec"), "<agent_code>", "exec"), safe_globals, local_vars)
    except Exception as exc:
        return f"Erreur lors de l'execution du code : {exc}", {"error": str(exc), "code": code}

    stdout_text = stdout_buffer.getvalue()
    result = local_vars.get("result")
    image_b64 = _capture_figure()

    artifact = {"code": code, "stdout": stdout_text, "image_b64": image_b64}

    if isinstance(result, pd.DataFrame):
        artifact["result_kind"] = "dataframe"
        artifact["result_df"] = result
        content = f"Resultat (table, {result.shape[0]} lignes x {result.shape[1]} colonnes) :\n{result.to_string()}"
    elif isinstance(result, pd.Series):
        artifact["result_kind"] = "series"
        artifact["result_df"] = result.to_frame()
        content = f"Resultat (serie) :\n{result.to_string()}"
    elif result is not None:
        artifact["result_kind"] = "scalar"
        artifact["result_value"] = result
        content = f"Resultat : {result}"
    elif image_b64 is not None:
        artifact["result_kind"] = "figure"
        content = "Graphique genere."
    else:
        artifact["result_kind"] = "none"
        content = "Le code s'est execute mais n'a pas affecte de variable `result`."

    if stdout_text:
        content += f"\n\nSortie standard :\n{stdout_text}"

    return content, artifact


TOOLS = [execute_python_code]
