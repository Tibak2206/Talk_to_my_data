"""Golden set de validation de l'agent (cahier des charges, point 14) :
au moins 10 questions dont la reponse est recalculee hors agent (pandas direct
sur les memes donnees que `app/agent/tools.py`) puis comparee au resultat
produit par l'agent. Necessite ANTHROPIC_API_KEY (facture des appels reels).

La comparaison est volontairement tolerante a la forme exacte du resultat
(dict, Series ou DataFrame multi-colonnes) et a l'echelle (fraction 0-1 vs
pourcentage 0-100), car le format precis depend de la formulation du code
generé par le modele : seule la valeur numerique compte pour la validation.

Usage : python -m scripts.validate_golden_set
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.agent.agent import ask
from app.agent.tools import _DATASET as df
from app.agent.tools import _MODEL_INFO as model_info

REL_TOL = 0.02
ABS_TOL_FLOOR = 0.05


def _normalize_label(label):
    text = unicodedata.normalize("NFKD", str(label))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.strip().lower()


def _values_match(value, expected):
    tol = max(ABS_TOL_FLOOR, REL_TOL * abs(expected))
    for candidate in (value, value / 100, value * 100):
        if abs(candidate - expected) <= tol:
            return True
    return False


def _all_numeric_values(artifact):
    values = []
    if artifact is None:
        return values
    kind = artifact.get("result_kind")
    if kind == "scalar":
        value = artifact.get("result_value")
        if isinstance(value, dict):
            values.extend(v for v in value.values() if isinstance(v, (int, float)))
        elif isinstance(value, (int, float)):
            values.append(float(value))
    elif kind in ("dataframe", "series"):
        result_df = artifact.get("result_df")
        if result_df is not None:
            for col in result_df.columns:
                values.extend(float(v) for v in result_df[col] if isinstance(v, (int, float)))
    return values


def _extract_segment_map(artifact):
    """Retourne {label_normalise: [valeurs numeriques associees]}, quelle que soit
    la forme du resultat (dict plat, ou DataFrame/Series indexe par segment)."""
    if artifact is None:
        return None
    kind = artifact.get("result_kind")
    segments = {}
    if kind == "scalar" and isinstance(artifact.get("result_value"), dict):
        for key, value in artifact["result_value"].items():
            if isinstance(value, (int, float)):
                segments.setdefault(_normalize_label(key), []).append(float(value))
    elif kind in ("dataframe", "series"):
        result_df = artifact.get("result_df")
        if result_df is not None:
            for idx, row in result_df.iterrows():
                values = [float(v) for v in row if isinstance(v, (int, float))]
                segments.setdefault(_normalize_label(idx), []).extend(values)
    return segments or None


def _numeric_values_from_steps(steps):
    values = []
    for step in steps:
        values.extend(_all_numeric_values(step.get("artifact")))
    return values


def _segment_map_from_steps(steps):
    merged = {}
    for step in steps:
        for key, values in (_extract_segment_map(step.get("artifact")) or {}).items():
            merged.setdefault(key, []).extend(values)
    return merged or None


def check_scalar(question, expected):
    def _check(result):
        values = _numeric_values_from_steps(result["steps"])
        match = next((v for v in values if _values_match(v, expected)), None)
        if match is None:
            return False, f"aucune valeur ne correspond (attendu {expected:.4f}, valeurs vues : {values[:8]})"
        return True, f"attendu {expected:.4f}, trouve {match:.4f}"

    return question, _check


def check_segments(question, expected_map):
    """expected_map : {"alias1|alias2|...": valeur_attendue} - plusieurs libelles
    possibles par segment separes par '|', pour tolerer le phrasage choisi par
    l'agent (ex. code brut "0"/"1" ou libelle "Sans defaut"/"En defaut")."""

    def _check(result):
        got = _segment_map_from_steps(result["steps"])
        if not got:
            return False, f"pas de resultat exploitable (attendu {expected_map})"
        mismatches = []
        for aliases_str, expected_value in expected_map.items():
            aliases = [_normalize_label(a) for a in aliases_str.split("|")]
            candidates = next((got[a] for a in aliases if a in got), None)
            if candidates is None:
                fuzzy_key = next((k for k in got for a in aliases if a and (a in k or k in a)), None)
                candidates = got.get(fuzzy_key) if fuzzy_key else None
            if candidates is None:
                mismatches.append(f"segment '{aliases_str}' absent (labels vus : {list(got)})")
                continue
            if not any(_values_match(v, expected_value) for v in candidates):
                mismatches.append(f"segment '{aliases_str}' : attendu {expected_value:.4f}, valeurs obtenues {candidates}")
        if mismatches:
            return False, "; ".join(mismatches)
        return True, f"tous les segments attendus correspondent ({expected_map})"

    return question, _check


def check_top_feature(question, expected_top_feature):
    def _check(result):
        texts = [result["answer"]]
        for step in result["steps"]:
            artifact = step.get("artifact") or {}
            texts.append(str(artifact.get("result_df")))
            texts.append(str(artifact.get("result_value", "")))
        ok = any(expected_top_feature in text for text in texts)
        return ok, f"feature attendue en tete : '{expected_top_feature}' {'trouvee' if ok else 'absente'} dans le resultat"

    return question, _check


def check_refusal(question):
    def _check(result):
        ok = result["refused"]
        return ok, f"refused={result['refused']}, nb_appels_outil={len(result['steps'])}"

    return question, _check


def build_golden_set():
    default_rate = float(df["default_payment_next_month"].mean())
    mean_age = float(df["age"].mean())
    rate_by_sex = df.groupby("sex_label")["default_payment_next_month"].mean().to_dict()
    rate_by_education = df.groupby("education_level_label")["default_payment_next_month"].mean().to_dict()
    rate_by_marital = df.groupby("marital_status_label")["default_payment_next_month"].mean().to_dict()
    limit_by_default = df.groupby("default_payment_next_month")["limit_balance"].mean().to_dict()
    age_median_by_default = df.groupby("default_payment_next_month")["age"].median().to_dict()
    top_feature = model_info["feature_importance"].iloc[0]["feature"]
    pr_auc = model_info["test_metrics"]["pr_auc"]

    return [
        check_scalar("Quel est le taux de defaut global sur l'ensemble des clients ?", default_rate),
        check_scalar("Quel est l'age moyen des clients dans le dataset ?", mean_age),
        check_segments(
            "Quel est le taux de defaut par sexe ?",
            {"Homme": rate_by_sex["Homme"], "Femme": rate_by_sex["Femme"]},
        ),
        check_segments(
            "Quel est le taux de defaut par niveau d'education ?",
            {k: v for k, v in rate_by_education.items()},
        ),
        check_segments(
            "Quel est le taux de defaut par statut marital ?",
            {k: v for k, v in rate_by_marital.items()},
        ),
        check_segments(
            "Quelle est la limite de credit moyenne, pour les clients en defaut compares aux clients sans defaut ?",
            {
                "0|sans defaut|non defaillant|sain": limit_by_default[0],
                "1|en defaut|defaillant": limit_by_default[1],
            },
        ),
        check_segments(
            "Quel est l'age median chez les clients en defaut compare aux clients sans defaut ?",
            {
                "0|sans defaut|non defaillant|sain": age_median_by_default[0],
                "1|en defaut|defaillant": age_median_by_default[1],
            },
        ),
        check_top_feature(
            "Quelle est la feature la plus importante du modele de scoring de credit ?", top_feature
        ),
        check_scalar(
            "Quel est le PR-AUC du modele de scoring sur le jeu de test ?", pr_auc
        ),
        check_scalar(
            "Combien de clients y a-t-il au total dans le dataset ?", float(len(df))
        ),
        check_refusal(
            "Quel est le revenu annuel moyen des clients ? (cette variable n'existe pas dans le dataset)"
        ),
        check_refusal(
            "Peux-tu te connecter a l'API de la banque pour recuperer les transactions en temps reel des clients ?"
        ),
    ]


def main():
    golden_set = build_golden_set()
    n_passed = 0

    for question, checker in golden_set:
        result = ask(question, history=None)
        passed, detail = checker(result)
        n_passed += int(passed)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {question}")
        print(f"       {detail}")
        print()

    print(f"Resultat : {n_passed}/{len(golden_set)} questions validees.")


if __name__ == "__main__":
    main()
