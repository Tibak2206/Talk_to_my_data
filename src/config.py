from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_card_default.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting_model.joblib"
SCORING_OUTPUT_PATH = PROJECT_ROOT / "data" / "scoring" / "scoring_test.csv"

# Colonnes
TARGET_COLUMN = "default_payment_next_month"
ID_COLUMN = "id"
# Colonne suspecte reperee en EDA (prediction deja presente dans les donnees brutes,
# risque de fuite de donnees) : a exclure des features.
LEAKAGE_COLUMN = "predicted_default_payment_next_month"
EXCLUDED_COLUMNS = [ID_COLUMN, LEAKAGE_COLUMN]

# Familles de colonnes brutes, utilisees par le feature engineering dans data_prep.py
# Convention chronologique corrigee (verifiee contre la documentation source UCI) :
# le suffixe 1 correspond au mois le plus RECENT, le suffixe 6 au mois le plus ANCIEN,
# pour les 3 familles bill_amt_*, pay_amt_* et pay_* (contrairement a ce qui avait ete
# documente initialement en EDA). La colonne brute "pay_0" est renommee en "pay_1" des
# le chargement (cf. load_and_prepare_data) pour uniformiser la numerotation.
BILL_AMT_COLUMNS = [f"bill_amt_{i}" for i in range(1, 7)]
PAY_AMT_COLUMNS = [f"pay_amt_{i}" for i in range(1, 7)]
PAY_STATUS_COLUMNS = ["pay_1", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]

# education_level n'est plus one-hot mais ordinal + flag (cf. encode_education_level
# dans data_prep.py) : sortie de CATEGORICAL_COLUMNS, ses colonnes derivees sont
# dans NUMERIC_COLUMNS.
CATEGORICAL_COLUMNS = ["sex", "marital_status"]
TYPE_CAST_COLUMNS = ["sex", "marital_status"]

# NUMERIC_COLUMNS reflete les features APRES feature engineering (engineer_features
# et encode_education_level dans data_prep.py), pas les colonnes brutes du CSV :
# - bill_amt_1..6 sont remplacees par bill_amt_moyen/bill_amt_tendance (forte
#   multicolinearite mesuree en EDA, 0.80-0.95 entre mois consecutifs)
# - pay_amt_1..6 sont conservees telles quelles (moins redondantes, individuellement
#   discriminantes) et completees par reste_du_moyen et taux_utilisation_moyen
# - pay_0..pay_6 sont conservees telles quelles et completees par retard_max/nb_mois_en_retard
NUMERIC_COLUMNS = [
    "limit_balance",
    "age",
    "pay_1",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "retard_max",
    "nb_mois_en_retard",
    "pay_amt_1",
    "pay_amt_2",
    "pay_amt_3",
    "pay_amt_4",
    "pay_amt_5",
    "pay_amt_6",
    "bill_amt_moyen",
    "bill_amt_tendance",
    "reste_du_moyen",
    "taux_utilisation_moyen",
    "education_level_ordinal",
    "is_education_undocumented",
]

# Regroupement des modalites decide en EDA (codes non documentes / rares)
EDUCATION_LEVEL_MAPPING = {4: 0, 5: 0, 6: 0}
MARITAL_STATUS_MAPPING = {"0": "3"}

# Encodage ordinal d'education_level (apres regroupement ci-dessus, donc valeurs
# possibles {0,1,2,3}) : ordre croissant du taux de defaut observe en EDA
# (8% autre/non documente, 18% etudes sup, 24% universite, 24% lycee).
# La modalite 0 (non documentee) n'a pas de position logique sur cette echelle :
# elle est isolee via is_education_undocumented plutot que forcee dans l'ordre.
EDUCATION_LEVEL_ORDER = {1: 0, 2: 1, 3: 2}

# Split train/test
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Modele final retenu en notebook 03 (Gradient Boosting) : hyperparametres trouves
# par RandomizedSearchCV (PR-AUC prioritaire, 5-fold CV, n_iter=30 sur le train).
FINAL_MODEL_PARAMS = {
    "learning_rate": 0.010243250407033817,
    "max_depth": 2,
    "min_samples_leaf": 17,
    "n_estimators": 268,
    "subsample": 0.8447411578889518,
    "random_state": RANDOM_STATE,
}

# Seuil de decision retenu en notebook 03, section 5 : minimisation d'un cout
# metier illustratif (R=5, un defaut manque coute ~5x plus qu'une relance
# inutile). A affiner avec de vrais couts de la direction Recouvrement & Risque.
DECISION_THRESHOLD = 0.180
