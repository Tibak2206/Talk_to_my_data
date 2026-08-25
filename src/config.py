from pathlib import Path

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_card_default.csv"

# Colonnes
TARGET_COLUMN = "default_payment_next_month"
ID_COLUMN = "id"
# Colonne suspecte reperee en EDA (prediction deja presente dans les donnees brutes,
# risque de fuite de donnees) : a exclure des features.
LEAKAGE_COLUMN = "predicted_default_payment_next_month"
EXCLUDED_COLUMNS = [ID_COLUMN, LEAKAGE_COLUMN]

CATEGORICAL_COLUMNS = ["sex", "marital_status", "education_level"]
TYPE_CAST_COLUMNS = ["sex", "marital_status"]
NUMERIC_COLUMNS = [
    "limit_balance",
    "age",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt_1",
    "bill_amt_2",
    "bill_amt_3",
    "bill_amt_4",
    "bill_amt_5",
    "bill_amt_6",
    "pay_amt_1",
    "pay_amt_2",
    "pay_amt_3",
    "pay_amt_4",
    "pay_amt_5",
    "pay_amt_6",
]

# Regroupement des modalites decide en EDA (codes non documentes / rares)
EDUCATION_LEVEL_MAPPING = {4: 0, 5: 0, 6: 0}
MARITAL_STATUS_MAPPING = {"0": "3"}

# Split train/test
TEST_SIZE = 0.2
RANDOM_STATE = 42
