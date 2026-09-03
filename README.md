# Talk to my Data

Projet de data science de bout en bout pour la direction **Recouvrement & Risque** d'une banque de détail, autour du dataset *Default of Credit Card Clients* (2 965 clients). Trois étapes : exploration/préparation des données, modélisation d'un score de défaut, et un assistant conversationnel (LangChain + Streamlit) pour interroger le dataset et le modèle en langage naturel.

**Propriétaire** : Tiéba Bamba

## Structure du projet

```
notebooks/
  01_setup_repo_et_eda.ipynb        Exploration des données, nettoyage, décisions de préparation
  02_modelisation_baseline.ipynb    Modèle de référence (LogisticRegression) + présélection de 4 candidats
  03_modelisation_advanced.ipynb    Optimisation, sélection du modèle final, seuil de décision, évaluation test

src/
  config.py       Constantes centralisées (colonnes, mappings, hyperparamètres, seuil)
  data_prep.py    Chargement, nettoyage, feature engineering, pipeline sklearn
  metrics.py      Métrique recall@topK
  train.py        Entraîne le modèle final et le sauvegarde dans models/
  infer.py        Score un jeu de données avec le modèle sauvegardé

app/
  streamlit_app.py     Interface Streamlit de l'assistant conversationnel
  agent/agent.py       Agent LangChain v1 (Claude) et boucle de conversation
  agent/tools.py        Outil unique d'exécution Python contrôlée (sandbox pandas)
  agent/prompts.py      Prompt système de l'agent

scripts/
  validate_golden_set.py   Golden set de validation de l'agent (12 questions, comparaison hors-agent)

data/raw/           Dataset brut (CSV)
data/scoring/        Fichier de scoring produit par infer.py (id, proba_default, label_pred)
models/               Modèle entraîné sauvegardé (.joblib)
```

## Installation

```bash
python -m venv venv
./venv/Scripts/activate       # Windows ; source venv/bin/activate sous Linux/Mac
pip install -r requirements.txt
```

## Notebooks (étapes 1 et 2)

Ouvrir les notebooks dans l'ordre (`01` → `02` → `03`) avec Jupyter/VS Code, environnement `venv` sélectionné. Chaque notebook est autonome et documente ses propres décisions (feature engineering, gestion du déséquilibre des classes, choix du seuil, etc.). Les graphiques sont exportés en PNG statique (`fig.show("png")`, via `kaleido`) pour s'afficher correctement aussi bien en local que dans la prévisualisation GitHub.

### Réentraîner le modèle / produire un fichier de scoring

```bash
python -m src.train    # entraîne le modèle final et le sauvegarde dans models/
python -m src.infer    # score le jeu de test et écrit data/scoring/scoring_test.csv
```

### Résultats du modèle final (Gradient Boosting)

Évaluation unique sur le jeu de test (593 clients), seuil de décision = 0.180 :

| Métrique | Valeur |
|---|---|
| PR-AUC | 0,534 |
| Accuracy | 0,727 |
| Precision | 0,404 |
| Recall | 0,583 |
| F1-score | 0,477 |

183 clients sur 593 (~31 %) sont signalés pour relance au seuil retenu. Détail complet et méthodologie (validation croisée imbriquée, choix du seuil par coût métier) dans `notebooks/03_modelisation_advanced.ipynb`.

## Assistant conversationnel (étape 3)

Agent LangChain v1 (Claude, via l'API Anthropic) qui répond en français à des questions simples sur le dataset et le modèle, en générant et exécutant du code pandas dans un environnement contrôlé (pas de SQL, pas d'accès réseau ni d'écriture disque dans le code exécuté).

### Configuration

Créer un fichier `.env` à la racine (voir `.env.example`) avec une clé API Anthropic :

```
ANTHROPIC_API_KEY=sk-ant-...
# Uniquement si necessaire (cle "identity-linked", erreur 400 sur
# anthropic-workspace-id) : ID du workspace, visible sur platform.claude.com.
ANTHROPIC_WORKSPACE_ID=wrkspc_...
```

### Lancer l'app

```bash
./venv/Scripts/python -m streamlit run app/streamlit_app.py
```

### Valider l'agent (golden set)

```bash
python -m scripts.validate_golden_set
```

Exécute 12 questions de référence (profilage, segmentation, comparaison, informations modèle, refus hors périmètre) et compare chaque résultat à un calcul pandas indépendant. Fait de vrais appels à l'API Anthropic (coût réel, de l'ordre de quelques dizaines de centimes).
