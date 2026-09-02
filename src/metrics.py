import numpy as np


def recall_at_k(y_true, y_proba, k):
    """Recall parmi les k% de clients au score de probabilite le plus eleve.

    Trie les clients par proba_default decroissante, prend les k% premiers
    (contrainte de capacite operationnelle, ex. l'equipe recouvrement ne peut
    relancer qu'une partie du portefeuille), et mesure quelle proportion des
    vrais defaillants est capturee dans ce groupe. Complementaire au Recall
    classique (qui fixe un seuil de probabilite plutot qu'un volume de clients).

    k : proportion entre 0 et 1 (ex. 0.2 pour les 20% de clients les plus a risque).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    n_top = int(np.ceil(len(y_proba) * k))
    top_indices = np.argsort(y_proba)[::-1][:n_top]

    return y_true[top_indices].sum() / y_true.sum()
