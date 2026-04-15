# 8INF436 - Prediction de Resiliation Client (Churn Telecom)
# UQAC - Hiver 2026

Equipe : Abdelaali Kaina, Mathias De Ridder, Lucas Matos, Talhatou Balde, Hajar Chakir

---

## Contenu du depot

```
|-- notebooks/
|   |-- 01_exploration_preparation.ipynb   Notebook 1 : preprocessing
|   |-- 02_reduction_dimension.ipynb       Notebook 2 : selection de features + PCA
|   `-- 03_modelisation_evaluation.ipynb   Notebook 3 : modeles + evaluation
|-- dashboard/
|   `-- app.py                   Interface Dash interactive
|-- src/
|   `-- calibration.py           Requis pour charger les modeles (pickle)
|-- models/                      Modeles entraines et prets a l'emploi
|-- data/
|   |-- raw/                     Dataset brut original
|   `-- processed/               Donnees preprocessees (sorties du notebook 1)
|-- requirements.txt
`-- run_dashboard.sh
```

---

## Lancer le dashboard interactif

### Prerequis
Python 3.10+ installe sur la machine.

### Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### Lancement

```bash
python dashboard/app.py
```

Ouvrir ensuite : http://127.0.0.1:8050

Le dashboard charge les modeles pre-entraines depuis `models/`.
Aucune execution des notebooks n'est necessaire pour le faire tourner.

---

## Rejouer les notebooks

Les notebooks doivent etre executes dans l'ordre :

1. `01_exploration_preparation.ipynb` -- lit `data/raw/`, ecrit dans `data/processed/`
2. `02_reduction_dimension.ipynb`    -- lit `data/processed/`, ecrit dans `models/`
3. `03_modelisation_evaluation.ipynb` -- lit `data/processed/` et `models/`, ecrit les modeles finaux

Installation identique, puis lancer Jupyter :

```bash
jupyter lab
```

---

## Resultats des modeles (jeu de test 20%)

| Modele        | Accuracy | F1-Score | ROC-AUC |
|---------------|----------|----------|---------|
| Random Forest | 0.905    | 0.758    | 0.953   |
| XGBoost       | 0.902    | 0.726    | 0.940   |
| MLP           | 0.878    | 0.627    | 0.899   |

Les modeles sont calibres par regression isotonique pour corriger le biais introduit par SMOTE.
