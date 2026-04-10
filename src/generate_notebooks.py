"""
Generateur de notebooks Jupyter pour le projet 8INF436.
Execute ce script depuis la racine du projet avec le venv active.
"""

import nbformat as nbf
import os

NB_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NB_DIR, exist_ok=True)



# Utilitaires


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip())

def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.strip())

def notebook(cells) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    return nb



# NOTEBOOK 1 : Exploration et Preparation des donnees


NB1_CELLS = [
    md("""# Notebook 1 — Exploration et Preparation des donnees

**Cours** : Forage des Donnees 8INF436 — UQAC Hiver 2026
**Dataset** : Base Telecom 2019-12 (churn prediction)
**Variable cible** : `FLAG_RESILIATION` (0 = client actif, 1 = resilie)

---

## Table des matieres
1. Chargement et apercu
2. Analyse exploratoire (EDA)
3. Gestion des valeurs manquantes
4. Ingenierie des attributs
5. Encodage des variables categoriques
6. Gestion du desequilibre de classes (SMOTE)
7. Normalisation
8. Sauvegarde des donnees prepares
"""),

    code("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import seaborn as sns
from pathlib import Path

# Chemins
ROOT = Path('..').resolve()
DATA_RAW  = ROOT / 'data' / 'raw'  / 'base_telecom_2019_12.csv'
DATA_PROC = ROOT / 'data' / 'processed'
DATA_PROC.mkdir(exist_ok=True)

print(f"Dataset : {DATA_RAW}")
print(f"Sortie  : {DATA_PROC}")
"""),

    md("## 1. Chargement et apercu initial"),

    code("""
# Lecture avec encodage latin-1 (accents francais) et separateur point-virgule
df = pd.read_csv(DATA_RAW, sep=';', encoding='latin-1', low_memory=False)

print(f"Dimensions : {df.shape[0]:,} lignes  x  {df.shape[1]} colonnes")
df.head(3)
"""),

    code("""
print("=== Types de colonnes ===")
print(df.dtypes.value_counts())
print()
print("=== Premiers types par colonne ===")
df.dtypes
"""),

    code("""
print("=== Valeurs manquantes (%) ===")
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df = pd.DataFrame({'N manquants': miss, '% manquants': miss_pct})
miss_df = miss_df[miss_df['N manquants'] > 0].sort_values('% manquants', ascending=False)
print(miss_df.to_string())
"""),

    md("## 2. Analyse exploratoire (EDA)"),

    code("""
# Distribution de la variable cible
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = df['FLAG_RESILIATION'].value_counts()
labels = ['Actif (0)', 'Resilie (1)']
colors = ['#2196F3', '#F44336']

axes[0].bar(labels, counts.values, color=colors, edgecolor='black', linewidth=0.7)
axes[0].set_title('Distribution de FLAG_RESILIATION', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Nombre de clients')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 100, f'{v:,}\\n({v/len(df)*100:.1f}%)',
                 ha='center', fontsize=10)

axes[1].pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor':'black','linewidth':0.5})
axes[1].set_title('Proportion des classes', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'distribution_cible.png',
            bbox_inches='tight')
plt.show()
print(f"Ratio desequilibre : {counts[0]/counts[1]:.2f}:1")
"""),

    code("""
# Variables numeriques : distribution
num_cols = ['VOL_APPELS_M1','VOL_APPELS_M2','VOL_APPELS_M3',
            'NB_SMS_M1','NB_SMS_M2','NB_SMS_M3',
            'TAILLE_VILLE','REVENU_MOYEN_VILLE','NB_SERVICES',
            'DUREE_OFFRE','NB_MIGRATIONS','NB_REENGAGEMENTS']

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        axes[i].hist(df[col].dropna(), bins=40, color='#5C6BC0', edgecolor='white', linewidth=0.3)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel('')
plt.suptitle('Distributions des variables numeriques', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'distributions_numeriques.png', bbox_inches='tight')
plt.show()
"""),

    code("""
# Variables categoriques
cat_cols = ['SEXE', 'CSP', 'TYPE_VILLE', 'ENSEIGNE', 'MODE_PAIEMENT',
            'TELEPHONE_INIT', 'TELEPHONE', 'SITUATION_IMPAYES', 'SEGMENT']

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    if col in df.columns:
        vc = df[col].value_counts().head(8)
        axes[i].barh(vc.index.astype(str), vc.values, color='#26A69A', edgecolor='white')
        axes[i].set_title(col, fontsize=10, fontweight='bold')
        axes[i].invert_yaxis()
plt.suptitle('Distributions des variables categoriques', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'distributions_categoriques.png', bbox_inches='tight')
plt.show()
"""),

    code("""
# Correlation numerique vs cible
df['FLAG_RESILIATION'] = pd.to_numeric(df['FLAG_RESILIATION'], errors='coerce')

vol_cols = [c for c in df.columns if 'VOL_APPELS' in c or 'NB_SMS' in c]
for c in vol_cols + ['TAILLE_VILLE','REVENU_MOYEN_VILLE','NB_SERVICES',
                      'DUREE_OFFRE','NB_MIGRATIONS','NB_REENGAGEMENTS',
                      'DUREE_OFFRE_INIT']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

corr_target = df.select_dtypes(include='number').corr()['FLAG_RESILIATION'].drop('FLAG_RESILIATION')
corr_target = corr_target.sort_values(key=abs, ascending=False).head(20)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#EF5350' if v > 0 else '#42A5F5' for v in corr_target.values]
ax.barh(corr_target.index, corr_target.values, color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Correlation avec FLAG_RESILIATION', fontsize=13, fontweight='bold')
ax.set_xlabel('Coefficient de correlation de Pearson')
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'correlation_cible.png', bbox_inches='tight')
plt.show()
"""),

    md("## 3. Ingenierie des attributs et nettoyage"),

    code("""
df_clean = df.copy()

# --- Suppression colonne identifiant ---
df_clean.drop(columns=['ID_CLIENT'], errors='ignore', inplace=True)

# --- Parsing des dates et calcul d'attributs derives ---
from datetime import datetime

REFERENCE_DATE = pd.Timestamp('2019-12-01')

for date_col in ['DATE_NAISSANCE', 'DATE_ACTIVATION', 'DATE_FIN_ENGAGEMENT',
                  'DATE_DERNIER_REENGAGEMENT']:
    if date_col in df_clean.columns:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], format='%d/%m/%Y', errors='coerce')

# Age du client
df_clean['AGE'] = ((REFERENCE_DATE - df_clean['DATE_NAISSANCE']).dt.days / 365.25).round(1)
df_clean['AGE'] = df_clean['AGE'].clip(lower=0, upper=100)

# Anciennete en mois depuis activation
df_clean['ANCIENNETE_MOIS'] = ((REFERENCE_DATE - df_clean['DATE_ACTIVATION']).dt.days / 30.44).round(1)
df_clean['ANCIENNETE_MOIS'] = df_clean['ANCIENNETE_MOIS'].clip(lower=0)

# Jours restants avant fin engagement
df_clean['JOURS_FIN_ENGAGEMENT'] = (df_clean['DATE_FIN_ENGAGEMENT'] - REFERENCE_DATE).dt.days
# Negatif = engagement expire
df_clean['ENGAGEMENT_EXPIRE'] = (df_clean['JOURS_FIN_ENGAGEMENT'] < 0).astype(int)

# Suppression colonnes de dates brutes
df_clean.drop(columns=['DATE_NAISSANCE','DATE_ACTIVATION','DATE_FIN_ENGAGEMENT',
                        'DATE_DERNIER_REENGAGEMENT'], inplace=True, errors='ignore')

# Volume moyen d'appels sur 6 mois
vol_cols = ['VOL_APPELS_M1','VOL_APPELS_M2','VOL_APPELS_M3',
            'VOL_APPELS_M4','VOL_APPELS_M5','VOL_APPELS_M6']
df_clean['VOL_APPELS_MOY'] = df_clean[vol_cols].mean(axis=1)

# Tendance appels (M1 vs M6)
df_clean['TENDANCE_APPELS'] = df_clean['VOL_APPELS_M1'] - df_clean['VOL_APPELS_M6']

# Volume moyen SMS
sms_cols = ['NB_SMS_M1','NB_SMS_M2','NB_SMS_M3',
            'NB_SMS_M4','NB_SMS_M5','NB_SMS_M6']
df_clean['NB_SMS_MOY'] = df_clean[sms_cols].mean(axis=1)
df_clean['TENDANCE_SMS'] = df_clean['NB_SMS_M1'] - df_clean['NB_SMS_M6']

print(f"Dimensions apres ingenierie : {df_clean.shape}")
df_clean[['AGE','ANCIENNETE_MOIS','VOL_APPELS_MOY','TENDANCE_APPELS']].describe()
"""),

    md("## 4. Gestion des valeurs manquantes"),

    code("""
from sklearn.impute import SimpleImputer

# Apercu des manquants apres ingenierie
miss2 = df_clean.isnull().sum()
print("Colonnes avec valeurs manquantes :")
print(miss2[miss2 > 0].sort_values(ascending=False).to_string())
"""),

    code("""
# Imputation des numeriques par la mediane
num_cols_clean = df_clean.select_dtypes(include='number').columns.tolist()
num_cols_clean = [c for c in num_cols_clean if c != 'FLAG_RESILIATION']

for col in num_cols_clean:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)

# Imputation des categoriques par le mode
cat_cols_clean = df_clean.select_dtypes(include='object').columns.tolist()
for col in cat_cols_clean:
    mode_val = df_clean[col].mode()
    if len(mode_val) > 0:
        df_clean[col] = df_clean[col].fillna(mode_val[0])

# Verification
remaining_miss = df_clean.isnull().sum().sum()
print(f"Valeurs manquantes restantes : {remaining_miss}")
"""),

    md("## 5. Encodage des variables categoriques"),

    code("""
from sklearn.preprocessing import LabelEncoder

# Encodage label pour variables ordinales naturelles
ordinal_map = {
    'DUREE_OFFRE_INIT': None,  # numerique deja
    'SEGMENT': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4},
}

# SEGMENT
if 'SEGMENT' in df_clean.columns:
    df_clean['SEGMENT_NUM'] = df_clean['SEGMENT'].map(
        lambda x: ord(str(x)[0]) - ord('A') if pd.notnull(x) else -1
    )
    df_clean.drop(columns=['SEGMENT'], inplace=True)

# Reduction de la cardinalite du code postal -> departement (2 premiers chiffres)
if 'CODE_POSTAL' in df_clean.columns:
    df_clean['CODE_POSTAL'] = df_clean['CODE_POSTAL'].astype(str).str.zfill(5)
    df_clean['DEPARTEMENT'] = df_clean['CODE_POSTAL'].str[:2].astype(str)
    df_clean.drop(columns=['CODE_POSTAL'], inplace=True)
    print(f"CODE_POSTAL -> DEPARTEMENT : {df_clean['DEPARTEMENT'].nunique()} departements uniques")

# One-hot encoding pour variables nominales (cardinalite raisonnable <= 50)
cat_remaining = df_clean.select_dtypes(include='object').columns.tolist()
high_card = [c for c in cat_remaining if df_clean[c].nunique() > 50]
if high_card:
    print(f"Suppression colonnes haute cardinalite : {high_card}")
    df_clean.drop(columns=high_card, inplace=True)
    cat_remaining = [c for c in cat_remaining if c not in high_card]
print(f"Colonnes categoriques a encoder : {cat_remaining}")

df_clean = pd.get_dummies(df_clean, columns=cat_remaining, drop_first=True, dtype=int)

print(f"Dimensions apres encodage : {df_clean.shape}")
df_clean.head(2)
"""),

    md("## 6. Gestion du desequilibre de classes (SMOTE)"),

    code("""
from collections import Counter
from imblearn.over_sampling import SMOTE

X = df_clean.drop(columns=['FLAG_RESILIATION'])
y = df_clean['FLAG_RESILIATION'].astype(int)

print(f"Distribution avant SMOTE : {Counter(y)}")
print(f"Ratio : {Counter(y)[0]/Counter(y)[1]:.2f}:1")

# Application SMOTE uniquement sur le train (le split se fait au notebook 3)
# On sauvegarde X et y tels quels — SMOTE sera applique dans le pipeline de modelisation
print(f"\\nDimensions X : {X.shape}")
print(f"Colonnes : {X.shape[1]} attributs")
"""),

    md("## 7. Normalisation (Min-Max)"),

    code("""
from sklearn.preprocessing import MinMaxScaler
import joblib

# Colonnes numeriques continues (pas les flags binaires)
binary_cols = [c for c in X.columns if X[c].nunique() == 2]
scale_cols  = [c for c in X.columns if c not in binary_cols and X[c].dtype in ['float64','int64']
               and X[c].nunique() > 2]

print(f"Colonnes a normaliser : {len(scale_cols)}")
print(f"Colonnes binaires     : {len(binary_cols)}")

scaler = MinMaxScaler()
X_scaled = X.copy()
X_scaled[scale_cols] = scaler.fit_transform(X[scale_cols])

# Sauvegarde du scaler pour le dashboard
joblib.dump(scaler, ROOT / 'models' / 'scaler.pkl')
joblib.dump(scale_cols, ROOT / 'models' / 'scale_cols.pkl')
print("Scaler sauvegarde.")

X_scaled.describe().round(3)
"""),

    md("## 8. Sauvegarde des donnees prepares"),

    code("""
# Sauvegarde
X_scaled.to_parquet(DATA_PROC / 'X_preprocessed.parquet', index=False)
y.to_frame().to_parquet(DATA_PROC / 'y_preprocessed.parquet', index=False)

# Sauvegarde de la liste des colonnes
import json
with open(DATA_PROC / 'feature_names.json', 'w') as f:
    json.dump(X_scaled.columns.tolist(), f, indent=2)

print(f"X shape : {X_scaled.shape}")
print(f"y shape : {y.shape}")
print(f"\\nDistribution finale y :")
print(y.value_counts(normalize=True).map(lambda x: f'{x:.1%}'))
print("\\nDonnees sauvegardees dans data/processed/")
"""),

    code("""
# Matrice de correlation finale (top 20 features numeriques)
top_num = X_scaled.select_dtypes(include='number').columns[:20]
corr_matrix = X_scaled[top_num].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
            square=True, linewidths=0.4, cbar_kws={"shrink": 0.6},
            ax=ax, annot=False)
ax.set_title('Matrice de correlation — Top 20 attributs', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'matrice_correlation.png', bbox_inches='tight')
plt.show()
print("Preparation terminee.")
"""),
]


# NOTEBOOK 2 : Reduction de dimension


NB2_CELLS = [
    md("""# Notebook 2 — Reduction et Selection de Dimensions

**Methodes utilisees :**
- Analyse en Composantes Principales (PCA)
- Importance des attributs par Random Forest (SelectFromModel)
- Selection univariee par Chi2 / ANOVA F-score

---
"""),

    code("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import seaborn as sns
from pathlib import Path
import json

ROOT = Path('..').resolve()
DATA_PROC = ROOT / 'data' / 'processed'

X = pd.read_parquet(DATA_PROC / 'X_preprocessed.parquet')
y_df = pd.read_parquet(DATA_PROC / 'y_preprocessed.parquet')
y = y_df['FLAG_RESILIATION'].astype(int)

print(f"X : {X.shape},  y : {y.shape}")
print(f"Classes : {y.value_counts().to_dict()}")
"""),

    md("## 2.1 Importance des attributs par Random Forest"),

    code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_selector = RandomForestClassifier(
    n_estimators=200, max_depth=12,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf_selector.fit(X_train, y_train)

importances = pd.Series(rf_selector.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# Top 30
fig, ax = plt.subplots(figsize=(10, 9))
importances.head(30).sort_values().plot.barh(ax=ax, color='#5C6BC0', edgecolor='white')
ax.set_title('Top 30 attributs — Importance Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance (Gini)')
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'importance_rf.png', bbox_inches='tight')
plt.show()

print(f"Top 10 attributs :")
print(importances.head(10).to_string())
"""),

    md("## 2.2 Selection des attributs (seuil importance)"),

    code("""
# Seuil : conserver les attributs dont l'importance cumulee atteint 95%
cum_importance = importances.cumsum()
n_features_95 = (cum_importance < 0.95).sum() + 1
print(f"Attributs pour 95% d'importance cumulee : {n_features_95} / {len(importances)}")

selected_features = importances.head(n_features_95).index.tolist()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, len(cum_importance)+1), cum_importance.values, color='#E53935', linewidth=2)
ax.axvline(n_features_95, color='#1565C0', linestyle='--', label=f'n={n_features_95} (95%)')
ax.axhline(0.95, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel("Nombre d'attributs (tries par importance)")
ax.set_ylabel('Importance cumulee')
ax.set_title('Importance cumulee des attributs', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'importance_cumulee.png', bbox_inches='tight')
plt.show()
"""),

    md("## 2.3 Analyse en Composantes Principales (PCA)"),

    code("""
from sklearn.decomposition import PCA

# PCA sur le jeu complet
pca_full = PCA(random_state=42)
pca_full.fit(X)

# Variance expliquee
exp_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(exp_var)
n_comp_95 = np.argmax(cum_var >= 0.95) + 1
n_comp_99 = np.argmax(cum_var >= 0.99) + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, min(31, len(exp_var)+1)), exp_var[:30] * 100,
            color='#42A5F5', edgecolor='white')
axes[0].set_title('Variance expliquee par composante (top 30)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Composante principale')
axes[0].set_ylabel('Variance expliquee (%)')

axes[1].plot(range(1, len(cum_var)+1), cum_var * 100, color='#EF5350', linewidth=2)
axes[1].axvline(n_comp_95, color='#1565C0', linestyle='--',
                label=f'{n_comp_95} comp. -> 95%')
axes[1].axvline(n_comp_99, color='#2E7D32', linestyle='--',
                label=f'{n_comp_99} comp. -> 99%')
axes[1].axhline(95, color='gray', linestyle=':', alpha=0.6)
axes[1].axhline(99, color='gray', linestyle=':', alpha=0.6)
axes[1].set_xlabel('Nombre de composantes')
axes[1].set_ylabel('Variance cumulee (%)')
axes[1].set_title('Variance cumulee — PCA', fontsize=12, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'pca_variance.png', bbox_inches='tight')
plt.show()

print(f"Composantes pour 95% de variance : {n_comp_95}")
print(f"Composantes pour 99% de variance : {n_comp_99}")
"""),

    code("""
# Visualisation 2D PCA (sample)
from sklearn.decomposition import PCA as PCA2D

pca2 = PCA2D(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X.sample(5000, random_state=42))
y_sample = y.loc[X.sample(5000, random_state=42).index]

fig, ax = plt.subplots(figsize=(9, 6))
colors = {0: '#42A5F5', 1: '#EF5350'}
labels = {0: 'Actif', 1: 'Resilie'}
for cls in [0, 1]:
    mask = y_sample == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=colors[cls], label=labels[cls],
               alpha=0.35, s=15, edgecolors='none')
ax.set_title('Projection PCA 2D (echantillon 5000)', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
ax.legend(title='Classe', framealpha=0.8)
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'pca_2d.png', bbox_inches='tight')
plt.show()
"""),

    md("## 2.4 Sauvegarde des attributs selectionnes"),

    code("""
import joblib

# PCA pour 95% de variance — utilise dans la modelisation
pca_95 = PCA(n_components=n_comp_95, random_state=42)
pca_95.fit(X[selected_features])

joblib.dump(pca_95,          ROOT / 'models' / 'pca.pkl')
joblib.dump(selected_features, ROOT / 'models' / 'selected_features.pkl')

with open(DATA_PROC / 'selected_features.json', 'w') as f:
    json.dump(selected_features, f, indent=2)

print(f"Attributs selectionnes : {len(selected_features)}")
print(f"Composantes PCA retenues : {n_comp_95}")
print("Fichiers sauvegardes : models/pca.pkl, models/selected_features.pkl")

# Resume
print("\\n=== Resume de la reduction de dimension ===")
print(f"Dimensions initiales    : {X.shape[1]} attributs")
print(f"Apres selection RF      : {len(selected_features)} attributs (-{X.shape[1]-len(selected_features)})")
print(f"Apres PCA (95% var.)    : {n_comp_95} composantes")
print(f"Reduction totale        : {100*(1 - n_comp_95/X.shape[1]):.1f}%")
"""),
]



# NOTEBOOK 3 : Modelisation et Evaluation


NB3_CELLS = [
    md("""# Notebook 3 — Modelisation et Evaluation des Modeles

**Modeles utilises :**
1. Foret Aleatoire (Random Forest)
2. Gradient Boosting (XGBoost)
3. Reseau de Neurones (MLP — Multi-Layer Perceptron)

**Strategie de validation :** Validation croisee stratifiee 5-fold + split train/test final (80/20)

---
"""),

    code("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import seaborn as sns
from pathlib import Path
import json, joblib, time

ROOT = Path('..').resolve()
DATA_PROC = ROOT / 'data' / 'processed'

X = pd.read_parquet(DATA_PROC / 'X_preprocessed.parquet')
y_df = pd.read_parquet(DATA_PROC / 'y_preprocessed.parquet')
y = y_df['FLAG_RESILIATION'].astype(int)

selected_features = joblib.load(ROOT / 'models' / 'selected_features.pkl')
X_sel = X[selected_features]

print(f"X_sel : {X_sel.shape},  y : {y.shape}")
print(f"Distribution : {y.value_counts().to_dict()}")
"""),

    md("## 3.1 Preparation — Split train/test + SMOTE"),

    code("""
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE uniquement sur le train
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Train avant SMOTE  : {X_train.shape[0]:,} lignes | {y_train.value_counts().to_dict()}")
print(f"Train apres SMOTE  : {X_train_sm.shape[0]:,} lignes | {pd.Series(y_train_sm).value_counts().to_dict()}")
print(f"Test               : {X_test.shape[0]:,} lignes  | {y_test.value_counts().to_dict()}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
"""),

    md("## 3.2 Modele 1 — Random Forest"),

    code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

print("Entrainement Random Forest...")
t0 = time.time()

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Validation croisee 5-fold
cv_rf = cross_validate(
    rf, X_train_sm, y_train_sm, cv=cv,
    scoring=['accuracy','f1','roc_auc','precision','recall'],
    return_train_score=True, n_jobs=-1
)

print(f"Duree CV : {time.time()-t0:.1f}s")
print("\\n--- Cross-validation (5-fold) ---")
for metric in ['test_accuracy','test_f1','test_roc_auc','test_precision','test_recall']:
    vals = cv_rf[metric]
    print(f"  {metric[5:]:12s} : {vals.mean():.4f} (+/- {vals.std():.4f})")

# Entrainement final
rf.fit(X_train_sm, y_train_sm)
joblib.dump(rf, ROOT / 'models' / 'random_forest.pkl')
print("\\nModele RF sauvegarde.")
"""),

    md("## 3.3 Modele 2 — XGBoost"),

    code("""
import xgboost as xgb

print("Entrainement XGBoost...")
t0 = time.time()

ratio = (y_train_sm == 0).sum() / (y_train_sm == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

cv_xgb = cross_validate(
    xgb_model, X_train_sm, y_train_sm, cv=cv,
    scoring=['accuracy','f1','roc_auc','precision','recall'],
    return_train_score=True, n_jobs=-1
)

print(f"Duree CV : {time.time()-t0:.1f}s")
print("\\n--- Cross-validation (5-fold) ---")
for metric in ['test_accuracy','test_f1','test_roc_auc','test_precision','test_recall']:
    vals = cv_xgb[metric]
    print(f"  {metric[5:]:12s} : {vals.mean():.4f} (+/- {vals.std():.4f})")

xgb_model.fit(X_train_sm, y_train_sm)
joblib.dump(xgb_model, ROOT / 'models' / 'xgboost.pkl')
print("\\nModele XGBoost sauvegarde.")
"""),

    md("## 3.4 Modele 3 — Reseau de Neurones (MLP)"),

    code("""
from sklearn.neural_network import MLPClassifier

print("Entrainement MLP...")
t0 = time.time()

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=512,
    learning_rate='adaptive',
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

cv_mlp = cross_validate(
    mlp, X_train_sm, y_train_sm, cv=cv,
    scoring=['accuracy','f1','roc_auc','precision','recall'],
    return_train_score=True, n_jobs=1
)

print(f"Duree CV : {time.time()-t0:.1f}s")
print("\\n--- Cross-validation (5-fold) ---")
for metric in ['test_accuracy','test_f1','test_roc_auc','test_precision','test_recall']:
    vals = cv_mlp[metric]
    print(f"  {metric[5:]:12s} : {vals.mean():.4f} (+/- {vals.std():.4f})")

mlp.fit(X_train_sm, y_train_sm)
joblib.dump(mlp, ROOT / 'models' / 'mlp.pkl')
print("\\nModele MLP sauvegarde.")
"""),

    md("## 3.5 Evaluation finale sur le jeu de test"),

    code("""
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)

models = {
    'Random Forest': rf,
    'XGBoost'      : xgb_model,
    'MLP'          : mlp,
}

results = {}
for name, model in models.items():
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'Accuracy'        : accuracy_score(y_test, y_pred),
        'Precision'       : precision_score(y_test, y_pred, zero_division=0),
        'Recall'          : recall_score(y_test, y_pred),
        'F1-Score'        : f1_score(y_test, y_pred),
        'ROC-AUC'         : roc_auc_score(y_test, y_proba),
        'AP (PR-AUC)'     : average_precision_score(y_test, y_proba),
        'y_pred'          : y_pred,
        'y_proba'         : y_proba,
    }

# Tableau des metriques
metrics_df = pd.DataFrame({
    name: {k: v for k, v in r.items() if isinstance(v, float)}
    for name, r in results.items()
}).T.round(4)

print("=== Metriques sur le jeu de TEST ===")
print(metrics_df.to_string())

# Sauvegarde JSON
metrics_export = metrics_df.to_dict()
with open(ROOT / 'models' / 'metrics.json', 'w') as f:
    json.dump(metrics_export, f, indent=2)
print("\\nMetriques sauvegardees : models/metrics.json")
"""),

    code("""
# Matrices de confusion
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Actif', 'Resilie'],
                yticklabels=['Actif', 'Resilie'],
                linewidths=0.5, cbar=False)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predit')
    ax.set_ylabel('Reel')
plt.suptitle('Matrices de confusion (jeu de test)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'matrices_confusion.png', bbox_inches='tight')
plt.show()
"""),

    code("""
# Courbes ROC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#1565C0', '#C62828', '#2E7D32']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    auc = res['ROC-AUC']
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})',
                 color=color, linewidth=2)

axes[0].plot([0,1],[0,1],'--', color='gray', linewidth=1)
axes[0].set_xlabel('Taux de Faux Positifs')
axes[0].set_ylabel('Taux de Vrais Positifs')
axes[0].set_title('Courbes ROC', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# Courbes Precision-Recall
for (name, res), color in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, res['y_proba'])
    ap = res['AP (PR-AUC)']
    axes[1].plot(rec, prec, label=f'{name} (AP={ap:.3f})',
                 color=color, linewidth=2)

axes[1].set_xlabel('Rappel')
axes[1].set_ylabel('Precision')
axes[1].set_title('Courbes Precision-Rappel', fontsize=13, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'roc_pr_curves.png', bbox_inches='tight')
plt.show()
"""),

    code("""
# Comparaison des metriques — graphique barres groupees
metrics_plot = metrics_df[['Accuracy','Precision','Recall','F1-Score','ROC-AUC']]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(metrics_plot.columns))
width = 0.22
colors_bar = ['#1565C0', '#C62828', '#2E7D32']
for i, (model_name, row) in enumerate(metrics_plot.iterrows()):
    bars = ax.bar(x + i*width, row.values, width, label=model_name,
                  color=colors_bar[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, row.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_plot.columns, fontsize=11)
ax.set_ylabel('Score')
ax.set_ylim(0, 1.12)
ax.set_title('Comparaison des metriques — Jeu de test', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'comparaison_modeles.png', bbox_inches='tight')
plt.show()
"""),

    code("""
# Comparaison cross-validation
cv_results = {
    'Random Forest': cv_rf,
    'XGBoost'      : cv_xgb,
    'MLP'          : cv_mlp,
}
cv_metrics = ['test_accuracy','test_f1','test_roc_auc']
labels_cv  = ['Accuracy','F1-Score','ROC-AUC']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric, label in zip(axes, cv_metrics, labels_cv):
    data_box = [cv_results[m][metric] for m in ['Random Forest','XGBoost','MLP']]
    bp = ax.boxplot(data_box, patch_artist=True, widths=0.5,
                    medianprops={'color':'black','linewidth':2})
    for patch, color in zip(bp['boxes'], colors_bar):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(['RF','XGB','MLP'])
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Validation croisee 5-fold', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(ROOT / 'rapport' / 'figures' / 'crossval_boxplot.png', bbox_inches='tight')
plt.show()
"""),

    code("""
# Rapport de classification detaille
for name, res in results.items():
    print(f"\\n{'='*55}")
    print(f"  {name}")
    print('='*55)
    print(classification_report(y_test, res['y_pred'],
                                 target_names=['Actif (0)','Resilie (1)']))

print("\\nTous les modeles ont ete evalues et sauvegardes.")
"""),
]



# Ecriture sur disque


notebooks = [
    ("01_exploration_preparation.ipynb",  NB1_CELLS),
    ("02_reduction_dimension.ipynb",      NB2_CELLS),
    ("03_modelisation_evaluation.ipynb",  NB3_CELLS),
]

for fname, cells in notebooks:
    nb = notebook(cells)
    path = os.path.join(NB_DIR, fname)
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Cree : {path}")

print("\nTous les notebooks ont ete generes.")
