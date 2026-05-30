"""
=============================================================================
SCRIPT DE GÉNÉRATION AUTOMATIQUE - DISEASE PREDICTION PROJECT
=============================================================================
1. Crée un dossier : Stage_ML_Disease/
2. Crée ce fichier : Stage_ML_Disease/setup_disease_project.py
3. Exécute : python setup_disease_project.py
4. Tout est créé automatiquement !
=============================================================================
"""

import os

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Création des dossiers
folders = ['src', 'notebooks', 'models', 'results', 'data']
for folder in folders:
    os.makedirs(os.path.join(PROJECT_ROOT, folder), exist_ok=True)

print("=" * 60)
print("GÉNÉRATION DU PROJET DISEASE PREDICTION")
print("=" * 60)

# =============================================================================
# 1. src/__init__.py
# =============================================================================
init_content = '''"""
Package src pour le projet Disease Prediction.
"""

__version__ = "1.0.0"
__author__ = "Othman Sebbar"
'''

# =============================================================================
# 2. src/data_loader.py
# =============================================================================
data_loader_content = '''"""
Module de chargement et prétraitement des données médicales.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_heart_disease_data():
    """
    Charge le dataset Heart Disease depuis UCI Repository.
    
    Returns:
        pd.DataFrame: Dataset complet
    """
    print("Chargement du dataset Heart Disease...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
        'ca', 'thal', 'target'
    ]
    
    df = pd.read_csv(url, names=column_names)
    print(f"  Dataset chargé: {df.shape[0]} patients, {df.shape[1]} features")
    
    return df


def clean_data(df):
    """
    Nettoie les données: remplace '?' par NaN, imputation médiane.
    
    Args:
        df: DataFrame brut
    
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    print("\\nNettoyage des données...")
    
    # Remplacement des '?' par NaN
    df = df.replace('?', np.nan)
    
    # Conversion en numérique
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Imputation par la médiane
    df = df.fillna(df.median())
    
    print(f"  Valeurs manquantes après imputation: {df.isnull().sum().sum()}")
    
    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Pipeline complet de prétraitement.
    
    Args:
        df: DataFrame nettoyé
        test_size: Proportion pour le test
        random_state: Graine aléatoire
    
    Returns:
        dict: Tous les jeux de données et objets de prétraitement
    """
    print("\\n" + "="*50)
    print("PRÉTRAITEMENT DES DONNÉES")
    print("="*50)
    
    # Conversion de la cible en binaire
    df['target_binary'] = (df['target'] > 0).astype(int)
    print(f"  Distribution: {df['target_binary'].value_counts().to_dict()}")
    
    # Séparation features / target
    X = df.drop(['target', 'target_binary'], axis=1)
    y = df['target_binary']
    
    print(f"  Features: {list(X.columns)}")
    print(f"  Target: maladie cardiaque (0=non, 1=oui)")
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\\n  Train: {X_train_scaled.shape[0]} échantillons")
    print(f"  Test:  {X_test_scaled.shape[0]} échantillons")
    
    # Sauvegarde du scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': list(X.columns),
        'scaler': scaler
    }


if __name__ == "__main__":
    df = load_heart_disease_data()
    df = clean_data(df)
    data = preprocess_data(df)
    print("\\n✅ Module data_loader testé avec succès!")
'''

# =============================================================================
# 3. src/models.py
# =============================================================================
models_content = '''"""
Module de construction des modèles de Machine Learning.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_models():
    """
    Initialise tous les modèles de classification.
    
    Returns:
        dict: Dictionnaire {nom: modèle}
    """
    print("\\nInitialisation des modèles...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    for name in models.keys():
        print(f"  ✓ {name}")
    
    return models


if __name__ == "__main__":
    models = get_models()
    print("\\n✅ Module models testé avec succès!")
'''

# =============================================================================
# 4. src/train.py
# =============================================================================
train_content = '''"""
Module d'entraînement et d'évaluation des modèles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import joblib
import os


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """
    Entraîne et évalue un modèle unique.
    
    Returns:
        tuple: (métriques dict, probabilités, modèle entraîné)
    """
    print(f"\\n{'='*50}")
    print(f"MODÈLE: {model_name}")
    print(f"{'='*50}")
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("\\n  Métriques:")
    for metric, value in metrics.items():
        print(f"    {metric:<12}: {value:.4f}")
    
    return metrics, y_pred_proba, model


def compare_models(results, roc_data, y_test, save_path='results/model_comparison.png'):
    """
    Compare tous les modèles et génère les visualisations.
    
    Args:
        results: dict {nom: métriques}
        roc_data: dict {nom: probabilités}
        y_test: vraies labels
        save_path: chemin de sauvegarde
    """
    print("\\n" + "="*60)
    print("COMPARAISON DES MODÈLES")
    print("="*60)
    
    # Tableau comparatif
    comparison_df = pd.DataFrame(results).T.round(4)
    print(f"\\n{comparison_df}")
    
    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 3, idx % 3]
        values = comparison_df[metric]
        bars = ax.bar(values.index, values, color=colors[idx], alpha=0.8)
        ax.set_title(f'{metric}', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Courbes ROC
    ax = axes[1, 2]
    for name, y_pred_proba in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('Taux de Faux Positifs')
    ax.set_ylabel('Taux de Vrais Positifs')
    ax.set_title('Courbes ROC', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\\n✅ Comparaison sauvegardée: {save_path}")
    
    return comparison_df


def save_best_model(trained_models, comparison_df, metric='ROC-AUC'):
    """
    Identifie et sauvegarde le meilleur modèle.
    
    Returns:
        tuple: (nom du meilleur modèle, modèle entraîné)
    """
    best_name = comparison_df[metric].idxmax()
    best_model = trained_models[best_name]
    
    print(f"\\n{'='*60}")
    print(f"MEILLEUR MODÈLE: {best_name}")
    print(f"{'='*60}")
    print(f"  {metric}: {comparison_df.loc[best_name, metric]:.4f}")
    
    joblib.dump(best_model, 'models/best_model.joblib')
    print(f"  ✅ Sauvegardé: models/best_model.joblib")
    
    return best_name, best_model


if __name__ == "__main__":
    print("✅ Module train chargé avec succès!")
'''

# =============================================================================
# 5. src/predict.py
# =============================================================================
predict_content = '''"""
Module de prédiction sur de nouveaux patients.
"""

import joblib
import numpy as np
import pandas as pd


def predict_patient(patient_data, feature_names, model_path='models/best_model.joblib', scaler_path='models/scaler.joblib'):
    """
    Prédit la présence de maladie pour un nouveau patient.
    
    Args:
        patient_data: dict avec les valeurs médicales
        feature_names: liste des noms de features
        model_path: chemin du modèle sauvegardé
        scaler_path: chemin du scaler sauvegardé
    
    Returns:
        tuple: (prédiction, probabilités)
    """
    # Chargement
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Préparation
    patient_df = pd.DataFrame([patient_data], columns=feature_names)
    patient_scaled = scaler.transform(patient_df)
    
    # Prédiction
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]
    
    # Affichage
    print("\\n" + "="*60)
    print("RÉSULTAT DE LA PRÉDICTION")
    print("="*60)
    print(f"  Âge: {patient_data['age']} ans")
    print(f"  Sexe: {'Homme' if patient_data['sex'] == 1 else 'Femme'}")
    print(f"  Cholestérol: {patient_data['chol']} mg/dl")
    print(f"  Pression artérielle: {patient_data['trestbps']} mm Hg")
    print(f"\\n  🔴 PRÉDICTION: {'MALADIE CARDIAQUE' if prediction == 1 else 'PAS DE MALADIE'}")
    print(f"  📊 Probabilité maladie: {probability[1]*100:.1f}%")
    print(f"  📊 Probabilité sain:    {probability[0]*100:.1f}%")
    print("="*60)
    
    return prediction, probability


if __name__ == "__main__":
    # Exemple de patient
    patient = {
        'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140,
        'chol': 240, 'fbs': 0, 'restecg': 0, 'thalach': 150,
        'exang': 1, 'oldpeak': 2.0, 'slope': 2, 'ca': 0, 'thal': 3
    }
    
    feature_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    print("✅ Module predict testé avec succès!")
    print("Utilisez predict_patient() après l'entraînement.")
'''

# =============================================================================
# 6. src/main.py (SCRIPT PRINCIPAL)
# =============================================================================
main_content = '''"""
=============================================================================
SCRIPT PRINCIPAL - Disease Prediction from Medical Data
=============================================================================
Pipeline complet: chargement -> prétraitement -> entraînement -> comparaison
=============================================================================
"""

import os
import sys
import warnings

# Suppression des warnings
warnings.filterwarnings('ignore')

# Ajout du dossier src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_heart_disease_data, clean_data, preprocess_data
from models import get_models
from train import train_and_evaluate, compare_models, save_best_model
from predict import predict_patient


def main():
    """
    Pipeline complet du projet Disease Prediction.
    """
    print("\\n" + "="*60)
    print("DISEASE PREDICTION FROM MEDICAL DATA")
    print("="*60)
    
    # Création des dossiers
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # --- 1. CHARGEMENT ET PRÉTRAITEMENT ---
    df = load_heart_disease_data()
    df = clean_data(df)
    data = preprocess_data(df)
    
    # --- 2. CONSTRUCTION DES MODÈLES ---
    models = get_models()
    
    # --- 3. ENTRAÎNEMENT ET ÉVALUATION ---
    results = {}
    roc_data = {}
    trained_models = {}
    
    for name, model in models.items():
        metrics, y_pred_proba, trained_model = train_and_evaluate(
            model, data['X_train'], data['X_test'],
            data['y_train'], data['y_test'], name
        )
        results[name] = metrics
        roc_data[name] = y_pred_proba
        trained_models[name] = trained_model
    
    # --- 4. COMPARAISON ---
    comparison_df = compare_models(results, roc_data, data['y_test'])
    
    # --- 5. SAUVEGARDE DU MEILLEUR MODÈLE ---
    best_name, best_model = save_best_model(trained_models, comparison_df)
    
    # --- 6. TEST SUR UN NOUVEAU PATIENT ---
    print("\\n" + "="*60)
    print("TEST SUR UN NOUVEAU PATIENT")
    print("="*60)
    
    nouveau_patient = {
        'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140,
        'chol': 240, 'fbs': 0, 'restecg': 0, 'thalach': 150,
        'exang': 1, 'oldpeak': 2.0, 'slope': 2, 'ca': 0, 'thal': 3
    }
    
    predict_patient(nouveau_patient, data['feature_names'])
    
    # --- 7. RÉCAPITULATIF ---
    print("\\n" + "="*60)
    print("PROJET TERMINÉ AVEC SUCCÈS !")
    print("="*60)
    print("Fichiers générés:")
    print("  ✅ models/best_model.joblib    (meilleur modèle)")
    print("  ✅ models/scaler.joblib         (standardiseur)")
    print("  ✅ results/model_comparison.png  (comparaison visuelle)")
    print("="*60)
    
    return best_name, best_model


if __name__ == "__main__":
    best_name, best_model = main()
'''

# =============================================================================
# 7. requirements.txt
# =============================================================================
requirements_content = '''pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==2.0.0
joblib==1.3.2
jupyter==1.0.0
'''

# =============================================================================
# 8. .gitignore
# =============================================================================
gitignore_content = '''__pycache__/
*.pyc
*.pyo
venv_disease/
.env
.ipynb_checkpoints/
*.joblib
data/raw/
data/processed/
'''

# =============================================================================
# GÉNÉRATION DES FICHIERS
# =============================================================================

files_to_create = {
    'src/__init__.py': init_content,
    'src/data_loader.py': data_loader_content,
    'src/models.py': models_content,
    'src/train.py': train_content,
    'src/predict.py': predict_content,
    'src/main.py': main_content,
    'requirements.txt': requirements_content,
    '.gitignore': gitignore_content
}

print("\nCréation des fichiers:\n")

for filepath, content in files_to_create.items():
    full_path = os.path.join(PROJECT_ROOT, filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    size = os.path.getsize(full_path)
    print(f"  ✅ {filepath:<30} ({size:>6} bytes)")

print("\n" + "=" * 60)
print("PROJET CRÉÉ AVEC SUCCÈS !")
print("=" * 60)
print("\nProchaines étapes:")
print("  1. cd Stage_ML_Disease")
print("  2. python -m venv venv_disease")
print("  3. venv_disease\\Scripts\\activate")
print("  4. pip install -r requirements.txt")
print("  5. python src/main.py")
print("\nPour GitHub:")
print("  git init")
print("  git add .")
print('  git commit -m "Initial commit - Disease Prediction"')
print("  git remote add origin https://github.com/othmansebbar/disease-prediction.git")
print("  git push -u origin main")
print("=" * 60)