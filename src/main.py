"""
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
    print("\n" + "="*60)
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
    print("\n" + "="*60)
    print("TEST SUR UN NOUVEAU PATIENT")
    print("="*60)
    
    nouveau_patient = {
        'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140,
        'chol': 240, 'fbs': 0, 'restecg': 0, 'thalach': 150,
        'exang': 1, 'oldpeak': 2.0, 'slope': 2, 'ca': 0, 'thal': 3
    }
    
    predict_patient(nouveau_patient, data['feature_names'])
    
    # --- 7. RÉCAPITULATIF ---
    print("\n" + "="*60)
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
