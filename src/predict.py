"""
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
    print("\n" + "="*60)
    print("RÉSULTAT DE LA PRÉDICTION")
    print("="*60)
    print(f"  Âge: {patient_data['age']} ans")
    print(f"  Sexe: {'Homme' if patient_data['sex'] == 1 else 'Femme'}")
    print(f"  Cholestérol: {patient_data['chol']} mg/dl")
    print(f"  Pression artérielle: {patient_data['trestbps']} mm Hg")
    print(f"\n  🔴 PRÉDICTION: {'MALADIE CARDIAQUE' if prediction == 1 else 'PAS DE MALADIE'}")
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
