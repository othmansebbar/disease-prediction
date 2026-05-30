"""
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
    print("\nNettoyage des données...")
    
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
    print("\n" + "="*50)
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
    
    print(f"\n  Train: {X_train_scaled.shape[0]} échantillons")
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
    print("\n✅ Module data_loader testé avec succès!")
