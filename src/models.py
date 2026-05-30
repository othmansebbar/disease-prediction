"""
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
    print("\nInitialisation des modèles...")
    
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
    print("\n✅ Module models testé avec succès!")
