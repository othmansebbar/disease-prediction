"""
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
    print(f"\n{'='*50}")
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
    
    print("\n  Métriques:")
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
    print("\n" + "="*60)
    print("COMPARAISON DES MODÈLES")
    print("="*60)
    
    # Tableau comparatif
    comparison_df = pd.DataFrame(results).T.round(4)
    print(f"\n{comparison_df}")
    
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
    
    print(f"\n✅ Comparaison sauvegardée: {save_path}")
    
    return comparison_df


def save_best_model(trained_models, comparison_df, metric='ROC-AUC'):
    """
    Identifie et sauvegarde le meilleur modèle.
    
    Returns:
        tuple: (nom du meilleur modèle, modèle entraîné)
    """
    best_name = comparison_df[metric].idxmax()
    best_model = trained_models[best_name]
    
    print(f"\n{'='*60}")
    print(f"MEILLEUR MODÈLE: {best_name}")
    print(f"{'='*60}")
    print(f"  {metric}: {comparison_df.loc[best_name, metric]:.4f}")
    
    joblib.dump(best_model, 'models/best_model.joblib')
    print(f"  ✅ Sauvegardé: models/best_model.joblib")
    
    return best_name, best_model


if __name__ == "__main__":
    print("✅ Module train chargé avec succès!")
