# 🏥 Disease Prediction from Medical Data

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-90%25-success)

## 🎯 Description

Système de prédiction de maladies cardiovasculaires utilisant le **Machine Learning** avec Scikit-learn. Le projet compare 5 algorithmes de classification pour identifier le meilleur modèle prédictif.

**Dataset**: Heart Disease (UCI Machine Learning Repository) - 303 patients, 13 features médicales.

## 🏥 Importance Médicale

Le Machine Learning révolutionne la médecine moderne :
- **Dépistage précoce** des maladies cardiovasculaires
- **Aide au diagnostic** pour les professionnels de santé
- **Réduction des coûts** en évitant les tests invasifs inutiles

## 🤖 Modèles Comparés

| Modèle | Accuracy | ROC-AUC | Forces |
|--------|----------|---------|--------|
| **XGBoost** | ~90% | ~92% | State-of-the-art |
| **Random Forest** | ~89% | ~91% | Robustesse |
| **SVM** | ~87% | ~90% | Haute dimension |
| **Logistic Regression** | ~85% | ~88% | Interprétabilité |
| **Decision Tree** | ~82% | ~85% | Visualisation |

## 🛠️ Technologies

| Technologie | Usage |
|-------------|-------|
| **Pandas/NumPy** | Manipulation de données médicales |
| **Scikit-learn** | Modèles de classification et évaluation |
| **XGBoost** | Gradient boosting avancé |
| **Matplotlib/Seaborn** | Visualisation des résultats |

## 📦 Installation

```bash
git clone https://github.com/othmansebbar/disease-prediction.git
cd disease-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt