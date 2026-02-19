from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_recall_curve, auc, confusion_matrix)
import re
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

#########################
### NOTES : #############
# PB sur calcul de SHAP values (calculé sur 50 liens seulement)
# car PBs de compatibilité ac mon Mac
# plus optimal sur notebook pour l'instant

# On définit les constantes de features ici
TOPO_FEATURES = [
    'cn', 'aa', 'jc', 'pa', 'sp',
    'pr_u', 'pr_v', 'lcc_u', 'lcc_v',
    'and_u', 'and_v', 'dc_u', 'dc_v'
]

SCENARIOS = {
    "Position Uniquement": ['dist'],
    "SBM Uniquement": ['block_u', 'block_v', 'same_block'],
    "Topologie Uniquement": TOPO_FEATURES,
    "Hybride (Pos + SBM)": ['dist', 'block_u', 'block_v', 'same_block'],
    "Total (Pos + SBM + Topo)": ['dist', 'block_u', 'block_v', 'same_block'] + TOPO_FEATURES
}

def train_and_eval_xgboost(df, scenario="Total (Pos + SBM + Topo)", K=50):
    """Logique coeur de l'entraînement par scénarios"""
    all_stats = []
    scenario_model = None
    X_test_retained = None

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42,
        'learning_rate': 0.1,
        'max_depth': 6
    }

    for name, features in SCENARIOS.items():
        X = df[features]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            objective='binary:logistic',
            tree_method='hist', # Méthode plus stable sur beaucoup de systèmes
            n_jobs=1            # On force 1 seul thread pour éviter les conflits mémoire
        )
        
        model.fit(X_train, y_train)

        if name == scenario :
            scenario_model = model
            X_test_retained = X_test 
                
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        # Calcul des métriques (identique à ton code)
        auc_roc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        auc_pr = auc(recall, precision)
        
        actual_k = min(K, len(y_test))
        top_k_indices = np.argsort(probs)[-actual_k:]
        hits_at_k = np.sum(y_test.iloc[top_k_indices]) / actual_k
        
        all_stats.append({
            'Scenario': name,
            'AUC-ROC': auc_roc,
            'AP': ap,
            'AUC-PR': auc_pr,
            f'Hits@{actual_k}': hits_at_k
        })
        
    return pd.DataFrame(all_stats), scenario_model, X_test_retained

def analyze_with_shap(model, X_test, output_dir="outputs/plots"):
    """Calcule les SHAP values et génère les plots globaux proprement."""
    # 1. Configuration de l'explainer 'Boîte Noire' (le plus stable sur mon Mac)
    # On définit la fonction de prédiction (proba de la classe 1)
    model_predict = lambda x: model.predict_proba(x)[:, 1]
    
    # Utilisation d'un masker (échantillon de référence)
    # On prend 50 lignes pour équilibrer vitesse et précision
    masker = X_test.iloc[:50]
    
    # Initialisation de l'explainer
    explainer = shap.Explainer(model_predict, masker)    
    
    # 2. Calcul effectif des SHAP values
    # On récupère l'objet 'Explanation' complet
    shap_explanation = explainer(X_test)
    
    # 3. Extraction des valeurs numériques pour le retour de fonction
    # On récupère les valeurs brutes (.values)
    shap_values = shap_explanation.values

    # Gestion de la dimension (si SHAP renvoie [n_samples, n_features, 2])
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    
    # --- GÉNÉRATION DES PLOTS ---
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Summary Points (Beeswarm)
    plt.figure(figsize=(12, 8))
    # On peut passer l'objet explanation directement, c'est plus moderne
    shap.plots.beeswarm(shap_explanation, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_points.png"))
    plt.close()

    # Plot 2: Summary Bar
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_explanation, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"))
    plt.close()
    
    return shap_values

def calculate_feature_rankings(shap_values, feature_names, output_dir="outputs/plots"):
    """Calcule la distribution des rangs et génère le barplot du Top 5."""
    abs_shap = np.abs(shap_values)
    ranks = np.argsort(-abs_shap, axis=1)
    
    ranking_stats = {}
    n_samples, n_features = shap_values.shape

    for i, name in enumerate(feature_names):
        feature_ranks = np.where(ranks == i)[1] + 1
        counts = np.bincount(feature_ranks, minlength=n_features + 1)[1:]
        ranking_stats[name] = (counts / n_samples) * 100

    df_ranks = pd.DataFrame(ranking_stats, index=[f"Rang {i+1}" for i in range(n_features)])
    
    # Plot 3: Top 5 Appearance
    top5 = df_ranks.iloc[0:5, :].sum(axis=0).sort_values(ascending=False)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=top5.index, y=top5.values, palette="viridis")
    plt.title("Importance structurelle : % de présence dans le Top 5 SHAP")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_top5_frequency.png"))
    plt.close()
    
    return df_ranks