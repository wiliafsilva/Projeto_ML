"""
Teste Rápido - Temporada 2023-2024
===================================

Treina modelos no período 2005-2014 e testa APENAS na temporada 2014-2015.

Autor: Projeto_ML
Data: Março 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocessing import load_all_data
from src.feature_engineering import calculate_team_stats
from src.train_models import prepare_features_by_model

def rps(y_true, y_prob):
    """Ranked Probability Score"""
    y_true = y_true.astype(int)
    y_true_onehot = np.eye(3)[y_true]
    y_true_cum = np.cumsum(y_true_onehot, axis=1)
    y_prob_cum = np.cumsum(y_prob, axis=1)
    k_minus_1 = y_prob.shape[1] - 1 if y_prob.shape[1] > 1 else 1
    return np.mean(np.sum((y_true_cum - y_prob_cum)**2, axis=1)) / k_minus_1

print("="*80)
print("TESTE RÁPIDO - TEMPORADA 2023-2024")
print("="*80)
print()
print("Metodologia:")
print("  - Treino: 2011-2023 (temporadas de treino)")
print("  - Teste: 2023-2024 (1 temporada)")
print("="*80)
print()

# Carregar dados
print("Carregando dados...")
df_all = load_all_data()
df_features = calculate_team_stats(df_all)

# Split treino/teste (novo split)
df_train = df_all[df_all['Season'] <= 2023].copy().reset_index(drop=True)
df_test_2023_2024 = df_all[df_all['Season'] == 2024].copy().reset_index(drop=True)

df_features_train = df_features[df_all['Season'] <= 2023].reset_index(drop=True)
df_features_test = df_features[df_all['Season'] == 2024].reset_index(drop=True)

print(f"   Treino: {len(df_train)} partidas")
print(f"   Teste (2023-2024): {len(df_test_2023_2024)} partidas")
print()

# Distribuição de classes
y_train_full = df_train['Result']
y_test = df_test_2023_2024['Result']

print("Distribuição de classes (2023-2024):")
print(f"  Vitória Casa (H): {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"  Empate (D): {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
print(f"  Vitória Fora (A): {(y_test == 2).sum()} ({(y_test == 2).sum()/len(y_test)*100:.1f}%)")
print()

# Calcular sample weights
sample_weights = compute_sample_weight('balanced', y_train_full)

# Modelos com hiperparâmetros otimizados
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'
    ),
    "XGBoost": XGBClassifier(
        eval_metric='mlogloss',
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42
    ),
    "NaiveBayes": GaussianNB(
        var_smoothing=1e-05
    ),
    "SVM": SVC(
        probability=True,
        kernel='rbf',
        C=0.1,
        gamma=0.001,
        random_state=42,
        class_weight='balanced'
    )
}

results = []

print("="*80)
print("TREINAMENTO E AVALIAÇÃO")
print("="*80)
print()

for name, model in models.items():
    print(f"{'='*60}")
    print(f"Modelo: {name}")
    print(f"{'='*60}")
    
    # Preparar features específicas
    df_train_model = prepare_features_by_model(df_features_train, name)
    df_test_model = prepare_features_by_model(df_features_test, name)
    
    X_train = df_train_model.drop(['Result', 'Season'], axis=1)
    y_train = df_train_model['Result']
    X_test = df_test_model.drop(['Result', 'Season'], axis=1)
    y_test_model = df_test_model['Result']
    
    print(f"Features: {X_train.shape[1]}")
    print(f"Treinando...")
    
    # Treinar
    if name in ["XGBoost", "NaiveBayes"]:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    
    # Predições
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    # Métricas
    acc = accuracy_score(y_test_model, preds)
    f1 = f1_score(y_test_model, preds, average='macro', zero_division=0)
    prec = precision_score(y_test_model, preds, average='macro', zero_division=0)
    rec = recall_score(y_test_model, preds, average='macro', zero_division=0)
    rps_score = rps(y_test_model.values, probs)
    
    print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   RPS:       {rps_score:.4f}")
    
    # Calibração (exceto SVM)
    if name in ["RandomForest", "XGBoost", "NaiveBayes"]:
        print(f"\n   Calibrando probabilidades...")
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        
        if name in ["XGBoost", "NaiveBayes"]:
            calibrated_model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            calibrated_model.fit(X_train, y_train)
        
        probs_cal = calibrated_model.predict_proba(X_test)
        preds_cal = calibrated_model.predict(X_test)
        
        acc_cal = accuracy_score(y_test_model, preds_cal)
        f1_cal = f1_score(y_test_model, preds_cal, average='macro', zero_division=0)
        rps_cal = rps(y_test_model.values, probs_cal)
        
        print(f"   Calibrado - Accuracy: {acc_cal:.4f} | F1: {f1_cal:.4f} | RPS: {rps_cal:.4f}")
        
        if rps_cal < rps_score:
            print(f"   ✓ Calibração melhorou RPS!")
            acc, f1, prec, rec, rps_score = acc_cal, f1_cal, prec, rec, rps_cal
        else:
            print(f"   ✗ Mantendo modelo base.")
    
    print()
    
    results.append({
        'Modelo': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'RPS': rps_score
    })

# Resumo final
print("="*80)
print("RESUMO FINAL - TEMPORADA 2023-2024")
print("="*80)
print()

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print()

# Identificar melhor modelo
best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
best_rps = results_df.loc[results_df['RPS'].idxmin()]
best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]

print("🏆 DESTAQUES:")
print(f"   Melhor Accuracy: {best_accuracy['Modelo']} ({best_accuracy['Accuracy']*100:.2f}%)")
print(f"   Melhor RPS: {best_rps['Modelo']} ({best_rps['RPS']:.4f})")
print(f"   Melhor F1-Score: {best_f1['Modelo']} ({best_f1['F1-Score']:.4f})")
print()

# Baseline
from collections import Counter
baseline_pred = Counter(y_train_full).most_common(1)[0][0]
baseline_preds = np.full(len(y_test), baseline_pred)
baseline_acc = accuracy_score(y_test, baseline_preds)

print(f"📊 COMPARAÇÃO COM BASELINE:")
print(f"   Baseline (sempre prever classe majoritária): {baseline_acc*100:.2f}%")
print(f"   Melhor ML ({best_accuracy['Modelo']}): {best_accuracy['Accuracy']*100:.2f}%")
print(f"   Ganho: +{(best_accuracy['Accuracy'] - baseline_acc)*100:.2f} pontos percentuais")
print()

print("="*80)
print("✅ TESTE CONCLUÍDO!")
print("="*80)
