"""
Script simples para ver feature importance das novas features Form + μₖ
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import load_all_data
from feature_engineering import calculate_team_stats
from train_models import prepare_features_by_model

def main():
    print("="*80)
    print("FEATURE IMPORTANCE - RandomForest")
    print("="*80)
    
    # Carregar e processar
    print("\n1. Carregando dados...")
    df_all = load_all_data()
    df_train = df_all[df_all['Season'] <= 2014].copy()
    
    print("2. Calculando features...")
    df_train = calculate_team_stats(df_train)
    df_train_rf = prepare_features_by_model(df_train, 'RandomForest')
    
    X_train = df_train_rf.drop(['Result', 'Season'], axis=1)
    print(f"\n✓ {X_train.shape[0]} partidas, {X_train.shape[1]} features")
    
    # Carregar modelo
    print("\n3. Carregando modelo RandomForest...")
    data = joblib.load('models/trained_models.pkl')
    rf_model = data['models']['RandomForest']['model']
    
    # Se for CalibratedClassifierCV, pegar o modelo base
    if hasattr(rf_model, 'calibrated_classifiers_'):
        # Pegar o primeiro calibrated classifier
        rf_base = rf_model.calibrated_classifiers_[0].estimator
        print("  (Extraindo RandomForest de CalibratedClassifierCV)")
    else:
        rf_base = rf_model
    
    # Feature importance
    print("\n4. Calculando feature importance...")
    importances = rf_base.feature_importances_
    feature_names = X_train.columns
    
    # DataFrame ordenado
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Top 20
    print("\n" + "="*80)
    print("TOP 20 FEATURES")
    print("="*80)
    print(f"\n{'Rank':<6} {'Feature':<30} {'Importance':<12} {'Bar'}")
    print("-"*80)
    
    max_imp = importance_df['Importance'].max()
    for idx, row in importance_df.head(20).iterrows():
        bar_len = int(row['Importance'] / max_imp * 40)
        print(f"#{idx+1:<5} {row['Feature']:<30} {row['Importance']:.6f}  {'='*bar_len}")
    
    # Novas features
    print("\n" + "="*80)
    print("RANKING DAS NOVAS FEATURES (Form + μₖ)")
    print("="*80)
    
    new_features = ['form_diff', 'corners_diff', 'shotsontarget_diff', 'shots_diff', 'goals_avg_diff']
    
    print(f"\n{'Feature':<30} {'Rank':<10} {'Importance':<12} {'% do Top 1'}")
    print("-"*80)
    
    for feat in new_features:
        if feat in feature_names.tolist():
            feat_row = importance_df[importance_df['Feature'] == feat]
            rank = feat_row.index[0] + 1
            importance = feat_row['Importance'].values[0]
            pct_of_top = importance / max_imp * 100
            print(f"{feat:<30} #{rank:<9} {importance:.6f}   {pct_of_top:5.1f}%")
        else:
            print(f"{feat:<30} NÃO ENCONTRADA")
    
    # Estatísticas das novas features
    print("\n" + "="*80)
    print("ESTATÍSTICAS DAS NOVAS FEATURES")
    print("="*80)
    
    print(f"\n{'Feature':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Unique'}")
    print("-"*100)
    
    for feat in new_features:
        if feat in X_train.columns:
            values = X_train[feat]
            print(f"{feat:<30} {values.mean():11.4f} {values.std():11.4f} {values.min():11.4f} {values.max():11.4f} {values.nunique():6d}")
    
    # Correlação entre novas features
    print("\n" + "="*80)
    print("CORRELAÇÃO DAS NOVAS FEATURES COM TOP 10")
    print("="*80)
    
    top_10_features = importance_df.head(10)['Feature'].tolist()
    
    corr_matrix = X_train.corr()
    
    for new_feat in new_features:
        if new_feat in X_train.columns:
            print(f"\n{new_feat}:")
            correlations = corr_matrix[new_feat][top_10_features].sort_values(ascending=False, key=abs)
            for feat, corr in correlations.items():
                if feat != new_feat:
                    print(f"  {feat:<30}: {corr:7.3f}")

if __name__ == "__main__":
    main()
