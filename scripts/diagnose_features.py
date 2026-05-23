import os
import sys

# garantir que o diretório `src` esteja no sys.path para imports
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import preprocessing
import train_models
from preprocessing import load_all_data
from train_models import prepare_features_by_model

import pandas as pd


def main():
    df_all = load_all_data()
    # features excluding Result and Season
    all_features = [c for c in df_all.columns if c not in ['Result','Season','Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR']]
    print(f"\nTotal de colunas no dataset (excluindo meta): {len(all_features)}")
    print(sorted(all_features))

    # Prepare for NaiveBayes and RandomForest via function
    df_nb = prepare_features_by_model(df_all, 'NaiveBayes')
    df_rf = prepare_features_by_model(df_all, 'RandomForest')

    cols_nb = [c for c in df_nb.columns if c not in ['Result','Season']]
    cols_rf = [c for c in df_rf.columns if c not in ['Result','Season']]

    print(f"\nNaiveBayes - features usadas: {len(cols_nb)}")
    print(cols_nb)
    print(f"\nRandomForest (Class B) - features usadas: {len(cols_rf)}")
    print(cols_rf)

    # Features expected by code (from prepare_features_by_model source)
    # We'll derive expected lists by calling with model name and comparing to df_all
    expected_nb = cols_nb  # already filtered
    expected_rf = cols_rf

    missing_in_nb = [f for f in expected_nb if f not in df_all.columns]
    missing_in_rf = [f for f in expected_rf if f not in df_all.columns]

    print(f"\nMissing in df for NaiveBayes (should be none): {missing_in_nb}")
    print(f"Missing in df for RandomForest (should be none): {missing_in_rf}")

    # List of features present in df_all but not used by either model
    used = set(cols_nb) | set(cols_rf)
    unused = [f for f in all_features if f not in used]
    print(f"\nFeatures presentes mas não usadas por NB/RF ({len(unused)}):")
    print(sorted(unused))

    # Check columns with all NaN or constant
    nan_cols = [c for c in all_features if df_all[c].isna().all()]
    const_cols = [c for c in all_features if df_all[c].nunique(dropna=True) <= 1]
    print(f"\nColunas com todos NaN: {nan_cols}")
    print(f"Colunas constantes (1 valor): {const_cols}")

if __name__ == '__main__':
    main()
