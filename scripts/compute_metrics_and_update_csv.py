import os
import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression

# Import preprocessing and feature engineering
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
import joblib

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(description="Compute metrics and update CSV")
    parser.add_argument("--model-path", default="models/trained_models.pkl")
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


args = parse_args()
output_dir = args.output_dir
if not os.path.isabs(output_dir):
    output_dir = os.path.join(BASE, output_dir)

model_path = args.model_path
if not os.path.isabs(model_path):
    model_path = os.path.join(BASE, model_path)
train_dir = os.path.join(BASE, 'data', 'data_2005_2014')
test_dir = os.path.join(BASE, 'data', 'data_2014_2016')

print('Carregando dados...')
df_train_raw = load_multiple_seasons(train_dir)
df_test_raw = load_multiple_seasons(test_dir)

print('Calculando features...')
features_train = calculate_team_stats(df_train_raw)
features_test = calculate_team_stats(df_test_raw)

# Models definitions (same as train_models)
models = {
    "SVM": SVC(probability=True, kernel='rbf', C=0.1, gamma=0.001, random_state=42, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', n_estimators=200, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=1.0, random_state=42),
    "NaiveBayes": GaussianNB(var_smoothing=1e-05),
}

# helper functions from train_models

def rps(y_true, y_prob):
    y_true = y_true.astype(int)
    y_true_onehot = np.eye(3)[y_true]
    y_true_cum = np.cumsum(y_true_onehot, axis=1)
    y_prob_cum = np.cumsum(y_prob, axis=1)
    k_minus_1 = y_prob.shape[1] - 1 if y_prob.shape[1] > 1 else 1
    return np.mean(np.sum((y_true_cum - y_prob_cum)**2, axis=1)) / k_minus_1

# train and compute probabilities
results = {}

# compute sample weights
# sample weights and training labels
sample_weights = compute_sample_weight('balanced', features_train['Result'])
y_train = features_train['Result']

# Nota: ensemble training/prediction (Voting/Stacking) não é realizado aqui.
# Se houver objetos de ensemble no arquivo models/trained_models.pkl, eles serão usados
# no momento da predição abaixo; caso contrário, o script pula ensembles.

# Now, compute metrics per season and update baseline_comparison.csv
csv_path = os.path.join(output_dir, 'baseline_comparison.csv')
if not os.path.exists(csv_path):
    print('CSV baseline_comparison.csv não encontrado. Saindo.')
    sys.exit(1)

df_baseline = pd.read_csv(csv_path)

# We'll add/overwrite columns Brier and ROC_AUC per row where applicable.
# The index has structure with season as first level in previous print; likely 'Temporada' index — but CSV index is first column.

# Build a map of (season, model) -> metrics
season_names = ['2014-2015', '2015-2016', 'All']
# Map season label in df to model names used in results keys
model_name_map = {
    'RandomForest': 'RandomForest',
    'XGBoost': 'XGBoost',
    'NaiveBayes': 'NaiveBayes',
    'SVM': 'SVM',
    'Voting_Equal': 'Voting_Equal',
    'Voting_Weighted': 'Voting_Weighted',
    'Stacking': 'Stacking'
}

# Helper to compute brier and roc for multiclass
from sklearn.preprocessing import label_binarize

for season in season_names:
    print(f'Processando temporada {season}...')
    # Filtrar linhas do CSV pela coluna 'Temporada'
    df_season_rows = df_baseline[df_baseline['Temporada'] == season]
    if df_season_rows.empty:
        print('  Nenhuma linha no CSV para', season)
        continue

    for i, row in df_season_rows.iterrows():
        model_label = str(row['Modelo'])
        # Tratar baselines explicitamente (Most Freq, Stratified, Always Draw)
        is_baseline = model_label.lower().startswith('baseline')

        if is_baseline:
            # usar labels do conjunto de teste para a temporada alvo
            if season == 'All':
                y_sel = features_test['Result']
            elif season == '2014-2015':
                y_sel = features_test[features_test['Season'] == 2015]['Result']
            else:
                y_sel = features_test[features_test['Season'] == 2016]['Result']

            from collections import Counter
            train_counts = Counter(y_train)
            total_train = sum(train_counts.values())
            # Probabilidades empíricas das classes (0,1,2)
            baseline_dist = np.array([train_counts.get(0, 0)/total_train,
                                      train_counts.get(1, 0)/total_train,
                                      train_counts.get(2, 0)/total_train])

            if 'most' in model_label.lower():
                # Prever sempre a classe majoritária do treino
                baseline_pred = int(train_counts.most_common(1)[0][0])
                preds_sel = np.full(len(y_sel), baseline_pred, dtype=int)
                probs_sel = np.tile(baseline_dist, (len(y_sel), 1))
            elif 'always draw' in model_label.lower() or 'always draw' in model_label:
                # Prever sempre empate (classe 1)
                preds_sel = np.full(len(y_sel), 1, dtype=int)
                onehot_draw = np.array([0.0, 1.0, 0.0])
                probs_sel = np.tile(onehot_draw, (len(y_sel), 1))
            else:
                # Stratified: amostrar segundo distribuição de treino (determinístico com seed)
                rng = np.random.RandomState(42)
                preds_sel = rng.choice([0,1,2], size=len(y_sel), p=baseline_dist)
                probs_sel = np.tile(baseline_dist, (len(y_sel), 1))

            y_true_vals = y_sel.values
        else:
            key = model_label
            # Determine X/y for this season and model
            from src.train_models import prepare_features_by_model
            if key in ['Voting_Equal', 'Voting_Weighted', 'Stacking']:
                df_season_model = prepare_features_by_model(features_test, 'RandomForest')
            else:
                df_season_model = prepare_features_by_model(features_test, key)

            if season == 'All':
                df_sel = df_season_model
            elif season == '2014-2015':
                df_sel = df_season_model[df_season_model['Season'] == 2015]
            else:
                df_sel = df_season_model[df_season_model['Season'] == 2016]

            X_sel = df_sel.drop(['Result', 'Season'], axis=1)
            y_sel = df_sel['Result']

            # Predict using trained model object (prefer trained_models.pkl)
            try:
                trained_models = {}
                try:
                    tm = joblib.load(model_path)
                    trained_models = tm.get('models', tm) if isinstance(tm, dict) else {}
                except Exception:
                    trained_models = {}

                if key == 'Voting_Equal' or key == 'Voting_Weighted' or key == 'Stacking':
                    # ensembles not available via trained_models.pkl in this script context
                    print(f'  Pulando ensemble {key} (não suportado aqui)')
                    continue
                else:
                    if key in trained_models:
                        model_obj = trained_models[key]['model']
                    elif key in results:
                        model_obj = results[key]['model']
                    else:
                        print('  Modelo', key, 'não encontrado nos resultados treinados')
                        continue

                probs_sel = model_obj.predict_proba(X_sel)
                preds_sel = model_obj.predict(X_sel)
                y_true_vals = y_sel.values
            except Exception as e:
                print('  Erro ao predizer para', key, season, e)
                continue

        # calcular Brier e ROC
        y_bin = label_binarize(y_true_vals, classes=[0,1,2])
        briers = []
        for c in range(y_bin.shape[1]):
            try:
                b = brier_score_loss(y_bin[:,c], probs_sel[:,c])
                briers.append(b)
            except Exception:
                briers.append(np.nan)
        brier_mean = np.nanmean(briers)

        try:
            roc = roc_auc_score(y_bin, probs_sel, average='macro', multi_class='ovr')
        except Exception:
            roc = np.nan
        
        # calcular RPS (já existe função rps definida acima)
        try:
            rps_val = rps(y_true_vals, probs_sel)
        except Exception:
            rps_val = np.nan

        # Add columns if necessary
        if 'Brier' not in df_baseline.columns:
            df_baseline['Brier'] = np.nan
        if 'ROC_AUC' not in df_baseline.columns:
            df_baseline['ROC_AUC'] = np.nan
        if 'RPS' not in df_baseline.columns:
            df_baseline['RPS'] = np.nan

        df_baseline.at[i, 'Brier'] = brier_mean
        df_baseline.at[i, 'ROC_AUC'] = roc
        df_baseline.at[i, 'RPS'] = rps_val
        print(f"  Atualizado {season} - {key}: Brier={brier_mean:.4f}, ROC_AUC={roc:.4f}")

# Save updated CSV
out_path = os.path.join(output_dir, 'baseline_comparison_with_metrics.csv')
df_baseline.to_csv(out_path)
print('CSV atualizado salvo em', out_path)
