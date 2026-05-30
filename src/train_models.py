
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def rps(y_true, y_prob):
    y_true = y_true.astype(int)  # Garantir que y_true é do tipo inteiro
    y_true_onehot = np.eye(3)[y_true]
    y_true_cum = np.cumsum(y_true_onehot, axis=1)
    y_prob_cum = np.cumsum(y_prob, axis=1)
    # Normalizar pelo número de categorias menos 1 (K-1) para ficar compatível
    # com a definição de RPS usada no artigo (valor entre 0 e 1).
    k_minus_1 = y_prob.shape[1] - 1 if y_prob.shape[1] > 1 else 1
    return np.mean(np.sum((y_true_cum - y_prob_cum)**2, axis=1)) / k_minus_1


def prepare_features_by_model(df, model_name):
    """
    Prepara features específicas de acordo com o modelo conforme artigo científico.
    
    Conforme Baboota & Kaur (2018):
    - Class A (valores individuais): Melhor para Naive Bayes devido à suposição 
      de independência condicional
    - Class B (diferenciais): Melhor para SVM, Random Forest, XGBoost
    
    Args:
        df: DataFrame com todas as features calculadas
        model_name: Nome do modelo ('NaiveBayes' ou outro)
    
    Returns:
        DataFrame com features apropriadas para o modelo
    """
    # Features Class B (diferenciais) - usadas por SVM, RF, XGBoost
    class_b_features = [
        # Baseline features (diferenciais)
        'gd_diff', 'streak_diff', 'weighted_diff',
        # Form differential
        'form_diff',
        # μₖ differentials
        'corners_diff', 'shotsontarget_diff', 'shots_diff', 'goals_avg_diff',
        # Ratings differentials (se existirem)
        'overall_diff', 'attack_diff', 'midfield_diff', 'defense_diff',
        # Position differentials
        'position_diff', 'points_diff',
        # Interaction features (DIA 9)
        'h2h_confidence', 'away_advantage', 'season_trend',
        'position_form_home', 'position_form_away', 'strength_balance',
        # Odds (não são diferenciais, mas são usadas por todos)
        'B365H', 'B365D', 'B365A',
        'prob_home', 'prob_draw', 'prob_away',
        'prob_home_norm', 'prob_draw_norm', 'prob_away_norm'
    ]
    
    # Features Class A (valores individuais) - usadas por Naive Bayes
    class_a_features = [
        # Form individuais
        'home_form', 'away_form',
        # Position individuais
        'home_position', 'away_position',
        'home_points', 'away_points',
        # Head-to-Head (não são diferenciais)
        'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
        'h2h_home_goals_avg', 'h2h_away_goals_avg', 'h2h_games',
        # Interaction features (DIA 9) - aplicáveis a ambos
        'h2h_confidence', 'away_advantage', 'season_trend',
        'position_form_home', 'position_form_away', 'strength_balance',
        # Odds (usadas por todos)
        'B365H', 'B365D', 'B365A',
        'prob_home', 'prob_draw', 'prob_away',
        'prob_home_norm', 'prob_draw_norm', 'prob_away_norm'
    ]
    
    # Selecionar features de acordo com o modelo
    if model_name == 'NaiveBayes':
        selected_features = class_a_features
        print(f"\n[Features] Usando Class A Features para {model_name}")
    else:
        selected_features = class_b_features
        print(f"\n[Features] Usando Class B Features para {model_name}")
    
    # Filtrar apenas features que existem no DataFrame
    available_features = [f for f in selected_features if f in df.columns]
    
    # Sempre incluir Result e Season
    feature_cols = available_features + ['Result', 'Season']
    
    # Verificar se alguma feature esperada está faltando
    missing_features = [f for f in selected_features if f not in df.columns and f not in ['Result', 'Season']]
    if missing_features:
        print(f"   ⚠️ Features ausentes: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
    
    print(f"   ✓ Features disponíveis: {len(available_features)}")
    print(f"     Exemplos: {available_features[:5]}")
    
    return df[feature_cols]


def train_models(df_train, df_test):
    """
    Treina e avalia modelos usando dados de treino e teste separados.
    Segue a metodologia do artigo:
    - Treino: 2005-2014 (9 temporadas)
    - Teste: 2014-2016 (2 temporadas)
    - Class A features para Naive Bayes (valores individuais)
    - Class B features para outros modelos (diferenciais)
    
    Args:
        df_train: DataFrame com features de treinamento
        df_test: DataFrame com features de teste
    """
    print(f"\n{'='*80}")
    print("SEPARAÇÃO DE FEATURES POR MODELO (Class A vs Class B)")
    print(f"{'='*80}")
    print("\nConforme artigo Baboota & Kaur (2018):")
    print("  - Class A (valores individuais): Naive Bayes")
    print("  - Class B (diferenciais): SVM, Random Forest, XGBoost")
    print(f"{'='*80}")
    
    # Preparar dados gerais (vamos filtrar por modelo depois)
    # Primeiro, vamos ver o que temos disponível
    print(f"\nDataset completo: {df_train.shape[1]-2} features (+ Result + Season)")
    print(f"Total features disponíveis: {sorted([c for c in df_train.columns if c not in ['Result', 'Season']])}")
    
    print(f"\nDados de Treinamento: {len(df_train)} partidas")
    print(f"Dados de Teste: {len(df_test)} partidas")
    
    # Pegar labels (comuns para todos os modelos)
    y_train_full = df_train['Result']
    y_test_full = df_test['Result']
    
    print(f"\nDistribuição de classes no treino:")
    print(f"  Vitória Casa (H): {(y_train_full == 0).sum()} ({(y_train_full == 0).sum()/len(y_train_full)*100:.1f}%)")
    print(f"  Empate (D): {(y_train_full == 1).sum()} ({(y_train_full == 1).sum()/len(y_train_full)*100:.1f}%)")
    print(f"  Vitória Fora (A): {(y_train_full == 2).sum()} ({(y_train_full == 2).sum()/len(y_train_full)*100:.1f}%)")
    print(f"\nDistribuição de classes no teste:")
    print(f"  Vitória Casa (H): {(y_test_full == 0).sum()} ({(y_test_full == 0).sum()/len(y_test_full)*100:.1f}%)")
    print(f"  Empate (D): {(y_test_full == 1).sum()} ({(y_test_full == 1).sum()/len(y_test_full)*100:.1f}%)")
    print(f"  Vitória Fora (A): {(y_test_full == 2).sum()} ({(y_test_full == 2).sum()/len(y_test_full)*100:.1f}%)")

    # Calcular sample weights para XGBoost e NaiveBayes
    sample_weights = compute_sample_weight('balanced', y_train_full)

    # Modelos com hiperparâmetros otimizados (DIA 5 + DIA 10 validação)
    # DIA 10: Validado com 43 features (Form + μₖ) → XGBoost RPS 0.4115 (melhor do projeto!)
    models = {
        "SVM": SVC(
            probability=True, 
            kernel='rbf',
            C=0.1,               # DIA 5: Otimizado via GridSearch
            gamma=0.001,         # DIA 5: Otimizado via GridSearch
            random_state=42, 
            class_weight='balanced'
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=50,          # DIA 5: Otimizado via GridSearch
            max_depth=5,              # DIA 5: Otimizado via GridSearch
            min_samples_split=2,      # DIA 5: Otimizado via GridSearch
            min_samples_leaf=1,       # DIA 5: Otimizado via GridSearch
            random_state=42, 
            class_weight='balanced'
        ),
        "XGBoost": XGBClassifier(
            eval_metric='mlogloss',
            n_estimators=200,         # DIA 5: Otimizado via GridSearch
            max_depth=3,              # DIA 5: Otimizado via GridSearch
            learning_rate=0.01,       # DIA 5: Otimizado via GridSearch
            subsample=0.8,            # DIA 5: Otimizado via GridSearch
            colsample_bytree=1.0,     # DIA 5: Otimizado via GridSearch
            random_state=42
        ),
        "NaiveBayes": GaussianNB(
            var_smoothing=1e-05       # DIA 5: Otimizado via GridSearch
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Treinando: {name}")
        print(f"{'='*60}")
        
        # Preparar features específicas para este modelo (Class A vs Class B)
        df_train_model = prepare_features_by_model(df_train, name)
        df_test_model = prepare_features_by_model(df_test, name)
        
        # Separar features e labels
        X_train = df_train_model.drop(['Result', 'Season'], axis=1)
        y_train = df_train_model['Result']
        X_test = df_test_model.drop(['Result', 'Season'], axis=1)
        y_test = df_test_model['Result']
        
        print(f"Features para treino: {X_train.shape[1]}")
        print(f"Amostras treino: {X_train.shape[0]}, Amostras teste: {X_test.shape[0]}")
        
        # Treinar com sample_weight para XGBoost e NaiveBayes
        if name in ["XGBoost", "NaiveBayes"]:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        score_rps = rps(y_test.values, probs)
        
        print(f"Modelo Base - Acurácia: {acc:.4f} | F1: {f1:.4f} | RPS: {score_rps:.4f}")

        # Calibrar probabilidades (exceto SVM que já tem boa calibração)
        if name in ["RandomForest", "XGBoost", "NaiveBayes"]:
            print(f"Aplicando calibração de probabilidades...")
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            
            if name in ["XGBoost", "NaiveBayes"]:
                calibrated_model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                calibrated_model.fit(X_train, y_train)
            
            probs_cal = calibrated_model.predict_proba(X_test)
            preds_cal = calibrated_model.predict(X_test)
            
            acc_cal = accuracy_score(y_test, preds_cal)
            f1_cal = f1_score(y_test, preds_cal, average='macro', zero_division=0)
            score_rps_cal = rps(y_test.values, probs_cal)
            
            print(f"Modelo Calibrado - Acurácia: {acc_cal:.4f} | F1: {f1_cal:.4f} | RPS: {score_rps_cal:.4f}")
            
            # Usar modelo calibrado se melhorar RPS
            if score_rps_cal < score_rps:
                print(f"✓ Calibração melhorou RPS em {(score_rps - score_rps_cal):.4f}! Usando modelo calibrado.")
                model = calibrated_model
                probs = probs_cal
                preds = preds_cal
                acc = acc_cal
                f1 = f1_cal
                score_rps = score_rps_cal
            else:
                print(f"✗ Calibração não melhorou RPS. Mantendo modelo base.")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "rps": score_rps,
            "feature_columns": list(X_train.columns)  # Salvar colunas usadas
        }
    
    # ======== ENSEMBLE METHODS (DIA 7) ========
    print(f"\n{'='*80}")
    print("ENSEMBLE METHODS - COMBINANDO MODELOS FORTES")
    print(f"{'='*80}")
    print("\nCombinando os 3 melhores modelos: RandomForest, XGBoost, NaiveBayes")
    print("Estratégias: Voting (soft) e Stacking (meta-learner)")
    print(f"{'='*80}")
    
    # Pegar modelos já treinados (Class B features para RF e XGB)
    rf_model = results['RandomForest']['model']
    xgb_model = results['XGBoost']['model']
    nb_model = results['NaiveBayes']['model']
    
    # Preparar dados para ensemble (usar Class B features - mais features = melhor)
    df_train_ensemble = prepare_features_by_model(df_train, 'RandomForest')  # Class B
    df_test_ensemble = prepare_features_by_model(df_test, 'RandomForest')
    
    X_train_ens = df_train_ensemble.drop(['Result', 'Season'], axis=1)
    y_train_ens = df_train_ensemble['Result']
    X_test_ens = df_test_ensemble.drop(['Result', 'Season'], axis=1)
    y_test_ens = df_test_ensemble['Result']
    
    # ========== 1. VOTING CLASSIFIER (Soft Voting) ==========
    print(f"\n{'='*60}")
    print("1. VOTING CLASSIFIER (Soft Voting)")
    print(f"{'='*60}")
    print("Estratégia: Média das probabilidades preditas por cada modelo")
    
    # Criar novos modelos base (não usar os já treinados para evitar problemas)
    rf_base = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=4,
        random_state=42, class_weight='balanced'
    )
    xgb_base = XGBClassifier(
        eval_metric='mlogloss', n_estimators=50, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, random_state=42
    )
    nb_base = GaussianNB()
    
    # Ensemble com pesos iguais
    voting_equal = VotingClassifier(
        estimators=[
            ('rf', rf_base),
            ('xgb', xgb_base),
            ('nb', nb_base)
        ],
        voting='soft',  # Média das probabilidades
        weights=[1, 1, 1]  # Pesos iguais
    )
    
    print("\nTreinando Voting Classifier (pesos iguais)...")
    voting_equal.fit(X_train_ens, y_train_ens)
    
    preds_vote_eq = voting_equal.predict(X_test_ens)
    probs_vote_eq = voting_equal.predict_proba(X_test_ens)
    
    acc_vote_eq = accuracy_score(y_test_ens, preds_vote_eq)
    f1_vote_eq = f1_score(y_test_ens, preds_vote_eq, average='macro', zero_division=0)
    rps_vote_eq = rps(y_test_ens.values, probs_vote_eq)
    
    print(f"Voting (pesos iguais) - Acurácia: {acc_vote_eq:.4f} | F1: {f1_vote_eq:.4f} | RPS: {rps_vote_eq:.4f}")
    
    results['Voting_Equal'] = {
        "model": voting_equal,
        "accuracy": acc_vote_eq,
        "f1": f1_vote_eq,
        "rps": rps_vote_eq,
        "feature_columns": list(X_train_ens.columns)  # Class B features
    }
    
    # Voting com pesos otimizados (RF melhor que outros)
    voting_weighted = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=4,
                random_state=42, class_weight='balanced'
            )),
            ('xgb', XGBClassifier(
                eval_metric='mlogloss', n_estimators=50, max_depth=3, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, random_state=42
            )),
            ('nb', GaussianNB())
        ],
        voting='soft',
        weights=[0.4, 0.3, 0.3]  # RF recebe mais peso (melhor RPS individual)
    )
    
    print("\nTreinando Voting Classifier (pesos otimizados: RF=0.4, XGB=0.3, NB=0.3)...")
    voting_weighted.fit(X_train_ens, y_train_ens)
    
    preds_vote_wt = voting_weighted.predict(X_test_ens)
    probs_vote_wt = voting_weighted.predict_proba(X_test_ens)
    
    acc_vote_wt = accuracy_score(y_test_ens, preds_vote_wt)
    f1_vote_wt = f1_score(y_test_ens, preds_vote_wt, average='macro', zero_division=0)
    rps_vote_wt = rps(y_test_ens.values, probs_vote_wt)
    
    print(f"Voting (pesos RF=0.4) - Acurácia: {acc_vote_wt:.4f} | F1: {f1_vote_wt:.4f} | RPS: {rps_vote_wt:.4f}")
    
    results['Voting_Weighted'] = {
        "model": voting_weighted,
        "accuracy": acc_vote_wt,
        "f1": f1_vote_wt,
        "rps": rps_vote_wt,
        "feature_columns": list(X_train_ens.columns)  # Class B features
    }
    
    # ========== 2. STACKING CLASSIFIER ==========
    print(f"\n{'='*60}")
    print("2. STACKING CLASSIFIER (Meta-learner)")
    print(f"{'='*60}")
    print("Estratégia: Logistic Regression aprende a combinar predições dos modelos base")
    
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=4,
                random_state=42, class_weight='balanced'
            )),
            ('xgb', XGBClassifier(
                eval_metric='mlogloss', n_estimators=50, max_depth=3, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, random_state=42
            )),
            ('nb', GaussianNB())
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        cv=3,  # Cross-validation para gerar predições do nível base
        stack_method='predict_proba'  # Usar probabilidades
    )
    
    print("\nTreinando Stacking Classifier (meta-learner: Logistic Regression)...")
    stacking_clf.fit(X_train_ens, y_train_ens)
    
    preds_stack = stacking_clf.predict(X_test_ens)
    probs_stack = stacking_clf.predict_proba(X_test_ens)
    
    acc_stack = accuracy_score(y_test_ens, preds_stack)
    f1_stack = f1_score(y_test_ens, preds_stack, average='macro', zero_division=0)
    rps_stack = rps(y_test_ens.values, probs_stack)
    
    print(f"Stacking Classifier - Acurácia: {acc_stack:.4f} | F1: {f1_stack:.4f} | RPS: {rps_stack:.4f}")
    
    results['Stacking'] = {
        "model": stacking_clf,
        "accuracy": acc_stack,
        "f1": f1_stack,
        "rps": rps_stack,
        "feature_columns": list(X_train_ens.columns)  # Class B features
    }

    print(f"\n{'='*80}")
    print("RESUMO FINAL - RESULTADOS POR TEMPORADA")
    print(f"{'='*80}")
    
    # Temporadas de teste (df_test já é o DataFrame de features)
    seasons_info = [
        ('2014-2015', 2015),
        ('2015-2016', 2016),
        ('All', None)  # None = todas as temporadas
    ]
    
    # Estrutura para armazenar resultados por temporada
    seasonal_results = {}
    
    for season_name, season_value in seasons_info:
        print(f"\n{'='*60}")
        print(f"TEMPORADA: {season_name}")
        print(f"{'='*60}")
        
        # Filtrar dados da temporada
        if season_value is None:
            # Todas as temporadas de teste
            df_test_season = df_test
        else:
            # Filtrar temporada específica
            df_test_season = df_test[df_test['Season'] == season_value].reset_index(drop=True)
        
        print(f"Total de jogos: {len(df_test_season)}\n")
        print(f"{'Modelo':<20} {'Acurácia':>10} {'F1':>10} {'RPS':>10}")
        print("-" * 60)
        
        # Inicializar dicionário para esta temporada
        seasonal_results[season_name] = {}
        
        for name, info in results.items():
            # Preparar features específicas para o modelo
            if name in ['Voting_Equal', 'Voting_Weighted', 'Stacking']:
                # Ensembles usam Class B features
                df_season_model = prepare_features_by_model(df_test_season, 'RandomForest')
            else:
                df_season_model = prepare_features_by_model(df_test_season, name)
            
            X_season = df_season_model.drop(['Result', 'Season'], axis=1)
            y_season = df_season_model['Result']
            
            # Fazer predições
            model = info['model']
            preds_season = model.predict(X_season)
            probs_season = model.predict_proba(X_season)
            
            # Calcular métricas
            acc_season = accuracy_score(y_season, preds_season)
            f1_season = f1_score(y_season, preds_season, average='macro', zero_division=0)
            rps_season = rps(y_season.values, probs_season)
            
            # Calcular métricas adicionais
            prec_season = precision_score(y_season, preds_season, average='macro', zero_division=0)
            rec_season = recall_score(y_season, preds_season, average='macro', zero_division=0)
            
            # Salvar métricas desta temporada
            seasonal_results[season_name][name] = {
                'accuracy': acc_season,
                'precision': prec_season,
                'recall': rec_season,
                'f1': f1_season,
                'rps': rps_season,
                'n_samples': len(y_season)
            }
            
            print(f"{name:<20} {acc_season:>10.4f} {f1_season:>10.4f} {rps_season:>10.4f}")
    
    print(f"\n{'='*80}")
    
    # Salvar resultados com informações sobre a divisão treino/teste
    results_metadata = {
        'models': results,
        'seasonal_results': seasonal_results,  # NOVO: Resultados por temporada
        'train_size': len(df_train),
        'test_size': len(df_test),
        'train_period': '2005-2014',
        'test_period': '2014-2016',
        'test_seasons': {
            '2014-2015': len(df_test[df_test['Season'] == 2015]),
            '2015-2016': len(df_test[df_test['Season'] == 2016]),
            'All': len(df_test)
        },
        'methodology': 'Replicação do artigo científico - Resultados separados por temporada'
    }
    
    joblib.dump(results_metadata, "models/trained_models.pkl")
    print(f"\n✓ Modelos salvos em models/trained_models.pkl")
    print(f"✓ Resultados salvos para: 2014-2015, 2015-2016, All")


class AutoencoderLatent(Model):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(latent_dim, activation="relu"),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid"),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _prepare_autoencoder_features(df):
    if "Result" not in df.columns:
        raise ValueError("Result column not found in features")

    y = df["Result"].astype(int).values
    seasons = df["Season"].values if "Season" in df.columns else None
    X = df.drop(columns=[c for c in ["Result", "Season"] if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.select_dtypes(include=[np.number]).copy()
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    return X, y, seasons


def train_models_autoencoder(df_train, df_test, latent_dim=8, output_dir="models/autoencoder_latent"):
    os.makedirs(output_dir, exist_ok=True)

    X_train, y_train, _ = _prepare_autoencoder_features(df_train)
    X_test, y_test, seasons_test = _prepare_autoencoder_features(df_test)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.values.astype(np.float32))

    autoencoder = AutoencoderLatent(X_train_scaled.shape[1], latent_dim=latent_dim)
    autoencoder.compile(optimizer="adam", loss="mae")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        min_delta=1e-4,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    autoencoder.fit(
        X_train_scaled,
        X_train_scaled,
        epochs=200,
        batch_size=min(2048, len(X_train_scaled)),
        validation_data=(X_test_scaled, X_test_scaled),
        shuffle=True,
        verbose=1,
        callbacks=[early_stop, reduce_lr],
    )

    X_train_latent = autoencoder.encoder(X_train_scaled).numpy()
    X_test_latent = autoencoder.encoder(X_test_scaled).numpy()

    sample_weights = compute_sample_weight('balanced', y_train)

    models = {
        "SVM": SVC(
            probability=True,
            kernel='rbf',
            C=0.1,
            gamma=0.001,
            random_state=42,
            class_weight='balanced'
        ),
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
    }

    results = {}
    for name, model in models.items():
        if name in ["XGBoost", "NaiveBayes"]:
            model.fit(X_train_latent, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_latent, y_train)

        preds = model.predict(X_test_latent)
        probs = model.predict_proba(X_test_latent)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        score_rps = rps(y_test, probs)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "rps": score_rps,
            "feature_columns": [f"latent_{i}" for i in range(X_train_latent.shape[1])]
        }

    # Ensembles on latent space
    rf_base = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=4,
        random_state=42, class_weight='balanced'
    )
    xgb_base = XGBClassifier(
        eval_metric='mlogloss', n_estimators=50, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, random_state=42
    )
    nb_base = GaussianNB()

    voting_equal = VotingClassifier(
        estimators=[('rf', rf_base), ('xgb', xgb_base), ('nb', nb_base)],
        voting='soft',
        weights=[1, 1, 1]
    )
    voting_equal.fit(X_train_latent, y_train)
    preds_vote_eq = voting_equal.predict(X_test_latent)
    probs_vote_eq = voting_equal.predict_proba(X_test_latent)
    results['Voting_Equal'] = {
        "model": voting_equal,
        "accuracy": accuracy_score(y_test, preds_vote_eq),
        "f1": f1_score(y_test, preds_vote_eq, average='macro', zero_division=0),
        "rps": rps(y_test, probs_vote_eq),
        "feature_columns": [f"latent_{i}" for i in range(X_train_latent.shape[1])]
    }

    voting_weighted = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=4,
                random_state=42, class_weight='balanced'
            )),
            ('xgb', XGBClassifier(
                eval_metric='mlogloss', n_estimators=50, max_depth=3, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, random_state=42
            )),
            ('nb', GaussianNB())
        ],
        voting='soft',
        weights=[0.4, 0.3, 0.3]
    )
    voting_weighted.fit(X_train_latent, y_train)
    preds_vote_wt = voting_weighted.predict(X_test_latent)
    probs_vote_wt = voting_weighted.predict_proba(X_test_latent)
    results['Voting_Weighted'] = {
        "model": voting_weighted,
        "accuracy": accuracy_score(y_test, preds_vote_wt),
        "f1": f1_score(y_test, preds_vote_wt, average='macro', zero_division=0),
        "rps": rps(y_test, probs_vote_wt),
        "feature_columns": [f"latent_{i}" for i in range(X_train_latent.shape[1])]
    }

    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=4,
                random_state=42, class_weight='balanced'
            )),
            ('xgb', XGBClassifier(
                eval_metric='mlogloss', n_estimators=50, max_depth=3, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, random_state=42
            )),
            ('nb', GaussianNB())
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        cv=3,
        stack_method='predict_proba'
    )
    stacking_clf.fit(X_train_latent, y_train)
    preds_stack = stacking_clf.predict(X_test_latent)
    probs_stack = stacking_clf.predict_proba(X_test_latent)
    results['Stacking'] = {
        "model": stacking_clf,
        "accuracy": accuracy_score(y_test, preds_stack),
        "f1": f1_score(y_test, preds_stack, average='macro', zero_division=0),
        "rps": rps(y_test, probs_stack),
        "feature_columns": [f"latent_{i}" for i in range(X_train_latent.shape[1])]
    }

    seasonal_results = {}
    seasons_info = [('2014-2015', 2015), ('2015-2016', 2016), ('All', None)]
    for season_name, season_value in seasons_info:
        if season_value is None or seasons_test is None:
            mask = np.ones(len(y_test), dtype=bool)
        else:
            mask = seasons_test == season_value
        if mask is not None and not np.any(mask):
            continue

        seasonal_results[season_name] = {}
        for name, info in results.items():
            model = info['model']
            preds_season = model.predict(X_test_latent[mask])
            probs_season = model.predict_proba(X_test_latent[mask])
            acc_season = accuracy_score(y_test[mask], preds_season)
            f1_season = f1_score(y_test[mask], preds_season, average='macro', zero_division=0)
            rps_season = rps(y_test[mask], probs_season)
            prec_season = precision_score(y_test[mask], preds_season, average='macro', zero_division=0)
            rec_season = recall_score(y_test[mask], preds_season, average='macro', zero_division=0)

            seasonal_results[season_name][name] = {
                'accuracy': acc_season,
                'precision': prec_season,
                'recall': rec_season,
                'f1': f1_season,
                'rps': rps_season,
                'n_samples': int(mask.sum())
            }

    results_df = pd.DataFrame([
        {
            "model": name,
            "accuracy": info["accuracy"],
            "f1": info["f1"],
            "rps": info["rps"],
        } for name, info in results.items()
    ])
    results_df.to_csv(os.path.join(output_dir, "latent_model_results.csv"), index=False)

    season_rows = []
    for season_name, models_dict in seasonal_results.items():
        for model_name, metrics in models_dict.items():
            season_rows.append({
                "season": season_name,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "rps": metrics["rps"],
                "n_samples": metrics["n_samples"],
            })
    pd.DataFrame(season_rows).to_csv(
        os.path.join(output_dir, "latent_model_results_by_season.csv"),
        index=False,
    )

    results_metadata = {
        'models': results,
        'seasonal_results': seasonal_results,
        'train_size': len(df_train),
        'test_size': len(df_test),
        'train_period': '2005-2014',
        'test_period': '2014-2016',
        'latent_dim': latent_dim,
        'feature_count': int(X_train_latent.shape[1]),
        'methodology': 'Autoencoder latent features for all models'
    }
    joblib.dump(results_metadata, os.path.join(output_dir, "trained_models_latent.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    autoencoder.save(os.path.join(output_dir, "autoencoder.keras"))
    autoencoder.encoder.save(os.path.join(output_dir, "encoder.keras"))


def train_models_with_decoder_hybrid(df_train, df_test, latent_dim=8, anomaly_percentile=95, output_dir="models/autoencoder_decoder_hybrid"):
    """
    Pipeline Robusto com Decoder Hybrid (Opção 4):
    
    ETAPA 1: Detecta anomalias usando reconstruction error
    ETAPA 2: Filtra dados ruins (outliers)
    ETAPA 3: Cria features híbridas (latent + reconstructed + error)
    ETAPA 4: Treina classificadores com features híbridas
    
    Args:
        df_train: DataFrame com features de treinamento
        df_test: DataFrame com features de teste
        latent_dim: Dimensão do latent space (padrão: 8)
        anomaly_percentile: Percentil para threshold de anomalias (padrão: 95)
        output_dir: Diretório para salvar resultados
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("PIPELINE ROBUSTO COM DECODER HYBRID")
    print("="*80)
    
    # ========== ETAPA 1: Preparar dados ==========
    print("\n[ETAPA 1] PREPARAÇÃO DOS DADOS")
    print("-" * 80)
    
    X_train, y_train, seasons_train = _prepare_autoencoder_features(df_train)
    X_test, y_test, seasons_test = _prepare_autoencoder_features(df_test)
    
    print(f"Dados de treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
    print(f"Dados de teste: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.values.astype(np.float32))
    
    # ========== ETAPA 2: Treinar Autoencoder e Detectar Anomalias ==========
    print("\n[ETAPA 2] TREINAMENTO DO AUTOENCODER E DETECÇÃO DE ANOMALIAS")
    print("-" * 80)
    
    print(f"Treinando autoencoder (latent_dim={latent_dim})...")
    autoencoder = AutoencoderLatent(X_train_scaled.shape[1], latent_dim=latent_dim)
    autoencoder.compile(optimizer="adam", loss="mae")
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        min_delta=1e-4,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0,
    )
    
    autoencoder.fit(
        X_train_scaled,
        X_train_scaled,
        epochs=200,
        batch_size=min(2048, len(X_train_scaled)),
        validation_data=(X_test_scaled, X_test_scaled),
        shuffle=True,
        verbose=0,
        callbacks=[early_stop, reduce_lr],
    )
    
    print("✓ Autoencoder treinado")
    
    # Computar reconstruction errors no treino
    X_train_reconstructed = autoencoder(X_train_scaled).numpy()
    reconstruction_errors_train = np.mean(np.abs(X_train_scaled - X_train_reconstructed), axis=1)
    
    # Definir threshold para anomalias
    threshold = np.percentile(reconstruction_errors_train, anomaly_percentile)
    anomaly_mask = reconstruction_errors_train >= threshold
    clean_mask = reconstruction_errors_train < threshold
    
    n_anomalies = anomaly_mask.sum()
    anomaly_ratio = (n_anomalies / len(reconstruction_errors_train)) * 100
    
    print(f"\nAnálise de Anomalias (threshold = {threshold:.6f}):")
    print(f"  • Anomalias detectadas: {n_anomalies} ({anomaly_ratio:.2f}%)")
    print(f"  • Dados limpos: {clean_mask.sum()} ({100-anomaly_ratio:.2f}%)")
    print(f"  • Reconstruction error - Min: {reconstruction_errors_train.min():.6f}, "
          f"Max: {reconstruction_errors_train.max():.6f}, "
          f"Mean: {reconstruction_errors_train.mean():.6f}")
    
    # ========== ETAPA 3: Criar Features Híbridas ==========
    print("\n[ETAPA 3] CRIAÇÃO DE FEATURES HÍBRIDAS")
    print("-" * 80)
    
    # Para dados de TREINO (limpos)
    X_train_latent = autoencoder.encoder(X_train_scaled).numpy()
    X_train_reconstructed = autoencoder(X_train_scaled).numpy()
    X_train_reconstruction_error = np.mean(np.abs(X_train_scaled - X_train_reconstructed), axis=1, keepdims=True)
    
    # Features híbridas: [latent (8D) + reconstructed (43D) + error (1D)] = 52D
    X_train_hybrid = np.hstack([
        X_train_latent,
        X_train_reconstructed,
        X_train_reconstruction_error
    ])
    
    # Para dados de TESTE
    X_test_latent = autoencoder.encoder(X_test_scaled).numpy()
    X_test_reconstructed = autoencoder(X_test_scaled).numpy()
    X_test_reconstruction_error = np.mean(np.abs(X_test_scaled - X_test_reconstructed), axis=1, keepdims=True)
    
    X_test_hybrid = np.hstack([
        X_test_latent,
        X_test_reconstructed,
        X_test_reconstruction_error
    ])
    
    print(f"Features híbridas criadas:")
    print(f"  • Latent space: {latent_dim}D")
    print(f"  • Features reconstruídas: {X_train_reconstructed.shape[1]}D")
    print(f"  • Reconstruction error: 1D")
    print(f"  • Total: {X_train_hybrid.shape[1]}D (ao invés de {X_train_scaled.shape[1]}D originais)")
    print(f"\nDados de treino: {X_train_hybrid.shape[0]} → {clean_mask.sum()} (após limpeza)")
    print(f"Dados de teste: {X_test_hybrid.shape[0]}")
    
    # Usar apenas dados limpos para treino
    X_train_hybrid_clean = X_train_hybrid[clean_mask]
    y_train_clean = y_train[clean_mask]
    seasons_train_clean = seasons_train[clean_mask] if seasons_train is not None else None
    
    # ========== ETAPA 4: Treinar Classificadores ==========
    print("\n[ETAPA 4] TREINAMENTO DOS CLASSIFICADORES COM FEATURES HÍBRIDAS")
    print("-" * 80)
    
    sample_weights_clean = compute_sample_weight('balanced', y_train_clean)
    
    models = {
        "SVM": SVC(
            probability=True,
            kernel='rbf',
            C=0.1,
            gamma=0.001,
            random_state=42,
            class_weight='balanced'
        ),
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
    }
    
    results = {}
    
    print(f"\nTreinamento com {X_train_hybrid_clean.shape[0]} amostras limpas:\n")
    print(f"{'Modelo':<20} {'Acurácia':>12} {'F1-Score':>12} {'RPS':>12}")
    print("-" * 60)
    
    for name, model in models.items():
        # Treinar com dados limpos
        if name in ["XGBoost", "NaiveBayes"]:
            model.fit(X_train_hybrid_clean, y_train_clean, sample_weight=sample_weights_clean)
        else:
            model.fit(X_train_hybrid_clean, y_train_clean)
        
        # Validar em dados de teste completos
        preds = model.predict(X_test_hybrid)
        probs = model.predict_proba(X_test_hybrid)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        score_rps = rps(y_test, probs)
        
        print(f"{name:<20} {acc:>12.4f} {f1:>12.4f} {score_rps:>12.4f}")
        
        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "rps": score_rps,
            "feature_columns": [f"h{i}" for i in range(X_train_hybrid.shape[1])]  # hybrid features
        }
    
    # ========== ETAPA 5: Análise por Temporada ==========
    print("\n[ETAPA 5] RESULTADOS POR TEMPORADA")
    print("-" * 80)
    
    seasonal_results = {}
    seasons_info = [
        ('2014-2015', 2015),
        ('2015-2016', 2016),
        ('All', None)
    ]
    
    for season_name, season_value in seasons_info:
        seasonal_results[season_name] = {}
        
        if season_value is None:
            mask = np.ones(len(y_test), dtype=bool)
        else:
            mask = seasons_test == season_value
        
        print(f"\n{season_name}: {mask.sum()} jogos")
        
        for name, info in results.items():
            model = info['model']
            preds_season = model.predict(X_test_hybrid[mask])
            probs_season = model.predict_proba(X_test_hybrid[mask])
            
            acc_season = accuracy_score(y_test[mask], preds_season)
            f1_season = f1_score(y_test[mask], preds_season, average='macro', zero_division=0)
            rps_season = rps(y_test[mask], probs_season)
            
            seasonal_results[season_name][name] = {
                'accuracy': acc_season,
                'f1': f1_season,
                'rps': rps_season,
                'n_samples': mask.sum()
            }
    
    # ========== Salvar Resultados ==========
    print("\n" + "="*80)
    print("SALVANDO RESULTADOS")
    print("="*80)
    
    results_df = pd.DataFrame([
        {
            "model": name,
            "accuracy": info["accuracy"],
            "f1": info["f1"],
            "rps": info["rps"],
        } for name, info in results.items()
    ])
    results_df.to_csv(os.path.join(output_dir, "hybrid_model_results.csv"), index=False)
    print(f"✓ Resultados salvos em: hybrid_model_results.csv")
    
    season_rows = []
    for season_name, models_dict in seasonal_results.items():
        for model_name, metrics in models_dict.items():
            season_rows.append({
                "season": season_name,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "rps": metrics["rps"],
                "n_samples": metrics["n_samples"],
            })
    pd.DataFrame(season_rows).to_csv(
        os.path.join(output_dir, "hybrid_model_results_by_season.csv"),
        index=False,
    )
    print(f"✓ Resultados por temporada salvos em: hybrid_model_results_by_season.csv")
    
    results_metadata = {
        'models': results,
        'seasonal_results': seasonal_results,
        'train_size': len(X_train_scaled),
        'train_size_clean': clean_mask.sum(),
        'test_size': len(X_test_scaled),
        'train_period': '2005-2014',
        'test_period': '2014-2016',
        'latent_dim': latent_dim,
        'anomaly_percentile': anomaly_percentile,
        'anomalies_detected': int(n_anomalies),
        'anomaly_ratio': float(anomaly_ratio),
        'reconstruction_threshold': float(threshold),
        'hybrid_features_count': X_train_hybrid.shape[1],
        'methodology': 'Decoder Hybrid: Latent Space + Reconstructed Features + Error Signal'
    }
    
    joblib.dump(results_metadata, os.path.join(output_dir, "trained_models_hybrid.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler_hybrid.joblib"))
    autoencoder.save(os.path.join(output_dir, "autoencoder_hybrid.keras"))
    autoencoder.encoder.save(os.path.join(output_dir, "encoder_hybrid.keras"))
    autoencoder.decoder.save(os.path.join(output_dir, "decoder_hybrid.keras"))
    
    print(f"✓ Modelos e metadados salvos em: {output_dir}/")
    
    print("\n" + "="*80)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f"\nResumo:")
    print(f"  • Anomalias detectadas e removidas: {n_anomalies} ({anomaly_ratio:.2f}%)")
    print(f"  • Dados de treino limpos: {clean_mask.sum()} amostras")
    print(f"  • Features híbridas: {X_train_hybrid.shape[1]}D")
    print(f"  • Modelos treinados: {list(results.keys())}")
    print(f"  • Diretório de saída: {output_dir}/")
    
    return results, results_metadata
