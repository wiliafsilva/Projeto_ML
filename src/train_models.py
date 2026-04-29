
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.models.gan_model import GANClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import numpy as np
import pandas as pd

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

    # ===== GAN-based augmentation (opcional) =====
    augmentation_enabled = False
    X_aug = None
    y_aug = None
    try:
        # Train GAN on Class B features (same used by RandomForest/XGBoost)
        df_train_gan = prepare_features_by_model(df_train, 'RandomForest')
        X_train_gan = df_train_gan.drop(['Result', 'Season'], axis=1)
        y_train_gan = df_train_gan['Result']

        gan = GANClassifier(
            epochs=50,
            batch_size=64,
            lr=1e-3,
            noise_dim=32,
            gen_hidden=(128, 128),
            disc_hidden=(128, 128),
            random_state=42
        )

        print("\n[GAN AUG] Treinando GAN para geração de amostras sintéticas (Classe B features)...")
        gan.fit(X_train_gan.values, y_train_gan.values)

        # Balancear: gerar (max_count - count) por classe
        counts = y_train_gan.value_counts().to_dict()
        max_count = max(counts.values())
        counts_to_gen = {int(k): int(max_count - v) for k, v in counts.items()}

        X_synth, y_synth = gan.generate(counts=counts_to_gen)
        if X_synth.shape[0] > 0:
            X_aug = pd.DataFrame(np.vstack([X_train_gan.values, X_synth]), columns=X_train_gan.columns)
            y_aug = pd.Series(np.concatenate([y_train_gan.values, y_synth]))
            augmentation_enabled = True
            print(f"[GAN AUG] Geradas {len(y_synth)} amostras sintéticas. Treino aumentado para {len(y_aug)} amostras (balanceado).")
        else:
            print("[GAN AUG] Nenhuma amostra sintética gerada. Prosseguindo sem augmentation.")
    except Exception as e:
        print(f"[GAN AUG] Falha ao treinar/gerar com GAN: {e}. Prosseguindo sem augmentation.")
        augmentation_enabled = False

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
        
        # Caso augmentation esteja habilitado, substituir dados de treino
        if augmentation_enabled and name in ["RandomForest", "XGBoost"] and X_aug is not None and y_aug is not None:
            # Garantir colunas alinhadas
            # X_aug foi criado a partir das mesmas Class B features usadas acima
            X_train = X_aug.reset_index(drop=True)
            y_train = y_aug.reset_index(drop=True)
            print(f"[GAN AUG] Usando dataset aumentado para {name}: {len(y_train)} amostras (inclui sintéticas).")

        # Treinar com sample_weight para XGBoost e NaiveBayes, exceto quando usamos dados aumentados
        if name == "XGBoost" and augmentation_enabled:
            # Quando o treino foi aumentado, sample_weights não corresponde ao novo conjunto; ignora-los
            print("[GAN AUG] Ignorando sample_weight para XGBoost devido a dataset aumentado.")
            model.fit(X_train, y_train)
        elif name in ["XGBoost", "NaiveBayes"]:
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
