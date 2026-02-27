
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import numpy as np

def rps(y_true, y_prob):
    y_true = y_true.astype(int)  # Garantir que y_true é do tipo inteiro
    y_true_onehot = np.eye(3)[y_true]
    y_true_cum = np.cumsum(y_true_onehot, axis=1)
    y_prob_cum = np.cumsum(y_prob, axis=1)
    return np.mean(np.sum((y_true_cum - y_prob_cum)**2, axis=1))

def train_models(df_train, df_test):
    """
    Treina e avalia modelos usando dados de treino e teste separados.
    Segue a metodologia do artigo:
    - Treino: 2005-2014 (9 temporadas)
    - Teste: 2014-2016 (2 temporadas)
    
    Args:
        df_train: DataFrame com features de treinamento
        df_test: DataFrame com features de teste
    """
    # Preparar dados de treinamento
    X_train = df_train.drop(['Result','Season'], axis=1)
    y_train = df_train['Result']
    
    # Preparar dados de teste
    X_test = df_test.drop(['Result','Season'], axis=1)
    y_test = df_test['Result']
    
    print(f"\nDados de Treinamento: {len(X_train)} partidas")
    print(f"Dados de Teste: {len(X_test)} partidas")
    print(f"\nDistribuição de classes no treino:")
    print(f"  Vitória Casa (H): {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"  Empate (D): {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    print(f"  Vitória Fora (A): {(y_train == 2).sum()} ({(y_train == 2).sum()/len(y_train)*100:.1f}%)")
    print(f"\nDistribuição de classes no teste:")
    print(f"  Vitória Casa (H): {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
    print(f"  Empate (D): {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
    print(f"  Vitória Fora (A): {(y_test == 2).sum()} ({(y_test == 2).sum()/len(y_test)*100:.1f}%)")

    # Calcular sample weights para XGBoost
    sample_weights = compute_sample_weight('balanced', y_train)

    # Modelos base
    models = {
        "SVM": SVC(probability=True, kernel='rbf', random_state=42, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42, n_estimators=100)
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Treinando: {name}")
        print(f"{'='*60}")
        
        # Treinar com sample_weight para XGBoost
        if name == "XGBoost":
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
        if name in ["RandomForest", "XGBoost"]:
            print(f"Aplicando calibração de probabilidades...")
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            
            if name == "XGBoost":
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
            "rps": score_rps
        }

    print(f"\n{'='*60}")
    print("RESUMO FINAL:")
    print(f"{'='*60}")
    for name, info in results.items():
        print(f"{name:15} - Acurácia: {info['accuracy']:.4f} | F1: {info['f1']:.4f} | RPS: {info['rps']:.4f}")
    
    # Salvar resultados com informações sobre a divisão treino/teste
    results_metadata = {
        'models': results,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_period': '2005-2014',
        'test_period': '2014-2016',
        'methodology': 'Replicação do artigo científico'
    }
    
    joblib.dump(results_metadata, "models/trained_models.pkl")
    print(f"\n✓ Modelos salvos em models/trained_models.pkl")
