
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import numpy as np

def rps(y_true, y_prob):
    y_true_onehot = np.eye(3)[y_true]
    y_true_cum = np.cumsum(y_true_onehot, axis=1)
    y_prob_cum = np.cumsum(y_prob, axis=1)
    return np.mean(np.sum((y_true_cum - y_prob_cum)**2, axis=1))

def train_models(df):

    train = df[df['Season'] <= 2018]
    test = df[df['Season'] > 2018]

    X_train = train.drop(['Result','Season'], axis=1)
    y_train = train['Result']
    X_test = test.drop(['Result','Season'], axis=1)
    y_test = test['Result']

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
    
    joblib.dump(results, "models/trained_models.pkl")
    print(f"\n✓ Modelos salvos em models/trained_models.pkl")
