
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
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

    # Modelos com suporte a class_weight
    models = {
        "SVM": SVC(probability=True, kernel='rbf', random_state=42, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        score_rps = rps(y_test.values, probs)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "rps": score_rps
        }

    joblib.dump(results, "models/trained_models.pkl")
