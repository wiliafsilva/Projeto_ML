import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from src.preprocessing import load_data
from src.feature_engineering import calculate_team_stats


def prepare_evaluation_data(path="data/epl.csv"):
    df = load_data(path)
    features = calculate_team_stats(df)
    train = features[features['Season'] <= 2018]
    test = features[features['Season'] > 2018]
    X_test = test.drop(['Result','Season'], axis=1)
    y_test = test['Result'].astype(int).values
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    y_bin = label_binarize(y_test, classes=[0,1,2])
    try:
        roc_auc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    except Exception:
        roc_auc = None

    try:
        avg_precision = average_precision_score(y_bin, probs, average='macro')
    except Exception:
        avg_precision = None

    # multiclass Brier score (mean squared error across classes)
    brier = np.mean(np.sum((y_bin - probs) ** 2, axis=1))

    # counts for debugging classes with no predictions
    y_test_counts = np.bincount(y_test)
    preds_counts = np.bincount(preds)

    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'brier_score': brier,
        'probs': probs,
        'preds': preds
        , 'y_test_counts': y_test_counts, 'preds_counts': preds_counts
    }


def plot_confusion_matrix(cm, labels=['H','D','A']):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_roc(probs, y_test):
    from sklearn.metrics import roc_curve, auc
    y_bin = label_binarize(y_test, classes=[0,1,2])
    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'classe {i} (AUC={roc_auc:.2f})')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curvas ROC')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_precision_recall(probs, y_test):
    from sklearn.metrics import precision_recall_curve
    y_bin = label_binarize(y_test, classes=[0,1,2])
    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(y_bin.shape[1]):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ax.plot(recall, precision, label=f'classe {i}')
    ax.set_xlabel('Revocação')
    ax.set_ylabel('Precisão')
    ax.set_title('Curvas Precisão-Revocação')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calibration_curve(probs, y_test, n_bins=10):
    # plot calibration for each class
    y_bin = label_binarize(y_test, classes=[0,1,2])
    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(y_bin.shape[1]):
        prob_pos = probs[:, i]
        frac_pos, mean_pred = calibration_curve(y_bin[:, i], prob_pos, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, marker='o', label=f'classe {i}')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('Valor médio predito')
    ax.set_ylabel('Fração de positivos')
    ax.set_title('Curvas de Calibração')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names):
    # prefer built-in importance from XGBoost/RandomForest
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None

    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(range(len(importances)), importances[idx])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(np.array(feature_names)[idx], rotation=90)
    ax.set_title('Importância das Features')
    fig.tight_layout()
    return fig
