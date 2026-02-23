
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.analysis import (
    prepare_evaluation_data, evaluate_model,
    plot_confusion_matrix, plot_roc, plot_precision_recall,
    plot_calibration_curve, plot_feature_importance
)
from src.preprocessing import load_data
from src.feature_engineering import calculate_team_stats

st.set_page_config(layout="wide")

st.title("Scientific Replica – Previsão da EPL (Streamlit)")

@st.cache_data
def load_models():
    try:
        return joblib.load("models/trained_models.pkl")
    except Exception:
        return {}

models = load_models()

page = st.sidebar.selectbox("Navegação", [
    "Visão Geral",
    "Comparação de Modelos",
    "Avaliação e Métricas",
    "Ajuste de Hiperparâmetros",
    "Distribuições e Importância de Features"
])

if page == "Visão Geral":
    st.header("Conjunto de Dados e Atributos")
    df = load_data("data/epl.csv")
    st.write("Amostra do conjunto de dados:")
    st.dataframe(df.head())
    features = calculate_team_stats(df)
    st.write("Amostra das features geradas:")
    st.dataframe(features.head())
    st.write("Temporadas nos dados:", sorted(df['Season'].unique()))

if page == "Comparação de Modelos":
    st.header("Resumo dos Modelos Treinados")
    if not models:
        st.warning("Nenhum modelo treinado encontrado. Execute `main.py` para treinar e gerar `models/trained_models.pkl`.")
    else:
        data = []
        for name, info in models.items():
            data.append([name, info.get('accuracy'), info.get('f1'), info.get('rps')])
        dfm = pd.DataFrame(data, columns=["Model","Accuracy","F1","RPS"])
        st.dataframe(dfm.set_index('Model'))
        st.subheader("Gráfico de barras: acurácia média de teste")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x='Model', y='Accuracy', data=dfm, ax=ax)
        ax.set_ylim(0,1)
        st.pyplot(fig)

if page == "Avaliação e Métricas":
    st.header("Avaliação do Modelo e Visualizações")
    X_test, y_test = prepare_evaluation_data("data/epl.csv")
    if not models:
        st.warning("Nenhum modelo treinado encontrado. Execute `main.py` para treinar os modelos primeiro.")
    else:
        model_names = list(models.keys())
        sel = st.selectbox("Selecione o modelo", model_names)
        info = models[sel]
        st.subheader(f"Métricas para {sel}")
        ev = evaluate_model(info['model'], X_test, y_test)
        # show classification report
        cr = ev['classification_report']
        st.write(pd.DataFrame(cr).transpose())
        st.write("Distribuição real das classes (H,D,A):", ev.get('y_test_counts'))
        st.write("Distribuição prevista das classes (H,D,A):", ev.get('preds_counts'))
        st.write("Pontuação Brier:", ev['brier_score'])
        if ev['roc_auc'] is not None:
            st.write("ROC AUC (macro):", ev['roc_auc'])

        # plots
        st.subheader("Matriz de Confusão")
        fig_cm = plot_confusion_matrix(ev['confusion_matrix'])
        st.pyplot(fig_cm)

        st.subheader("Curvas ROC")
        fig_roc = plot_roc(ev['probs'], y_test)
        st.pyplot(fig_roc)

        st.subheader("Precisão-Recall")
        fig_pr = plot_precision_recall(ev['probs'], y_test)
        st.pyplot(fig_pr)

        st.subheader("Calibração")
        fig_cal = plot_calibration_curve(ev['probs'], y_test)
        st.pyplot(fig_cal)

        st.subheader("Importância das Features (se disponível)")
        feat_plot = plot_feature_importance(info['model'], X_test.columns)
        if feat_plot is not None:
            st.pyplot(feat_plot)
        else:
            st.write("Importância das features não disponível para este modelo.")

if page == "Ajuste de Hiperparâmetros":
    st.header("Ajuste rápido de hiperparâmetros (grades pequenas)")
    df = load_data("data/epl.csv")
    features = calculate_team_stats(df)
    train = features[features['Season'] <= 2018]
    X_train = train.drop(['Result','Season'], axis=1)
    y_train = train['Result']

    tune_model = st.selectbox("Escolha o modelo para ajustar", ["SVM","RandomForest","XGBoost"])
    if st.button("Executar ajuste"):
        with st.spinner("Executando busca em grade (pode levar um momento)..."):
            st.info("Ajuste iniciado — pode demorar. Verifique o terminal para logs do GridSearch.")
            if tune_model == 'SVM':
                param_grid = {'C':[0.5,1,2], 'gamma':[0.01,0.05,0.1]}
                gs = GridSearchCV(SVC(probability=True), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
            elif tune_model == 'RandomForest':
                param_grid = {'n_estimators':[50,100], 'max_depth':[5,10,None]}
                gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
            else:
                param_grid = {'learning_rate':[0.01,0.05], 'max_depth':[3,6]}
                gs = GridSearchCV(XGBClassifier(eval_metric='mlogloss', use_label_encoder=False), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
            try:
                gs.fit(X_train, y_train)
            except Exception as e:
                st.error("O ajuste falhou — verifique os logs para detalhes.")
                st.exception(e)
            else:
                st.write("Melhores parâmetros:", gs.best_params_)
                results = pd.DataFrame(gs.cv_results_)
                st.dataframe(results[['params','mean_test_score','std_test_score']])
            # small heatmap if 2 params
            if len(param_grid) == 2:
                p1, p2 = list(param_grid.keys())
                # build pivot table
                results['p1'] = results['param_' + p1].astype(str)
                results['p2'] = results['param_' + p2].astype(str)
                pivot = results.pivot(index='p1', columns='p2', values='mean_test_score')
                fig, ax = plt.subplots()
                sns.heatmap(pivot, annot=True, fmt='.3f', ax=ax)
                ax.set_xlabel(p2)
                ax.set_ylabel(p1)
                st.pyplot(fig)

if page == "Distribuições e Importância de Features":
    st.header("Distribuições das features e importância")
    df = load_data("data/epl.csv")
    features = calculate_team_stats(df)
    st.subheader("Distribuições univariadas")
    cols = ['gd_diff','streak_diff','weighted_diff']
    for c in cols:
        fig, ax = plt.subplots(figsize=(6,3))
        sns.kdeplot(features[c].dropna(), shade=True, ax=ax)
        ax.set_title(c)
        st.pyplot(fig)

    st.subheader("Importância das features pelo RandomForest")
    if 'RandomForest' in models:
        rf = models['RandomForest']['model']
        fi_fig = plot_feature_importance(rf, features.drop(['Result','Season'],axis=1).columns)
        if fi_fig is not None:
            st.pyplot(fi_fig)
    else:
        st.write("Treine o RandomForest (execute main.py) para mostrar importâncias.")
