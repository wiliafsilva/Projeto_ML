
import streamlit as st
import joblib
import pandas as pd
from src.analysis import (
    prepare_evaluation_data, evaluate_model,
    plot_confusion_matrix, plot_roc, plot_precision_recall,
    plot_calibration_curve, plot_feature_importance
)

st.set_page_config(layout="wide")

st.title("Scientific Replica – EPL Prediction")

models = joblib.load("models/trained_models.pkl")

page = st.sidebar.selectbox("Navigation", ["Model Comparison", "Evaluation & Metrics"]) 

if page == "Model Comparison":
    data = []
    for name, info in models.items():
        data.append([name, info['accuracy'], info['f1'], info['rps']])
    df = pd.DataFrame(data, columns=["Model","Accuracy","F1","RPS"])
    st.dataframe(df)

if page == "Evaluation & Metrics":
    st.header("Model Evaluation & Visualizations")
    X_test, y_test = prepare_evaluation_data("data/epl.csv")
    model_names = list(models.keys())
    sel = st.selectbox("Select model", model_names)
    info = models[sel]
    st.subheader(f"Metrics for {sel}")
    ev = evaluate_model(info['model'], X_test, y_test)
    # show classification report
    cr = ev['classification_report']
    st.write(pd.DataFrame(cr).transpose())
    # show distribution of true and predicted labels to help debug 'no predicted samples' issues
    st.write("True label distribution (H,D,A):", ev.get('y_test_counts'))
    st.write("Predicted label distribution (H,D,A):", ev.get('preds_counts'))
    st.write("Brier score:", ev['brier_score'])
    if ev['roc_auc'] is not None:
        st.write("ROC AUC (macro):", ev['roc_auc'])

    # plots
    st.subheader("Confusion Matrix")
    fig_cm = plot_confusion_matrix(ev['confusion_matrix'])
    st.pyplot(fig_cm)

    st.subheader("ROC Curves")
    fig_roc = plot_roc(ev['probs'], y_test)
    st.pyplot(fig_roc)

    st.subheader("Precision-Recall")
    fig_pr = plot_precision_recall(ev['probs'], y_test)
    st.pyplot(fig_pr)

    st.subheader("Calibration")
    fig_cal = plot_calibration_curve(ev['probs'], y_test)
    st.pyplot(fig_cal)

    st.subheader("Feature Importance (if available)")
    feat_plot = plot_feature_importance(info['model'], X_test.columns)
    if feat_plot is not None:
        st.pyplot(feat_plot)
    else:
        st.write("Feature importance not available for this model.")
