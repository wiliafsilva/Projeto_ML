
# Scientific Replica – Premier League Match Prediction

This project replicates a scientific study predicting match outcomes in the English Premier League using Machine Learning.

## Methodology
- Temporal split (no random split)
- Incremental feature engineering (no data leakage)
- Comparison of 4 models:
    - GaussianNB
    - SVM (RBF)
    - Random Forest
    - XGBoost

## Features Implemented
- Cumulative Goal Difference
- Last 5 matches average goals
- Streak (last 5 matches)
- Weighted Streak
- Home vs Away feature differences

## Train/Test Split
Train: 1993–2018  
Test: 2019–2023

## Run
Create venv:
python -m venv .venv

Ativar venv
.\.venv\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt

Run training pipeline:
python main.py

Run interface:
& .\.venv\Scripts\Activate.ps1
streamlit run app.py
