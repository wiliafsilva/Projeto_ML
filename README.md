
# Scientific Replica – Premier League Match Prediction

This project replicates a scientific study predicting match outcomes in the English Premier League using Machine Learning.

## Methodology
- Temporal split (no random split)
- Incremental feature engineering (no data leakage)
- Comparison of 4 models:
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

## Instalação

### 1. Criar ambiente virtual:
```bash
python -m venv .venv
```

### 2. Ativar ambiente virtual:
```bash
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependências:
```bash
pip install -r requirements.txt
```

## Comandos para Executar

### Pipeline Principal

**Treinar modelos:**
```bash
python main.py
```

**Executar interface Streamlit:**
```bash
streamlit run app.py
```

### Scripts de Análise e Verificação

**Verificar todos os dados do projeto:**
```bash
python scripts\verify_all.py
```
Exibe estatísticas completas do dataset, features, modelos, divisão treino/teste e distribuição por temporada.

**Mostrar métricas dos modelos:**
```bash
python scripts\show_metrics.py
```
Mostra acurácia, F1 Score e RPS de todos os modelos treinados.

**Testar cálculo de features:**
```bash
python scripts\test_features.py
```
Testa o cálculo das features e mostra estatísticas detalhadas sobre valores zeros.

**Inspecionar dataset EPL:**
```bash
python scripts\inspect_epl.py
```
Analisa o dataset bruto: total de partidas, temporadas, gols, e distribuição de resultados.

**Debug de features:**
```bash
python scripts\debug_features.py
```
Script de debug para investigar problemas no cálculo de features.

**Análise SHAP (explicabilidade):**
```bash
python scripts\shap_analysis.py
```
Analisa a importância das features usando SHAP valores. Gera gráficos mostrando quais features têm maior impacto nas previsões.

**GridSearch avançado (otimização):**
```bash
python scripts\gridsearch_advanced.py
```
Otimiza hiperparâmetros dos modelos usando validação temporal. *Atenção: pode demorar bastante!*

## Estrutura do Projeto

```
Projeto_ML/
├── data/
│   └── epl.csv              # Dataset com 12,026 partidas da EPL (1993-2023)
├── models/
│   └── trained_models.pkl   # Modelos treinados (SVM, RandomForest, XGBoost)
├── scripts/
│   ├── verify_all.py        # Verificação completa do projeto
│   ├── show_metrics.py      # Métricas dos modelos
│   ├── test_features.py     # Teste de features
│   ├── inspect_epl.py       # Inspeção do dataset
│   ├── debug_features.py    # Debug de features
│   ├── shap_analysis.py     # Análise SHAP de explicabilidade
│   └── gridsearch_advanced.py  # Otimização de hiperparâmetros
├── src/
│   ├── preprocessing.py     # Carregamento e preparação dos dados
│   ├── feature_engineering.py  # Cálculo das features
│   ├── train_models.py      # Treinamento dos modelos
│   └── analysis.py          # Avaliação e visualizações
├── app.py                   # Interface Streamlit
├── main.py                  # Pipeline de treinamento
├── requirements.txt         # Dependências
└── README.md               # Este arquivo
```

## Resultados

**Métricas dos Modelos (Test Set 2019-2023):**
- **SVM:** Acurácia 46.53% | F1 0.4403 | RPS 0.4342 ⭐
- **XGBoost:** Acurácia 47.26% | F1 0.3760 | RPS 0.4522
- **RandomForest:** Acurácia 44.37% | F1 0.4048 | RPS 0.5203

**Sobre as Métricas:**
- **RPS (Ranked Probability Score):** Quanto menor, melhor. Mede a qualidade das probabilidades preditas.
- **O SVM** apresenta o melhor RPS, indicando melhor calibração de probabilidades.

## Melhorias Implementadas

### 1. Calibração de Probabilidades
Os modelos RandomForest e XGBoost agora usam **calibração isotônica** para melhorar as probabilidades preditas. A calibração é aplicada automaticamente durante o treinamento e o modelo calibrado é usado apenas se melhorar o RPS.

### 2. Balanceamento de Classes
- **SVM e RandomForest:** Usam `class_weight='balanced'`
- **XGBoost:** Usa `sample_weight` calculado para balancear as classes

### 3. Análise de Explicabilidade (SHAP)
Execute `python scripts\shap_analysis.py` para:
- Ver ranking de importância das features por modelo
- Gerar gráficos SHAP mostrando impacto de cada feature
- Entender quais características mais influenciam as previsões

### 4. Otimização de Hiperparâmetros
Execute `python scripts\gridsearch_advanced.py` para:
- Busca em grade com validação temporal (TimeSeriesSplit)
- Otimização focada em minimizar RPS
- Salva melhores parâmetros em `models/optimized_models.pkl`

## Sobre os Modelos

### SVM (Support Vector Machine)
- **Kernel RBF** para capturar relações não-lineares
- **Vantagens:** Boa calibração de probabilidades, eficaz em espaços de alta dimensão
- **Desvantagens:** Lento em grandes datasets, sensível à escala
- **Uso:** Melhor quando probabilidades calibradas são importantes

### RandomForest
- **Ensemble de árvores** (bagging)
- **Vantagens:** Robusto, fornece importâncias de features, pouco pré-processamento
- **Desvantagens:** Probabilidades podem não ser bem calibradas
- **Uso:** Bom para análise exploratória e interpretabilidade

### XGBoost
- **Gradient boosting otimizado**
- **Vantagens:** Alta acurácia, captura padrões complexos, rápido
- **Desvantagens:** Requer ajuste de hiperparâmetros, pode sobreajustar
- **Uso:** Melhor quando acurácia máxima é prioridade
