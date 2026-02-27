
# Scientific Replica – Premier League Match Prediction

This project replicates a scientific study predicting match outcomes in the English Premier League using Machine Learning.

## Methodology
- Temporal split (no random split)
- Incremental feature engineering (no data leakage)
- Comparison of 3 models:
    - SVM (RBF)
    - Random Forest
    - XGBoost

## Features Implemented
- Cumulative Goal Difference
- Last 5 matches average goals
- Streak (last 5 matches)
- Weighted Streak
- Home vs Away feature differences

## Train/Test Split (Conforme Artigo Científico)
**Train:** 2005–2014 (9 temporadas)  
**Test:** 2014–2016 (2 temporadas)

Esta divisão segue exatamente a metodologia descrita no artigo científico *"Predictive analysis and modelling football results using"*.

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

## 🚀 Sequência de Execução Completa

### Para Primeiros Passos (Setup Inicial):

```bash
# 1. Verificar integridade dos dados
python scripts\verify_all.py

# 2. Treinar todos os modelos
python main.py

# 3. Visualizar métricas dos modelos
python scripts\show_metrics.py

# 4. Abrir interface Streamlit
streamlit run app.py
```

### Para Gerar Conteúdo do Artigo Científico:

```bash
# Opção 1: Gerar tudo de uma vez (RECOMENDADO)
python scripts\generate_all.py

# Opção 2: Gerar individualmente
# 1. Gerar todas as tabelas consolidadas
python scripts\generate_tables.py

# 2. Gerar todas as visualizações (300 DPI)
python scripts\generate_figures.py

# 3. Visualizar no Streamlit → "Análise Científica Consolidada"
streamlit run app.py
```

### Ordem Recomendada de Execução:

1. **Verificação** → `verify_all.py` - Valida estrutura dos dados
2. **Treinamento** → `main.py` - Treina e salva modelos
3. **Métricas** → `show_metrics.py` - Exibe performance
4. **Tabelas & Figuras** → `generate_all.py` - Gera tudo de uma vez
5. **Visualização** → `streamlit run app.py` - Interface interativa

**Ou use os botões no Streamlit:**
- Na página "Análise Científica Consolidada", clique em:
  - 🔄 "Gerar Tabelas" para criar os CSVs
  - 🎨 "Gerar Figuras" para criar os PNGs
  - Os arquivos aparecem automaticamente após a geração!

---

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

**Gerar tabelas consolidadas (NOVO):**
```bash
python scripts\generate_tables.py
```
Gera tabelas consolidadas para o artigo científico:
- Resumo completo do dataset
- Estatísticas descritivas das features
- Comparação completa de modelos (incluindo baseline)
- Performance por temporada
- Matrizes de confusão detalhadas
- Classificação por classe

**Gerar visualizações avançadas (NOVO):**
```bash
python scripts\generate_figures.py
```
Gera figuras de alta qualidade (300 DPI) para o artigo científico:
- Radar chart de comparação multi-métrica
- Heatmap de correlação entre features
- Boxplots de features por resultado
- Comparação de feature importance (RF vs XGBoost)
- Curvas de calibração comparativas
- Gráfico de barras de métricas

**Gerar tudo de uma vez (NOVO):**
```bash
python scripts\generate_all.py
```
Executa `generate_tables.py` e `generate_figures.py` em sequência.
Gera todos os 10 CSVs + 6 PNGs de uma vez só!

## Estrutura do Projeto

```
Projeto_ML/
├── data/
│   ├── data_2005_2014/      # Dados de treino (9 temporadas)
│   │   ├── Season_2005_2006.csv
│   │   ├── ... (outras temporadas)
│   │   └── Season_2013_2014.csv
│   └── data_2014_2016/      # Dados de teste (2 temporadas)
│       ├── Season_2014_2015.csv
│       └── Season_2015_2016.csv
├── models/
│   ├── trained_models.pkl   # Modelos treinados (SVM, RandomForest, XGBoost)
│   ├── tabela*.csv          # Tabelas consolidadas para artigo
│   └── figures/             # Visualizações científicas (PNG 300 DPI)
│       ├── fig1_radar_comparison.png
│       ├── fig2_feature_correlation.png
│       ├── fig3_boxplots_by_result.png
│       ├── fig4_feature_importance_comparison.png
│       ├── fig5_calibration_comparison.png
│       └── fig6_metrics_comparison_bars.png
├── scripts/
│   ├── verify_all.py        # Verificação completa do projeto
│   ├── show_metrics.py      # Métricas dos modelos
│   ├── test_features.py     # Teste de features
│   ├── inspect_epl.py       # Inspeção do dataset
│   ├── debug_features.py    # Debug de features
│   ├── shap_analysis.py     # Análise SHAP de explicabilidade
│   ├── gridsearch_advanced.py  # Otimização de hiperparâmetros
│   ├── generate_tables.py   # 🆕 Geração de tabelas científicas
│   ├── generate_figures.py  # 🆕 Geração de visualizações avançadas
│   └── generate_all.py      # 🆕 Gera tudo de uma vez (tabelas + figuras)
├── src/
│   ├── preprocessing.py     # Carregamento e preparação dos dados
│   ├── feature_engineering.py  # Cálculo das features
│   ├── train_models.py      # Treinamento dos modelos
│   └── analysis.py          # Avaliação e visualizações
├── app.py                   # Interface Streamlit (6 páginas)
├── main.py                  # Pipeline de treinamento
├── requirements.txt         # Dependências
├── README.md               # Este arquivo
├── MELHORIAS.md            # Documentação de melhorias
└── ANALISE_COMPLETUDE.md   # 🆕 Análise de completude vs artigo
```

## Resultados

**Métricas dos Modelos (Test Set 2014-2016):**
- **SVM:** Acurácia 44.08% | F1 0.2898 | RPS 0.4342 ⭐
- **RandomForest:** Acurácia 43.82% | F1 0.2796 | RPS 0.4556
- **XGBoost:** Acurácia 41.84% | F1 0.3035 | RPS 0.4468
- **Baseline:** Acurácia 46.05% (sempre prever "Vitória Casa")

**Sobre as Métricas:**
- **RPS (Ranked Probability Score):** Quanto menor, melhor. Mede a qualidade das probabilidades preditas.
- **Baseline:** Modelo trivial que sempre prevê a classe majoritária (Vitória Casa). Serve como referência mínima.
- **O SVM** apresenta o melhor RPS, indicando melhor calibração de probabilidades.

## 🆕 Análise Científica Consolidada

### Tabelas para o Artigo
Execute `python scripts\generate_tables.py` para gerar 6 tabelas em CSV:
1. **Resumo do Dataset:** Total de partidas, distribuição de resultados, split treino/teste
2. **Estatísticas Descritivas:** Mean, Std, Min, Quartis, Max para cada feature
3. **Comparação de Modelos:** Todas as métricas (Accuracy, Precision, Recall, F1, RPS, Brier, ROC AUC)
4. **Matrizes de Confusão:** Contagens absolutas e percentuais para cada modelo
5. **Performance por Temporada:** Acurácia de cada modelo por temporada de teste
6. **Classificação Detalhada:** Precision, Recall, F1, Support por classe e modelo

### Visualizações para o Artigo
Execute `python scripts\generate_figures.py` para gerar 6 figuras em PNG (300 DPI):
1. **Radar Chart:** Comparação multi-métrica visual dos 3 modelos
2. **Heatmap:** Correlação linear entre as features
3. **Boxplots:** Distribuição de cada feature por tipo de resultado
4. **Feature Importance:** Comparação RF vs XGBoost lado a lado
5. **Calibração:** Curvas de calibração dos 3 modelos por classe
6. **Barras:** Comparação visual de Accuracy, Precision, Recall, F1

### Visualizar no Streamlit
Acesse a nova página **"Análise Científica Consolidada"** no Streamlit para:
- Visualizar todas as tabelas interativamente
- Explorar as figuras com legendas descritivas
- Fazer download dos CSVs para LaTeX/Word
- Ver instruções de uso para o artigo

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
