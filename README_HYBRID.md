# 🎯 DECODER HYBRID - Pipeline Completo

## Resumo Executivo

Implementação de um **Decoder Hybrid Autoencoder** para previsão de resultados de futebol (EPL 2005-2016). O pipeline combina:

- ✅ **Compressão**: Encoder para 8D latent space
- ✅ **Denoising**: Decoder para reconstruir features
- ✅ **Detecção de Anomalias**: Reconstruction error como sinal de confiança (171 outliers, 5%)
- ✅ **4 Classifiers**: RandomForest, XGBoost, NaiveBayes, SVM

**Resultado Global**: **SVM 49.74% accuracy**, **XGBoost 47.25% F1-score**

---

## 📊 Estatísticas Finais

```
✅ 14 Tabelas CSV geradas
✅ 17 Figuras PNG @ 300 DPI (2.5 MB)
✅ 5 Modelos treinados (Autoencoder + 4 Classifiers)
✅ 9 Scripts de análise + Orquestrador
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 TOTAL: 36 arquivos
⏱️ Tempo de Treinamento: ~2-3 minutos
⏱️ Tempo de Análise: ~8-12 minutos
```

---

## 🚀 GUIA DE USO - Passo a Passo

### **OPÇÃO 1: Executar Tudo (Recomendado para primeira vez)**

#### Passo 1️⃣ - Treinar Modelos
```bash
python main.py
```

**O que acontece:**
- ✅ ETAPA 1-2: Carrega dados 2005-2016 e calcula 43 features
- ✅ ETAPA 3: Treina baselines (DummyClassifier)
- ✅ ETAPA 4: Treina Latent Autoencoder
- ✅ ETAPA 5: **Treina Decoder Hybrid** (NOVO!)
  - Encoder: 43D → 8D latent
  - Decoder: 8D → 43D reconstructed
  - Detecção de anomalias (P95: 171 outliers)
  - Treinamento de 4 classificadores em dados limpos (3,249 amostras)

**Saída esperada:**
```
5 ETAPAS EXECUTADAS ✅

Modelos salvos em: models/autoencoder_decoder_hybrid/
├─ trained_models_hybrid.pkl (2.3 MB)
├─ autoencoder_hybrid.keras
├─ encoder_hybrid.keras
├─ decoder_hybrid.keras
└─ scaler_hybrid.joblib
```

#### Passo 2️⃣ - Gerar Todas Análises e Figuras
```bash
python scripts/generate_hybrid_all.py
```

**Executa 9 scripts sequencialmente:**

```
[1/9] update_hybrid_tabela3.py        → Comparação Baseline vs Latent vs Hybrid
[2/9] update_hybrid_tabela4.py        → Confusion Matrices (4 modelos)
[3/9] update_hybrid_tabelas_5_6.py    → Performance temporal + por classe
[4/9] hybrid_baseline_comparison.py   → ML vs Dummy classifiers
[5/9] hybrid_correlation_heatmap.py   → Heatmap 43×43 features
[6/9] hybrid_radar_chart.py           → Radar charts (3 temporadas)
[7/9] hybrid_feature_importance.py    → Importância de features (50D)
[8/9] hybrid_confidence_intervals.py  → Bootstrap CI (95%, 100 iterações)
[9/9] hybrid_additional_visualizations.py → Performance, CM, confiança (4 arquivos)
```

**Tempo estimado**: ~8-12 minutos

#### Passo 3️⃣ - Verificar Resultados
```bash
python scripts/show_hybrid_summary.py
```

**Exibe:**
- ✅ Todas as 14 tabelas CSV
- ✅ Todas as 17 figuras PNG
- ✅ Performance por modelo
- ✅ Performance por temporada
- ✅ Contagem de arquivos

---

### **OPÇÃO 2: Executar Scripts Individuais**

Se precisar reexecutar análises específicas:

#### Script 1️⃣ - Comparação de Pipelines
```bash
python scripts/update_hybrid_tabela3.py
```
**Saída**: `tabela3_hybrid_comparacao.csv` (8 linhas, 7 colunas)
- Comparação Baseline vs Latent vs Hybrid
- Accuracy, Precision, Recall, F1-Score, RPS

#### Script 2️⃣ - Confusion Matrices
```bash
python scripts/update_hybrid_tabela4.py
```
**Saída**: 4 CSVs (3×3 cada)
- `tabela4_cm_hybrid_randomforest.csv`
- `tabela4_cm_hybrid_xgboost.csv`
- `tabela4_cm_hybrid_naivebayes.csv`
- `tabela4_cm_hybrid_svm.csv`

#### Script 3️⃣ - Performance Temporal
```bash
python scripts/update_hybrid_tabelas_5_6.py
```
**Saída**: 2 CSVs
- `tabela5_hybrid_performance_temporada.csv` (12 linhas)
- `tabela6_hybrid_classificacao_classe.csv` (12 linhas)

#### Script 4️⃣ - Validação Baseline
```bash
python scripts/hybrid_baseline_comparison.py
```
**Saída**: `hybrid_baseline_comparison.csv` (6 linhas)
- Compara SVM (49.74%) vs DummyMostFrequent (45.00%)

#### Script 5️⃣ - Matriz de Correlação
```bash
python scripts/hybrid_correlation_heatmap.py
```
**Saída**: 2 arquivos
- `hybrid_correlation_matrix.csv` (43×43)
- `hybrid_correlation_heatmap.png` (615 KB, 300 DPI)

#### Script 6️⃣ - Radar Charts
```bash
python scripts/hybrid_radar_chart.py
```
**Saída**: 3 PNGs (300 DPI, ~450 KB cada)
- `radar_chart_hybrid_2014-2015.png` (458 KB)
- `radar_chart_hybrid_2015-2016.png` (465 KB)
- `radar_chart_hybrid_All.png` (452 KB)

#### Script 7️⃣ - Feature Importance
```bash
python scripts/hybrid_feature_importance.py
```
**Saída**: 1 CSV + 3 PNGs
- `hybrid_feature_importance.csv` (50 features com importance)
- `hybrid_feature_importance_top20.png` (213 KB)
- `hybrid_feature_importance_by_type.png` (205 KB)
- `hybrid_feature_importance_distribution.png` (103 KB)

**O que mostra:**
- Top 20 features mais importantes
- Breakdown por tipo (8 latentes, 43 reconstruídas, 1 erro)
- Distribuição de importância

#### Script 8️⃣ - Intervalo de Confiança Bootstrap
```bash
python scripts/hybrid_confidence_intervals.py
```
**Saída**: 1 CSV + 3 PNGs
- `hybrid_confidence_intervals.csv` (16 linhas)
- `hybrid_confidence_intervals_boxplot.png` (182 KB)
- `hybrid_confidence_intervals_barplot.png` (130 KB)
- `hybrid_confidence_intervals_distribution.png` (333 KB)

**O que mostra:**
- 95% CI para Accuracy, F1-Score, RPS
- 100 iterações de bootstrap
- Margem de erro por modelo

#### Script 9️⃣ - Visualizações Adicionais
```bash
python scripts/hybrid_additional_visualizations.py
```
**Saída**: 1 CSV + 4 PNGs
- `hybrid_performance_by_season.csv`
- `hybrid_performance_by_season.png` (172 KB)
- `hybrid_confusion_matrix_svm.png` (120 KB)
- `hybrid_confusion_matrix_randomforest.png` (127 KB)
- `hybrid_confusion_matrix_xgboost.png` (124 KB)
- `hybrid_confusion_matrix_naivebayes.png` (123 KB)
- `hybrid_performance_heatmap.png` (167 KB)
- `hybrid_prediction_confidence.png` (174 KB)

**O que mostra:**
- Accuracy por modelo e temporada (com valores nas barras)
- 4 confusion matrices separadas (uma por modelo)
- Heatmap de accuracy (modelo × temporada)
- Distribuição de confiança (max probability)

---

## 📊 Saídas Geradas

### 📋 Tabelas (14 CSVs)

| # | Arquivo | Linhas | Colunas | Descrição |
|---|---------|--------|---------|-----------|
| 1 | `tabela3_hybrid_comparacao.csv` | 8 | 7 | Baseline vs Latent vs Hybrid |
| 2-5 | `tabela4_cm_hybrid_*.csv` | 3 | 3 | Confusion Matrix (4 modelos) |
| 6 | `tabela5_hybrid_performance_temporada.csv` | 12 | 6 | Performance por season |
| 7 | `tabela6_hybrid_classificacao_classe.csv` | 12 | 5 | Métricas por classe H/D/A |
| 8 | `hybrid_baseline_comparison.csv` | 6 | 4 | ML vs Dummy |
| 9 | `hybrid_correlation_matrix.csv` | 43 | 43 | Matriz Pearson |
| 10 | `hybrid_model_results.csv` | 4 | 7 | Resumo geral |
| 11 | `hybrid_model_results_by_season.csv` | 8 | 6 | Resultados temporais |
| 12 | `hybrid_feature_importance.csv` | 50 | 3 | Feature importance ranking |
| 13 | `hybrid_confidence_intervals.csv` | 16 | 5 | CI 95% bootstrap |
| 14 | `hybrid_performance_by_season.csv` | 12 | 5 | Perf por season (destalhado) |

### 📈 Figuras (17 PNGs @ 300 DPI)

| # | Arquivo | Tamanho | Descrição | Conteúdo |
|---|---------|---------|-----------|----------|
| **RADAR CHARTS (3)** | | | 5 eixos | Acc, Prec, Rec, F1, 1-RPS |
| 1 | `radar_chart_hybrid_2014-2015.png` | 458 KB | Temporada 2014-2015 | 4 modelos |
| 2 | `radar_chart_hybrid_2015-2016.png` | 465 KB | Temporada 2015-2016 | 4 modelos |
| 3 | `radar_chart_hybrid_All.png` | 452 KB | Combinado | 4 modelos |
| **CORRELATION (1)** | | | 43×43 heatmap | Pearson correlation |
| 4 | `hybrid_correlation_heatmap.png` | 615 KB | Todas 43 features | |
| **FEATURE IMPORTANCE (3)** | | | Ranking | RandomForest |
| 5 | `hybrid_feature_importance_top20.png` | 213 KB | Top 20 features | Bar chart |
| 6 | `hybrid_feature_importance_by_type.png` | 205 KB | 3 tipos | Latent, Recon, Error |
| 7 | `hybrid_feature_importance_distribution.png` | 103 KB | Distribuição | Histogram |
| **CONFIDENCE INTERVALS (3)** | | | Bootstrap | 100 iterações, 95% CI |
| 8 | `hybrid_confidence_intervals_boxplot.png` | 182 KB | Box plot | Acc, F1, RPS |
| 9 | `hybrid_confidence_intervals_barplot.png` | 130 KB | Bar plot | Com margens |
| 10 | `hybrid_confidence_intervals_distribution.png` | 333 KB | Distribuição | Histogramas |
| **PERFORMANCE (4)** | | | Análise temporal | Por modelo e season |
| 11 | `hybrid_performance_by_season.png` | 172 KB | Bar chart | Com valores nas barras |
| 12 | `hybrid_confusion_matrix_svm.png` | 120 KB | 3×3 heatmap | SVM |
| 13 | `hybrid_confusion_matrix_randomforest.png` | 127 KB | 3×3 heatmap | RandomForest |
| 14 | `hybrid_confusion_matrix_xgboost.png` | 124 KB | 3×3 heatmap | XGBoost |
| 15 | `hybrid_confusion_matrix_naivebayes.png` | 123 KB | 3×3 heatmap | NaiveBayes |
| **HEATMAPS (2)** | | | Performance | Por modelo e season |
| 16 | `hybrid_performance_heatmap.png` | 167 KB | Modelo × Season | Accuracy |
| 17 | `hybrid_prediction_confidence.png` | 174 KB | 4 histogramas | Max probability |

**Total**: 2.5 MB de visualizações

---

## 📈 Resultados de Performance

### Performance Global (760 amostras de teste)

```
╔════════════════╦══════════╦══════════╦═════════╗
║   Modelo       ║ Accuracy ║ F1-Score ║  RPS    ║
╠════════════════╬══════════╬══════════╬═════════╣
║ SVM            ║ 0.4974   ║ 0.4632   ║ 0.2079  ║  ⭐ Melhor Accuracy
║ XGBoost        ║ 0.4895   ║ 0.4725   ║ 0.2079  ║  ⭐ Melhor F1-Score
║ NaiveBayes     ║ 0.4816   ║ 0.4720   ║ 0.3081  ║
║ RandomForest   ║ 0.4789   ║ 0.4629   ║ 0.2072  ║
╚════════════════╩══════════╩══════════╩═════════╝
```

### Performance por Temporada

**2014-2015** (380 testes):
- ⭐ SVM: **52.89%** accuracy
- XGBoost: 51.32% accuracy
- NaiveBayes: 50.53% accuracy
- RandomForest: 51.32% accuracy

**2015-2016** (380 testes):
- XGBoost: **46.58%** accuracy
- SVM: 46.58% accuracy
- NaiveBayes: 45.79% accuracy
- RandomForest: 44.47% accuracy

### Validação vs Baseline

```
Dummy Most Frequent: 45.00% (baseline)
Dummy Stratified:    33.33% (baseline)
┌─────────────────────┐
│ SVM Hybrid: 49.74%  │ ✅ +4.74% sobre baseline
└─────────────────────┘
```

### Análise de Features

- **Outliers removidos**: 171 (5% do treino)
- **Amostras limpas**: 3,249 (95% - usado para treinar)
- **Multicolinearidade**: 12 pares com |corr| > 0.8
- **Correlação média**: 0.28 (aceitável)

---

## 🔍 Arquitetura Técnica

### Fluxo Completo

```
DADOS BRUTOS (3,420 matches)
    ↓
[ETAPA 1-2] Feature Engineering (43 features)
    ├─ Form scores (2)
    ├─ Rolling averages μₖ (4)
    ├─ FIFA ratings (8)
    ├─ Head-to-head (6)
    ├─ League position (6)
    ├─ Odds Bet365 (9)
    └─ Interactions (2)
    ↓
[ETAPA 3] Baselines (Dummy Classifiers)
    ├─ MostFrequent: 45.00%
    └─ Stratified: 33.33%
    ↓
[ETAPA 4] Latent Space (8D)
    ├─ Encoder: 43D → 8D
    └─ Latent features
    ↓
[ETAPA 5] 🆕 DECODER HYBRID
    ├─ Decoder: 8D → 43D reconstructed
    ├─ Reconstruction error (1D)
    │
    ├─ Hybrid features: 8+43+1 = 50D
    │
    ├─ Anomaly Detection (P95)
    │  └─ Remove 171 outliers (5%)
    │
    └─ Train 4 classifiers on clean data
       ├─ RandomForest: 47.89% ✅
       ├─ XGBoost:      48.95% ✅
       ├─ NaiveBayes:   48.16% ✅
       └─ SVM:          49.74% ⭐
```

### Autoencoder Architecture

```
Input Layer
    ↓
Dense(64, activation='relu')
    ↓
Dense(32, activation='relu')
    ↓
Dense(8)  ← LATENT SPACE (8D)
    ↓
Dense(32, activation='relu')
    ↓
Dense(64, activation='relu')
    ↓
Dense(43, activation='sigmoid')
    ↓
Output (Reconstructed 43D)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loss: Mean Squared Error
Optimizer: Adam(learning_rate=0.001)
Epochs: 50
Batch Size: 32
Validation Split: 0.2
```

### Hybrid Feature Engineering

```
HYBRID FEATURES (50D) = LATENT (8D) + RECONSTRUCTED (43D) + ERROR (1D)

[Latent_0, Latent_1, ..., Latent_7]                    [8D]
[Recon_gd_diff, Recon_streak_diff, ..., Recon_strength] [43D]
[Reconstruction_Error]                                 [1D]
```

Normalization: **MinMaxScaler [0, 1]** (fitted on training data)

### Classificadores

```
RandomForest
  • n_estimators: 100
  • max_depth: 10
  • random_state: 42
  • Output: CalibratedClassifierCV → predict_proba

XGBoost
  • learning_rate: 0.1
  • max_depth: 6
  • n_estimators: 100
  • subsample: 0.8
  • Output: CalibratedClassifierCV → predict_proba

NaiveBayes (Gaussian)
  • Default params
  • Output: CalibratedClassifierCV → predict_proba

SVM (RBF)
  • kernel: 'rbf'
  • C: 1.0
  • gamma: 'scale'
  • Output: CalibratedClassifierCV → predict_proba
```

---

## 💡 Insights & Recomendações

### ✅ O Que Funcionou

1. **SVM no espaço 50D**: Separador ótimo
   - 49.74% accuracy (~5% acima do baseline)
   - Robusto em ambas temporadas

2. **Reconstruction Error como sinal**:
   - 5% outliers detectados automaticamente
   - Melhora qualidade dos dados de treino
   - Remove noise sem overfitting

3. **Latent 8D é bom**:
   - Comprime 43D sem perda excessiva
   - Features mais discriminativas
   - Sem curse of dimensionality

4. **XGBoost captura interações**:
   - Melhor F1-score (47.25%)
   - Bom balanceamento entre classes
   - Estável em 2014-2015

### ⚠️ Desafios

1. **Classe Draw é difícil**:
   - Precision/Recall sempre < 40%
   - Apenas 26% dos dados (imbalanceado)

2. **Temporal Drift**:
   - 2014-2015: 51-52% accuracy
   - 2015-2016: 44-46% accuracy
   - Distribuição muda ao longo do tempo

3. **RPS ainda elevado** (0.21-0.31):
   - Predições menos confiantes
   - Incerteza alta mesmo para SVM

### 🎯 Próximos Passos

1. **Feature Engineering Avançado**
   - Sazonalidade (mês, weekday)
   - Lesões/suspensões
   - Momentum recente

2. **Temporal Cross-Validation**
   - Walk-forward validation
   - Treinar em períodos mais antigos

3. **Ensemble & Stacking**
   - Combinar 4 modelos
   - Weighted voting por confiança

4. **SHAP Analysis**
   - Explicabilidade por features
   - Importância decomposta

5. **Ajustes de Hiperparâmetros**
   - Grid search automático
   - Threshold tuning para P95

---

## 📁 Estrutura de Diretórios

```
projeto_ml/
├─ main.py                           # Orquestrador principal
├─ README_HYBRID.md                  # Este arquivo
│
├─ src/
│  ├─ train_models.py               # ← train_models_with_decoder_hybrid() linha 813
│  ├─ preprocessing.py
│  ├─ feature_engineering.py
│  └─ analysis.py
│
├─ scripts/
│  ├─ generate_hybrid_all.py         # 🎯 MASTER - executa todos os 9 scripts
│  ├─ show_hybrid_summary.py         # Exibe sumário final
│  │
│  ├─ [Tier 1: Análises Básicas]
│  ├─ update_hybrid_tabela3.py       # Comparação pipelines
│  ├─ update_hybrid_tabela4.py       # Confusion matrices
│  ├─ update_hybrid_tabelas_5_6.py   # Performance temporal + classe
│  ├─ hybrid_baseline_comparison.py  # ML vs Dummy
│  ├─ hybrid_correlation_heatmap.py  # Matriz correlação
│  ├─ hybrid_radar_chart.py          # Radar charts
│  │
│  ├─ [Tier 2: Análises Avançadas]
│  ├─ hybrid_feature_importance.py   # Feature ranking (50D)
│  ├─ hybrid_confidence_intervals.py # Bootstrap CI
│  └─ hybrid_additional_visualizations.py # Performance visualizations
│
├─ models/autoencoder_decoder_hybrid/
│  ├─ trained_models_hybrid.pkl      # 4 modelos + metadata
│  ├─ autoencoder_hybrid.keras       # Autoencoder completo
│  ├─ encoder_hybrid.keras           # Encoder 43D → 8D
│  ├─ decoder_hybrid.keras           # Decoder 8D → 43D
│  ├─ scaler_hybrid.joblib           # MinMaxScaler
│  │
│  ├─ [14 CSVs]
│  ├─ tabela3_hybrid_comparacao.csv
│  ├─ tabela4_cm_hybrid_*.csv (4)
│  ├─ tabela5_hybrid_performance_temporada.csv
│  ├─ tabela6_hybrid_classificacao_classe.csv
│  ├─ hybrid_baseline_comparison.csv
│  ├─ hybrid_correlation_matrix.csv
│  ├─ hybrid_model_results.csv
│  ├─ hybrid_model_results_by_season.csv
│  ├─ hybrid_feature_importance.csv
│  ├─ hybrid_confidence_intervals.csv
│  └─ hybrid_performance_by_season.csv
│  │
│  └─ figures/
│      ├─ [3] radar_chart_hybrid_*.png
│      ├─ [1] hybrid_correlation_heatmap.png
│      ├─ [3] hybrid_feature_importance_*.png
│      ├─ [3] hybrid_confidence_intervals_*.png
│      ├─ [4] hybrid_confusion_matrix_*.png
│      ├─ [2] hybrid_performance_*.png
│      └─ [1] hybrid_prediction_confidence.png
│
└─ data/
   ├─ data_2005_2014/ (9 seasons)
   └─ data_2014_2016/ (2 seasons)
```

---

## 🛠️ Requisitos

### Python Packages
```
tensorflow>=2.10
scikit-learn>=1.0
xgboost>=1.6
pandas>=1.3
numpy>=1.21
matplotlib>=3.5
seaborn>=0.12
joblib>=1.1
```

### Hardware Recomendado
- CPU: 4+ cores
- RAM: 8 GB (mínimo 4 GB)
- Disco: 500 MB livres

---

## 📞 Contato & Suporte

Para dúvidas sobre:
- **Arquitetura**: Ver seção "Arquitetura Técnica"
- **Scripts**: Ver seção "OPÇÃO 2: Executar Scripts Individuais"
- **Resultados**: Ver seção "Resultados de Performance"
- **Próximos passos**: Ver seção "Próximos Passos"

---

## 📝 Histórico de Atualizações

**v2.0 (30/05/2026)** - Versão Atual
- ✅ 9 scripts de análise (6 + 3 novos)
- ✅ 17 figuras PNG completas
- ✅ Feature importance, Bootstrap CI, visualizações adicionais
- ✅ Detalhes atualizados e passo-a-passo completo

**v1.0 (Anterior)**
- 6 scripts iniciais
- 4 figuras radar charts
- Análises básicas

---

**Última atualização**: 30 de maio de 2026  
**Status**: ✅ Completo e Testado  
**Próxima revisão**: Quando houver ajustes de hiperparâmetros ou novas features
