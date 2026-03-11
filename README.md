
# 🏆 Predição de Resultados da Premier League - Machine Learning

Projeto completo de Machine Learning para predição de resultados de partidas da Premier League (2005-2016), com **43 features**, **4 modelos calibrados**, e **análises científicas** prontas para publicação.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Processo de Execução Completo](#processo-de-execução-completo)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Scripts Disponíveis](#scripts-disponíveis)
- [Resultados](#resultados)

---

## 🎯 Visão Geral

Este projeto implementa um pipeline completo de Machine Learning para predição de resultados (Vitória Casa/Empate/Vitória Visitante) de partidas da Premier League.

**Características principais:**
- ✅ **43 features** engenheiradas (33 do artigo + 10 adicionais)
- ✅ **4 modelos**: RandomForest, XGBoost, NaiveBayes, SVM (todos calibrados)
- ✅ **Split temporal**: Treino 2005-2014, Teste 2014-2016 (sem data leakage)
- ✅ **Hiperparâmetros otimizados** via GridSearch CV
- ✅ **Análises estatísticas**: Baseline, correlação, confidence intervals, confusion matrices
- ✅ **14 tabelas científicas** + **3 visualizações** prontas para publicação
- ✅ **Interface Streamlit** interativa

---

## 🔧 Pré-requisitos

- **Python 3.8+**
- **pip** (gerenciador de pacotes)
- **Windows PowerShell** (para ativar ambiente virtual)

---

## 📦 Instalação

### Passo 1: Clonar/baixar o projeto

```bash
cd c:\Users\seu_usuario\Desktop
git clone https://github.com/wiliafsilva/Projeto_ML.git
cd Projeto_ML
```

### Passo 2: Criar ambiente virtual

```bash
python -m venv .venv
```

### Passo 3: Ativar ambiente virtual

**Windows PowerShell:**
```bash
.\.venv\Scripts\Activate.ps1
```

**Prompt de Comando (cmd):**
```bash
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Passo 4: Instalar dependências

```bash
pip install -r requirements.txt
```

**Principais pacotes instalados:**
- `scikit-learn` (modelos ML)
- `xgboost` (Gradient Boosting)
- `pandas`, `numpy` (manipulação de dados)
- `matplotlib`, `seaborn` (visualizações)
- `streamlit` (interface web)
- `joblib` (salvamento de modelos)

---

## 🚀 Processo de Execução Completo

Siga esta sequência para executar o projeto do zero até os resultados finais.


### ⚡ Execução Rápida (Quick Start)

Se você já tem tudo instalado e só quer treinar e visualizar:

```bash
# 1. Ativar ambiente
.\.venv\Scripts\Activate.ps1

# 2. Treinar modelos
python main.py

# 3. Abrir interface
streamlit run app.py
```

Acesse http://localhost:8501 no navegador e explore!

---

### 📊 Execução Completa (Passo a Passo Detalhado)

#### **ETAPA 1: Verificação dos Dados**

Antes de tudo, verifique se os dados estão corretos:

```bash
python scripts/verify_all.py
```

**O que este script faz:**
- ✅ Verifica estrutura das pastas `data/data_2005_2014/` e `data/data_2014_2016/`
- ✅ Conta partidas por temporada (380 por temporada esperado)
- ✅ Valida presença de todas as colunas necessárias
- ✅ Mostra estatísticas do dataset (3420 treino + 760 teste)
- ✅ Exibe distribuição de resultados (Home Win: 43.3%, Draw: 26.3%, Away Win: 30.4%)

**Saída esperada:**
```
================================================================================
VERIFICACAO COMPLETA DO PROJETO
================================================================================

[OK] Total geral: 4180 partidas (3420 treino + 760 teste)
[OK] Features implementadas: 43 (33 artigo + 10 bonus)
[OK] Modelos disponíveis: 4 (RandomForest, XGBoost, NaiveBayes, SVM)
```

---

#### **ETAPA 2: Treinamento dos Modelos**

Agora treine os 4 modelos Machine Learning:

```bash
python main.py
```

**O que este script faz:**
1. **Carrega os dados** das temporadas 2005-2016
2. **Calcula 43 features** para cada partida:
   - Form features (5 últimas partidas)
   - Goal difference acumulada
   - Ratings FIFA (Overall, Attack, Midfield, Defense)
   - Head-to-head histórico
   - League position & points
   - Betting odds (Bet365)
   - Interaction features (6 combinações)
3. **Divide temporalmente**: Treino 2005-2014, Teste 2014-2016
4. **Treina 4 modelos** com hiperparâmetros otimizados:
   - RandomForest (50 trees, max_depth=5)
   - XGBoost (200 estimators, lr=0.01)
   - NaiveBayes (Gaussian, var_smoothing=1e-05)
   - SVM (RBF kernel, C=0.1, gamma=0.001)
5. **Aplica calibração** (isotônica) nas probabilidades
6. **Salva modelos** em `models/trained_models.pkl`

**Tempo estimado:** 2-5 minutos

**Saída esperada:**
```
================================================================================
RESULTADOS FINAIS - TEST SET (2014-2016: 760 partidas)
================================================================================

Modelo            Accuracy   Precision     Recall   F1-Score        RPS
RandomForest        0.5066      0.3336     0.4417     0.3773     0.4132
XGBoost             0.4934      0.4740     0.4751     0.4733     0.4142
NaiveBayes          0.4697      0.4665     0.4629     0.4604     0.4194
SVM                 0.4632      0.4499     0.4513     0.4488     0.4271

[OK] Modelos salvos em: models/trained_models.pkl
```

---

#### **ETAPA 3: Visualizar Métricas**

Veja um resumo das métricas dos modelos treinados:

```bash
python scripts/show_metrics.py
```

**O que este script faz:**
- Carrega `models/trained_models.pkl`
- Exibe tabela formatada com todas as métricas
- Destaca o melhor modelo por métrica

**Saída esperada:**
```
Modelo            Accuracy      F1      RPS
RandomForest       50.66%   0.3773   0.4132  <- MELHOR RPS
XGBoost            49.34%   0.4733   0.4142
NaiveBayes         46.97%   0.4604   0.4194
SVM                46.32%   0.4488   0.4271
```

---

#### **ETAPA 4: Gerar Análises Científicas**

##### **4a) Baseline Comparison (Validação Científica)**

Compare os modelos ML com preditores triviais:

```bash
python scripts/baseline_comparison.py
```

**Resultado:**
- Cria `models/baseline_comparison.csv`
- Prova que ML é **17% melhor** que preditores triviais
- RandomForest 50.7% vs Baseline 43.3% ✅

---

##### **4b) Correlation Heatmap (Multicolinearidade)**

Analise correlação entre features:

```bash
python scripts/correlation_heatmap.py
```

**Resultado:**
- Cria `models/correlation_matrix.csv` (41×41 matriz)
- Cria `models/figures/correlation_heatmap.png` (visualização)
- Identifica 44 pares com correlação > 0.8
- Descobre que `overall_diff ↔ midfield_diff = 0.999` (redundantes!)

---

##### **4c) Confidence Intervals (Bootstrap)**

Calcule intervalos de confiança 95% via bootstrap:

```bash
python scripts/confidence_intervals_fast.py
```

**Resultado:**
- Cria `models/confidence_intervals.csv`
- Bootstrap com 100 iterações
- Mostra CI para Accuracy, F1, RPS por modelo
- Exemplo: RandomForest 0.507 [0.469, 0.545]

---

##### **4d) Radar Chart (Comparação Multi-Métrica)**

Crie gráfico radar comparando modelos:

```bash
python scripts/radar_chart.py
```

**Resultado:**
- Cria `models/figures/radar_chart.png`
- Polígono 5-axis para cada modelo
- Visualiza trade-offs entre Accuracy, Precision, Recall, F1, RPS

---

##### **4e) Atualizar Tabelas 3, 5, 6**

Regenere tabelas com performance atual:

```bash
python scripts/update_tabela3.py
python scripts/update_tabelas_5_6.py
```

**Resultado:**
- Tabela 3: Comparação de modelos atualizada
- Tabela 5: Performance por temporada (2014-2015, 2015-2016)
- Tabela 6: Classificação por classe (H/D/A) para cada modelo

---

##### **4f) Confusion Matrices (Tabela 4)**

Regenere matrizes de confusão:

```bash
python scripts/update_tabela4.py
```

**Resultado:**
- 4 arquivos: `models/tabela4_cm_*.csv`
- Matrizes 3×3 (Home Win, Draw, Away Win)
- Cada modelo tem sua própria matriz


---

#### **ETAPA 5: Interface Streamlit (Visualização Interativa)**

Inicie a aplicação web para explorar os resultados:

```bash
streamlit run app.py
```

**Acesse:** http://localhost:8501 no navegador

**6 Páginas disponíveis:**

1. **📊 Dashboard Principal**
   - Visão geral das métricas
   - Gráficos de comparação
   - Distribuição de resultados

2. **🔮 Preditor Interativo** (Em fase de implantação)
   - Insira dados de uma partida manualmente
   - Veja predições em tempo real dos 4 modelos
   - Probabilidades para H/D/A

3. **📈 Análise Comparativa**
   - Comparação detalhada entre modelos
   - Gráficos de Accuracy, F1, RPS
   - Matrizes de confusão interativas

4. **🎯 Features & Importância**
   - Ranking das 43 features
   - Gráficos de importância para RF e XGBoost
   - Análise SHAP (se disponível)

5. **📊 Análise Científica Consolidada**
   - Todas as 14 tabelas CSV
   - 3 visualizações PNG (300 DPI)
   - Download de arquivos
   - Botões para gerar tabelas/figuras

6. **ℹ️ Sobre o Projeto**
   - Metodologia
   - Descrição das features
   - Split temporal
   - Tecnologias utilizadas

---

#### **ETAPA 6: Otimização de Hiperparâmetros (Opcional)**

Se quiser testar novos hiperparâmetros:

##### **Teste Rápido (5-10 minutos):**

```bash
python scripts/gridsearch_quick_test.py
```

**O que faz:**
- Testa poucas combinações de hiperparâmetros
- TimeSeriesSplit com 3 folds
- Gera arquivo `models/RECOMENDACOES_HIPERPARAMETROS.txt`
- Salva resultados em `models/quick_test_results.pkl`

##### **GridSearch Completo (1-2 horas):**

```bash
python scripts/gridsearch_advanced.py
```

**O que faz:**
- Teste exhaustivo de hiperparâmetros
- TimeSeriesSplit com 5 folds
- Salva resultados em `models/gridsearch_results.csv`
- ⚠️ **Muito lento!** Use apenas se tiver tempo

---

#### **ETAPA 7: Análises Adicionais (Opcional)**

##### **Feature Importance (Análise):**

```bash
python scripts/feature_importance_simple.py
```

Mostra ranking das features mais importantes para RF e XGBoost.

##### **SHAP Analysis (Explicabilidade):**

```bash
python scripts/shap_analysis.py
```

Gera gráficos SHAP mostrando impacto de cada feature nas predições.

##### **Testar Features:**

```bash
python scripts/test_features.py
```

Testa cálculo das features e mostra estatísticas detalhadas.

##### **Inspecionar Dataset:**

```bash
python scripts/inspect_epl.py
```

Analisa dataset bruto: partidas, temporadas, gols, distribuição de resultados.

---

### 📂 Arquivos Gerados

Após executar todos os scripts, você terá:

#### **Modelos:**
- `models/trained_models.pkl` - 4 modelos calibrados + metadados

#### **Tabelas Científicas (14 CSVs):**
1. `models/baseline_comparison.csv` - Comparação ML vs baselines
2. `models/tabela3_comparacao_modelos.csv` - Comparação completa
3. `models/tabela4_cm_randomforest.csv` - Confusion Matrix RF
4. `models/tabela4_cm_xgboost.csv` - Confusion Matrix XGBoost
5. `models/tabela4_cm_naivebayes.csv` - Confusion Matrix NB
6. `models/tabela4_cm_svm.csv` - Confusion Matrix SVM
7. `models/tabela5_performance_temporada.csv` - Performance por season
8. `models/tabela6_classificacao_randomforest.csv` - Métricas por classe RF
9. `models/tabela6_classificacao_xgboost.csv` - Métricas por classe XGBoost
10. `models/tabela6_classificacao_naivebayes.csv` - Métricas por classe NB
11. `models/tabela6_classificacao_svm.csv` - Métricas por classe SVM
12. `models/correlation_matrix.csv` - Matriz 41×41
13. `models/confidence_intervals.csv` - Bootstrap CIs
14. `models/gridsearch_results.csv` - Resultados GridSearch (se executado)

#### **Visualizações (3 PNGs):**
1. `models/figures/correlation_heatmap.png` - Heatmap 41×41 features
2. `models/figures/radar_chart.png` - Spider plot 5 métricas
3. `models/figures/fig3_boxplots_by_result.png` - Boxplots features por resultado

---

## 📁 Estrutura do Projeto


```
Projeto_ML/
│
├── 📂 data/                          # Dados das partidas Premier League
│   ├── data_2005_2014/              # TREINO (9 temporadas, 3420 partidas)
│   │   ├── Season_2005_2006.csv
│   │   ├── Season_2006_2007.csv
│   │   ├── ...
│   │   └── Season_2013_2014.csv
│   │
│   └── data_2014_2016/              # TESTE (2 temporadas, 760 partidas)
│       ├── Season_2014_2015.csv
│       └── Season_2015_2016.csv
│
├── 📂 src/                           # Código-fonte principal
│   ├── preprocessing.py             # Carregamento e preparação dos dados
│   ├── feature_engineering.py       # Cálculo das 43 features
│   ├── train_models.py              # Treinamento dos 4 modelos
│   └── analysis.py                  # Avaliação e visualizações
│
├── 📂 scripts/                       # Scripts de análise (19 essenciais)
│   ├── verify_all.py                # ✅ Verificação completa do projeto
│   ├── show_metrics.py              # 📊 Exibir métricas dos modelos
│   ├── baseline_comparison.py       # 🔬 Comparação com baselines
│   ├── correlation_heatmap.py       # 🌡️ Análise de multicolinearidade
│   ├── confidence_intervals_fast.py # 📈 Bootstrap CIs (100 iter)
│   ├── radar_chart.py               # 🎯 Radar chart multi-métrica
│   ├── update_tabela3.py            # 📋 Atualizar tabela 3
│   ├── update_tabelas_5_6.py        # 📋 Atualizar tabelas 5 e 6
│   ├── update_tabela4.py            # 📋 Atualizar matrizes confusão
│   ├── gridsearch_quick_test.py     # ⚡ Otimização rápida hiperparâmetros
│   ├── gridsearch_advanced.py       # 🔍 GridSearch completo (lento)
│   ├── gridsearch_43features.py     # 🔍 GridSearch com 43 features
│   ├── feature_importance_simple.py # 📊 Ranking de features
│   ├── shap_analysis.py             # 🔬 Análise SHAP (explicabilidade)
│   ├── test_features.py             # ✔️ Testar cálculo de features
│   ├── inspect_epl.py               # 🔍 Inspecionar dataset bruto
│   └── generate_*.py                # 📊 Scripts de geração (3 arquivos)
│
├── 📂 models/                        # Modelos treinados e resultados
│   ├── trained_models.pkl           # 🧠 4 modelos calibrados + metadados
│   │
│   ├── 📊 Tabelas Científicas (14 CSVs):
│   ├── baseline_comparison.csv
│   ├── tabela3_comparacao_modelos.csv
│   ├── tabela4_cm_*.csv (4 arquivos)
│   ├── tabela5_performance_temporada.csv
│   ├── tabela6_classificacao_*.csv (4 arquivos)
│   ├── correlation_matrix.csv
│   ├── confidence_intervals.csv
│   ├── gridsearch_results.csv
│   │
│   └── 📂 figures/                  # Visualizações (PNG 300 DPI)
│       ├── correlation_heatmap.png
│       ├── radar_chart.png
│       └── fig3_boxplots_by_result.png
│
├── 📄 main.py                        # ⚙️ Pipeline principal de treinamento
├── 📄 app.py                         # 🌐 Interface Streamlit (6 páginas)
├── 📄 requirements.txt               # 📦 Dependências Python
├── 📄 README.md                      # 📖 Este arquivo
```

---

## 📊 Scripts Disponíveis

### 🔵 **Essenciais (Execute nesta ordem):**

| Script | Comando | Descrição | Tempo |
|--------|---------|-----------|-------|
| 1. Verificação | `python scripts/verify_all.py` | Valida estrutura dos dados | 5s |
| 2. Treinamento | `python main.py` | Treina 4 modelos ML | 2-5min |
| 3. Métricas | `python scripts/show_metrics.py` | Exibe performance | 2s |
| 4. Interface | `streamlit run app.py` | Abre app web | - |

### 🟢 **Análises Científicas:**

| Script | Comando | Resultado |
|--------|---------|-----------|
| Baseline | `python scripts/baseline_comparison.py` | `baseline_comparison.csv` |
| Correlação | `python scripts/correlation_heatmap.py` | `correlation_matrix.csv` + PNG |
| Bootstrap CI | `python scripts/confidence_intervals_fast.py` | `confidence_intervals.csv` |
| Radar Chart | `python scripts/radar_chart.py` | `radar_chart.png` |
| Tabela 3 | `python scripts/update_tabela3.py` | `tabela3_comparacao_modelos.csv` |
| Tabelas 5&6 | `python scripts/update_tabelas_5_6.py` | `tabela5_*.csv` + `tabela6_*.csv` |
| Tabela 4 | `python scripts/update_tabela4.py` | `tabela4_cm_*.csv` (4 arquivos) |

### 🟡 **Otimização (Opcional):**

| Script | Comando | Tempo Estimado |
|--------|---------|----------------|
| GridSearch Rápido | `python scripts/gridsearch_quick_test.py` | 5-10 min |
| GridSearch Completo | `python scripts/gridsearch_advanced.py` | 1-2 horas ⚠️ |

### 🟠 **Análises Extras:**

| Script | Descrição |
|--------|-----------|
| `feature_importance_simple.py` | Ranking de features (RF + XGBoost) |
| `shap_analysis.py` | Análise SHAP (explicabilidade) |
| `test_features.py` | Testa cálculo das 43 features |
| `inspect_epl.py` | Inspeciona dataset bruto EPL |

---

## 🏆 Resultados

### **Performance dos Modelos (Test Set 2014-2016)**

| Modelo | Accuracy | Precision | Recall | F1-Score | RPS ⬇️ |
|--------|----------|-----------|--------|----------|--------|
| **RandomForest** 🥇 | **50.66%** | 33.36% | 44.17% | 37.73% | **0.4132** ⭐ |
| **XGBoost** 🥈 | 49.34% | 47.40% | 47.51% | 47.33% | 0.4142 |
| **NaiveBayes** 🥉 | 46.97% | 46.65% | 46.29% | 46.04% | 0.4194 |
| **SVM** | 46.32% | 44.99% | 45.13% | 44.88% | 0.4271 |
| *Baseline* | 43.29% | - | - | - | - |

**Notas:**
- 🎯 **RPS (Ranked Probability Score)**: Quanto **menor**, melhor (0 = perfeito)
- ✅ RandomForest é o **melhor modelo** (menor RPS)
- 🔬 ML supera baseline em **+17%** (validação científica ✅)

### **Confidence Intervals 95% (Bootstrap)**

| Modelo | Accuracy CI | F1 CI | RPS CI |
|--------|-------------|-------|--------|
| RandomForest | [0.469, 0.545] | [0.356, 0.404] | [0.394, 0.432] |
| XGBoost | [0.458, 0.533] | [0.440, 0.510] | [0.401, 0.428] |
| NaiveBayes | [0.437, 0.504] | [0.429, 0.496] | [0.402, 0.433] |
| SVM | [0.433, 0.503] | [0.424, 0.488] | [0.409, 0.446] |

**Interpretação:**
- ✅ Intervalos estreitos (largura ~7%) → estimativas confiáveis
- ✅ Nenhum CI sobrepõe baseline → diferença significativa

### **Performance por Temporada**

| Temporada | RF | XGBoost | NaiveBayes | SVM |
|-----------|-----|---------|------------|-----|
| 2014-2015 | **53.95%** 🔥 | 51.84% | 47.37% | 50.00% |
| 2015-2016 | 47.37% | 46.84% | 46.58% | 42.63% |

**Observação:** Performance varia entre temporadas (mudanças táticas, transferências).

### **Principais Descobertas Científicas**

1. ✅ **ML vs Baseline**: RandomForest 50.7% vs Baseline 43.3% = **+17% gain**
2. 🔬 **Multicolinearidade**: `overall_diff ↔ midfield_diff = 0.999` (redundantes!)
3. 📊 **Trade-off**: RF melhor Accuracy, XGBoost mais balanceado
4. 📈 **Temporal**: Performance 2014-2015 superior à 2015-2016
5. ✅ **Robustez**: Confidence Intervals estreitos (estimativas confiáveis)

---

## 🔬 Metodologia

### **Split Temporal (Sem Data Leakage)**

```
Treino:  2005-2014 (9 temporadas) → 3420 partidas
Teste:   2014-2016 (2 temporadas) → 760 partidas
```

✅ **Sem random split** - Evita data leakage temporal  
✅ **Features incrementais** - Calculadas sequencialmente  
✅ **Validação cruzada temporal** - TimeSeriesSplit no GridSearch

### **43 Features Implementadas**

#### **Classe A: Features Individuais (14)**
- `home_form`, `away_form` (últimas 5 partidas)
- `home_position`, `away_position` (posição na liga)
- `home_points`, `away_points` (pontos acumulados)
- `h2h_*` (6 features head-to-head)

#### **Classe B: Features Diferenciais (29)**
- `form_diff`, `gd_diff`, `streak_diff`, `weighted_diff`
- `overall_diff`, `attack_diff`, `midfield_diff`, `defense_diff` (FIFA ratings)
- `corners_diff`, `shots_diff`, `shotsontarget_diff`, `goals_avg_diff`
- `position_diff`, `points_diff`
- `prob_home_norm`, `prob_draw_norm`, `prob_away_norm` (odds)
- 6 interaction features (DIA 9)

### **4 Modelos Calibrados**

| Modelo | Hiperparâmetros Otimizados | Calibração |
|--------|----------------------------|------------|
| RandomForest | n_estimators=50, max_depth=5 | Isotônica ✅ |
| XGBoost | n_estimators=200, lr=0.01, max_depth=3 | Isotônica ✅ |
| NaiveBayes | var_smoothing=1e-05 | Isotônica ✅ |
| SVM | C=0.1, gamma=0.001, kernel=RBF | Não aplicável |

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **scikit-learn** - Modelos ML, métricas, validação
- **XGBoost** - Gradient Boosting otimizado
- **pandas** - Manipulação de dados
- **numpy** - Operações numéricas
- **matplotlib + seaborn** - Visualizações
- **streamlit** - Interface web interativa
- **joblib** - Serialização de modelos
- **SHAP** (opcional) - Explicabilidade

---

---

## 🎯 Próximos Passos Sugeridos

### **Análises Pendentes (Para Publicação):**

1. **Significance Tests** (2h)
   - McNemar test (comparação de modelos)
   - Wilcoxon signed-rank test
   
2. **VIF Multicolinearity** (1h)
   - Calcular Variance Inflation Factor
   - Feature selection baseada em VIF
   
3. **Residual Analysis** (1.5h)
   - Padrões de erro por tipo de jogo
   - Quando os modelos falham mais

4. **Feature-Target Correlation** (1h)
   - Correlação de cada feature com Result
   - Identificar features mais preditivas

---

**Arquivos essenciais preservados:**
- ✅ `main.py`, `app.py`, `requirements.txt`, `README.md`
- ✅ Pasta `src/` (4 módulos principais)
- ✅ Pasta `data/` (todos os CSVs)
- ✅ Pasta `scripts/` (19 scripts essenciais)
- ✅ Pasta `models/` (resultados gerados)
