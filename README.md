## 📋 Índice

- [Visão Geral](#visão-geral)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Execução Rápida (Quick Start)](#execução-rápida-quick-start)
- [Execução Completa — Gerar Tudo](#execução-completa--gerar-tudo)
- [Passo a Passo Detalhado (Análises)](#passo-a-passo-detalhado-análises)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Scripts Principais](#scripts-principais)
- [Saída Esperada / Arquivos gerados](#saída-esperada--arquivos-gerados)
- [Notas e Troubleshooting](#notas-e-troubleshooting)

---

## 🎯 Visão Geral

Pipeline para predizer o resultado de partidas (Home Win / Draw / Away Win) com foco em reprodutibilidade científica e geração de artefatos prontos para publicação.

Principais pontos:
- 43 features (33 do artigo base + 10 extras)
- 4 modelos principais: RandomForest, XGBoost, NaiveBayes, SVM
- Split temporal: treino 2005–2014, teste 2014–2016 (evita data leakage)
- Otimização de hiperparâmetros com TimeSeriesSplit
- Calibração de probabilidades (quando aplicável)
- Geração de tabelas CSV e figuras PNG para relatórios
- Interface interativa via Streamlit

---

## 🔧 Pré-requisitos

- Python 3.8 ou superior
- pip
- Recomendo executar em ambiente virtual (`venv`) ou conda
- Em Windows use PowerShell ou cmd para ativar o `venv`

---

## 📦 Instalação

### Passo 1: Clonar / obter o repositório

```bash
cd C:\Users\seu_usuario\Desktop
git clone https://github.com/wiliafsilva/Projeto_ML.git
cd Projeto_ML
```

### Passo 2: Criar ambiente virtual

```bash
python -m venv .venv
```

### Passo 3: Ativar ambiente virtual

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Prompt (cmd.exe):

```cmd
.venv\Scripts\activate.bat
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Passo 4: Instalar dependências

```bash
pip install -r requirements.txt
```

Pacotes importantes: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`, `joblib`, `shap` (opcional para explicabilidade).

---

## 🚀 Processo de Execução Completo

Siga esta sequência para executar o projeto do zero até os resultados finais.


## 🚀 Execução Rápida (Quick Start)

Se já instalou as dependências e quer executar o pipeline principal (treinamento + salvar modelos) e abrir a interface:

```powershell
# 1) Ative o venv (PowerShell)
.\.venv\Scripts\Activate.ps1

# 2) Treine modelos e salve resultados
python main.py

# 3) Abra a interface (em outra janela/terminal)
streamlit run app.py
```

Abra http://localhost:8501 no navegador.
## Execução Completa — Gerar Tudo (comando único)

Se quiser gerar todos os artefatos (verificação, treinamento, tabelas, figuras e métricas) em sequência, use o script central `scripts/generate_all.py`. Ele executa os passos na ordem correta e salva os resultados em `models/`.

```powershell
# Ativar venv (PowerShell)
.\.venv\Scripts\Activate.ps1

# Gerar tudo (pode demorar alguns minutos)
python scripts/generate_all.py
```

Nota: `generate_all.py` invoca os scripts individuais; verifique `scripts/` se quiser rodar etapas separadamente.

## Passo a passo detalhado (rodando etapas individualmente)

1) Verificação dos dados

```powershell
'python scripts/verify_all.py'
```

2) Treinamento (pipeline principal)

```powershell
'python main.py'
```

3) Visualizar métricas resumo

```powershell
'python scripts/show_metrics.py'
```

4) Gerar análises científicas (tabelas e figuras)

```powershell
'python scripts/baseline_comparison.py
python scripts/correlation_heatmap.py
python scripts/confidence_intervals_fast.py
python scripts/radar_chart.py
python scripts/update_tabela3.py
python scripts/update_tabelas_5_6.py
python scripts/update_tabela4.py'
```

5) Análises extras (opcionais)

```powershell
'python scripts/feature_importance_simple.py
python scripts/shap_analysis.py
python scripts/test_features.py
python scripts/inspect_epl.py'
```

6) Otimização de hiperparâmetros (opcional)

```powershell
# Rápido (5-10min)
'python scripts/gridsearch_quick_test.py'

# Completo (1-2h)
'python scripts/gridsearch_advanced.py'
```

## Scripts principais

- `main.py` — pipeline principal (feature engineering, treino, calibração, salvamento)
- `scripts/generate_all.py` — executa toda geração de artefatos em sequência
- `scripts/verify_all.py` — valida integridade dos dados
- `scripts/show_metrics.py` — resumo das métricas salvo/carregado de `models/`
- `scripts/*` — scripts individuais para cada análise (ver lista completa na pasta `scripts/`)

## Saída esperada / arquivos gerados

Todos os arquivos são gerados dentro da pasta `models/`:

- `models/trained_models.pkl` — modelos treinados e metadados
- `models/*.csv` — tabelas científicas (tabela3, tabela4, tabela5, tabela6, baseline, confidence intervals, gridsearch)
- `models/figures/*.png` — heatmap, radar chart e boxplots

## Notas e troubleshooting

- Se faltar algum CSV de dados, o `verify_all.py` apontará arquivos/colunas ausentes.
- Em Windows, prefira PowerShell para ativar o `venv` com `Activate.ps1`.
- Se ocorrer erro de memória no GridSearch completo, reduza o número de folds ou use `gridsearch_quick_test.py`.
- Para usar a análise SHAP instale `shap` e execute `python scripts/shap_analysis.py` (pode ser lento).


#### **Interface Streamlit (Visualização Interativa)**

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

## 🔬 Metodologia

### **Split Temporal (Sem Data Leakage)**

```
Treino:  2005-2014 (9 temporadas) → 3420 partidas
Teste:   2014-2016 (2 temporadas) → 760 partidas
```

✅ **Sem random split** - Evita data leakage temporal  
✅ **Features incrementais** - Calculadas sequencialmente  
✅ **Validação cruzada temporal** - TimeSeriesSplit no GridSearch

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

**Arquivos essenciais preservados:**
- ✅ `main.py`, `app.py`, `requirements.txt`, `README.md`
- ✅ Pasta `src/` (4 módulos principais)
- ✅ Pasta `data/` (todos os CSVs)
- ✅ Pasta `scripts/` (19 scripts essenciais)
- ✅ Pasta `models/` (resultados gerados)
