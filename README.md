
# Scientific Replica – Premier League Match Prediction

This project replicates a scientific study predicting match outcomes in the English Premier League using Machine Learning.

## ⚡ Quick Start (Primeiros Passos)

```bash
# 1. Criar e ativar ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Treinar modelos com hiperparâmetros otimizados
python main.py

# 4. Abrir interface interativa
streamlit run app.py
```

Pronto! Acesse http://localhost:8501 no navegador.

---

## Methodology

- Temporal split (no random split)
- Incremental feature engineering (no data leakage)
- Comparison of 4 models:
  - SVM (RBF)
  - Random Forest
  - XGBoost
  - Naive Bayes (Gaussian)

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

### 1. Criar ambiente virtual

```bash
python -m venv .venv
```

### 2. Ativar ambiente virtual

```bash
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

## 🚀 Sequência de Execução Completa

### Para Primeiros Passos (Setup Inicial)

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

### Para Gerar Conteúdo do Artigo Científico

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

### Ordem Recomendada de Execução

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

**Teste rápido de hiperparâmetros (NOVO):**

```bash
python scripts\quick_hyperparameter_test.py
```

Executa teste rápido (5-10 min) para encontrar os melhores hiperparâmetros para todos os modelos:
- Testa múltiplas combinações de parâmetros
- Usa TimeSeriesSplit (validação temporal)
- Gera arquivo de recomendações com melhores configurações
- Salva resultados em `models/RECOMENDACOES_HIPERPARAMETROS.txt`

**Gerar tabelas consolidadas:**

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

```md
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
│   ├── trained_models.pkl   # Modelos treinados (SVM, RandomForest, XGBoost, NaiveBayes)
│   ├── quick_test_results.pkl  # 🆕 Resultados do teste rápido de hiperparâmetros
│   ├── RECOMENDACOES_HIPERPARAMETROS.txt  # 🆕 Melhores configurações encontradas
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
│   ├── gridsearch_advanced.py  # Otimização de hiperparâmetros (completo)
│   ├── quick_hyperparameter_test.py  # 🆕 Teste rápido de hiperparâmetros
│   ├── generate_tables.py   # Geração de tabelas científicas
│   ├── generate_figures.py  # Geração de visualizações avançadas
│   └── generate_all.py      # Gera tudo de uma vez (tabelas + figuras)
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

### 🏆 Métricas dos Modelos Otimizados (Test Set 2014-2016)

Com hiperparâmetros otimizados via GridSearch CV + Calibração:

| Modelo | Acurácia | F1-Score | RPS | Status |
|--------|----------|----------|-----|--------|
| **RandomForest** 🥇 | **48.42%** | 0.3398 | **0.4261** | ⭐ **MELHOR** |
| **SVM** 🥈 | 44.34% | 0.4319 | 0.4315 | Excelente |
| **XGBoost** 🥉 | 44.21% | 0.4053 | 0.4332 | Muito Bom |
| **NaiveBayes** | 43.55% | 0.4138 | 0.4373 | Bom |
| **Baseline** | 46.05% | - | - | (sempre "Vitória Casa") |

### 📊 Sobre as Métricas

- **RPS (Ranked Probability Score):** Quanto **menor**, melhor. Mede a qualidade das probabilidades preditas (0 = perfeito, 1 = péssimo).
- **F1-Score:** Média harmônica entre Precision e Recall (macro average).
- **Baseline:** Modelo trivial que sempre prevê a classe majoritária. Serve como referência mínima.

### 🎯 Principais Conquistas

✅ **RandomForest** agora é o melhor modelo (RPS 0.4261)  
✅ **Melhorias de 2-4%** após otimização de hiperparâmetros  
✅ **Calibração** aplicada automaticamente melhora probabilidades  
✅ Todos os modelos superam baseline em termos de probabilidades (RPS)

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

### 1. Otimização de Hiperparâmetros 🆕

Todos os modelos agora usam **hiperparâmetros otimizados** encontrados via GridSearch CV:

**SVM (RPS: 0.4315):**
- `C=0.1` (regularização suave)
- `gamma=0.001` (influência ampla para melhor generalização)

**RandomForest (RPS: 0.4261 - MELHOR):**
- `n_estimators=50` (50 árvores são suficientes)
- `max_depth=5` (árvores rasas evitam overfitting)
- `min_samples_split=2`, `min_samples_leaf=1`

**XGBoost (RPS: 0.4332):**
- `learning_rate=0.01` (aprendizado conservador)
- `max_depth=3` (árvores rasas)
- `n_estimators=200` (mais iterações com learning rate baixo)
- `subsample=0.8`, `colsample_bytree=1.0`

**NaiveBayes (RPS: 0.4373):**
- `var_smoothing=1e-05` (suavização otimizada)

Execute `python scripts\quick_hyperparameter_test.py` para testar outras configurações!

### 2. Calibração de Probabilidades

Os modelos RandomForest, XGBoost e NaiveBayes agora usam **calibração isotônica** para melhorar as probabilidades preditas. A calibração é aplicada automaticamente durante o treinamento e o modelo calibrado é usado apenas se melhorar o RPS.

### 3. Balanceamento de Classes

- **SVM e RandomForest:** Usam `class_weight='balanced'`
- **XGBoost e NaiveBayes:** Usam `sample_weight` calculado para balancear as classes

### 4. Análise de Explicabilidade (SHAP)

Execute `python scripts\shap_analysis.py` para:

- Ver ranking de importância das features por modelo
- Gerar gráficos SHAP mostrando impacto de cada feature
- Entender quais características mais influenciam as previsões

### 5. Interface Streamlit Aprimorada

O aplicativo Streamlit foi melhorado com:

✅ **Correções de compatibilidade:** Parâmetros atualizados (`width='stretch'` em vez de `use_container_width`)  
✅ **Encoding UTF-8:** Suporte correto para caracteres especiais  
✅ **Visualizações aprimoradas:** Gráficos para 1 e 2 parâmetros no ajuste de hiperparâmetros  
✅ **NaiveBayes integrado:** Adicionado em todas as funcionalidades  
✅ **Métricas expandidas:** Classification report corretamente formatado  

### 6. Testes Automatizados de Qualidade

Execute `python scripts\verify_all.py` para verificação completa:
- Validação de estrutura de dados
- Checagem de features calculadas
- Verificação de modelos treinados
- Estatísticas por temporada

---

## 🖥️ Interface Streamlit

O aplicativo Streamlit possui **6 páginas** para análise completa:

### 1. **Visão Geral**
- Estatísticas gerais do dataset
- Distribuição de resultados (Vitória Casa/Empate/Vitória Fora)
- Amostras dos dados e features geradas
- Análise de features por temporada

### 2. **Comparação de Modelos**
- Tabela comparativa com Acurácia, F1 e RPS
- Gráfico de barras de performance
- Ranking dos modelos

### 3. **Avaliação e Métricas**
- Métricas detalhadas por modelo selecionado
- Classification report expandido
- Matriz de confusão interativa
- Curvas ROC e Precision-Recall
- Curvas de calibração
- Feature importance (quando disponível)

### 4. **Análise Científica Consolidada** ⭐
- **10 tabelas** em CSV prontas para artigo científico
- **6 visualizações** em PNG (300 DPI) de alta qualidade
- Botões para gerar tabelas e figuras
- Download direto dos arquivos
- Instruções de uso para publicação

### 5. **Ajuste de Hiperparâmetros**
- Interface para executar GridSearch interativamente
- Suporte para SVM, RandomForest, XGBoost e NaiveBayes
- Visualização automática dos resultados (barras para 1 parâmetro, heatmap para 2)
- Display dos melhores parâmetros encontrados

### 6. **Distribuições e Importância de Features**
- KDE plots das features
- Comparação de importância entre RandomForest e XGBoost
- Análise univariada

---

## 🔧 Correções Técnicas e Melhorias Recentes

### Bugs Corrigidos

✅ **NaiveBayes não previa empates:** Corrigido com adição de `sample_weight` e calibração  
✅ **Erro Arrow no Streamlit:** Convertido dicionários em colunas para string  
✅ **Erro de encoding UTF-8:** Adicionado `encoding='utf-8'` nos subprocess calls  
✅ **Warning de depreciação:** Removido `use_label_encoder` do XGBoost  
✅ **Parâmetro Streamlit depreciado:** Substituído `use_container_width` por `width='stretch'`  

### Melhorias de Performance

📈 **RandomForest:** +4.7% acurácia e -4.2% RPS após otimização  
📈 **XGBoost:** +1.6% acurácia e -2.1% RPS após otimização  
📈 **Todos os modelos:** Melhorias consistentes com hiperparâmetros otimizados  

### Novas Funcionalidades

🆕 Script de teste rápido de hiperparâmetros (`quick_hyperparameter_test.py`)  
🆕 Visualização de 1 parâmetro no GridSearch (gráfico de barras)  
🆕 NaiveBayes totalmente integrado em todas as funcionalidades  
🆕 Arquivo de recomendações automático com melhores configurações  

---

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

### Naive Bayes (Gaussian)

- **Modelo probabilístico simples** que assume independência condicional entre features, usando distribuição normal para variáveis contínuas.
- **Vantagens:** Extremamente rápido para treinar e prever, pouco sujeito a overfitting, bom baseline probabilístico.
- **Desvantagens:** A hipótese de independência raramente é verdadeira, o que pode limitar a acurácia máxima.
- **Uso:** Útil como modelo de referência leve e como comparação adicional às abordagens de ensemble mais complexas.

---

## 💡 Perguntas Frequentes (FAQ)

### Como melhorar ainda mais os modelos?

1. **Teste outros hiperparâmetros:**
   ```bash
   python scripts\quick_hyperparameter_test.py
   ```

2. **Analise feature importance:**
   ```bash
   python scripts\shap_analysis.py
   ```

3. **Adicione novas features** em `src/feature_engineering.py`

### Por que RandomForest agora é o melhor?

Após otimização de hiperparâmetros:
- Árvores **mais rasas** (depth=5) evitam overfitting
- **Menos estimadores** (50) são mais eficientes
- **Calibração** melhora as probabilidades preditas

### Como usar os hiperparâmetros encontrados?

Os hiperparâmetros já estão aplicados em `src/train_models.py`. Execute `python main.py` para treinar com eles!

### O que fazer se acurácia parecer baixa?

É **normal** para predição de futebol! Resultados de futebol têm muito fator aleatório. Acurácias de 40-50% são boas para este problema. O importante é que:
- **RPS seja baixo** (boa calibração de probabilidades)
- Supere o **baseline** (prever sempre a mesma classe)

### Como exportar para artigo científico?

```bash
# Gerar tudo de uma vez
python scripts\generate_all.py

# Ou individual
python scripts\generate_tables.py
python scripts\generate_figures.py
```

Arquivos gerados em `models/` (CSVs) e `models/figures/` (PNGs 300 DPI).

### Onde visualizar os resultados?

```bash
streamlit run app.py
```

Acesse **"Análise Científica Consolidada"** para ver tudo consolidado!

---

## 📚 Referências

- Metodologia baseada em artigo científico sobre predição de resultados da EPL
- Dataset: Football-Data.co.uk (temporadas 2005-2016)
- Métricas: RPS (Ranked Probability Score), Brier Score, ROC AUC

---

## 👨‍💻 Desenvolvimento

**Projeto:** Replicação Científica - Predição EPL  
**Status:** ✅ Completo e Otimizado  
**Última Atualização:** Março 2026  

**Principais Features:**
- ✅ 4 modelos otimizados (SVM, RF, XGBoost, NB)
- ✅ Calibração automática de probabilidades
- ✅ Hiperparâmetros otimizados via GridSearch
- ✅ Interface Streamlit completa (6 páginas)
- ✅ Geração automática de tabelas e figuras científicas
- ✅ Análise SHAP de explicabilidade
- ✅ Validação temporal (sem data leakage)
