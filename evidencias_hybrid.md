# Evidências: Decoder Hybrid (projeto Projeto_ML)

Objetivo: catalogar todo o material científico e experimental disponível relacionado ao módulo/abordagem "Hybrid" (Decoder Hybrid - reconstructed-only 43D).

---

**Visão Geral do Projeto**

- Fonte principal: pipeline "Decoder Hybrid" implementado em `src/train_models.py` (função `train_models_with_decoder_hybrid`) que treina um autoencoder e usa o output do decoder (43 features reconstruídas) como features "hybrid" para classificadores.
- Local dos artefatos: [models/autoencoder_decoder_hybrid](models/autoencoder_decoder_hybrid)
- Scripts de geração: vários scripts em `scripts/` com prefixo `hybrid_*` e `generate_hybrid_all.py` para orquestrar.

---

**Arquivos Relacionados ao Hybrid**

- `src/train_models.py` — função `train_models_with_decoder_hybrid` (pipeline descrito). (aprox. linhas 820-1160)
- `models/autoencoder_decoder_hybrid/` — diretório contendo pesos, modelos, CSVs, figuras e artefatos: lista completa:
  - `autoencoder_hybrid_weights.h5`
  - `encoder_hybrid.keras`
  - `decoder_hybrid.keras`
  - `scaler_hybrid.joblib`
  - `trained_models_hybrid.pkl` (metadados + modelos treinados)
  - `hybrid_model_results.csv`
  - `hybrid_model_results_by_season.csv`
  - `hybrid_feature_importance.csv`
  - `hybrid_confidence_intervals.csv`
  - `hybrid_baseline_comparison.csv`
  - `hybrid_correlation_matrix.csv`
  - `tabela3_hybrid_comparacao.csv`
  - `tabela4_cm_hybrid_*.csv` (confusion matrices por modelo)
  - `tabela5_hybrid_performance_temporada.csv`
  - `tabela6_hybrid_classificacao_classe.csv`
  - `figures/` — múltiplas PNGs (confusion matrices, heatmaps, radar charts, feature importance, CI plots etc.)
- `scripts/generate_hybrid_all.py` — orquestra geração de todos os artefatos Hybrid.
- `scripts/hybrid_baseline_comparison.py` — compara modelos Hybrid com baselines dummy.
- `scripts/hybrid_confidence_intervals.py` — calcula bootstrap (N=100) e gera plots.
- `scripts/hybrid_feature_importance.py` — extrai importâncias (RandomForest) e gera gráficos.
- `scripts/hybrid_correlation_heatmap.py` — gera matriz de correlação e heatmap.
- `scripts/hybrid_radar_chart.py`, `scripts/hybrid_additional_visualizations.py`, `scripts/hybrid_performance_*` — visualizações e tabelas auxiliares.

---

**Metodologia Identificada**

- Pipeline (fonte): `src/train_models.py` — etapas principais:
  1. Preparação de features originais via `src/feature_engineering.calculate_team_stats` (produz 43 features, inclui Form ELO-style, μk, H2H, posição na tabela, odds, interactions).
  2. Escalonamento (`MinMaxScaler`).
  3. Treinamento de Autoencoder (`AutoencoderLatent`) com `latent_dim` (padrão 8), loss MAE, callbacks `EarlyStopping` e `ReduceLROnPlateau`.
  4. Cálculo de reconstruction error; detecção de anomalias por percentil (`anomaly_percentile`, padrão 95) e filtragem de amostras (treino) consideradas anômalas.
  5. Criação de features híbridas: atualmente o pipeline usa APENAS as features reconstruídas pelo decoder (`X_train_hybrid = X_train_reconstructed`) — 43D reconstructed features.
  6. Treinamento de classificadores com features híbridas (SVM, RandomForest, XGBoost, NaiveBayes) e avaliação.

---

**Algoritmos Implementados**

- Autoencoder (classe `AutoencoderLatent` em `src/train_models.py` / possivelmente em `src/analysis.py` ou módulos relacionados). Arquivos: `models/autoencoder_decoder_hybrid/autoencoder_hybrid_weights.h5`, `encoder_hybrid.keras`, `decoder_hybrid.keras`.
- Classificadores:
  - Support Vector Machine (`sklearn.svm.SVC`) — `SVM`
  - Random Forest (`sklearn.ensemble.RandomForestClassifier`) — `RandomForest`
  - XGBoost (`xgboost.XGBClassifier`) — `XGBoost`
  - Gaussian Naive Bayes (`sklearn.naive_bayes.GaussianNB`) — `NaiveBayes`

Fonte de declaração dos algoritmos: `src/train_models.py` (definição do dicionário `models`) e `scripts/hybrid_baseline_comparison.py`.

---

**Parâmetros Utilizados**

- Parâmetros gerais do autoencoder:
  - `latent_dim`: default 8 (ver `train_models_with_decoder_hybrid` e metadata salvo em `trained_models_hybrid.pkl`).
  - Otimização: `optimizer='adam'`, `loss='mae'`, `epochs=200`, `batch_size=min(2048, len(X_train_scaled))`, callbacks `EarlyStopping(patience=10,min_delta=1e-4)` e `ReduceLROnPlateau(factor=0.5,patience=5,min_lr=1e-6)`.
- Anomalias:
  - `anomaly_percentile`: default 95 (threshold definido via `np.percentile(reconstruction_errors_train, anomaly_percentile)`).
- Classificadores — parâmetros por padrão (podem ser sobrescritos por `models/optimized_models.pkl` se existir):
  - SVM: `probability=True, kernel='rbf', C=0.1, gamma=0.001, random_state=42, class_weight='balanced'`.
  - RandomForest: `n_estimators=50, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42, class_weight='balanced'`.
  - XGBoost: `eval_metric='mlogloss', n_estimators=200, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=1.0, random_state=42`.
  - NaiveBayes: `var_smoothing=1e-05`.
  - Sample weighting: `sample_weights_clean = compute_sample_weight('balanced', y_train_clean)` é usado para alguns modelos (XGBoost, NaiveBayes em trecho presente).

Observação: função `load_optimized_params(path='models/optimized_models.pkl')` tenta aplicar parâmetros otimizados caso o arquivo exista; catalogar se `models/optimized_models.pkl` existe (não listado anteriormente). Se faltar, os parâmetros acima são os usados.

---

**Configurações Experimentais**

- Divisão temporal utilizada (constante no projeto):
  - Treino: seasons 2005-2014 (diretório `data/data_2005_2014`)
  - Teste: seasons 2014-2016 (diretório `data/data_2014_2016`)
  - O código registra `train_period='2005-2014'` e `test_period='2014-2016'` nos metadados.
- Validação estatística: bootstrap (script `scripts/hybrid_confidence_intervals.py`):
  - `N_ITERATIONS = 100` (observação: CSV salvo `hybrid_confidence_intervals.csv` mostra margens e CI para métricas).
- Grid search / otimização:
  - Há suporte para carregar resultados otimizados (`models/optimized_models.pkl`) via `load_optimized_params`, e existe arquivo `models/gridsearch_advanced_por_temporada.csv` e `models/gridsearch_results.csv` no repositório (indicando que gridsearch foi executado em outros scripts), mas não há evidência direta de que esses resultados foram aplicados ao pipeline Hybrid além do carregamento condicional.

---

**Métricas de Avaliação**

- Métricas calculadas e salvas:
  - `accuracy` (accuracy_score)
  - `f1` (macro F1, `average='macro'`)
  - `rps` (Ranked Probability Score — implementado em `src/train_models.py` função `rps`) — utilizado como métrica probabilística principal em vários lugares.
  - `precision`, `recall` (usadas em `hybrid_baseline_comparison` e tabelas por classe)
  - Além disso, `hybrid_confidence_intervals.py` calcula média, std e intervalos CI_Lower/CI_Upper para `ACCURACY`, `F1`, `RPS`.

Fontes: `models/autoencoder_decoder_hybrid/hybrid_model_results.csv`, `hybrid_model_results_by_season.csv`, `hybrid_confidence_intervals.csv`, `hybrid_baseline_comparison.csv`, `tabela5_hybrid_performance_temporada.csv`, `tabela6_hybrid_classificacao_classe.csv`.

---

**Conjuntos de Dados Utilizados**

- Dados de partidas brutas: arquivos CSV em `data/data_2005_2014/` e `data/data_2014_2016/` (cada temporada um CSV, ex.: `Season_2014_2015.csv`).
- A pipeline usa `src.preprocessing.load_multiple_seasons` para agregar estes diretórios.
- Recursos externos: há menção a Ratings FIFA e odds Bet365 nas features; `src/feature_engineering` adiciona ratings e odds apenas se colunas existirem no dataset de entrada (checagem de presença). No workspace há `data/fifa_ratings.csv` — indica disponibilidade parcial de ratings.

---

**Gráficos Gerados**

- Diretório `models/autoencoder_decoder_hybrid/figures` contém (lista parcial):
  - `hybrid_confidence_intervals_barplot.png`
  - `hybrid_confidence_intervals_boxplot.png`
  - `hybrid_confidence_intervals_distribution.png`
  - `hybrid_confusion_matrix_naivebayes.png`
  - `hybrid_confusion_matrix_randomforest.png`
  - `hybrid_confusion_matrix_svm.png`
  - `hybrid_confusion_matrix_xgboost.png`
  - `hybrid_correlation_heatmap.png`
  - `hybrid_feature_importance_by_type.png`
  - `hybrid_feature_importance_distribution.png`
  - `hybrid_feature_importance_top20.png`
  - `hybrid_performance_by_season.png`
  - `hybrid_performance_heatmap.png`
  - `hybrid_prediction_confidence.png`
  - `radar_chart_hybrid_2014-2015.png`, `radar_chart_hybrid_2015-2016.png`, `radar_chart_hybrid_All.png`

Cada figura gerada por scripts em `scripts/` (ver `hybrid_confidence_intervals.py`, `hybrid_feature_importance.py`, `hybrid_correlation_heatmap.py`, `hybrid_radar_chart.py`).

---

**Tabelas Disponíveis**

- `models/autoencoder_decoder_hybrid/hybrid_model_results.csv` — resumo por modelo (`model,accuracy,f1,rps`).
- `models/autoencoder_decoder_hybrid/hybrid_model_results_by_season.csv` — resultados por temporada.
- `models/autoencoder_decoder_hybrid/hybrid_confidence_intervals.csv` — média, std e intervalos de confiança por métrica e modelo.
- `models/autoencoder_decoder_hybrid/hybrid_baseline_comparison.csv` — comparação com baselines (Most Frequent, Stratified).
- `models/autoencoder_decoder_hybrid/tabela3_hybrid_comparacao.csv` — comparação Latent vs Hybrid (e.g. 8D vs reconstructed 43D) com métricas summary.
- `models/autoencoder_decoder_hybrid/tabela5_hybrid_performance_temporada.csv` — performance por temporada (2014-2015, 2015-2016, All).
- `models/autoencoder_decoder_hybrid/tabela6_hybrid_classificacao_classe.csv` — métricas por classe (Precision, Recall, F1, Support) por modelo.
- `models/autoencoder_decoder_hybrid/hybrid_feature_importance.csv` — importância das features (RandomForest feature_importances_).
- `models/autoencoder_decoder_hybrid/hybrid_correlation_matrix.csv` — matriz de correlação 43×43.

---

**Resultados Quantitativos**

- Valores extraídos (arquivo `hybrid_model_results.csv`):
  - SVM: accuracy=0.4986842105, f1=0.4898310475, rps=0.2070908681
  - RandomForest: accuracy=0.4802631579, f1=0.4476847586, rps=0.2094619232
  - XGBoost: accuracy=0.5, f1=0.4625694031, rps=0.2074255417
  - NaiveBayes: accuracy=0.4789473684, f1=0.4725036244, rps=0.3045887074

- Por temporada (`hybrid_model_results_by_season.csv` / `tabela5_hybrid_performance_temporada.csv`):
  - 2014-2015: SVM acc=0.52105, RF=0.51053, XGB=0.51316, NB=0.50263 (n=380)
  - 2015-2016: SVM acc=0.47632, RF=0.45, XGB=0.48684, NB=0.45526 (n=380)

- Tabelas por classe (`tabela6_hybrid_classificacao_classe.csv`):
  - Exemplo: RandomForest: H precision=0.5691 recall=0.6261 F1=0.5962 (support 329); D precision=0.3095 recall=0.26 F1=0.2826 (support 200); A precision=0.4652 recall=0.4632 F1=0.4642 (support 231).

- Intervalos de confiança (`hybrid_confidence_intervals.csv`): contém média, std e CI_Lower/CI_Upper para ACCURACY, F1, RPS por modelo (bootstrap N=100). Ex.: SVM ACCURACY Mean=0.50016 CI=[0.47230,0.53365].

---

**Comparações Realizadas**

- Hybrid (43D reconstructed) vs Latent (8D) — `tabela3_hybrid_comparacao.csv` contém linhas comparativas para cada modelo: RandomForest, XGBoost, NaiveBayes, SVM com métricas para Hybrid (43D) e Latent (8D).
- Hybrid models vs Baselines (Most Frequent, Stratified) — `hybrid_baseline_comparison.csv` e script `scripts/hybrid_baseline_comparison.py`.
- Comparações por temporada (2014-2015 vs 2015-2016) — `hybrid_model_results_by_season.csv` e `tabela5_hybrid_performance_temporada.csv`.
- Comparações de métricas probabilísticas e intervalos via bootstrap — `hybrid_confidence_intervals.csv` e gráficos correspondentes.

---

**Conclusões Registradas (textuais/implícitas)**

- Sumário em `scripts/show_hybrid_summary.py` e nas mensagens impressas em `train_models_with_decoder_hybrid` indicam que:
  - Pipeline escolhido para Hybrid usa apenas features reconstruídas do decoder (43D reconstructed).
  - Modelos ML (Hybrid) superam baselines simples em acurácia e F1 (e.g., SVM/XGBoost frequentemente melhores que Most Frequent baseline).
  - Variação por temporada observada (2014-2015 mais previsível que 2015-2016).
  - RandomForest tem viés em algumas execuções (no outro pipeline original RF às vezes não prevê empates), mas nesta variante híbrida RF oferece métricas intermediárias.
  - Feature importance (RandomForest) mostra ranking de features reconstruídas — top features `Recon_2`, `Recon_26`, `Recon_36`, etc., indicadas em `hybrid_feature_importance.csv`.

Fonte: mensagens e prints em `src/train_models.py`, `scripts/show_hybrid_summary.py`, `scripts/hybrid_baseline_comparison.py`, e os CSVs gerados.

---

**Localização Aproximada das Evidências (arquivos / seções)**

- Implementação pipeline Hybrid: `src/train_models.py` linhas ~820-1160 (função `train_models_with_decoder_hybrid`).
- Parâmetros de treino e modelos: `src/train_models.py` início (definições de `svm_params`, `rf_params`, `xgb_base_params`, `nb_params`) e `load_optimized_params` no topo.
- Tabelas e resultados resumidos: `models/autoencoder_decoder_hybrid/hybrid_model_results.csv`, `hybrid_model_results_by_season.csv`, `tabela3_hybrid_comparacao.csv`, `tabela5_hybrid_performance_temporada.csv`, `tabela6_hybrid_classificacao_classe.csv`.
- Métricas de incerteza: `models/autoencoder_decoder_hybrid/hybrid_confidence_intervals.csv` e figuras correspondentes em `figures/`.
- Importância de features: `models/autoencoder_decoder_hybrid/hybrid_feature_importance.csv` e figuras `hybrid_feature_importance_top20.png`.
- Matriz de correlação: `models/autoencoder_decoder_hybrid/hybrid_correlation_matrix.csv` e `figures/hybrid_correlation_heatmap.png`.
- Scripts utilitários e orquestração: `scripts/generate_hybrid_all.py`, `scripts/hybrid_*`.

---

**Limitações da Evidência Encontrada**

- Não inventar: somente cataloguei artefatos existentes.
- Incertezas / pontos não explicitamente documentados nos artefatos:
  1. Não há descrição textual longa no repositório que explique racionalmente por que o pipeline final usa apenas as features reconstruídas do decoder (mudança marcada como "UPDATED: usar apenas as features reconstruídas do decoder"). Necessário documentar justificativa experimental (por que não usar latent + reconstr error etc.).
  2. O arquivo `models/optimized_models.pkl` é referenciado (e há CSVs de gridsearch), mas não verifiquei se esse pickle existe e foi aplicado ao pipeline Hybrid; precisa confirmar presença e conteúdo de `models/optimized_models.pkl` se pretende usar parâmetros otimizados.
  3. O número de iterações do bootstrap é 100 (em `scripts/hybrid_confidence_intervals.py`), enquanto no resumo grande (RESUMO.md) foi mencionado 1.000 iterações; há discrepância entre documento RESUMO.md e os scripts reais (anotar). [INCERTEZA: discrepância detectada]
  4. Algumas métricas probabilísticas (ex.: Brier Score, ROC AUC) mencionadas no `RESUMO.md` não aparecem nos CSVs Hybrid (os arquivos atuais incluem accuracy, f1, rps; `hybrid_baseline_comparison.csv` inclui precision/recall). Confirmar se Brier/ROC foram calculados em pipelines alternativos.
  5. Possível ausência de `models/optimized_models.pkl` ou `models/autoencoder_hybrid.keras` (autoencoder salvo como `autoencoder_hybrid_weights.h5` e `encoder_hybrid.keras`/`decoder_hybrid.keras` existem) — confirmar integridade dos modelos salvos se for reproduzir experimentos.

---

**Informações Ausentes para um Artigo Científico (necessárias para completar)**

- Justificativa/motivação detalhada da escolha do pipeline "decoder-only" (por que usar apenas reconstructed features e não latent features combinadas).
- Descrição formal do Autoencoder: arquitetura, número de camadas, unidades por camada, funções de ativação, número de parâmetros — código da classe `AutoencoderLatent` (não lido aqui) deve ser inserido no manuscrito.
- Verificação de reprodução: seeds, runtime (hardware), duração de treino, e versão exata das bibliotecas usadas (requirements.txt fornece versões gerais, mas metadados experimentais poderiam ser mais explícitos no diretório `models/autoencoder_decoder_hybrid/trained_models_hybrid.pkl`).
- Evidência completa de hyperparameter tuning: existência e conteúdo de `models/optimized_models.pkl` e `models/gridsearch_advanced_por_temporada.csv` (há CSVs, mas o pickle deve ser inspecionado).
- Resultados complementares solicitáveis: Brier score, ROC AUC, matriz de confusão numérica (CSV já existe), análise de calibração (calibration plots), estatísticas de importância SHAP (há `shap_analysis.py` no repositório — verificar se rodou para Hybrid).
- Racional e testes para tratamento de classes desbalanceadas (e.g., SMOTE, threshold tuning) — não parece ter sido sistematicamente testado para Hybrid (apenas `class_weight='balanced'` e sample weights em alguns modelos).

---

**Avaliação da Qualidade da Evidência Disponível**

- Pontos fortes:
  - Repositório contém artefatos completos de execuções (modelos Keras salvos, pickles com modelos, CSVs de resultados, figuras) permitindo ver métricas e reproduzir a maior parte dos experimentos.
  - Pipeline implementado com cuidado para evitar data leakage (features calculadas apenas com dados anteriores, resets por temporada) — evidência em `src/feature_engineering.py`.
  - Comparações temporais e bootstrap para incerteza estão presentes (scripts e CSVs).

- Limitações:
  - Falta documentação textual consolidada que justifique escolhas experimentais (por exemplo, por que usar somente reconstructed features).
  - Pequenas discrepâncias documentais (RESUMO.md menciona 1.000 iterações bootstrap; script usa N=100) que devem ser reconciliadas.
  - Algumas métricas mencionadas em resumos podem não ter sido calculadas no pipeline Hybrid mostrado (p.ex. Brier, ROC AUC), exigir confirmação.

Conclusão qualitativa: evidência experimental suficiente para construir uma seção de Métodos e Resultados com tabelas, figuras e intervalos de confiança; porém falta documentação técnica detalhada (arquitetura do autoencoder, justificativas metodológicas, log de hyperparameter tuning aplicado ao Hybrid) e confirmação de algumas métricas para uma submissão científica robusta.

---

**Itens recomendados a produzir/confirmar antes de escrever o artigo**

1. Inspecionar/validar conteúdo de `models/optimized_models.pkl` (se existir) e, se aplicável, explicar quais hiperparâmetros foram usados para os resultados Hybrid.
2. Extrair a definição completa da arquitetura do autoencoder (`AutoencoderLatent`) e salvar em `models/autoencoder_decoder_hybrid/autoencoder_architecture.txt` ou similar.
3. Reconciliar número de iterações de bootstrap entre `RESUMO.md` e `scripts/hybrid_confidence_intervals.py`; padronizar (recomendo N=1000 para robustez) e re-gerar CSVs/figuras se necessário.
4. Rodar SHAP analysis para modelos Hybrid (há `scripts/shap_analysis.py`) e salvar valores SHAP médios por feature para incluir discussão de interpretabilidade.
5. Calcular e registrar métricas adicionais (Brier score, ROC AUC macro, curves de calibração) se forem necessárias no manuscrito.
6. Documentar decisões de pré-processamento (ausência/imputação de ratings FIFA para times promovidos), seeds e ambiente (versões de libs), e incluir no repositório (ex.: `models/autoencoder_decoder_hybrid/experiment_metadata.json`).

---

Arquivo gerado com base nos artefatos existentes no repositório (escopo: `models/autoencoder_decoder_hybrid`, `src/train_models.py`, scripts `scripts/hybrid_*.py` e `src/feature_engineering.py`).

Se desejar, prossigo com:
- (A) inspecionar `models/optimized_models.pkl` e `trained_models_hybrid.pkl` e listar parâmetros exatos; e/ou
- (B) executar `scripts/hybrid_confidence_intervals.py` com N=1000 para fortalecer CI.

Fim do inventário.
# Auditoria: Evidências do módulo "Hybrid"

Objetivo: mapear e catalogar todo o material científico / experimental disponível relacionado ao pipeline/abordagem denominada "Hybrid" (Decoder Hybrid / reconstructed-only 43D) para uso na redação de artigo científico.

Data da auditoria: 2026-06-15
Repositório: workspace local

---

## Visão Geral do Projeto

Resumo: O repositório contém um pipeline denominado "Decoder Hybrid" que cria features híbridas a partir de um autoencoder (encoder + decoder). O fluxo principal detecta anomalias via reconstruction error, filtra amostras, usa as saídas do decoder (43 features reconstruídas) como novo espaço de features e treina classificadores (SVM, RandomForest, XGBoost, NaiveBayes). Resultados, tabelas, figuras e artefatos treinados são salvos em `models/autoencoder_decoder_hybrid`.

Arquivos-chave explorados: `src/train_models.py`, `scripts/generate_hybrid_all.py`, `scripts/hybrid_* .py`, e o diretório `models/autoencoder_decoder_hybrid/`.

---

## Arquivos Relacionados ao Hybrid

- `src/train_models.py` (caminho: [src/train_models.py](src/train_models.py#L1-L140) e trecho do pipeline em [src/train_models.py](src/train_models.py#L820-L1160))
  - Implementa `train_models_with_decoder_hybrid(...)` (pipeline completo: preparar dados, treinar autoencoder, detectar anomalias, criar features híbridas a partir do decoder, treinar classificadores, avaliar por temporada, salvar resultados e metadados).
  - Uso: fonte principal do processo experimental.

- `models/autoencoder_decoder_hybrid/` (diretório)
  - Artefatos salvos:
    - `trained_models_hybrid.pkl` (modelos treinados + metadados)
    - `scaler_hybrid.joblib` (MinMaxScaler)
    - `encoder_hybrid.keras`, `decoder_hybrid.keras`, `autoencoder_hybrid_weights.h5` (modelos Keras)
    - CSVs: `hybrid_model_results.csv`, `hybrid_model_results_by_season.csv`, `hybrid_baseline_comparison.csv`, `hybrid_confidence_intervals.csv`, `hybrid_feature_importance.csv`, `hybrid_correlation_matrix.csv`, `tabela3_hybrid_comparacao.csv`, `tabela4_cm_hybrid_*.csv`, `tabela5_hybrid_performance_temporada.csv`, `tabela6_hybrid_classificacao_classe.csv`, entre outros.
    - Figures: em `models/autoencoder_decoder_hybrid/figures/` (vários PNG — ver seção "Gráficos e Figuras Disponíveis").
  - Uso: principal repositório de evidências quantitativas e imagens.

- `scripts/generate_hybrid_all.py` (scripts orchestrator)
  - Executa em sequência scripts: atualizar tabelas (update_hybrid_tabela3/4/5_6), baseline comparison, correlation heatmap, radar charts, feature importance, bootstrap CI, visualizações adicionais.

- `scripts/hybrid_baseline_comparison.py` (scripts/hybrid_baseline_comparison.py)
  - Compara modelos Hybrid com baselines (`Most Frequent`, `Stratified`) usando conjuntos: `data/data_2005_2014` (treino) e `data/data_2014_2016` (teste).

- `scripts/hybrid_confidence_intervals.py` (scripts/hybrid_confidence_intervals.py)
  - Implementa bootstrap (N_ITERATIONS = 100 no código) para obter CI 95% de `accuracy`, `f1`, `rps` e gera figuras (boxplot, barplot, distribuições).

- `scripts/hybrid_feature_importance.py` (scripts/hybrid_feature_importance.py)
  - Extrai `feature_importances_` do RandomForest salvo e produz CSV `hybrid_feature_importance.csv` e gráficos (top20, por tipo, distribuição).

- Outros scripts relacionados:
  - `scripts/hybrid_correlation_heatmap.py` (gera `hybrid_correlation_matrix.csv` e heatmap PNG)
  - `scripts/hybrid_radar_chart.py`, `scripts/hybrid_additional_visualizations.py`, `scripts/show_hybrid_summary.py` (sumário dos artefatos)
  - `scripts/generate_hybrid_all.py` (orquestrador)
  - `scripts/update_hybrid_tabela3.py`, `scripts/update_hybrid_tabela4.py`, `scripts/update_hybrid_tabelas_5_6.py` (geram as tabelas finais)

Observação: muitos scripts carregam os modelos a partir de `models/autoencoder_decoder_hybrid/trained_models_hybrid.pkl` e utilizam `encoder_hybrid.keras` + `decoder_hybrid.keras` + `scaler_hybrid.joblib` para recomputar features híbridas.

---

## Metodologia Identificada

Fontes primárias: `src/train_models.py` (implementação do pipeline) e scripts em `scripts/` que geram tabelas e figuras.

Pipeline (concreto, conforme `train_models_with_decoder_hybrid`):
- ETAPA 1: Preparar dados — `src._prepare_autoencoder_features` (calcula features originais via `src/feature_engineering.calculate_team_stats`) e aplica `MinMaxScaler`.
- ETAPA 2: Treinar AutoencoderLatent (classe `AutoencoderLatent`) com `latent_dim` (default 8). Treino com `optimizer='adam'`, `loss='mae'`, `epochs=200` (callbacks: EarlyStopping e ReduceLROnPlateau).
- ETAPA 3: Detectar anomalias via reconstruction error; threshold definido por `anomaly_percentile` (padrão 95); remove amostras com error >= threshold (usa apenas dados limpos para treino dos classificadores).
- ETAPA 4: Criar features híbridas:
  - `X_train_hybrid` = saída do decoder (reconstructed features) — 43 dimensões segundo scripts.
  - `latent` é calculado mas não utilizado diretamente para treinar (no fluxo atual usam apenas reconstructed features). Há variantes no repositório (latent-only experiments) mas o foco Hybrid usa reconstructed-only 43D.
- ETAPA 5: Treinar classificadores sobre as features híbridas limpas e avaliar no conjunto de teste completo (não filtrado): SVM, RandomForest, XGBoost, NaiveBayes.
- ETAPA 6: Avaliação por temporada e bootstrap para intervalos de confiança.

Implementações complementares:
- Feature engineering e prevenção de data leakage implementadas em `src/feature_engineering.py` (cálculo de Form, μₖ, H2H, posição na tabela, odds quando presentes, e interaction features).
- `scripts/*` geram tabelas e imagens a partir dos modelos treinados.

---

## Parâmetros Experimentais

Parâmetros explicitamente codificados (fonte: `src/train_models.py` e scripts):

Autoencoder / Anomalia:
- `latent_dim`: default 8 (em `train_models_with_decoder_hybrid` signature)
- Autoencoder compile: `optimizer='adam'`, `loss='mae'`
- Treino: `epochs=200`, `batch_size=min(2048, len(X_train_scaled))`
- Callbacks: `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-4)`
- `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)`
- `anomaly_percentile`: default 95 (threshold percentile para reconstruction error)

Classificadores (valores default explicitados no código):
- SVM (`svm_params`): `probability=True, kernel='rbf', C=0.1, gamma=0.001, random_state=42, class_weight='balanced'` (pode ser sobrescrito por `models/optimized_models.pkl` via `load_optimized_params`).
- RandomForest (`rf_params`): `n_estimators=50, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42, class_weight='balanced'`.
- XGBoost (`xgb_base_params`): `eval_metric='mlogloss', n_estimators=200, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=1.0, random_state=42`.
- NaiveBayes (`nb_params`): `var_smoothing=1e-05`.
- `sample_weights` para treino: `compute_sample_weight('balanced', y_train_clean)` aplicado em XGBoost e NaiveBayes em `train_models_with_decoder_hybrid`.
- Possibilidade de carregar parâmetros otimizados: `load_optimized_params(path='models/optimized_models.pkl')` — arquivo `models/optimized_models.pkl` existe na raiz `models/`.

Experiment config / splits:
- Treino: `data/data_2005_2014` (carregado por `src.preprocessing.load_multiple_seasons` em scripts)
- Teste: `data/data_2014_2016`
- Avaliação por temporada: 2014-2015, 2015-2016 e agregado (All)

Observações de inconsistência:
- O arquivo de resumo `RESUMO.md` menciona bootstrap com 1000 iterações; o script `scripts/hybrid_confidence_intervals.py` define `N_ITERATIONS = 100`. Marcar como incerteza (ver seção "Limitações").

---

## Métricas de Avaliação

Métricas utilizadas (fontes: `src/train_models.py`, `scripts/hybrid_baseline_comparison.py`, `scripts/hybrid_confidence_intervals.py`):
- Accuracy (acurácia)
- Precision, Recall (macro) — calculados nas comparações com baselines e em tabelas por classe
- F1-Score (média macro)
- Ranked Probability Score (RPS) — função `rps(y_true, y_prob)` implementada em `src/train_models.py` e variante em `scripts/hybrid_confidence_intervals.py` (implementação compatível)
- Brier Score: não é rotineiramente salvo nos CSVs do Hybrid (aparece em RESUMO.md mas não foi encontrado como saída explícita nos CSVs do diretório Hybrid)
- Intervalos de Confiança (CI 95%) via bootstrap (accuracy, f1, rps) — salvado em `hybrid_confidence_intervals.csv`.

Observação: Para problemas multiclasses (H/D/A), as funções em código usam `zero_division=0` para evitar exceções quando uma classe não é prevista.

---

## Conjuntos de Dados Utilizados

- Dados brutos: arquivos CSV em `data/data_2005_2014/` (Season_2005_2006.csv ... Season_2013_2014.csv) — usados para treino/feature calc.
- Dados de teste: `data/data_2014_2016/` (Season_2014_2015.csv, Season_2015_2016.csv).
- Feature pipeline: `src/feature_engineering.calculate_team_stats()` constrói as 43 features (inclui form, μₖ k=6, H2H window=5, tabela de posições, odds se existirem, e interaction features).

Arquivos de origem onde os caminhos aparecem: `scripts/hybrid_* .py` e `scripts/hybrid_baseline_comparison.py` (mostram explicitamente `train_dir = "data/data_2005_2014"` e `test_dir = "data/data_2014_2016"`).

---

## Gráficos e Figuras Disponíveis

Local: `models/autoencoder_decoder_hybrid/figures/` — lista de arquivos (existentes no repositório):
- hybrid_confidence_intervals_barplot.png
- hybrid_confidence_intervals_boxplot.png
- hybrid_confidence_intervals_distribution.png
- hybrid_confusion_matrix_naivebayes.png
- hybrid_confusion_matrix_randomforest.png
- hybrid_confusion_matrix_svm.png
- hybrid_confusion_matrix_xgboost.png
- hybrid_correlation_heatmap.png
- hybrid_feature_importance_by_type.png
- hybrid_feature_importance_distribution.png
- hybrid_feature_importance_top20.png
- hybrid_performance_by_season.png
- hybrid_performance_heatmap.png
- hybrid_prediction_confidence.png
- radar_chart_hybrid_2014-2015.png
- radar_chart_hybrid_2015-2016.png
- radar_chart_hybrid_All.png

Uso sugerido em artigo: figuras de importância de features, heatmap de correlação, matrizes de confusão por modelo, gráficos de performance por temporada e intervalos de confiança.

---

## Tabelas Disponíveis

Local: `models/autoencoder_decoder_hybrid/` — CSVs relevantes (existentes):
- `hybrid_model_results.csv` — resumo de `model, accuracy, f1, rps` (All)
- `hybrid_model_results_by_season.csv` — resultados por temporada (2014-2015, 2015-2016, All)
- `hybrid_baseline_comparison.csv` — comparação com baselines (Most Frequent, Stratified)
- `hybrid_confidence_intervals.csv` — média, std, CI_lower, CI_upper, margin por modelo e métrica
- `hybrid_feature_importance.csv` — feature importances do RandomForest (recon_*)
- `hybrid_correlation_matrix.csv` — matriz 43×43
- `tabela3_hybrid_comparacao.csv` — comparação Latent vs Hybrid vs (resumo)
- `tabela4_cm_hybrid_*.csv` — confusion matrices (por modelo)
- `tabela5_hybrid_performance_temporada.csv` — performance por temporada (temporal breakdown)
- `tabela6_hybrid_classificacao_classe.csv` — métricas por classe (Precision/Recall/F1/Support)

Cada tabela contém o cabeçalho e os valores salvos — ver `models/autoencoder_decoder_hybrid/`.

---

## Resultados Quantitativos (principais extraídos dos CSVs)

Fonte primária: `models/autoencoder_decoder_hybrid/hybrid_model_results.csv` e `hybrid_model_results_by_season.csv`.

Resultados (Agregado All = 2014-2016, N=760):
- XGBoost: Accuracy = 0.5000, F1 = 0.4626, RPS = 0.2074 (`models/autoencoder_decoder_hybrid/hybrid_model_results.csv`)
- SVM: Accuracy = 0.4987, F1 = 0.4898, RPS = 0.2071
- RandomForest: Accuracy = 0.4803, F1 = 0.4477, RPS = 0.2095
- NaiveBayes: Accuracy = 0.4789, F1 = 0.4725, RPS = 0.3046

Por temporada (exemplos — ver `tabela5_hybrid_performance_temporada.csv`):
- 2014-2015 (N=380): SVM 0.5211, RandomForest 0.5105, XGBoost 0.5132, NaiveBayes 0.5026
- 2015-2016 (N=380): SVM 0.4763, RandomForest 0.4500, XGBoost 0.4868, NaiveBayes 0.4553

Confusion / class breakdown (`tabela6_hybrid_classificacao_classe.csv`):
- Ex.: RandomForest — Home (H): Precision 0.5691, Recall 0.6261, F1 0.5962, Support 329; Draw (D): Precision 0.3095, Recall 0.2600, F1 0.2826, Support 200; Away (A): F1 0.4642, Support 231.
- Observação: desempenho em classe empate (D) é consistentemente pior que H/A para vários modelos — ver tabela.

Intervalos de confiança (bootstrap) — `hybrid_confidence_intervals.csv` (N_ITERATIONS=100 no código):
- Ex.: SVM ACCURACY Mean=0.50016, CI [0.47230, 0.53365]
- Ex.: RandomForest ACCURACY Mean=0.48266, CI [0.44799, 0.51391]
- Ex.: XGBoost ACCURACY Mean=0.50209, CI [0.47155, 0.53289]

Feature importance top (RandomForest) — `hybrid_feature_importance.csv`:
- Top features (nomes 'Recon_i' porque são saídas reconstruídas do decoder). Ex.: Recon_2, Recon_26, Recon_36 aparecem entre os mais importantes (importances ~0.03 cada).

Comparação com baselines — `hybrid_baseline_comparison.csv`:
- Baseline Most Frequent: Accuracy ~0.4329
- Hybrid ML models outperform baselines (ex.: XGBoost 0.50 > 0.4329)

---

## Comparações Realizadas

- Comparação contra baselines (`Most Frequent`, `Stratified`) — `scripts/hybrid_baseline_comparison.py` e `hybrid_baseline_comparison.csv`.
- Comparação entre pipelines (Latent 8D vs Hybrid reconstructed 43D) — `tabela3_hybrid_comparacao.csv` contém linhas para ambos os pipelines (Latent vs Hybrid) e mostra métricas agregadas por modelo.
- Avaliações por temporada (2014-2015 vs 2015-2016) — `hybrid_model_results_by_season.csv` e `tabela5_hybrid_performance_temporada.csv`.
- Intervalos de confiança via bootstrap (CI 95%) — `hybrid_confidence_intervals.csv` e figuras.

---

## Conclusões Registradas nos Arquivos

- Os scripts e CSVs indicam que os modelos Hybrid (decoder reconstructed 43D) obtêm ganhos sobre baselines (ver `hybrid_baseline_comparison.csv`).
- Não há um único "campeão" consistente em todas as métricas; dependendo da métrica e temporada, SVM/XGBoost/RandomForest aparecem melhores — ver `hybrid_model_results.csv` e `hybrid_model_results_by_season.csv`.
- As tabelas de confusão (`tabela4_cm_hybrid_*.csv`) e `tabela6_hybrid_classificacao_classe.csv` mostram baixa performance para classe Empate (D), sugerindo problema de desequilíbrio.
- `scripts/hybrid_feature_importance.py` e `hybrid_feature_importance.csv` mostram que as saídas reconstruídas do decoder têm importância variada; nomes são genéricos (`Recon_i`) — possível necessidade de mapear reconstructions → features originais para interpretação.

Observação: há um documento de alto nível `RESUMO.md` que traz uma versão narrativa dos resultados (inclui estatísticas e conclusões). Entretanto, existem pequenas discrepâncias numéricas e de configuração entre o `RESUMO.md` e os CSVs/ scripts (ver seção "Limitações da Evidência Encontrada").

---

## Limitações da Evidência Encontrada

- Inconsistências/duvidas identificadas:
  - `RESUMO.md` menciona bootstrap com 1.000 iterações; o script `scripts/hybrid_confidence_intervals.py` define `N_ITERATIONS = 100`. Não há evidência no repositório de que 1.000 iterações tenham sido executadas (nenhum CSV com N=1000). Registrar como incerteza.
  - `RESUMO.md` relata números (ex.: RandomForest accuracy 49.74%, RPS 0.2066) que não coincidem exatamente com os CSVs em `models/autoencoder_decoder_hybrid` (ex.: RandomForest Accuracy 0.4803 no CSV). Pode haver múltiplas execuções/variações (latent vs hybrid vs tuned params) — incerteza: origem exata das cifras no `RESUMO.md`.
  - As features no `hybrid_feature_importance.csv` aparecem como `Recon_i` (índices) — falta mapeamento claro entre cada `Recon_i` e a feature original explicável (por exemplo: qual Recon_i corresponde a `gd_diff` ou `B365D`). Isso dificulta interpretação científica direta.
  - `models/optimized_models.pkl` existe, mas não é possível inspecionar seu conteúdo textual sem carregar (é um pickle). Scripts tentam usá-lo via `load_optimized_params()`; não há documentação clara dos hiperparâmetros finais usados quando ele existe.
  - Algumas métricas mencionadas em `RESUMO.md` (ex.: Brier Score, RPS agregado com certo valor) aparecem em alto-nível mas não sempre presentes nos CSVs do pipeline Hybrid.

- Ausência de documentação reproducibility steps explícitos:
  - Não existe um notebook ou script único que replique passo-a-passo a execução completa com comandos e ambiente (embora `scripts/generate_hybrid_all.py` orquestre os scripts, faltam instruções de ambiente exato — `requirements.txt` existe mas não há `Makefile` / `README` com comando único reproduzível e versão de pacotes exatas além do `RESUMO.md`).

---

## Informações Ausentes para um Artigo Científico

Itens que precisam ser produzidos ou confirmados antes de submeter um artigo:

1. Mapeamento dos `Recon_i` → features humanas (descrição semântica das 43 dimensões reconstruídas) para permitir interpretação (SHAP / correspondência ou anotações que relacionem decoder output às features originais).
2. Registro preciso da versão final dos hiperparâmetros (se `models/optimized_models.pkl` foi usado, exportar um CSV/JSON com os valores exatos aplicados a cada execução usada para gerar as tabelas finais).
3. Registro do número exato de iterações do bootstrap (100 vs 1000) e, se houver execuções com 1000, incluir os CSVs resultantes e as sementes usadas.
4. Script/README com instruções reprodutíveis (comando único, ambiente, seed, tempo de execução esperado).
5. Explicitação de quais variantes do pipeline foram comparadas (Latent-only 8D vs Decoder Hybrid 43D vs outros) e confirmação de como as comparações foram feitas (mesma semente, mesmos folds temporais, mesm o pré-processamento).
6. Mapeamento entre as tabelas/figuras e as perguntas de pesquisa (e.g., como foi escolhido o melhor modelo — por F1, Accuracy, RPS?) — atualmente múltiplas métricas são computadas sem priorização publicada.
7. Logs experimentais (ou arquivo `trained_models_hybrid.pkl` contendo apenas os metadados mas não o registro textual da execução) — seria útil um arquivo `run_metadata.json` com data/hora, git commit, comando executado, versão de dependências.

---

## Avaliação da Qualidade da Evidência Disponível

Força da evidência:
- Pontos fortes:
  - Resultados numéricos (CSV) e figuras estão salvos e organizados de forma consistente em `models/autoencoder_decoder_hybrid`.
  - Scripts que geram os artefatos estão presentes e são executáveis localmente (orquestrador `scripts/generate_hybrid_all.py`).
  - Pipeline implementa controles anti-leakage na engenharia de features e divisão temporal (treino: 2005-2014; teste: 2014-2016).
  - Métricas apropriadas para problema multiclasses (Accuracy, Precision/Recall, F1-macro, RPS) são calculadas e CIs por bootstrap são fornecidos.

- Pontos fracos / lacunas:
  - Faltam artefatos de reprodutibilidade completos (registro de parâmetros finais, mapeamento Recon_i → feature original, run metadata, instruções passo-a-passo com ambiente exato).
  - Pequenas discrepâncias entre documentação narrativa (`RESUMO.md`) e os CSVs/ scripts (bootstrap N=100 vs 1000; números de performance diferentes) exigem verificação e confirmação de qual execução é a referência para o artigo.
  - Interpretação das features reconstruídas é limitada devido a nomes genéricos (`Recon_i`).

Conclusão qualitativa: A evidência experimental existente é sólida em termos de execução e organização (scripts, CSVs, figuras presentes), suficiente para fundamentar uma primeira versão de resultados em um artigo. Entretanto, antes de submeter, é recomendável produzir documentação adicional para fechar lacunas de reprodutibilidade e esclarecer discrepâncias numéricas/paramétricas.

---

## Próximos Passos Recomendados (curto prazo)

- Exportar o conteúdo de `models/optimized_models.pkl` para JSON/CSV para fixar hiperparâmetros otimizados usados (se aplicável).
- Executar (ou reexecutar) `scripts/hybrid_confidence_intervals.py` com `N_ITERATIONS=1000` (se o objetivo for reproduzir o número citado em `RESUMO.md`) e salvar os CSVs gerados (documentar tempo de execução e seed).
- Gerar mapeamento Recon_i → feature original: executar um script que correlacione cada dimensão reconstruída com features originais (ex.: correlação Pearson entre decoder output e cada feature original) e salvar `reconstruction_feature_mapping.csv`.
- Criar `run_metadata.json` contendo git commit, data/hora, comandos executados, versões de pacotes (p.ex. `pip freeze`) e parâmetros principais (latent_dim, anomaly_percentile, classifier params). Salvar em `models/autoencoder_decoder_hybrid/`.

---

## Anexos / Arquivos de origem citados (exemplos)

- [src/train_models.py](src/train_models.py#L820-L1160) — função `train_models_with_decoder_hybrid`
- [scripts/hybrid_baseline_comparison.py](scripts/hybrid_baseline_comparison.py#L1-L200)
- [scripts/hybrid_confidence_intervals.py](scripts/hybrid_confidence_intervals.py#L1-L40)
- [scripts/hybrid_feature_importance.py](scripts/hybrid_feature_importance.py#L1-L40)
- [models/autoencoder_decoder_hybrid/hybrid_model_results.csv](models/autoencoder_decoder_hybrid/hybrid_model_results.csv)
- [models/autoencoder_decoder_hybrid/hybrid_confidence_intervals.csv](models/autoencoder_decoder_hybrid/hybrid_confidence_intervals.csv)
- [models/autoencoder_decoder_hybrid/hybrid_feature_importance.csv](models/autoencoder_decoder_hybrid/hybrid_feature_importance.csv)
- [models/autoencoder_decoder_hybrid/tabela3_hybrid_comparacao.csv](models/autoencoder_decoder_hybrid/tabela3_hybrid_comparacao.csv)
- [models/autoencoder_decoder_hybrid/tabela5_hybrid_performance_temporada.csv](models/autoencoder_decoder_hybrid/tabela5_hybrid_performance_temporada.csv)
- [models/autoencoder_decoder_hybrid/tabela6_hybrid_classificacao_classe.csv](models/autoencoder_decoder_hybrid/tabela6_hybrid_classificacao_classe.csv)

(Links referem-se a paths existentes no workspace.)

---

Se desejar, prossigo com qualquer uma das ações recomendadas (exportar `optimized_models.pkl`, gerar mapeamento Recon→feature, reexecutar bootstrap com 1000 iterações, ou formatar um `run_metadata.json`).
