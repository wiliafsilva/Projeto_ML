## RESUMO DO PROJETO: Replica Científica — Previsão da Premier League

Este arquivo resume o projeto, incluindo metodologias empregadas, resultados obtidos e uma comparação concisa entre a implementação atual e o artigo científico base.

**Resumo rápido**

- Escopo: replicação científica de um estudo para prever resultados da Premier League (EPL) usando features históricas, formatação de time, odds e estatísticas agregadas.
- Dados: partidas das temporadas 2005–2016, com split de treino (2005–2014) e teste (2014–2016).
- Implementação: pipeline completo (pré-processamento, engenharia de features, treino, calibração e ensembles) e interface interativa em Streamlit.

---

**1) Metodologias usadas**

- Pré-processamento e enrichments:
  - Integração de ratings (FIFA), odds (Bet365) e preenchimento de valores ausentes por mediana.
  - Normalizações simples e criação de probabilidades implícitas a partir de odds.

- Engenharia de features:
  - `calculate_team_stats()` produz duas famílias de features:
  - `calculate_team_stats()` produz duas famílias de features:
    - Class A (features individuais, usadas pelo NaiveBayes): exemplos legíveis — Forma do mandante (`home_form`), Forma do visitante (`away_form`), Posição do mandante na tabela (`home_position`), Posição do visitante na tabela (`away_position`), Pontos do mandante (`home_points`), etc. (≈27 features).
    - Class B (features diferenciais, usadas por RF/XGBoost/SVM e ensembles): diferenças entre mandante/visitante — Diferença de forma (`form_diff`), Diferença de saldo de gols (`gd_diff`), Diferença de posição (`position_diff`), Diferença de finalizações (`shots_diff`), Probabilidade normalizada do mandante (`prob_home_norm`), etc. (≈29 features).

  - Lista completa de features (extraída do código `src/train_models.py`):
    - Class A (valores individuais) — 27 features:
      - `home_form`
      - `away_form`
      - `home_position`
      - `away_position`
      - `home_points`
      - `away_points`
      - `h2h_home_wins`
      - `h2h_draws`
      - `h2h_away_wins`
      - `h2h_home_goals_avg`
      - `h2h_away_goals_avg`
      - `h2h_games`
      - `h2h_confidence`
      - `away_advantage`
      - `season_trend`
      - `position_form_home`
      - `position_form_away`
      - `strength_balance`
      - `B365H`
      - `B365D`
      - `B365A`
      - `prob_home`
      - `prob_draw`
      - `prob_away`
      - `prob_home_norm`
      - `prob_draw_norm`
      - `prob_away_norm`

    - Class B (diferenciais) — 29 features:
      - `gd_diff`
      - `streak_diff`
      - `weighted_diff`
      - `form_diff`
      - `corners_diff`
      - `shotsontarget_diff`
      - `shots_diff`
      - `goals_avg_diff`
      - `overall_diff`
      - `attack_diff`
      - `midfield_diff`
      - `defense_diff`
      - `position_diff`
      - `points_diff`
      - `h2h_confidence`
      - `away_advantage`
      - `season_trend`
      - `position_form_home`
      - `position_form_away`
      - `strength_balance`
      - `B365H`
      - `B365D`
      - `B365A`
      - `prob_home`
      - `prob_draw`
      - `prob_away`
      - `prob_home_norm`
      - `prob_draw_norm`

- Modelagem:
  - Modelos individuais: SVM, RandomForest, XGBoost, NaiveBayes.
  - Calibração de probabilidades: `CalibratedClassifierCV` aplicada quando indicada (melhora RPS/qualidade probabilística).
  - Ensembles:
    - `Voting_Equal`: soft-voting com pesos iguais entre RF, XGB e NB.
    - `Voting_Weighted`: soft-voting com pesos ajustados (RF maior peso baseado em RPS).
    - `Stacking`: stacking com meta-learner (Logistic Regression) que aprende a combinar as previsões dos modelos base.

- Métricas usadas:
  - Acurácia, F1 (macro), RPS (Ranked Probability Score — métrica principal do artigo), Brier score e ROC AUC (quando aplicável).

---

**2) Dados e split**

- Treino: temporadas 2005–2014 (≈3420 partidas)
- Teste: temporadas 2014–2016 (≈760 partidas)
- Total aproximado: 4180 partidas (soma dos subconjuntos acima)

---

**3) Resultados (resumo numérico atualizado)**

- Visão geral (métricas agregadas — arquivo de origem: [models/tabela3_comparacao_modelos.csv](models/tabela3_comparacao_modelos.csv)):
  - RandomForest: Accuracy=0.4974 | Precision=0.3299 | Recall=0.4281 | F1=0.3656 | RPS=0.2066 | Brier=0.6059 | ROC AUC=0.6458
  - XGBoost: Accuracy=0.4947 | Precision=0.4670 | Recall=0.4704 | F1=0.4645 | RPS=0.2071 | Brier=0.6089 | ROC AUC=0.6562
  - NaiveBayes: Accuracy=0.4776 | Precision=0.4779 | Recall=0.4723 | F1=0.4698 | RPS=0.2097 | Brier=0.6195 | ROC AUC=0.6456
  - SVM: Accuracy=0.4618 | Precision=0.4468 | Recall=0.4492 | F1=0.4464 | RPS=0.2136 | Brier=0.6192 | ROC AUC=0.6274
  - Voting_Equal: Accuracy=0.4750 | F1=0.4489 | RPS=0.2167
  - Voting_Weighted: Accuracy=0.4750 | F1=0.4474 | RPS=0.2149
  - Stacking: Accuracy=0.4974 | F1=0.4576 | RPS=0.2080

- Desempenho por temporada (fonte: [models/tabela5_performance_temporada.csv](models/tabela5_performance_temporada.csv)):
  - 2015-2016 (380 jogos): Baseline=45.26% | SVM=49.21% | RF=52.11% | XGB=52.11% | NB=48.68% | Voting_Equal=49.21% | Voting_Weighted=49.47% | Stacking=51.58%
  - 2016-2017 (380 jogos): Baseline=41.32% | SVM=43.16% | RF=47.37% | XGB=46.84% | NB=46.84% | Voting_Equal=45.79% | Voting_Weighted=45.53% | Stacking=47.89%
  - All (760 jogos): Baseline=43.29% | SVM=46.18% | RF=49.74% | XGB=49.47% | NB=47.76% | Voting_Equal=47.50% | Voting_Weighted=47.50% | Stacking=49.74%

- Comparação com baselines por temporada e por modelo completa: [models/baseline_comparison_with_metrics.csv](models/baseline_comparison_with_metrics.csv)

- Intervalos de confiança (95%) — resumo (fonte: [models/confidence_intervals.csv](models/confidence_intervals.csv)):
  - All — RandomForest: Accuracy mean=0.5065 (CI 0.4690–0.5448) | F1 mean=0.3781 (CI 0.3560–0.4040) | RPS mean=0.4119 (CI 0.3939–0.4319)
  - All — XGBoost: Accuracy mean=0.4949 (CI 0.4584–0.5329) | F1 mean=0.4748 (CI 0.4402–0.5098) | RPS mean=0.4149 (CI 0.4005–0.4281)
  - All — NaiveBayes: Accuracy mean=0.4704 (CI 0.4368–0.5039) | F1 mean=0.4614 (CI 0.4290–0.4962) | RPS mean=0.4187 (CI 0.4020–0.4333)
  - All — SVM: Accuracy mean=0.4646 (CI 0.4335–0.5034) | F1 mean=0.4507 (CI 0.4237–0.4885) | RPS mean=0.4277 (CI 0.4090–0.4457)

- Observações sobre classificação por classe (exemplos — arquivos: [models/tabela6_classificacao_randomforest.csv](models/tabela6_classificacao_randomforest.csv), [models/tabela6_classificacao_xgboost.csv](models/tabela6_classificacao_xgboost.csv)):
  - RandomForest (All): acurácia global 0.4974, má performance para a classe "Empate" (precision=0.0, recall=0.0 — indica viés a prever vitórias/derrotas em detrimento do empate).
  - XGBoost (All): melhor equilíbrio entre classes; para "Empate" precision≈0.338 e f1≈0.294, melhor que RF para essa classe.

---

**3.1) Importância de features e SHAP (sumário)**

- Top features por Importance (RandomForest) — fonte: [models/feature_importance_randomforest.csv](models/feature_importance_randomforest.csv):
  1. `h2h_games` — Importance=0.1063
  2. `B365D` — Importance=0.1002
  3. `points_diff` — Importance=0.0970
  4. `away_position` — Importance=0.0704
  5. `position_diff` — Importance=0.0545

- Top features por SHAP (impacto médio) — fonte: [models/shap_importance_randomforest.csv](models/shap_importance_randomforest.csv):

  ---

  **Apêndice — Tabelas completas de resultados**

  Obs: as tabelas abaixo são extraídas dos CSVs gerados pelo pipeline e mantidas em `models/`.

  **A) Comparação por modelo (tabela3_comparacao_modelos.csv)**

  | Modelo | Accuracy | Precision | Recall | F1 | RPS | Brier | ROC AUC |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Baseline (Majoritário) | 0.4329 | - | - | - | - | - | - |
  | SVM | 0.4618 | 0.4468 | 0.4492 | 0.4464 | 0.2136 | 0.6192 | 0.6274 |
  | RandomForest | 0.4974 | 0.3299 | 0.4281 | 0.3656 | 0.2066 | 0.6059 | 0.6458 |
  | XGBoost | 0.4947 | 0.4670 | 0.4704 | 0.4645 | 0.2071 | 0.6089 | 0.6562 |
  | NaiveBayes | 0.4776 | 0.4779 | 0.4723 | 0.4698 | 0.2097 | 0.6195 | 0.6456 |
  | Voting_Equal | 0.4750 | 0.4484 | 0.4545 | 0.4489 | 0.2167 | 0.6329 | 0.6500 |
  | Voting_Weighted | 0.4750 | 0.4467 | 0.4537 | 0.4474 | 0.2149 | 0.6279 | 0.6495 |
  | Stacking | 0.4974 | 0.4607 | 0.4670 | 0.4576 | 0.2080 | 0.6123 | 0.6587 |

  **B) Desempenho por temporada (tabela5_performance_temporada.csv)**

  | Temporada | Jogos | Baseline | SVM | RandomForest | XGBoost | NaiveBayes | Voting_Equal | Voting_Weighted | Stacking |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 2015-2016 | 380 | 45.26% | 49.21% | 52.11% | 52.11% | 48.68% | 49.21% | 49.47% | 51.58% |
  | 2016-2017 | 380 | 41.32% | 43.16% | 47.37% | 46.84% | 46.84% | 45.79% | 45.53% | 47.89% |
  | All | 760 | 43.29% | 46.18% | 49.74% | 49.47% | 47.76% | 47.50% | 47.50% | 49.74% |

  **C) Comparação com baselines por temporada (baseline_comparison_with_metrics.csv) — amostra**

  | Temporada | Modelo | Accuracy | F1 | Precision | Recall | Tipo | Brier | ROC_AUC | RPS |
  |---|---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 2014-2015 | RandomForest | 0.5395 | 0.3977 | 0.3544 | 0.4588 | ML | 0.1943 | 0.6717 | 0.2002 |
  | 2014-2015 | XGBoost | 0.5184 | 0.5013 | 0.5055 | 0.5039 | ML | 0.1973 | 0.6789 | 0.2027 |
  | 2014-2015 | NaiveBayes | 0.4737 | 0.4562 | 0.4628 | 0.4597 | ML | 0.2030 | 0.6659 | 0.2076 |
  | 2014-2015 | SVM | 0.5000 | 0.4816 | 0.4832 | 0.4857 | ML | 0.1996 | 0.6486 | 0.2081 |
  | 2014-2015 | Baseline (Most Freq) | 0.4526 | 0.2077 | 0.1509 | 0.3333 | Baseline | 0.2149 | 0.5 | 0.2299 |
  | 2015-2016 | RandomForest | 0.4737 | 0.3565 | 0.3127 | 0.4242 | ML | 0.2096 | 0.6159 | 0.2130 |
  | 2015-2016 | XGBoost | 0.4684 | 0.4450 | 0.4465 | 0.4500 | ML | 0.2086 | 0.6296 | 0.2115 |
  | 2015-2016 | NaiveBayes | 0.4658 | 0.4616 | 0.4673 | 0.4635 | ML | 0.2100 | 0.6240 | 0.2118 |

  *Arquivo completo:* [models/baseline_comparison_with_metrics.csv](models/baseline_comparison_with_metrics.csv)

  **D) Intervalos de confiança (confidence_intervals.csv) — amostra**

  | Temporada | Modelo | Accuracy_Mean | Accuracy_CI_Lower | Accuracy_CI_Upper | F1_Mean | F1_CI_Lower | F1_CI_Upper | RPS_Mean | RPS_CI_Lower | RPS_CI_Upper |
  |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | All | RandomForest | 0.5065 | 0.4690 | 0.5448 | 0.3781 | 0.3560 | 0.4040 | 0.4119 | 0.3939 | 0.4319 |
  | All | XGBoost | 0.4949 | 0.4584 | 0.5329 | 0.4748 | 0.4402 | 0.5098 | 0.4149 | 0.4005 | 0.4281 |
  | All | NaiveBayes | 0.4704 | 0.4368 | 0.5039 | 0.4614 | 0.4290 | 0.4962 | 0.4187 | 0.4020 | 0.4333 |
  | All | SVM | 0.4646 | 0.4335 | 0.5034 | 0.4507 | 0.4237 | 0.4885 | 0.4277 | 0.4090 | 0.4457 |

  *Arquivo completo:* [models/confidence_intervals.csv](models/confidence_intervals.csv)

  **E) Importância de features e SHAP (amostra)**

  | Feature | Importance (RF) | SHAP Impact |
  |---|---:|---:|
  | h2h_games | 0.1063225 | 0.0327563 |
  | B365D | 0.1002363 | 0.0305912 |
  | points_diff | 0.0970143 | 0.0263853 |
  | away_position | 0.0703913 | 0.0202130 |
  | position_diff | 0.0545449 | 0.0198647 |

  *Arquivos completos:* [models/feature_importance_randomforest.csv](models/feature_importance_randomforest.csv), [models/shap_importance_randomforest.csv](models/shap_importance_randomforest.csv)

  **F) Matrizes de confusão / classificação por classe (amostra)**

  - RandomForest (fonte: `models/tabela6_classificacao_randomforest.csv`):

  | Classe | precision | recall | f1-score | support |
  |---|---:|---:|---:|---:|
  | Vitória Casa | 0.5037 | 0.8298 | 0.6269 | 329 |
  | Empate | 0.0 | 0.0 | 0.0 | 200 |
  | Vitória Visitante | 0.4861 | 0.4545 | 0.4698 | 231 |

  *Arquivos completos de classification: veja `models/tabela6_classificacao_*.csv`*

  ---

  **Fim do Apêndice**

