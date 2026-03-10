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

**3) Resultados (resumo numérico)**

- Desempenho final (valores obtidos na execução atual do pipeline):
  - SVM: Acurácia ≈ 0.4618 | F1 ≈ 0.4464 | RPS ≈ 0.4271
  - RandomForest: Acurácia ≈ 0.4974 (modelo calibrado usado) | F1 ≈ 0.3656 | RPS ≈ 0.4132
  - XGBoost: Acurácia ≈ 0.4947 | F1 ≈ 0.4645 | RPS ≈ 0.4142
  - NaiveBayes: Acurácia ≈ 0.4776 (calibrado) | F1 ≈ 0.4698 | RPS ≈ 0.4194
  - Voting_Equal: Acurácia ≈ 0.4750 | F1 ≈ 0.4489 | RPS ≈ 0.4334
  - Voting_Weighted: Acurácia ≈ 0.4750 | F1 ≈ 0.4474 | RPS ≈ 0.4298
  - Stacking: Acurácia ≈ 0.4974 | F1 ≈ 0.4576 | RPS ≈ 0.4161

  **Tabelas completas de resultados (métricas por modelo)**

  Os valores abaixo foram extraídos do arquivo `models/trained_models.pkl` gerado pelo pipeline atual.

  | Modelo | Acurácia | F1 (macro) | RPS | Nº de features usadas |
  |---|---:|---:|---:|---:|
  | SVM | 0.4618 | 0.4464 | 0.4271 | 29 |
  | RandomForest | 0.4974 | 0.3656 | 0.4132 | 29 |
  | XGBoost | 0.4947 | 0.4645 | 0.4142 | 29 |
  | NaiveBayes | 0.4776 | 0.4698 | 0.4194 | 27 |
  | Voting_Equal | 0.4750 | 0.4489 | 0.4334 | 29 |
  | Voting_Weighted | 0.4750 | 0.4474 | 0.4298 | 29 |
  | Stacking | 0.4974 | 0.4576 | 0.4161 | 29 |

  Observação: os valores refletem a execução atual do pipeline na sua máquina (os números podem variar ligeiramente por seeds, versões de bibliotecas e pré-processamento exato).

  ---

**4) Comparação com o artigo (diferenças e alinhamentos)**

- Alinhamentos com o artigo original:
  - Mesma divisão temporal (treino/teste por temporadas) e uso de features derivadas de forma, gols e odds.
  - Uso de RPS como métrica central para avaliar qualidade probabilística das previsões.

- Diferenças e adaptações nesta implementação:
  - Ajustes práticos em hiperparâmetros e execução (algumas buscas em grid foram realizadas localmente).
  - Implementação de ensembles (voting e stacking) para combinar forças dos modelos, o que pode não corresponder 1:1 ao setup experimental do artigo.
  - Pipeline e código disponibilizados com interface Streamlit e scripts auxiliares para geração automática de tabelas e figuras, facilitando reprodução e apresentação.
  - Correções e robustez: foram adicionados metadados (`feature_columns`) aos modelos salvos para garantir compatibilidade entre treino e avaliação — bug identificado e corrigido durante o desenvolvimento.

---

**5) Estrutura do repositório (pontos-chave)**

- Código principal e módulos: `src/` — contém `preprocessing.py`, `feature_engineering.py`, `train_models.py`, `analysis.py`.
- Scripts auxiliares: `scripts/` — geração de tabelas, figuras, gridsearch, inspeções e testes.
- Aplicação interativa: `app.py` (Streamlit) — permite seleção de modelos, visualizações e avaliação.
- Modelos e artefatos: `models/trained_models.pkl` e outputs em `models/` (`figures/`, tabelas `.csv`).

---

**6) Observações finais e próximos passos sugeridos**

- Reprodutibilidade: re-treinar com `python main.py` assegura `models/trained_models.pkl` atualizado com `feature_columns`.
- Validar estabilidade: rodar o pipeline com seeds diferentes e versões de dependências para estimar variância dos resultados.
- Explorar melhorias:
  - Feature selection adicional / regularização para reduzir overfitting.
  - Testar modelos probabilísticos alternativos e calibração mais refinada.
  - Avaliação por janela temporal (rolling) para verificar estabilidade em tempo.

---

