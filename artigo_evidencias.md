## Evidências para artigo: Decoder Hybrid

Este documento reúne as evidências extraídas do repositório para fundamentar um artigo científico sobre o pipeline "Decoder Hybrid". É material bruto — não é o artigo final.

### 1. Metodologia (pipeline)
- Construção de features originais: função `calculate_team_stats()` em `src/feature_engineering.py` (form, μ_k, H2H window=5, posição na tabela, odds, interações).
- Treino do Autoencoder (AE): encoder → decoder. O Decoder é usado para reconstruir as 43 features originais e as features reconstruídas servem como entrada para classificadores (pipeline "Hybrid").
- Detecção de anomalias/filtragem: cálculo de erro de reconstrução (MAE média por amostra) com limiar estabelecido pelo percentil (p95) — amostras com erro alto tratadas como anomalias (removidas/etiquetadas).
- Treino de classificadores sobre features reconstruídas: SVM, RandomForest, XGBoost, GaussianNB (NaiveBayes). Também há ensembling (Voting e Stacking no pipeline Latent).
- Avaliação: métricas Accuracy, F1-macro (macro), RPS (Ranked Probability Score). Bootstrap para IC; testes pareados (McNemar) para comparação de acurácia.

### 2. Arquitetura do Autoencoder
- Modelos salvos: models/autoencoder_decoder_hybrid/encoder_hybrid.keras, models/autoencoder_decoder_hybrid/decoder_hybrid.keras, pesos em `autoencoder_hybrid_weights.h5`.
- Dimensão latente: valor registrado em `models/autoencoder_decoder_hybrid/trained_models_hybrid.pkl` (campo `latent_dim`).
- Perda / otimização: treino com perda `mae` (Mean Absolute Error); parâmetros de treino (epochs, batch_size) estão registrados nos metadados gerados pelo pipeline.

### 3. Hiperparâmetros principais
- `latent_dim`: conforme `trained_models_hybrid.pkl`.
- `loss` do AE: `mae`.
- `epochs`: definido no script de treino (`src/train_models.py`) e nos metadados do modelo salvo.
- `anomaly_percentile`: 95 (p95 para filtragem de amostras anômalas).
- Classificadores: SVM, RandomForest, XGBoost, NaiveBayes; existe também `models/optimized_models.pkl` com configurações otimizadas.

### 4. Datasets
- Treino: `data/data_2005_2014/` (Season_2005_2006.csv … Season_2013_2014.csv).
- Teste: `data/data_2014_2016/` (Season_2014_2015.csv, Season_2015_2016.csv).
- Conjunto de teste usado nos relatórios: 760 amostras (pipeline).  

### 5. Tabelas (CSV relevantes)
- `models/autoencoder_decoder_hybrid/hybrid_model_results.csv`
- `models/autoencoder_decoder_hybrid/hybrid_model_results_by_season.csv`
- `models/autoencoder_decoder_hybrid/hybrid_confidence_intervals.csv`
- `models/autoencoder_decoder_hybrid/hybrid_feature_importance.csv`
- `models/autoencoder_decoder_hybrid/tabela3_hybrid_comparacao.csv`
- `models/autoencoder_latent/latent_model_results.csv`

### 6. Figuras (arquivos)
- `models/autoencoder_decoder_hybrid/figures/hybrid_confidence_intervals_*.png`
- `models/autoencoder_decoder_hybrid/figures/hybrid_feature_importance_top20.png`
- Matrizes de confusão e heatmaps em `models/autoencoder_decoder_hybrid/figures/`.

### 7. Resultados (resumo)
- Arquivos de resultados agregados: ver `models/autoencoder_decoder_hybrid/*_results*.csv`.
- Validação estatística gerada: `models/statistical_validation/validacao_estatistica_hybrid.md` (McNemar + bootstrap N=1000). Resumo breve:
  - SVM: Accuracy Hybrid = 0.4987 vs Latent = 0.4724; Δ=0.0263; McNemar p=0.0245; bootstrap acc p=0.018 (IC 95% do Δ ≈ [0.0039, 0.0474]).
  - RandomForest: sem diferença significativa.
  - XGBoost: sem diferença significativa.
  - NaiveBayes: acurácia igual; discrepância no RPS entre pipelines (investigar).

### 8. Validação estatística (procedimento)
- McNemar (par de previsões corretas/incorretas) para testar diferença de acurácia pareada.
- Bootstrap (N=1000) sobre conjunto de teste para diferenças em Accuracy, F1-macro e RPS; IC 95% via percentis.
- Predições por amostra salvas em: `models/statistical_validation/preds_hybrid_<MODEL>.csv` e `preds_latent_<MODEL>.csv`.

### 9. Limitações
- Tamanho e representatividade do conjunto de teste (760 amostras).
- Possível desequilíbrio de classes afetando métricas macro.
- Imputações (ratings, odds) podem introduzir viés.
- Interpretabilidade reduzida das features `Recon_i` produzidas pelo decoder.

### 10. Ameaças à validade
- Vazamento temporal ou de informação via pre-processamento; confirmar que nenhum feature usa informação do futuro.
- Múltiplas comparações sem correção estatística aumentam risco de falsos positivos.
- Calibração das probabilidades (impacto no RPS): alguns modelos podem produzir probabilidades não calibradas.

### 11. Referências internas (arquivos consultados)
- `src/train_models.py`
- `src/feature_engineering.py`
- `scripts/hybrid_confidence_intervals.py`
- `scripts/hybrid_feature_importance.py`
- `models/autoencoder_decoder_hybrid/trained_models_hybrid.pkl`
- `models/autoencoder_latent/trained_models_latent.pkl`
- `models/optimized_models.pkl`
- `models/statistical_validation/validacao_estatistica_hybrid.md`

### 12. Recomendações / próximos passos
- Investigar RPS anômalo do `NaiveBayes` (alinhar `classes_`, verificar calibração das probabilidades).
- Documentar valores exatos de `latent_dim`, `epochs`, `reconstruction_threshold` diretamente no documento final do artigo.
- Aplicar correção para múltiplas comparações ao reportar significância estatística.

---
Timestamp extração: 2026-06-15
