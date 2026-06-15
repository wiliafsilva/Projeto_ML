# Narrativa Científica Recomendada — Decoder Hybrid

Base: análise estrita dos artefatos em `models/autoencoder_decoder_hybrid/`, `src/train_models.py`, `src/feature_engineering.py` e scripts `scripts/hybrid_*.py`.

1. Hipóteses realmente suportadas pelos resultados
- H1 suportada (condicional): O espaço reconstruído pelo decoder (Decoder Hybrid 43D) **pode** produzir aumentos de Accuracy em alguns classificadores comparado ao espaço latente 8D — evidenciado por SVM (0.4724 → 0.4987, +5.6% relativo) e XGBoost (0.4947 → 0.5000, +1.1% relativo) em `tabela3_hybrid_comparacao.csv`.
- H2 suportada: A pipeline inclui filtragem de amostras por reconstruction error (anomaly_percentile=95) — evidência do código e dos metadados; isto demonstra uso de uma etapa explícita de limpeza/robustez antes do treino.

2. Hipóteses NÃO suportadas
- Redução de ruído generalizada: não há análise direta que comprove redução de ruído nas features reconstruídas (nenhum artefato mede SNR, variância residual ou correlação Recon→original). Logo, **não suportado**.
- Melhoria de generalização ampla: resultados são mistos; CIs do bootstrap exibem sobreposição. Não há evidência estatística robusta que comprove melhoria de generalização universal do Hybrid sobre Latent.
- Melhora consistente em métricas probabilísticas (RPS) ou F1-macro: não suportada — RPS e F1 variam por modelo e às vezes pioram (ex.: NaiveBayes RPS pior no Hybrid).

3. Afirmações que um revisor poderia questionar
- Falta de mapeamento semântico de `Recon_i` (interpretabilidade limitada).
- Discrepância documentada sobre bootstrap (RESUMO.md vs script: 1000 vs 100 iterações).
- Ausência de registro claro dos hiperparâmetros finais (conteúdo de `models/optimized_models.pkl` não exposto em formato legível no artigo).
- Falta de testes estatísticos formais de significância entre pipelines (apenas CIs com sobreposição mostradas).
- Possível seleção de melhor métrica (Accuracy) sem justificar prioridade sobre RPS/F1 para problema probabilístico.

4. Três títulos científicos propostos
- (A) "Decoder-based Feature Reconstruction Improves Accuracy for Some Classifiers in Premier League Match Prediction"
- (B) "Decoder Hybrid Representations vs Latent Embeddings: An Empirical Study on Football Match Outcomes"
- (C) "Reconstructed Feature Spaces from Autoencoders as Competitive Inputs for Sports Outcome Classification"

5. Três contribuições científicas principais (sustentadas)
- (1) Evidência empírica de que a saída do decoder (43D) pode melhorar Accuracy para SVM e XGBoost frente ao latent 8D, em testes temporais (2005–2016 splits) — `tabela3_hybrid_comparacao.csv`.
- (2) Implementação e validação prática de um pipeline que combina autoencoder, filtragem por reconstruction error e avaliação por temporada com bootstrap CI (artefatos: modelos Keras, CSVs, figuras de CI).
- (3) Disponibilização de artefatos reprodutíveis (modelos, scaler, CSVs e scripts) que permitem análise detalhada por modelo e por classe.

6. Mensagem principal recomendada
- O Decoder Hybrid é uma alternativa competitiva ao espaço latente para tarefas de classificação esportiva: em alguns classificadores (SVM, XGBoost) produz ganhos modestos de Accuracy, mas as melhorias não são universais nem conclusivas em todas as métricas; recomenda-se investigação adicional para interpretação e confirmação estatística.

7. Resultados que devem aparecer no resumo
- Ganhos de Accuracy observados (SVM +5.6% relativo; XGBoost +1.1% relativo) com referência explícita a `tabela3_hybrid_comparacao.csv`.
- Observação de que gains são pontuais e que RPS/F1 não melhoram de forma consistente; mencionar bootstrap CIs e sobreposição de intervalos (`hybrid_confidence_intervals.csv`) como limitação.

8. Resultados que NÃO devem aparecer no resumo
- Afirmar redução de ruído, melhoria de generalização ampla, ou aumento universal de desempenho — todas não comprovadas pelos artefatos.

9. Estrutura de figuras/tabelas recomendada (prioridade para submissão PESQBASE)
- Tabela 1 (principal): comparação Latent 8D vs Decoder Hybrid 43D por modelo — métricas Accuracy, F1, RPS (`tabela3_hybrid_comparacao.csv`).
- Figura 1: barplot de Accuracy com CI95 (bootstrap) para cada modelo (`figures/hybrid_confidence_intervals_barplot.png`).
- Figura 2: feature importance top20 (decoder features) (`figures/hybrid_feature_importance_top20.png`), acompanhado por um anexo com correlações Recon→original (recomendar gerar).
- Tabela 2: métricas por classe H/D/A para o melhor modelo e para comparativo (`tabela6_hybrid_classificacao_classe.csv`).
- Figuras suplementares: confusion matrices por modelo, correlation heatmap, radar charts por temporada.

# Tese Científica Recomendada

Com base nas evidências do repositório, a tese defensável é: a representação produzida pelo decoder de um autoencoder, quando combinada com filtragem por reconstruction error, constitui uma alternativa competitiva ao espaço latente de baixa dimensão para predição de resultados da Premier League — particularmente, ela produz ganhos observáveis de Accuracy para SVM e XGBoost sob validação temporal; contudo, tais ganhos são modestos, dependem do classificador, e não implicam melhoria generalizada em métricas probabilísticas (RPS) ou F1. Portanto, o trabalho contribui com evidência experimental de uma via promissora para engenharia de features em tarefas esportivas, mas exige análises adicionais (mapeamento interpretável das dimensões reconstruídas e testes estatísticos de significância) antes de reclamações mais amplas sobre redução de ruído ou generalização.
