# Contribuição Científica do Pipeline Decoder Hybrid

Objetivo: identificar, com base nas evidências do repositório, qual é a contribuição científica real do pipeline "Decoder Hybrid" e reunir as evidências que a sustentam.

Fontes principais: `models/autoencoder_decoder_hybrid/tabela3_hybrid_comparacao.csv`, `models/autoencoder_decoder_hybrid/hybrid_model_results.csv`, `models/autoencoder_decoder_hybrid/hybrid_confidence_intervals.csv`, `src/train_models.py`, `src/feature_engineering.py`, `scripts/hybrid_feature_importance.py`.

1) Qual problema o Decoder Hybrid resolve
- Evidência: o pipeline transforma features originais (pré-processadas e sujeitas a ruído e alta dimensionalidade) em um espaço reconstruído pelo decoder (43D) e treina classificadores sobre esse espaço; isso visa reduzir dimensionalidade/ruído e produzir features mais robustas a outliers (veja `src/train_models.py` — etapa de detecção de anomalias por reconstruction error e filtragem com `anomaly_percentile=95`).

2) Limitações das abordagens tradicionais que motivam o Hybrid
- Evidência: as abordagens tradicionais incluem uso do espaço original (43 features) e uso do espaço latente (8D). O repositório implementa e compara essas variantes (`tabela3_hybrid_comparacao.csv`) indicando que diferentes representações podem afetar desempenho por modelo; isto motiva investigar o uso da saída do decoder como alternativa.

3) Hipótese testada
- Evidência (implícita): features reconstruídas pelo decoder (43D) podem preservar informação discriminativa enquanto atenuam ruído e melhorar desempenho geral/robustez dos classificadores em relação ao espaço latente (8D) ou às features originais. A hipótese é operacionalizada comparando métricas entre pipelines (Latent 8D vs Decoder Hybrid 43D) em `tabela3_hybrid_comparacao.csv`.

4–6) Diferença entre as representações (evidências no código)
- Features originais (43D): calculadas por `src/feature_engineering.calculate_team_stats()` — Form, μ_k, H2H, posições, odds, interactions.
- Espaço latente (8D): saída do `encoder` (`autoencoder.encoder(...)`) dimensionada para `latent_dim=8` (default) — usado em experiências "Latent" (CSV `tabela3_hybrid_comparacao.csv`).
- Decoder Hybrid (43D reconstruído): saída do `decoder` aplicada ao latent e usada como features finais (`X_train_hybrid = X_train_reconstructed` no `train_models_with_decoder_hybrid`). Evidência: `src/train_models.py` e arquivos em `models/autoencoder_decoder_hybrid/`.

7) O Hybrid supera o Latent? (usando `tabela3_hybrid_comparacao.csv`)
- Dados (Accuracy):
  - RandomForest: Hybrid 0.4803 vs Latent 0.4882 → Latent melhor.
  - XGBoost: Hybrid 0.5000 vs Latent 0.4947 → Hybrid melhor.
  - NaiveBayes: Hybrid 0.4789 vs Latent 0.4789 → iguais.
  - SVM: Hybrid 0.4987 vs Latent 0.4724 → Hybrid melhor.

8) Em quais métricas?
- A comparação acima usa **Accuracy** (coluna `Accuracy` em `tabela3_hybrid_comparacao.csv`). Outras métricas na mesma tabela: `F1` e `RPS` (ver abaixo).

9) Em quais modelos?
- Hybrid supera Latent em Accuracy para **SVM** e **XGBoost**; Empata em **NaiveBayes**; perde para **RandomForest**.

10) Qual o ganho percentual? (ganho relativo em Accuracy = (Hybrid-Latent)/Latent)
- SVM: (0.4987 − 0.4724) / 0.4724 = 0.0557 → **+5.57% relativo** (acurácia relativa).
- XGBoost: (0.5000 − 0.4947) / 0.4947 = 0.0107 → **+1.07% relativo**.
- NaiveBayes: 0.0% (empate).
- RandomForest: (0.4803 − 0.4882) / 0.4882 = −0.0162 → **−1.62% relativo** (Latent superior).

11–13) Avaliar se os resultados justificam afirmar que houve redução de ruído / melhoria de generalização / aumento de desempenho preditivo
- Evidência disponível e limitações:
  - Redução de ruído: não há análise direta quantificando ruído residual (por exemplo, correlação decoder→feature original ou medidas de SNR). Embora `src/train_models.py` aplique filtragem por reconstruction error (anomaly removal), não existe no repositório um artefato que comprove redução de ruído objetiva (ex.: menor variance residual por feature). Portanto **não há evidência suficiente para afirmar redução de ruído** além da hipótese operacional.
  - Melhoria de generalização: evidência mista. Alguns modelos (SVM, XGBoost) melhoram em Accuracy ao usar Decoder Hybrid; entretanto, outras métricas probabilísticas (RPS) na `tabela3_hybrid_comparacao.csv` mostram que Latent frequentemente tem RPS igual ou melhor (ex.: RF RPS Latent 0.2068 < Hybrid 0.2095; XGBoost RPS 0.2067 < 0.2074). Além disso, bootstrap CIs (`models/autoencoder_decoder_hybrid/hybrid_confidence_intervals.csv`) indicam sobreposição entre CIs das accuracies médias (ex.: XGBoost CI [0.4715,0.5329], SVM CI [0.4723,0.5336]) — isto reduz a confiança em diferenças estatisticamente significativas. Conclusão: **evidência insuficiente para afirmar melhoria de generalização de forma generalizada; há ganhos pontuais que exigem teste estatístico robusto adicional**.
  - Aumento de desempenho preditivo: resultados mostram **melhora de Accuracy para SVM (+5.6% relativo) e XGBoost (+1.1% relativo)**; porém, medidas probabilísticas (RPS) e F1 nem sempre melhoram. Logo, **não é correto afirmar aumento generalizado do desempenho**; o que se pode afirmar, com base nos CSVs, é que o Decoder Hybrid **pode** melhorar Accuracy para alguns modelos (SVM, XGBoost) mas não de forma consistente para todos os classificadores ou métricas.

14) Três resultados mais fortes (evidência direta)
- (i) SVM: ganho de Accuracy de 0.4724 → 0.4987 (≈ +5.6% relativo) ao usar Decoder Hybrid (`tabela3_hybrid_comparacao.csv`).
- (ii) XGBoost: leve ganho de Accuracy de 0.4947 → 0.5000 (≈ +1.1% relativo) com Decoder Hybrid (`tabela3_hybrid_comparacao.csv`).
- (iii) Existência de avaliação de incerteza: bootstrap CIs salvos em `models/autoencoder_decoder_hybrid/hybrid_confidence_intervals.csv` e figuras (`figures/hybrid_confidence_intervals_*.png`), mostrando tentativa de caracterizar estabilidade dos resultados.

15) Três pontos mais vulneráveis que um revisor poderia questionar
- (i) Falta de mapeamento semântico das `Recon_i`: `hybrid_feature_importance.csv` lista features como `Recon_0..Recon_42` sem indicar qual feature original elas representam — prejudica interpretabilidade.
- (ii) Inconsistência/documentação insuficiente sobre bootstrap (RESUMO.md menciona 1000 iterações; o script usa `N_ITERATIONS=100`) e potencial falta de registro de parâmetros finais (`models/optimized_models.pkl` precisa ser inspecionado) — afeta reprodutibilidade.
- (iii) Métricas probabilísticas (RPS) e F1 não melhoram consistentemente com Hybrid — em alguns casos RPS piora (p.ex. NaiveBayes: Hybrid RPS 0.3046 vs Latent 0.2302) — revisão exigirá explicação ou análises suplementares (calibração, análise SHAP, teste estatístico de significância).

16) Figuras e tabelas a incluir no artigo principal (prioridade)
- Tabela principal: `tabela3_hybrid_comparacao.csv` (Latent 8D vs Hybrid 43D — Accuracy, F1, RPS por modelo).
- Tabela suplementar: `tabela6_hybrid_classificacao_classe.csv` (métricas por classe H/D/A) para discutir viés por classe.
- Figura: `figures/hybrid_feature_importance_top20.png` (top features reconstruídas) + um anexo com correlação entre `Recon_i` e features originais (recomendado gerar).
- Figura: `figures/hybrid_confidence_intervals_barplot.png` (accuracy com CI 95%) para discutir estabilidade.
- Figura: confusion matrices por modelo (`figures/hybrid_confusion_matrix_*.png`).

17) Contribuição Científica Proposta (≤ 500 palavras)

Baseado estritamente nas evidências disponíveis, propõe-se a seguinte contribuição científica: o pipeline Decoder Hybrid demonstra que a representação produzida pelo decoder de um autoencoder (reconstructed 43D features), combinada com filtragem de anomalias por reconstruction error, constitui uma alternativa competitiva ao espaço latente de baixa dimensão (8D) para tarefas de classificação de resultados de partidas da Premier League. Evidências empíricas em `models/autoencoder_decoder_hybrid/tabela3_hybrid_comparacao.csv` mostram que, em termos de Accuracy, o Decoder Hybrid melhora o desempenho para SVM (+5.6% relativo) e XGBoost (+1.1% relativo) enquanto empata com NaiveBayes e perde ligeiramente para RandomForest. Além disso, o repositório inclui avaliação de incerteza (bootstrap CIs em `hybrid_confidence_intervals.csv`), importâncias de features reconstruídas (`hybrid_feature_importance.csv`) e análises por temporada, fornecendo um conjunto de evidências experimentais bem documentadas. Contudo, a evidência é mista: métricas probabilísticas (RPS) e F1-macro nem sempre favorecem o Decoder Hybrid, e CIs apresentados mostram sobreposição que reduz a força estatística das diferenças observadas. Além disso, falta um mapeamento interpretável entre dimensões reconstruídas e features originais, e algumas discrepâncias documentais (por exemplo, número de iterações do bootstrap) requerem esclarecimento para reprodutibilidade. Assim, a contribuição científica suportada pelos dados é uma evidência experimental de que a saída do decoder pode, em certos classificadores, melhorar a acurácia preditiva em comparação ao espaço latente compacto, e que essa alternativa merece investigação aprofundada (análises de significância, calibração de probabilidades, e interpretação das features reconstruídas) antes de reivindicar ganhos gerais de redução de ruído, generalização ou desempenho.

---

Observação: todas as afirmações acima são estritamente baseadas nos arquivos citados; onde não há suporte direto, a limitação foi explicitamente indicada.
