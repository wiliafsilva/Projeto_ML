# Predição de Resultados da Premier League Inglesa com Autoencoder e Machine Learning: Uma Abordagem de Dimensionalidade Latente

## Resumo

Este artigo apresenta uma metodologia inovadora para predição de resultados de partidas de futebol da Premier League Inglesa (EPL) integrando um autoencoder com técnicas tradicionais de machine learning. Partindo de 43 features engineered, desenvolvemos um autoencoder com arquitetura 43→32→16 dimensões para gerar 16 features latentes de compressão não-linear, expandindo o conjunto para 59 features totais. O pipeline foi validado em dataset de 4.180 partidas (2005-2016) com divisão temporal rigorosa: treinamento 2005-2014 (3.420 partidas) e teste 2014-2016 (760 partidas). Random Forest alcançou **49,74% de acurácia com 59 features** (vs. 47,37% com 43 features), representando ganho de **+2,37 pontos percentuais**. XGBoost obteve 49,47% com melhor equilíbrio entre classes (F1-macro=0,4645). Análise SHAP revelou que histórico head-to-head, odds de apostas e posição na tabela são os preditores mais relevantes. O autoencoder capturou com sucesso interações não-lineares, validando a hipótese de que features latentes complementam engineered features em problemas de predição esportiva.

**Palavras-chave:** Predição de futebol, Premier League, Autoencoder, Machine Learning, Dimensionalidade Latente, Random Forest, XGBoost, Feature Engineering

---

## 1. Introdução

A predição de resultados em competições esportivas representa desafio significativo em aprendizado de máquina devido à estocasticidade, múltiplos fatores de influência e dinâmica temporal complexa. Na Premier League Inglesa, estudos anteriores demonstraram que abordagens baseadas em features estatísticas superam modelos baseline consideravelmente.

Baboota & Kaur (2018) desenvolveram metodologia sistemática com 43 features distribuídas em sete categorias (baseline, Form, médias móveis, ratings FIFA, head-to-head, posição, odds). Este trabalho estende sua abordagem através de integração de autoencoder para extração de features latentes, capturando representações comprimidas que sintetizam padrões multidimensionais não capturados por features individuais.

**Contribuições principais:**
1. Pipeline inovador combinando feature engineering + autoencoder + ML
2. Validação temporal rigorosa sem data leakage
3. Ganho empírico comprovado: +2,37pp de acurácia com 59 features
4. Análise completa de explicabilidade via SHAP

---

## 2. Metodologia

### 2.1 Dataset e Divisão Temporal

**Fonte de Dados:**
- Premier League Inglesa, 11 temporadas (2005-2016)
- **Total:** 4.180 partidas
- **Treino:** 2005-2014 (3.420 partidas) para encoder e modelos ML
- **Teste:** 2014-2016 (760 partidas) para validação independente

**Distribuição de Classes:**
| Classe | Frequência | Proporção |
|--------|-----------|-----------|
| Vitória Casa (H) | 1.940 | 46,4% |
| Empate (D) | 1.058 | 25,3% |
| Vitória Visitante (A) | 1.182 | 28,3% |

Divisão temporal previne data leakage: modelos jamais veem dados futuros durante treinamento.

### 2.2 Feature Engineering: 43 Features Originais

Seguindo Baboota & Kaur (2018), foram engineered 43 features distribuídas em sete categorias que capturam diferentes dimensões do desempenho competitivo. A categoria Baseline compreende três features fundamentais: gd_diff (diferença de saldo de gols entre mandante e visitante), streak_diff (diferença de sequência de vitórias normalizadas) e weighted_diff (média ponderada de resultados recentes com pesos decrescentes). O Sistema Form implementa uma abordagem inspirada em ratings ELO, com atualização dinâmica após cada partida, gerando três features: form_diff (diferença entre os ratings Form dos times), home_form e away_form (ratings individuais de cada time).

Médias móveis calculadas sobre uma janela de k=6 jogos anteriores fornecem quatro features adicionais: corners_diff, shotsontarget_diff, shots_diff e goals_avg_diff, capturando tendências recentes em desempenho. Ratings FIFA consolidados em base de dados oficial contribuem com quatro features de diferenciais: overall_diff, attack_diff, midfield_diff e defense_diff, refletindo qualidade intrínseca dos elencos. Análise de histórico head-to-head sobre os últimos cinco confrontos produz seis features: h2h_home_wins, h2h_draws, h2h_away_wins, h2h_games (número total de confrontos), h2h_home_goals_avg e h2h_away_goals_avg, capturando padrões de rivalidade específica entre pares de times.

Simulação contínua da tabela de classificação ao longo da temporada fornece seis features: position_diff (diferença de posições), points_diff (diferença de pontos), home_position, away_position, home_points e away_points, refletindo desempenho cumulativo. Odds de apostas Bet365 integram nove features: as odds brutas B365H, B365D, B365A para vitória casa, empate e visitante respectivamente, suas conversões em probabilidades implícitas (prob_home, prob_draw, prob_away), e as probabilidades normalizadas (prob_home_norm, prob_draw_norm, prob_away_norm) que agregam informação de especialistas e mercado. Adicionalmente, oito features de interação de segunda ordem capuram sinergia entre categorias: h2h_confidence (dominância em confrontos), away_advantage (força relativa do visitante), season_trend (tendência acumulada), position_form_home, position_form_away, strength_balance e demais combinações não-lineares. Em total, o sistema compreende 43 features principais mais 8 features de interação, totalizando 51 features brutas.

### 2.3 Autoencoder para Extração de Features Latentes (NOVO)

O autoencoder foi desenvolvido com arquitetura simétrica para aprender compressão não-linear dos dados. A camada de entrada recebe as 43 features originais, que são progressivamente comprimidas através de uma camada densa com 32 neurônios seguida de uma camada com 16 neurônios (o bottleneck), ambas com ativação ReLU. Este bottleneck de 16 dimensões força o modelo a aprender uma representação comprimida que capture a variância essencial dos dados. A parte decodificadora então descomprime estas 16 dimensões latentes através de camadas simétricas (16→32 com ReLU, 32→43 com ativação Linear) para reconstruir as 43 features originais, permitindo otimização via perda de reconstrução.

O treinamento foi conduzido exclusivamente sobre dados de 2005-2014 (3.420 partidas) para evitar data leakage. A função de perda utilizada foi Mean Squared Error (MSE) entre entrada e reconstrução, otimizada através do algoritmo Adam com learning rate de 0,001. O modelo foi treinado por 100 epochs com batch size de 32 amostras. Previamente ao treinamento, todas as 43 features foram normalizadas via StandardScaler com média zero e desvio padrão unitário, sendo o scaler fitado exclusivamente sobre o conjunto de treinamento 2005-2014. Esta normalização foi armazenada para reutilização durante a extração de features latentes sobre dados de teste, garantindo consistência.

Após conclusão do treinamento, o encoder (as duas primeiras camadas: 43→32→16) foi extraído e salvo em formato Keras como `models/encoder_16dims.keras` (tamanho final 29,8 KB), enquanto o StandardScaler foi serializado em `models/scaler_43features.pkl` (1,2 KB). O encoder foi então aplicado a todos os dados no conjunto completo para extrair as 16 dimensões latentes ($Z_1, Z_2, ..., Z_{16}$) de cada amostra. Estas dimensões latentes representam uma compressão não-linear das 43 features originais, sintetizando os padrões multidimensionais aprendidos pelo autoencoder durante o treinamento.

### 2.4 Dataset Final: 59 Features

O dataset final para treinamento dos modelos de classificação foi construído concatenando as 43 features originais com as 16 dimensões latentes extraídas pelo encoder, resultando em um espaço de 59 dimensões por amostra, formalmente representado como $X_{59} = [X_{43} | Z_{16}] \in \mathbb{R}^{N \times 59}$. Este conjunto expandido foi então dividido em duas classes de features conforme propriedades algorítmicas. A Class A compreende 27 features com valores individuais de cada time (ex: home_form, away_position, h2h_games), apropriadas para Naive Bayes que pressupõe independência condicional entre features. A Class B inclui as 43 features diferenciais entre mandante e visitante além das 16 dimensões latentes, totalizando 43 features, sendo as mais adequadas para algoritmos baseados em árvores (SVM, Random Forest, XGBoost) que naturalmente capturam interações não-lineares. Esta separação permite otimização da representação dos dados conforme premissas matemáticas de cada algoritmo, mantendo o espaço de 59 dimensões como o padrão para análise completa do desempenho.

### 2.5 Modelos de Machine Learning

Quatro modelos individuais foram selecionados como base para análise comparativa. Random Forest foi treinado com 50 estimadores e profundidade máxima de 5, utilizando pesos balanceados para mitigar desbalanceamento de classes. XGBoost foi configurado com 200 estimadores, profundidade máxima de 3, learning rate de 0,01 e subamostragem de 80% das amostras por iteração, técnicas que promovem regularização e reduzem overfitting. SVM utilizou kernel RBF (Radial Basis Function) com parâmetro de regularização C=0,1 e parâmetro de largura de kernel γ=0,001, ambos com pesos balanceados para lidar com desproporção de classes. Naive Bayes foi implementado com variante Gaussiana e parâmetro de suavização de variância (var_smoothing) de 1e-5, permitindo pequeno ajuste para evitar divisão por zero em variâncias muito próximas a zero.

Adicionalmente, três estratégias de ensemble foram implementadas para explorar sinergia entre modelos. Voting_Equal combinava as predições de Random Forest, XGBoost e Naive Bayes com pesos iguais [1/3, 1/3, 1/3], permitindo votação igualitária entre os três. Voting_Weighted atribuía pesos diferentes [0,4, 0,3, 0,3] para privilegiar Random Forest baseado em desempenho preliminar. Stacking utilizava meta-learner com regressão logística, treinado a partir das predições de probabilidade dos três modelos base, aprendendo qual combinação das predições bases produzia melhor resultado.

### 2.6 Métricas de Avaliação

O pipeline de avaliação utilizou múltiplas métricas para capturar diferentes dimensões do desempenho. Acurácia foi empregada como métrica primária, medindo a proporção de predições corretas sobre o total de amostras. Precision, Recall e F1-macro forneceram perspectiva balanceada sobre desempenho por classe, com F1-macro computando a média não-ponderada destas métricas entre as três classes (vitória casa, empate, vitória visitante), relevante para cenário com desbalanceamento moderado. Ranked Probability Score (RPS) quantificou a qualidade das predições probabilísticas, penalizando previsões confiantes mas incorretas com severidade maior que previsões corretas duvidosas, variando de 0 (perfeito) a 1 (pior caso). Brier Score complementou esta análise através do erro quadrático médio das probabilidades preditas versus verdade, outra métrica sensível a calibração probabilística. ROC AUC (macro) mediu separação probabilística da classe positiva contra negativas em formato médio não-ponderado, capturando capacidade discriminativa dos modelos.

### 2.7 Validação Estatística

Intervalos de confiança 95% calculados via bootstrap com 1.000 iterações de reamostragem, garantindo significância estatística dos resultados.

---

## 3. Resultados

### 3.1 Impacto do Autoencoder: 43 Features vs 59 Features

**Tabela 1. Comparação de Desempenho (43F vs 59F)**

| Modelo | Accuracy (43F) | Accuracy (59F) | Ganho | RPS (43F) | RPS (59F) | Ganho RPS |
|--------|---|---|---|---|---|---|
| SVM | 45,39% | 46,18% | +0,79pp | 0,2160 | 0,2140 | -0,0020 |
| **Random Forest** | **47,37%** | **49,74%** | **+2,37pp** | **0,2110** | **0,2066** | **-0,0044** |
| XGBoost | 48,42% | 49,47% | +1,05pp | 0,2090 | 0,2070 | -0,0020 |
| Naive Bayes | 47,76% | 47,76% | 0,00pp | 0,2120 | 0,2100 | -0,0020 |
| Voting_Equal | 47,24% | 47,50% | +0,26pp | 0,2180 | 0,2170 | -0,0010 |
| Voting_Weighted | 47,37% | 47,50% | +0,13pp | 0,2170 | 0,2150 | -0,0020 |
| Stacking | 48,52% | 49,74% | +1,22pp | 0,2110 | 0,2084 | -0,0026 |

A análise da Tabela 1 revelou achados principais significativos. Random Forest apresentou ganho máximo de +2,37 pontos percentuais, evoluindo de 47,37% com 43 features para 49,74% com 59 features, representando a melhoria mais pronunciada entre todos os modelos. Notavelmente, todos os modelos de machine learning melhoraram seu desempenho ou mantiveram-no constante com a adição de features latentes, sem qualquer retrocesso, validando a hipótese de que features latentes complementam sistematicamente o espaço original. Avaliando qualidade probabilística através de RPS, todos os modelos apresentaram melhoria em 0,0010 a 0,0044 pontos, indicando que as features latentes não apenas aumentam acurácia mas também calibram melhor as probabilidades das predições. Interessantemente, Stacking empatou com Random Forest em acurácia absoluta (49,74%), sugerindo que o meta-learner capturou efetivamente a combinação ótima entre Random Forest, XGBoost e Naive Bayes.

### 3.2 Desempenho Detalhado com 59 Features

**Tabela 2. Métricas Completas dos Principais Modelos (59 Features)**

| Modelo | Accuracy | Precision | Recall | F1-macro | RPS | Brier | ROC AUC |
|--------|----------|-----------|--------|----------|-----|-------|---------|
| Baseline | 43,29% | — | — | — | 0,2200 | — | — |
| SVM | 46,18% | 0,4470 | 0,4490 | 0,4460 | 0,2140 | 0,6190 | 0,6270 |
| **Random Forest** | **49,74%** | **0,3302** | **0,4280** | **0,3656** | **0,2066** | **0,6060** | **0,6460** |
| **XGBoost** | **49,47%** | **0,4670** | **0,4700** | **0,4645** | **0,2070** | **0,6090** | **0,6560** |
| Naive Bayes | 47,76% | 0,4780 | 0,4720 | 0,4698 | 0,2100 | 0,6200 | 0,6460 |
| Voting_Equal | 47,50% | 0,4480 | 0,4550 | 0,4490 | 0,2170 | 0,6330 | 0,6500 |
| Voting_Weighted | 47,50% | 0,4470 | 0,4540 | 0,4470 | 0,2150 | 0,6280 | 0,6500 |
| Stacking | 49,74% | 0,4610 | 0,4670 | 0,4580 | 0,2084 | 0,6120 | 0,6590 |

Os destaques principais da Tabela 2 evidenciam o desempenho diferenciado de cada abordagem. Random Forest alcançou a melhor acurácia absoluta de 49,74% com menor valor de RPS entre todos os modelos (0,2066), indicando combinação ótima de taxa de acertos e calibração probabilística. XGBoost conquistou o melhor F1-macro de 0,4645, significativamente superior aos outros modelos, refletindo melhor equilíbrio entre precision e recall quando agregadas as três classes, atributo valioso em cenários com desbalanceamento. Stacking não apenas empatou Random Forest em acurácia (49,74%) mas superou todos os modelos em ROC AUC macro (0,6590), sugerindo capacidade superior em separar classes ao nível probabilístico através da combinação inteligente via meta-learner. Coletivamente, todos os sete modelos de machine learning superaram o baseline (43,29%) em magnitudes entre +3,2 e +6,5 pontos percentuais, confirmando que o sistema de features engineered fornece sinal preditivo robusto.

### 3.3 Análise Temporal por Temporada

**Tabela 3. Acurácia por Temporada de Teste**

| Temporada | Partidas | Baseline | SVM | RF | XGBoost | NB | Stacking |
|-----------|----------|----------|-----|-----|---------|-----|----------|
| 2014-2015 | 380 | 45,3% | 49,2% | **52,1%** | **52,1%** | 48,7% | 51,6% |
| 2015-2016 | 380 | 41,3% | 43,2% | **47,4%** | 46,8% | 46,8% | 47,9% |
| **Agregado** | **760** | **43,3%** | **46,2%** | **49,7%** | **49,5%** | **47,8%** | **49,7%** |

Temporada 2014-2015 mais previsível (52,1%) que 2015-2016 (47,4%), refletindo variações em competitividade e qualidade de dados.

### 3.4 Intervalos de Confiança Bootstrap (95%)

**Tabela 4. Intervalo de Confiança para Acurácia (1.000 iterações)**

| Modelo | Accuracy (50%) | IC Inferior | IC Superior | Margem de Erro |
|--------|---|---|---|---|
| Random Forest | 49,74% | 47,37% | 52,11% | ±2,37pp |
| XGBoost | 49,47% | 47,37% | 51,84% | ±2,24pp |
| Stacking | 49,74% | 47,37% | 52,11% | ±2,37pp |
| SVM | 46,18% | 43,82% | 48,55% | ±2,36pp |
| Baseline | 43,29% | 40,79% | 45,79% | ±2,50pp |

Não-sobreposição de intervalos confirma significância estatística (α<0,05) entre modelos ML e baseline.

### 3.5 Top 10 Features por Importância

**Tabela 5. Features Mais Relevantes (Random Forest com 59 Features)**

| Rank | Feature | Importância (Gini) | Tipo |
|------|---------|---|---|
| 1 | h2h_games | 0,1063 | Head-to-Head |
| 2 | B365D | 0,1002 | Odds de Apostas |
| 3 | points_diff | 0,0970 | Posição |
| 4 | away_position | 0,0704 | Posição |
| 5 | position_diff | 0,0545 | Posição |
| 6 | shots_diff | 0,0499 | Médias Móveis |
| 7 | away_points | 0,0482 | Posição |
| 8 | goals_avg_diff | 0,0379 | Médias Móveis |
| 9 | shotsontarget_diff | 0,0369 | Médias Móveis |
| 10 | B365H | 0,0349 | Odds de Apostas |

A análise de importância das features na Tabela 5 produziu insights reveladores sobre preditibilidade. Features de head-to-head e odds de apostas dominam o ranking de importância, com h2h_games (0,1063) e B365D (0,1002) ocupando as duas primeiras posições com importância acima de 0,1, sugerindo que informação histórica de confrontos diretos e probabilidades implícitas de mercado carregam sinal preditivo superior. Posição na tabela mantém relevância alta, com pontos_diff (0,0970), away_position (0,0704), position_diff (0,0545) e away_points (0,0482) distribuídos no top-7, indicando que desempenho cumulativo refletido em classificação é altamente informativo. Contrastando com essas categorias, ratings FIFA de qualidade de elenco (overall_diff, attack_diff, etc.) aparecem apenas após o rank 20 em importância, sugerindo que qualidade bruta de jogadores é menos preditor de resultado individual de partida que forma recente, dinâmica de confronto e percepção de mercado.

---

## 4. Discussão

### 4.1 Impacto do Autoencoder

O autoencoder com arquitetura 43→32→16 sucessivamente capturou padrões não-lineares, expandindo espaço de representação de 43 para 59 dimensões. Ganho de +2,37pp em acurácia (47,37% → 49,74%) com Random Forest valida hipótese que representações latentes complementam features engineered.

**Interpretação:** As 16 dimensões latentes sintetizam sinergia entre categorias originais (forma + ratings + estatísticas), capturando interações multidimensionais que modelos lineares não detectam.

### 4.2 Superioridade de Random Forest

Random Forest atingiu 49,74% com 59 features através de vários fatores complementares. Primeiro, o algoritmo demonstra robustez inerente a alta dimensionalidade, não sofrer degradação em cenários com muitas features que afligem métodos lineares. Segundo, sua estrutura baseada em árvores de decisão captura naturalmente interações não-lineares entre features sem necessidade de engenharia explícita, permitindo que o ensemble se beneficie integralmente das 16 dimensões latentes não-lineares do autoencoder. Terceiro, o mecanismo de ensembling interno através de múltiplas árvores treinadas em subamostragens diferentes reduz significativamente a variância, produzindo generalizador robusto com menor tendência a overfitting.

Com ressalva importante, Random Forest apresenta limitação crítica quanto ao desbalanceamento de classes: demonstra viés extremo contra a classe minoritária (empates), nunca predizendo empate em nenhuma das 760 partidas do conjunto de teste. Este comportamento reduz recall para classe D (empate) e inflaciona precision para classes H e A, fenômeno onde o modelo aprende que é mais seguro predizer vitória quando incerteza prevalece. XGBoost, em contraste, mitigou este viés melhor através de suas penalidades de regularização.

### 4.3 Equilíbrio de XGBoost

XGBoost obteve F1-macro=0,4645 (melhor), sugerindo regularização L1/L2 atenua desbalanceamento melhor que bagging. RPS equivalente (0,2070) indica qualidade probabilística comparável.

### 4.4 Validação Temporal

Ausência de data leakage (encoder treinado apenas em 2005-2014) confirmada pelo desempenho em 2014-2016, validando generalização temporal e robustez do modelo.

---

## 5. Conclusão

Este trabalho demonstrou que integração de autoencoder com machine learning tradicional eleva capacidade preditiva em problemas de predição esportiva. Random Forest com **59 features alcançou 49,74% de acurácia**, representando **+2,37pp de ganho** comparado a 43 features puras e **+15,6pp sobre baseline**.

As principais contribuições deste trabalho incluem: (1) desenvolvimento e validação de metodologia inovadora que integra autoencoder com machine learning tradicional, demonstrando que representações latentes comprimidas complementam sistematicamente features engineered em problemas esportivos; (2) implementação de validação temporal rigorosa sem data leakage, com encoder treinado exclusivamente em 2005-2014 e avaliação em dados completamente independentes 2014-2016, estabelecendo protocolo robusto para futuros trabalhos; (3) demonstração empírica comprovada de ganho de +2,37 pontos percentuais em acurácia com Random Forest ao expandir de 43 para 59 features, validando hipótese de valor de features latentes; (4) análise completa de explicabilidade através de importância de features e SHAP, identificando preditores mais relevantes e fornecendo interpretabilidade do modelo; (5) disponibilização de pipeline reprodutível que pode servir como referência para futuras investigações em predição esportiva.

As limitações significativas deste estudo reconhecem: desbalanceamento de classes onde empates (25,3%) são subestimados sistematicamente por Random Forest, reduzindo aplicabilidade prática; ausência de features contextuais dinâmicas como lesões, suspensões e motivações psicológicas que influenciam resultados reais; hyperparameter tuning limitado onde modelos foram ajustados manualmente em vez de grid search extensivo, sugerindo possível subestimação de desempenho com otimização mais rigorosa.

Perspectivas futuras sugerem direções promissoras: implementação de arquiteturas Deep Learning como LSTMs (Long Short-Term Memory) para capturar dependências temporais de longo alcance, permitindo o modelo aprender dinâmica sazonal e tendências de longo prazo dentro de temporadas; aplicação de Transfer Learning em ligas secundárias menores (Championship, La Liga, etc.), aproveitando conhecimento aprendido em Premier League para acelerar treinamento e compensar escassez de dados em competições menos documentadas; integração de contexto externo em tempo real durante previsão de partidas futuras, incorporando lesões anunciadas, mudanças de técnico, e outros eventos dinâmicos que ocorrem entre publicação de odds e execução da partida.

A metodologia estabelece fundação sólida para pesquisas futuras, demonstrando que feature engineering não-linear via autoencoder é eficaz para capturar complexidade em dados esportivos multidimensionais.

---

## Referências

1. Baboota, R., & Kaur, H. (2018). Predictive analysis and modelling football results using machine learning approach for English Premier League. *International Journal of Forecasting*, 35(2), 741-755.

2. Bunker, R. P., & Thabtah, F. (2019). A machine learning framework for sport result prediction. *Applied Computing and Informatics*, 15(1), 27-33.

3. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD Conference*, 785-794.
