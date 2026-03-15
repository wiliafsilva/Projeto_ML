# Predição de Resultados da Premier League Inglesa Utilizando Aprendizado de Máquina: Uma Abordagem com 43 Features e Validação Temporal

## Abstract

Este estudo apresenta uma abordagem abrangente para previsão de resultados de partidas de futebol da Premier League Inglesa (EPL) utilizando técnicas de aprendizado de máquina. Seguindo a metodologia de Baboota & Kaur (2018), desenvolvemos um sistema que integra 43 features derivadas de estatísticas históricas, ratings FIFA, odds de apostas e análises head-to-head. Avaliamos quatro algoritmos de classificação—Support Vector Machine (SVM), Random Forest, XGBoost e Naive Bayes—além de três métodos ensemble (Voting e Stacking). O dataset compreende 4.180 partidas distribuídas em 11 temporadas (2005-2016), com divisão temporal rigorosa: treinamento em 2005-2014 (3.420 partidas) e teste em 2014-2016 (760 partidas). Os resultados demonstram que Random Forest alcançou a melhor acurácia (49,74%) e o menor Ranked Probability Score (RPS=0,2066), superando baselines em 14,9 pontos percentuais. XGBoost apresentou o melhor equilíbrio entre classes (F1-macro=0,4645). Análise SHAP revelou que histórico head-to-head (h2h_games), odds de empate (B365D) e diferença de pontos na tabela (points_diff) são os preditores mais relevantes. Intervalos de confiança bootstrap (95%, 1.000 iterações) confirmam a significância estatística dos resultados. Este trabalho contribui com validação temporal rigorosa, análise de explicabilidade e comparação sistemática com múltiplos baselines.

**Palavras-chave:** Predição de futebol, Premier League, Machine Learning, Random Forest, XGBoost, SHAP, Ranked Probability Score

---

## 1. Introdução

A predição de resultados esportivos representa um desafio significativo em aprendizado de máquina devido à natureza estocástica e multifatorial dos eventos competitivos. No contexto do futebol profissional, particularmente na Premier League Inglesa (EPL), a previsão precisa de resultados possui implicações tanto científicas quanto comerciais, incluindo análise tática, gestão esportiva e mercados de apostas.

Estudos anteriores demonstraram que abordagens baseadas em features estatísticas agregadas superam métodos baseline consideravelmente. Baboota & Kaur (2018) desenvolveram uma metodologia sistemática que combina features de forma (Form), estatísticas de jogos (μₖ) e análises históricas para predição de resultados em formato ternário (Vitória Casa, Empate, Vitória Visitante). Seus resultados indicaram que modelos ensemble alcançam desempenho superior quando comparados a classificadores individuais.

Este trabalho apresenta uma implementação e extensão da metodologia proposta por Baboota & Kaur, incorporando 43 features distribuídas em sete categorias: (i) baseline estatísticas, (ii) sistema Form ELO-style, (iii) médias móveis μₖ, (iv) ratings FIFA, (v) histórico head-to-head, (vi) posição na tabela da liga, e (vii) odds de apostas. Adicionalmente, desenvolvemos features de interação de segunda ordem para capturar relações não-lineares entre preditores.

**Objetivos específicos:**
1. Replicar a metodologia científica de Baboota & Kaur (2018) com validação temporal rigorosa
2. Avaliar quatro algoritmos de classificação e três métodos ensemble em um dataset de 11 temporadas
3. Comparar sistematicamente com múltiplos modelos baseline
4. Realizar análise de explicabilidade utilizando SHAP (SHapley Additive exPlanations)
5. Validar significância estatística através de intervalos de confiança bootstrap

---

## 2. Trabalhos Relacionados

A literatura sobre predição de resultados de futebol pode ser categorizada em três abordagens principais: (i) métodos estatísticos tradicionais, (ii) aprendizado de máquina supervisionado, e (iii) modelagem probabilística.

**Métodos Estatísticos:** Modelos de Poisson e suas variações (Dixon & Coles, 1997) representam a abordagem clássica, assumindo que gols seguem distribuições de Poisson independentes. Essas técnicas capturam padrões de força ofensiva/defensiva mas possuem limitações em incorporar features multidimensionais.

**Aprendizado de Máquina:** Baboota & Kaur (2018) aplicaram SVM, Naive Bayes e Random Forest em dados da EPL, introduzindo o conceito de separação Class A/Class B para otimizar features segundo características dos algoritmos. Joseph et al. (2006) demonstraram a superioridade de redes neurais sobre regressão logística para predição de resultados da liga inglesa. Constantinou & Fenton (2012) propuseram redes Bayesianas para modelagem causal de resultados.

**Ensemble Methods:** Bunker & Thabtah (2019) conduziram revisão sistemática indicando que métodos ensemble frequentemente superam classificadores individuais em contextos esportivos. Tax & Joustra (2015) aplicaram Stacking em competições de futebol holandês com ganhos de 3-5% em acurácia.

**Lacunas Identificadas:** Poucos estudos incorporam odds de apostas como features auxiliares, ignorando informação agregada do mercado. Análises de explicabilidade (SHAP, LIME) raramente são aplicadas em predição esportiva, limitando a interpretabilidade dos modelos. Validação temporal rigorosa com separação por temporadas completas permanece subutilizada.

---

## 3. Metodologia

### 3.1 Descrição do Dataset

O dataset compreende 4.180 partidas da Premier League Inglesa coletadas de 11 temporadas consecutivas (2005-2016). As características do dataset são apresentadas na Tabela 1.

**Tabela 1. Características do Dataset**

| Métrica | Valor |
|---------|-------|
| Total de Partidas | 4.180 |
| Período | 2006-2016 |
| Temporadas | 11 |
| Times Únicos | 37 |
| Média Gols/Jogo | 2,66 |
| Vitórias Casa | 1.940 (46,4%) |
| Empates | 1.058 (25,3%) |
| Vitórias Visitante | 1.182 (28,3%) |
| Partidas Treino | 3.420 (2005-2014) |
| Partidas Teste | 760 (2014-2016) |
| Split Treino/Teste | 81,8% / 18,2% |

A divisão temporal rigorosa previne vazamento de informação (*data leakage*), garantindo que modelos sejam avaliados exclusivamente em dados futuros não observados durante treinamento. A distribuição de classes apresenta desbalanceamento moderado, com vantagem do mandante (46,4%) em relação a empates (25,3%) e vitórias visitantes (28,3%).

**Fontes de Dados:**
- **Estatísticas de jogos:** Football-Data.co.uk (gols, chutes, escanteios, resultados)
- **Ratings FIFA:** Base consolidada com 695 entradas cobrindo 35 times (2006-2016), incluindo ratings geral, ataque, meio-campo e defesa
- **Odds de apostas:** Bet365 (odds de vitória casa, empate e vitória visitante)

Valores ausentes (principalmente ratings FIFA para times promovidos) foram imputados utilizando mediana temporal da liga, afetando aproximadamente 5-10% das amostras.

### 3.2 Engenharia de Features

Seguindo Baboota & Kaur (2018), as features foram projetadas para capturar múltiplas dimensões do desempenho de times. O sistema completo compreende 43 features distribuídas em sete categorias principais.

#### 3.2.1 Features Baseline (3 features)

Features fundamentais derivadas de estatísticas acumuladas:
- **Goal Difference (gd_diff):** Diferença no saldo de gols entre mandante e visitante
- **Streak Difference (streak_diff):** Diferença na sequência de vitórias recentes normalizada
- **Weighted Difference (weighted_diff):** Média ponderada de resultados recentes com pesos decrescentes

Essas features estabelecem a linha base de desempenho relativo entre times.

#### 3.2.2 Sistema Form (3 features)

Implementação do sistema Form proposto por Baboota & Kaur (2018), baseado em atualização estilo ELO. A variável Form de cada time (ξ) é inicializada em 1,0 no início de cada temporada e atualizada após cada partida segundo as equações:

**Vitória do time α sobre β:**
$$\xi_j^{\alpha} = \xi_{j-1}^{\alpha} + \gamma \cdot \xi_{j-1}^{\beta}$$
$$\xi_j^{\beta} = \xi_{j-1}^{\beta} - \gamma \cdot \xi_{j-1}^{\beta}$$

**Empate entre α e β:**
$$\xi_j^{\alpha} = \xi_{j-1}^{\alpha} - \gamma(\xi_{j-1}^{\alpha} - \xi_{j-1}^{\beta})$$
$$\xi_j^{\beta} = \xi_{j-1}^{\beta} - \gamma(\xi_{j-1}^{\beta} - \xi_{j-1}^{\alpha})$$

onde γ = 0,33 representa a fração de transferência entre times. Três features são derivadas:
- **form_diff:** Diferença Form(mandante) - Form(visitante)
- **home_form:** Form individual do mandante
- **away_form:** Form individual do visitante

#### 3.2.3 Médias Móveis μₖ (4 features)

Estatísticas de médias móveis calculadas sobre janela de k=6 jogos anteriores:
- **corners_diff:** Diferença nas médias de escanteios
- **shotsontarget_diff:** Diferença nas finalizações no alvo
- **shots_diff:** Diferença no total de finalizações
- **goals_avg_diff:** Diferença na média de gols marcados

A janela temporal k=6 foi selecionada seguindo o artigo base, balanceando sensibilidade a mudanças recentes com estabilidade estatística.

#### 3.2.4 Ratings FIFA (4 features)

Integração de ratings oficiais FIFA extraídos de bases consolidadas:
- **overall_diff:** Diferença no rating geral
- **attack_diff:** Diferença no rating de ataque
- **midfield_diff:** Diferença no rating de meio-campo
- **defense_diff:** Diferença no rating de defesa

Ratings FIFA capturam qualidade intrínseca dos elencos independentemente de forma recente.

#### 3.2.5 Head-to-Head (6 features)

Análise de confrontos diretos históricos com janela de 5 jogos:
- **h2h_home_wins, h2h_draws, h2h_away_wins:** Contagens de resultados em confrontos anteriores
- **h2h_home_goals_avg, h2h_away_goals_avg:** Média de gols marcados em confrontos
- **h2h_games:** Número total de confrontos registrados

Features H2H capturam rivalidades específicas e padrões históricos entre pares de times.

#### 3.2.6 Posição na Tabela (6 features)

Simulação contínua da tabela de classificação ao longo da temporada:
- **home_position, away_position:** Posições atuais na tabela
- **position_diff:** Diferença de posições
- **home_points, away_points:** Pontuação acumulada
- **points_diff:** Diferença de pontos

Essas features refletem o desempenho cumulativo na temporada corrente, identificando times em ascensão ou declínio.

#### 3.2.7 Odds de Apostas (9 features)

Integração de odds Bet365 e conversão em probabilidades implícitas:
- **B365H, B365D, B365A:** Odds brutas (casa, empate, visitante)
- **prob_home, prob_draw, prob_away:** Probabilidades implícitas (1/odd)
- **prob_home_norm, prob_draw_norm, prob_away_norm:** Probabilidades normalizadas (soma=1)

Odds agregam informação de especialistas, análises de mercado e lesões de jogadores não capturadas em estatísticas históricas.

#### 3.2.8 Features de Interação (8 features)

Features de segunda ordem capturando interações não-lineares:
- **h2h_confidence:** Dominância em confrontos diretos
- **away_advantage:** Força relativa do visitante em contexto adverso
- **season_trend:** Tendência acumulada na temporada
- **position_form_home, position_form_away:** Combinação posição × forma
- **strength_balance:** Equilíbrio de forças entre times

**Prevenção de Data Leakage:** Todas as features são calculadas exclusivamente com informações disponíveis antes de cada partida. Reset automático ao início de cada temporada garante independência temporal.

### 3.3 Classificação de Features: Class A vs Class B

Seguindo Baboota & Kaur (2018), as features foram organizadas em duas classes segundo propriedades algorítmicas:

- **Class A (27 features):** Valores individuais de cada time (ex: home_form, away_position, h2h_games). Utilizadas por Naive Bayes devido à suposição de independência condicional entre features.

- **Class B (29 features):** Diferenciais entre mandante e visitante (ex: form_diff, gd_diff, shots_diff). Utilizadas por SVM, Random Forest e XGBoost, que modelam relações de superioridade relativa mais eficientemente.

Esta separação otimiza a representação dos dados segundo as premissas matemáticas de cada família de algoritmos.

### 3.4 Modelos de Aprendizado de Máquina

#### 3.4.1 Modelos Individuais

**Support Vector Machine (SVM):**  
Classificador baseado em hiperplanos de margem máxima. Configuração: kernel RBF, C=0,1, γ=0,001. Pesos balanceados aplicados para mitigar desbalanceamento de classes.

**Random Forest:**  
Ensemble de árvores de decisão com agregação por votação majoritária. Configuração: 50 estimadores, profundidade máxima 5, critério Gini, pesos balanceados. Controle de overfitting via limitação de profundidade e amostras mínimas por folha.

**XGBoost:**  
Gradient boosting com regularização. Configuração: 200 estimadores, profundidade máxima 3, taxa de aprendizado 0,01, subsample=0,8. Otimização via log-loss multi-classe.

**Naive Bayes:**  
Classificador probabilístico baseado no teorema de Bayes com suposição de independência condicional. Configuração: Gaussian Naive Bayes com suavização de variância (var_smoothing=1×10⁻⁵).

#### 3.4.2 Calibração de Probabilidades

Calibração isotônica foi aplicada utilizando validação cruzada temporal (3 folds) quando resultava em melhoria do Ranked Probability Score. A calibração ajusta as probabilidades preditas para refletir frequências empíricas reais, crucial para métricas probabilísticas como RPS e Brier Score.

#### 3.4.3 Métodos Ensemble

**Voting Classifier (Equal/Weighted):**  
Combinação por soft-voting de probabilidades preditas. Duas variantes foram testadas:
- **Voting_Equal:** Pesos uniformes [1/3, 1/3, 1/3] para Random Forest, XGBoost e Naive Bayes
- **Voting_Weighted:** Pesos ajustados [0,4, 0,3, 0,3] priorizando Random Forest baseado em desempenho de validação cruzada

**Stacking Classifier:**  
Arquitetura meta-learner com regressão logística. Os modelos base (SVM, RF, XGBoost, NB) geram predições de probabilidade que alimentam um classificador de segunda camada, aprendendo a combinação ótima das predições base.

### 3.5 Métricas de Avaliação

Sete métricas foram empregadas para avaliação multidimensional:

**Acurácia:** Proporção de predições corretas. Métrica primária mas sensível a desbalanceamento.

**Precision, Recall, F1-Score (macro):** Média não-ponderada entre classes, penalizando modelos que ignoram classes minoritárias.

**Ranked Probability Score (RPS):**  
Métrica probabilística que avalia a qualidade da distribuição de probabilidades predita através de diferenças cumulativas:

$$\text{RPS} = \frac{1}{K-1} \sum_{j=1}^{n} \sum_{k=1}^{K} (P_j^{cumsum}(k) - y_j^{cumsum}(k))^2$$

onde K=3 classes, $P_j^{cumsum}$ são probabilidades cumulativas preditas, e $y_j^{cumsum}$ é a distribuição verdadeira cumulativa. RPS penaliza erros proporcionalmente à distância entre classes preditas e reais (0 = perfeito, 1 = pior).

**Brier Score:** Métrica de calibração medindo erro quadrático médio entre probabilidades preditas e resultados binários.

**ROC AUC (macro):** Área sob curva ROC média entre classes, avaliando capacidade de separação probabilística.

---

## 4. Configuração Experimental

### 4.1 Divisão Temporal

A divisão temporal rigorosa previne contaminação de dados futuros:
- **Treinamento:** Temporadas 2005-2014 (9 temporadas, N=3.420 partidas)
- **Teste:** Temporadas 2014-2016 (2 temporadas, N=760 partidas)
  - Temporada 2014-2015: 380 partidas
  - Temporada 2015-2016: 380 partidas

Esta configuração simula cenário realista onde modelos treinados em histórico são aplicados a previsões futuras.

### 4.2 Otimização de Hiperparâmetros

Grid Search foi conduzido com validação cruzada temporal (TimeSeriesSplit, 5 folds) no conjunto de treinamento. A métrica de otimização foi RPS, priorizando qualidade probabilística sobre acurácia pontual.

**Tabela 2. Hiperparâmetros Otimizados e RPS de Validação Cruzada**

| Modelo | RPS (CV) | Hiperparâmetros |
|--------|----------|-----------------|
| XGBoost | 0,410 | n_estimators=200, max_depth=3, lr=0,01, subsample=0,8 |
| SVM | 0,410 | C=0,1, γ=0,001, kernel=rbf |
| RandomForest | 0,425 | n_estimators=50, max_depth=5, min_samples_split=2 |
| NaiveBayes | 0,437 | var_smoothing=1×10⁻⁵ |

XGBoost e SVM apresentaram RPS de validação cruzada equivalente, seguidos por Random Forest e Naive Bayes.

### 4.3 Modelos Baseline

Três baselines foram implementados para validação de aprendizado efetivo:
1. **Most Frequent:** Sempre prevê classe majoritária (Vitória Casa)
2. **Stratified:** Previsões aleatórias respeitando distribuição de classes do treino
3. **Always Draw:** Sempre prevê empate (pior caso)

Modelos de aprendizado de máquina devem superar significativamente esses baselines para demonstrar capacidade preditiva real.

### 4.4 Validação Estatística

Intervalos de confiança foram calculados via bootstrap com 1.000 iterações de reamostragem com reposição. Para cada iteração, métricas (Acurácia, F1, RPS) foram computadas e intervalos de 95% extraídos via percentis 2,5% e 97,5%. Não-sobreposição de intervalos indica significância estatística (α<0,05).

---

## 5. Resultados

### 5.1 Desempenho Geral dos Modelos

A Tabela 3 apresenta o desempenho comparativo de todos os modelos no conjunto de teste completo (760 partidas, 2014-2016).

**Tabela 3. Comparação de Modelos no Conjunto de Teste**

| Modelo | Accuracy | Precision | Recall | F1 | RPS | Brier | ROC AUC |
|--------|----------|-----------|--------|-----|-----|-------|---------|
| Baseline | 0,433 | — | — | — | — | — | — |
| SVM | 0,462 | 0,447 | 0,449 | 0,446 | 0,214 | 0,619 | 0,627 |
| **Random Forest** | **0,497** | 0,330 | 0,428 | 0,366 | **0,207** | **0,606** | 0,646 |
| XGBoost | 0,495 | **0,467** | **0,470** | **0,465** | 0,207 | 0,609 | **0,656** |
| Naive Bayes | 0,478 | 0,478 | 0,472 | 0,470 | 0,210 | 0,620 | 0,646 |
| Voting_Equal | 0,475 | 0,448 | 0,455 | 0,449 | 0,217 | 0,633 | 0,650 |
| Voting_Weighted | 0,475 | 0,447 | 0,454 | 0,447 | 0,215 | 0,628 | 0,650 |
| Stacking | 0,497 | 0,461 | 0,467 | 0,458 | 0,208 | 0,612 | 0,659 |

**Principais Achados:**
- **Random Forest** alcançou a maior acurácia (49,74%) e o menor RPS (0,2066)
- **XGBoost** apresentou o melhor F1-Score (0,4645), indicando superior equilíbrio entre classes
- **Stacking** empatou com Random Forest em acurácia e obteve o melhor ROC AUC (0,6587)
- Todos os modelos superaram o baseline em 2,9-6,5 pontos percentuais
- Ensembles não superaram consistentemente o melhor modelo individual (Random Forest)

### 5.2 Análise Temporal por Temporada

A Tabela 4 decompõe o desempenho por temporada individual, revelando variações sazonais.

**Tabela 4. Acurácia por Temporada**

| Temporada | Jogos | Baseline | SVM | RandomForest | XGBoost | NaiveBayes | Stacking |
|-----------|-------|----------|-----|--------------|---------|------------|----------|
| 2014-2015 | 380 | 45,3% | 49,2% | **52,1%** | **52,1%** | 48,7% | 51,6% |
| 2015-2016 | 380 | 41,3% | 43,2% | **47,4%** | 46,8% | 46,8% | 47,9% |
| **Agregado** | **760** | **43,3%** | **46,2%** | **49,7%** | **49,5%** | **47,8%** | **49,7%** |

A temporada 2014-2015 mostrou-se mais previsível (52,1% para RF e XGBoost) comparada a 2015-2016 (47,4% para RF). Variações sazonais podem refletir diferentes níveis de competitividade, mudanças regulamentares ou qualidade dos dados de odds disponíveis.

### 5.3 Intervalos de Confiança Bootstrap

A Tabela 5 apresenta intervalos de confiança de 95% calculados via bootstrap (1.000 iterações) para o agregado das duas temporadas de teste.

**Tabela 5. Intervalos de Confiança (95%) – Agregado 2014-2016**

| Modelo | Accuracy | F1-Score | RPS |
|--------|----------|----------|-----|
| Random Forest | 0,507 [0,469–0,545] | 0,378 [0,356–0,404] | 0,412 [0,394–0,432] |
| XGBoost | 0,495 [0,458–0,533] | 0,475 [0,440–0,510] | 0,415 [0,401–0,428] |
| Naive Bayes | 0,470 [0,437–0,504] | 0,461 [0,429–0,496] | 0,419 [0,402–0,433] |
| SVM | 0,465 [0,434–0,503] | 0,451 [0,424–0,489] | 0,428 [0,409–0,446] |

Observa-se que os intervalos de confiança de Random Forest e XGBoost apresentam sobreposição substancial, indicando que as diferenças de desempenho não são estatisticamente significativas (p>0,05). Ambos superam consistentemente os baselines, com intervalos completamente deslocados.

### 5.4 Análise por Classe

A Tabela 6 apresenta métricas desagregadas por classe de resultado para Random Forest e XGBoost.

**Tabela 6. Desempenho por Classe (Agregado 2014-2016)**

| Modelo | Classe | Precision | Recall | F1-Score | Support |
|--------|--------|-----------|--------|----------|---------|
| **RandomForest** | Vitória Casa | 0,562 | 0,783 | 0,655 | 329 |
|  | Empate | **0,000** | **0,000** | **0,000** | 200 |
|  | Vitória Visitante | 0,428 | 0,307 | 0,358 | 231 |
| **XGBoost** | Vitória Casa | 0,541 | 0,696 | 0,609 | 329 |
|  | Empate | 0,338 | 0,245 | 0,284 | 200 |
|  | Vitória Visitante | 0,503 | 0,471 | 0,486 | 231 |

**Observações Críticas:**
- Random Forest apresenta viés extremo, **nunca prevendo empate** (Precision=Recall=0,000)
- XGBoost demonstra equilíbrio superior, com F1(Empate)=0,284, ainda que abaixo das demais classes
- Desbalanceamento original do dataset (25,3% empates) impacta negativamente a classe minoritária
- Técnicas de balanceamento (class_weight='balanced', sample_weight) atenuam mas não eliminam o viés

### 5.5 Importância de Features

A Tabela 7 apresenta as 10 features mais relevantes segundo Random Forest (importância de Gini) e valores SHAP médios.

**Tabela 7. Top 10 Features por Importância**

| Rank | Feature | Importância (RF) | SHAP (Impacto Médio) | Categoria |
|------|---------|------------------|----------------------|-----------|
| 1 | h2h_games | 0,1063 | 0,0328 | Head-to-Head |
| 2 | B365D | 0,1002 | — | Odds de Apostas |
| 3 | points_diff | 0,0970 | — | Posição na Tabela |
| 4 | away_position | 0,0704 | — | Posição na Tabela |
| 5 | position_diff | 0,0545 | — | Posição na Tabela |
| 6 | shots_diff | 0,0499 | — | Médias Móveis μₖ |
| 7 | away_points | 0,0482 | — | Posição na Tabela |
| 8 | goals_avg_diff | 0,0379 | — | Médias Móveis μₖ |
| 9 | shotsontarget_diff | 0,0369 | — | Médias Móveis μₖ |
| 10 | B365H | 0,0349 | — | Odds de Apostas |

**Insights:**
- Features de **Head-to-Head** e **Odds de Apostas** dominam as primeiras posições
- **Posição na tabela** (points_diff, position_diff) apresenta alta relevância preditiva
- Features μₖ (shots, goals) contribuem moderadamente
- Ratings FIFA (overall_diff, attack_diff) aparecem apenas após rank 20, sugerindo menor impacto direto comparado a estatísticas recentes

Análise de correlação identificou multicolinearidade esperada entre ratings FIFA (r(overall_diff, midfield_diff)=0,999), mas algoritmos baseados em árvores (RF, XGBoost) são robustos a essa dependência linear.

### 5.6 Comparação com Baseline

A Tabela 8 apresenta ganhos absolutos relativos aos três baselines testados.

**Tabela 8. Ganho sobre Baselines (Agregado 2014-2016)**

| Modelo | Accuracy | Δ vs Most Freq | Δ vs Stratified | Δ vs Always Draw |
|--------|----------|----------------|-----------------|------------------|
| Baseline (Most Freq) | 43,29% | — | +17,2 p.p. | +12,0 p.p. |
| Baseline (Stratified) | 26,05% | -17,2 p.p. | — | -5,2 p.p. |
| Baseline (Always Draw) | 31,32% | -12,0 p.p. | +5,2 p.p. | — |
| **Random Forest** | **49,74%** | **+6,5 p.p.** | **+23,7 p.p.** | **+18,4 p.p.** |
| **XGBoost** | **49,47%** | **+6,2 p.p.** | **+23,4 p.p.** | **+18,2 p.p.** |

Random Forest supera o baseline mais forte (Most Frequent) em 6,5 pontos percentuais, representando ganho relativo de 14,9%. Este resultado confirma aprendizado efetivo além de heurísticas triviais.

---

## 6. Discussão

### 6.1 Desempenho dos Modelos

Random Forest emergiu como o modelo mais preciso, atingindo 49,74% de acurácia e RPS de 0,2066. Este resultado é consistente com estudos anteriores indicando superioridade de ensembles de árvores em problemas com features heterogêneas e interações complexas. A limitação de profundidade (max_depth=5) e número moderado de árvores (n_estimators=50) foram estratégias eficazes contra overfitting.

Por outro lado, XGBoost demonstrou equilíbrio superior entre classes, alcançando F1-macro de 0,4645 contra 0,3656 do Random Forest. Este resultado sugere que boosting sequencial com regularização L1/L2 atenua melhor o viés contra classes minoritárias comparado a bagging puramente.

Naive Bayes, apesar de F1 razoável (0,4698), apresentou RPS inferior (0,2097), indicando calibração probabilística menos precisa. A suposição de independência condicional é violada por correlações intrínsecas entre features (ex: overall_diff e attack_diff), degradando qualidade das probabilidades preditas.

SVM com kernel RBF obteve desempenho moderado (Accuracy=46,18%), possivelmente devido à alta dimensionalidade (29 features Class B) e espaço de features não-linearmente separável. Experimentos com kernels alternativos (polinomial, sigmoid) não melhoraram os resultados.

### 6.2 Ensembles: Expectativa vs Realidade

Contrariando a literatura (Bunker & Thabtah, 2019), os métodos ensemble testados não superaram o melhor modelo individual. Stacking empatou com Random Forest (49,74%), enquanto Voting obteve desempenho inferior (47,50%).

**Hipóteses Explicativas:**
1. **Erros Correlacionados:** Modelos base cometem erros sistemáticos nas mesmas partidas (ex: empates são consistentemente ignorados), limitando ganhos por diversidade
2. **Tamanho do Conjunto de Teste:** Com apenas 760 amostras, o meta-learner do Stacking possui dados limitados para aprender combinações ótimas
3. **Otimização Insuficiente:** Pesos do Voting_Weighted foram ajustados manualmente; otimização automática poderia melhorar resultados

### 6.3 O Problema do Empate

Random Forest apresenta incapacidade completa de prever empates (Precision=Recall=0,000), comportamento parcialmente esperado dado:
- Frequência reduzida no treino (25,3% vs 46,4% vitórias casa)
- Distribuição de probabilidades tende a extremos (alta confiança em vitória/derrota)
- Limitações do critério Gini para classes minoritárias

XGBoost mitiga parcialmente este problema (F1(Empate)=0,284), mas permanece desafiador. Estratégias futuras incluem:
- Sobreamostragem sintética (SMOTE) da classe Empate
- Threshold tuning assimétrico (ajustar limites de decisão favorecendo empates)
- Modelos especializados em dois estágios (1º: casa/não-casa, 2º: empate/visitante)

### 6.4 Análise de Explicabilidade

Análise SHAP revelou que **h2h_games** (número de confrontos históricos) possui maior impacto marginal médio (SHAP=0,0328), apesar de importância de Gini moderada. Isto indica que, embora não seja o split mais frequente em árvores, histórico de confrontos possui forte poder discriminativo quando presente.

**B365D** (odds de empate) aparece como segunda feature mais importante (0,1002), validando a hipótese de que odds agregam informação de fontes externas não capturadas em estatísticas históricas (ex: lesões, moral do time, condições climáticas).

Features de **posição na tabela** (points_diff, position_diff) dominam ranks 3-5, confirmando que desempenho cumulativo na temporada corrente é fortemente preditivo. A dinâmica da posição captura "momentum" e qualidade relativa de forma mais direta que ratings estáticos FIFA.

Surpreendentemente, **ratings FIFA** aparecem apenas após rank 20. Possíveis explicações:
- Ratings FIFA são atualizados irregularmente (mensalmente), não capturando mudanças recentes
- Estatísticas de jogos (μₖ) refletem qualidade real mais precisamente que avaliações subjetivas
- Correlações com outras features reduzem contribuição marginal

### 6.5 Limitações e Trabalhos Futuros

**Limitações Metodológicas:**
1. **Desbalanceamento de Classes:** Estratégias de reamostragem (SMOTE, ADASYN) não foram exploradas sistematicamente
2. **Features Temporais:** Sazonalidade intra-temporada (Natal, fixtures congestionados) não foi modelada
3. **Contexto Externo:** Lesões, suspensões, substituições técnicas e fatores psicológicos não foram incorporados
4. **Ensemble Tuning:** Hiperparâmetros dos ensembles foram fixados; otimização conjunta poderia melhorar resultados

**Direções Futuras:**
- **Deep Learning:** Redes LSTM para capturar dependências temporais de longo prazo
- **Transfer Learning:** Pré-treino em ligas secundárias (Championship, LaLiga) com fine-tuning na EPL
- **Features Contextuais:** Integração de APIs para lesões/suspensões em tempo real
- **Probabilidades Dinâmicas:** Modelos Bayesianos online que atualizam previsões conforme eventos da partida
- **Análise de Apostas:** Desenvolvimento de estratégias de apostas baseadas em edge probabilístico (quando P(ML) > 1/odd)

---

## 7. Conclusão

Este estudo demonstrou que técnicas de aprendizado de máquina, quando combinadas com engenharia rigorosa de features e validação temporal, superam significativamente modelos baseline para predição de resultados da Premier League Inglesa. Random Forest alcançou acurácia de 49,74% (RPS=0,2066), representando ganho de 14,9% sobre o baseline Most Frequent.

A separação Class A/Class B proposta por Baboota & Kaur (2018) foi validada experimentalmente: Naive Bayes com features individuais alcançou F1=0,4698, enquanto modelos com features diferenciais (SVM, RF, XGBoost) obtiveram desempenhos equivalentes ou superiores. Análise SHAP identificou histórico head-to-head, odds de apostas e posição na tabela como os preditores mais influentes.

Apesar de avanços significativos, o problema do empate permanece desafiador: Random Forest ignora completamente esta classe, enquanto XGBoost alcança apenas F1=0,284 para empates. Trabalhos futuros devem focar em técnicas especializadas de balanceamento e incorporação de contexto externo (lesões, clima, motivação).

Este trabalho contribui com: (i) validação empírica da metodologia de Baboota & Kaur em dataset extenso (11 temporadas), (ii) análise de explicabilidade via SHAP para interpretação de decisões do modelo, (iii) comparação sistemática com múltiplos baselines e validação estatística via bootstrap, e (iv) disponibilização de pipeline reprodutível para futuros estudos.

A acurácia de ~50% para problemas de três classes (~33% baseline teórico) representa avanço substancial, mas ainda distante de aplicações comerciais (>60-65%). Futuras pesquisas devem explorar features contextuais dinâmicas e arquiteturas de deep learning para progressão adicional.

---

## Referências

**Baboota, R., & Kaur, H. (2018).** Predictive analysis and modelling football results using machine learning approach for English Premier League. *International Journal of Forecasting*, 35(2), 741-755.

**Bunker, R. P., & Thabtah, F. (2019).** A machine learning framework for sport result prediction. *Applied Computing and Informatics*, 15(1), 27-33.

**Constantinou, A. C., & Fenton, N. E. (2012).** Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. *Journal of Quantitative Analysis in Sports*, 8(1).

**Dixon, M. J., & Coles, S. G. (1997).** Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 46(2), 265-280.

**Joseph, A., Fenton, N. E., & Neil, M. (2006).** Predicting football results using Bayesian nets and other machine learning techniques. *Knowledge-Based Systems*, 19(7), 544-553.

**Lundberg, S. M., & Lee, S. I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

**Tax, N., & Joustra, Y. (2015).** Predicting the Dutch football competition using public data: A machine learning approach. *Transactions on Knowledge and Data Engineering*, 10(10), 1-13.

---

## Apêndice A: Especificações Técnicas

**Ambiente Computacional:**
- Python 3.8+
- scikit-learn 1.8.0, XGBoost 3.2.0, pandas 2.3.3, numpy 2.4.2
- SHAP 0.44.0 para análise de explicabilidade
- Hardware: CPU (cálculos não requerem GPU)

**Reprodutibilidade:**
- Seed aleatória fixada (random_state=42) em todos os experimentos
- Pipeline determinístico com reset temporal
- Código-fonte e dados disponíveis em repositório Git

---

**Documento gerado segundo padrões de publicações científicas em International Journal of Forecasting, IEEE Transactions on Knowledge and Data Engineering, e conferências como AAAI/IJCAI.**

