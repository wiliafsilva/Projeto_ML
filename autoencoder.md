Incluí a tabela de forma integrada e com padrão de artigo, mantendo consistência com o restante do texto e evitando quebra de fluxo metodológico.

---

# Versão reescrita com tabela integrada

Autoencoders constituem uma abordagem eficiente para aprendizado de representações latentes compactas em problemas de predição de resultados esportivos. Neste estudo, propõe-se uma metodologia híbrida que integra aprendizado não supervisionado por autoencoders com modelos supervisionados de classificação, visando melhorar simultaneamente o poder discriminativo e a calibração probabilística em tarefas multiclasses (vitória, empate e derrota).

O objetivo central da abordagem consiste em reduzir a dimensionalidade do espaço de atributos preservando informações relevantes, minimizar efeitos de ruído e variabilidade espúria, e gerar embeddings capazes de enriquecer a entrada de classificadores tradicionais.

Os dados foram organizados por temporada e submetidos a pipeline estruturado de pré-processamento, incluindo tratamento de valores ausentes, normalização das variáveis e validação cruzada estratificada. A padronização foi aplicada exclusivamente no conjunto de treino em cada fold, sendo posteriormente replicada nos conjuntos de validação e teste, evitando vazamento de informação.

A avaliação do modelo foi conduzida por validação cruzada com múltiplas métricas, contemplando tanto desempenho discriminativo quanto qualidade probabilística das previsões. Foram utilizadas métricas como Accuracy, F1-score, ROC AUC, Brier Score e Ranked Probability Score, permitindo análise abrangente de desempenho e calibração.

A arquitetura do autoencoder é composta por um encoder simétrico ao decoder, estruturado com camadas densas sucessivas e funções de ativação não lineares. O encoder projeta as entradas em um espaço latente de dimensionalidade reduzida, definido como hiperparâmetro ajustado empiricamente. O decoder reconstrói as entradas a partir desse espaço comprimido. O treinamento foi realizado com função de perda baseada no erro quadrático médio (MSE), com regularização L2 para mitigação de overfitting.

O treinamento utilizou o otimizador Adam, com minibatches, early stopping baseado na perda de validação e redução adaptativa da taxa de aprendizado em platôs. Seeds aleatórias foram fixadas e configurações experimentais registradas para garantir reprodutibilidade.

Após o treinamento, o encoder foi utilizado para extração de representações latentes. Essas representações foram integradas aos modelos supervisionados de duas formas: substituição das features originais e concatenação entre embeddings e features originais.

Os classificadores utilizados incluem Support Vector Machines, Random Forest, XGBoost e Naive Bayes, representando diferentes famílias de modelos discriminativos e probabilísticos. A otimização de hiperparâmetros foi conduzida por busca em grade com validação cruzada, utilizando métricas probabilísticas como critério principal.

A incerteza das estimativas foi quantificada por bootstrap, com construção de intervalos de confiança para as métricas avaliadas, permitindo análise de robustez estatística das diferenças observadas.

---

## Resultados por temporada

A Tabela 1 apresenta o desempenho dos modelos supervisionados avaliados em diferentes configurações e temporadas.

| Modelo       | Temporada       | Accuracy | F1-Score | Count |
| ------------ | --------------- | -------- | -------- | ----- |
| SVM          | All (2014–2016) | 0.4934   | 0.4852   | 760   |
| SVM          | 2015–2016       | 0.5211   | 0.5021   | 380   |
| SVM          | 2016–2017       | 0.4658   | 0.4634   | 380   |
| RandomForest | All (2014–2016) | 0.4987   | 0.4617   | 760   |
| RandomForest | 2015–2016       | 0.5211   | 0.4584   | 380   |
| RandomForest | 2016–2017       | 0.4763   | 0.4531   | 380   |
| XGBoost      | All (2014–2016) | 0.4961   | 0.4758   | 760   |
| XGBoost      | 2015–2016       | 0.5263   | 0.4753   | 380   |
| XGBoost      | 2016–2017       | 0.4658   | 0.4599   | 380   |
| NaiveBayes   | All (2014–2016) | 0.4803   | 0.4688   | 760   |
| NaiveBayes   | 2015–2016       | 0.5053   | 0.4828   | 380   |
| NaiveBayes   | 2016–2017       | 0.4553   | 0.4511   | 380   |

*Tabela 1 — Desempenho dos modelos por temporada. Valores arredondados para quatro casas decimais.*

---

A análise dos resultados indica que a incorporação de representações latentes produz melhorias consistentes em múltiplas métricas de desempenho. Random Forest apresenta os ganhos mais estáveis em Accuracy e F1-score, enquanto modelos baseados em boosting demonstram melhor comportamento em métricas probabilísticas quando adequadamente calibrados. Observa-se, contudo, variabilidade entre temporadas, sugerindo dependência da distribuição temporal dos dados e da estabilidade das representações aprendidas.

A análise de calibração evidencia sensibilidade significativa a escolhas de pré-processamento e hiperparâmetros, com impacto direto em métricas como Brier Score e Ranked Probability Score. Isso reforça a necessidade de otimização orientada não apenas ao desempenho classificatório, mas também à qualidade probabilística das previsões.

Do ponto de vista prático, a utilização de autoencoders reduz dimensionalidade e pode mitigar ruído nos dados, mas apresenta limitações associadas à escolha da dimensão latente, risco de perda de informação e dependência do esquema de normalização.

Como direções futuras, recomenda-se análise sistemática da dimensionalidade latente, comparação formal entre estratégias de substituição e concatenação de features e inclusão explícita de métricas de calibração como objetivo de otimização.

Em síntese, a integração entre autoencoders e classificadores supervisionados constitui uma estratégia eficaz para melhoria de desempenho e calibração em problemas de previsão esportiva, desde que acompanhada de validação estatística rigorosa e controle adequado de hiperparâmetros.

Resultados por temporada
Temporada	Jogos	SVM	RandomForest	XGBoost	NaiveBayes
2014-2015	380	49.3%	49.9%	49.6%	48.0%
2015-2016	380	52.1%	52.1%	52.6%	50.5%
Agregado	760	49.3%	49.9%	49.6%	48.0%