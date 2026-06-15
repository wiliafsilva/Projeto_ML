# Validação Estatística: Decoder Hybrid vs Latent

Comparações por modelo (Hybrid - Latent). McNemar testa diferença em acurácia (pares corretos/incorretos). Bootstrap N=1000 para diferenças em Accuracy, F1-macro e RPS.

## Modelo: SVM

- Accuracy Hybrid: 0.4987  | Latent: 0.4724  | Diff: 0.0263
- McNemar b: 46, c: 26, p-value: 0.0245
- Accuracy diff 95% CI (bootstrap): [0.0039, 0.0474]  | p_boot: 0.0180
- F1-macro Hybrid: 0.4898  | Latent: 0.4696  | Diff: 0.0203
- F1 diff 95% CI: [-0.0011, 0.0414]  | p_boot: 0.0620
- RPS Hybrid: 0.414182  | Latent: 0.411832  | Diff: 0.002349
- RPS diff 95% CI: [-0.001989, 0.006580]  | p_boot: 0.2440

## Modelo: RandomForest

- Accuracy Hybrid: 0.4803  | Latent: 0.4882  | Diff: -0.0079
- McNemar b: 69, c: 75, p-value: 0.6771
- Accuracy diff 95% CI (bootstrap): [-0.0368, 0.0250]  | p_boot: 0.6760
- F1-macro Hybrid: 0.4477  | Latent: 0.4566  | Diff: -0.0089
- F1 diff 95% CI: [-0.0423, 0.0282]  | p_boot: 0.6540
- RPS Hybrid: 0.418924  | Latent: 0.413608  | Diff: 0.005316
- RPS diff 95% CI: [-0.003122, 0.013412]  | p_boot: 0.2180

## Modelo: XGBoost

- Accuracy Hybrid: 0.5000  | Latent: 0.4947  | Diff: 0.0053
- McNemar b: 69, c: 65, p-value: 0.7956
- Accuracy diff 95% CI (bootstrap): [-0.0224, 0.0368]  | p_boot: 0.7720
- F1-macro Hybrid: 0.4626  | Latent: 0.4663  | Diff: -0.0038
- F1 diff 95% CI: [-0.0376, 0.0347]  | p_boot: 0.8400
- RPS Hybrid: 0.414851  | Latent: 0.413338  | Diff: 0.001513
- RPS diff 95% CI: [-0.002246, 0.005318]  | p_boot: 0.4560

## Modelo: NaiveBayes

- Accuracy Hybrid: 0.4789  | Latent: 0.4789  | Diff: 0.0000
- McNemar b: 30, c: 30, p-value: 1.0000
- Accuracy diff 95% CI (bootstrap): [-0.0184, 0.0211]  | p_boot: 1.0000
- F1-macro Hybrid: 0.4725  | Latent: 0.4706  | Diff: 0.0019
- F1 diff 95% CI: [-0.0167, 0.0226]  | p_boot: 0.8820
- RPS Hybrid: 0.609177  | Latent: 0.460420  | Diff: 0.148757
- RPS diff 95% CI: [0.128018, 0.169626]  | p_boot: 0.0000
