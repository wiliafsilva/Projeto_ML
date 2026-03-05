#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script para gerar tabelas consolidadas para o artigo científico"""

import sys
import os
from pathlib import Path

# Forçar UTF-8 no Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import joblib
import pandas as pd
import numpy as np
from src.preprocessing import load_all_data, load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print("="*80)
print("GERAÇÃO DE TABELAS CONSOLIDADAS PARA ARTIGO CIENTÍFICO")
print("="*80)

# Carregar dados
df_all = load_all_data()
df_train = load_multiple_seasons("data/data_2005_2014")
df_test = load_multiple_seasons("data/data_2014_2016")

# Carregar modelos
try:
    results_metadata = joblib.load("models/trained_models.pkl")
    models = results_metadata.get('models', results_metadata)
except:
    print("\n⚠️  ERRO: Modelos não encontrados. Execute 'python main.py' primeiro.")
    sys.exit(1)

# Preparar features
features_train = calculate_team_stats(df_train)
features_test = calculate_team_stats(df_test)

X_test = features_test.drop(['Result', 'Season'], axis=1)
y_test = features_test['Result']

print("\n" + "="*80)
print("TABELA 1: RESUMO DO DATASET")
print("="*80)

# Estatísticas gerais
total_partidas = len(df_all)
temporadas = df_all['Season'].nunique()
times_unicos = pd.concat([df_all['HomeTeam'], df_all['HomeTeam']]).nunique()
media_gols = (df_all['FTHG'] + df_all['FTAG']).mean()

# Distribuição de resultados
result_counts = df_all['FTR'].value_counts()
vitorias_casa = result_counts.get('H', 0)
empates = result_counts.get('D', 0)
vitorias_fora = result_counts.get('A', 0)

tabela1 = pd.DataFrame({
    'Métrica': [
        'Total de Partidas',
        'Período',
        'Temporadas',
        'Times Únicos',
        'Média Gols/Jogo',
        'Vitórias Casa',
        'Empates',
        'Vitórias Visitante',
        '',
        'Partidas Treino',
        'Partidas Teste',
        'Split Treino/Teste'
    ],
    'Valor': [
        f'{total_partidas:,}',
        f'{int(df_all["Season"].min())}-{int(df_all["Season"].max())}',
        f'{temporadas}',
        f'{times_unicos}',
        f'{media_gols:.2f}',
        f'{vitorias_casa:,} ({vitorias_casa/total_partidas*100:.1f}%)',
        f'{empates:,} ({empates/total_partidas*100:.1f}%)',
        f'{vitorias_fora:,} ({vitorias_fora/total_partidas*100:.1f}%)',
        '',
        f'{len(df_train):,} (2005-2014)',
        f'{len(df_test):,} (2014-2016)',
        f'{len(df_train)/total_partidas*100:.1f}% / {len(df_test)/total_partidas*100:.1f}%'
    ]
})

print(tabela1.to_string(index=False))
tabela1.to_csv('models/tabela1_resumo_dataset.csv', index=False)
print("\n✓ Salva em: models/tabela1_resumo_dataset.csv")

print("\n" + "="*80)
print("TABELA 2: ESTATÍSTICAS DESCRITIVAS DAS FEATURES")
print("="*80)

features_all = calculate_team_stats(df_all)
feature_cols = ['gd_diff', 'streak_diff', 'weighted_diff']

stats_data = []
for col in feature_cols:
    data = features_all[col].dropna()
    stats_data.append({
        'Feature': col,
        'Mean': f'{data.mean():.4f}',
        'Std': f'{data.std():.4f}',
        'Min': f'{data.min():.4f}',
        '25%': f'{data.quantile(0.25):.4f}',
        '50%': f'{data.quantile(0.50):.4f}',
        '75%': f'{data.quantile(0.75):.4f}',
        'Max': f'{data.max():.4f}'
    })

tabela2 = pd.DataFrame(stats_data)
print(tabela2.to_string(index=False))
tabela2.to_csv('models/tabela2_estatisticas_features.csv', index=False)
print("\n✓ Salva em: models/tabela2_estatisticas_features.csv")

print("\n" + "="*80)
print("TABELA 3: COMPARAÇÃO COMPLETA DE MODELOS")
print("="*80)

# Calcular baseline (sempre prever classe majoritária)
from collections import Counter
y_train = features_train['Result']
baseline_pred = Counter(y_train).most_common(1)[0][0]  # Classe mais frequente no treino
baseline_preds = np.full(len(y_test), baseline_pred)
baseline_acc = accuracy_score(y_test, baseline_preds)

# Coletar métricas de todos os modelos
comparison_data = []

# Adicionar baseline primeiro
comparison_data.append({
    'Modelo': 'Baseline (Majoritário)',
    'Accuracy': f'{baseline_acc:.4f}',
    'Precision': '-',
    'Recall': '-',
    'F1': '-',
    'RPS': '-',
    'Brier': '-',
    'ROC AUC': '-'
})

# Adicionar modelos treinados
for name, info in models.items():
    model = info['model']
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    # Métricas básicas
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    
    # Brier score
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    brier = np.mean(np.sum((y_bin - probs) ** 2, axis=1))
    
    # ROC AUC
    from sklearn.metrics import roc_auc_score
    try:
        roc_auc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    except:
        roc_auc = None
    
    comparison_data.append({
        'Modelo': name,
        'Accuracy': f'{acc:.4f}',
        'Precision': f'{prec:.4f}',
        'Recall': f'{rec:.4f}',
        'F1': f'{f1:.4f}',
        'RPS': f'{info.get("rps", 0):.4f}',
        'Brier': f'{brier:.4f}',
        'ROC AUC': f'{roc_auc:.4f}' if roc_auc else '-'
    })

tabela3 = pd.DataFrame(comparison_data)
print(tabela3.to_string(index=False))
tabela3.to_csv('models/tabela3_comparacao_modelos.csv', index=False)
print("\n✓ Salva em: models/tabela3_comparacao_modelos.csv")

print("\n" + "="*80)
print("TABELA 4: MATRIZ DE CONFUSÃO DETALHADA (POR MODELO)")
print("="*80)

from sklearn.metrics import confusion_matrix

classes = ['Vitória Casa', 'Empate', 'Vitória Visitante']

for name, info in models.items():
    model = info['model']
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    
    print(f"\n{name}:")
    print("-" * 60)
    
    # Criar DataFrame com a matriz
    cm_df = pd.DataFrame(cm, 
                         index=[f'Real: {c}' for c in classes],
                         columns=[f'Pred: {c}' for c in classes])
    
    # Adicionar totais
    cm_df['Total'] = cm_df.sum(axis=1)
    
    # Adicionar percentuais
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    cm_pct_df = pd.DataFrame(cm_pct,
                              index=[f'Real: {c}' for c in classes],
                              columns=[f'Pred: {c}' for c in classes])
    
    print("\nContagens absolutas:")
    print(cm_df.to_string())
    
    print("\nPercentuais por linha (%):")
    print(cm_pct_df.round(1).to_string())
    
    # Salvar
    cm_df.to_csv(f'models/tabela4_cm_{name.lower()}.csv')
    print(f"\n✓ Salva em: models/tabela4_cm_{name.lower()}.csv")

print("\n" + "="*80)
print("TABELA 5: PERFORMANCE POR TEMPORADA")
print("="*80)

# Separar features de teste por temporada
temporada_data = []

for season in sorted(features_test['Season'].unique()):
    season_features = features_test[features_test['Season'] == season]
    X_season = season_features.drop(['Result', 'Season'], axis=1)
    y_season = season_features['Result']
    
    # Baseline para essa temporada
    baseline_preds_season = np.full(len(y_season), baseline_pred)
    baseline_acc_season = accuracy_score(y_season, baseline_preds_season)
    
    row = {
        'Temporada': f'{int(season)}-{int(season)+1}',
        'Jogos': len(y_season),
        'Baseline': f'{baseline_acc_season*100:.2f}%'
    }
    
    # Acurácia de cada modelo nessa temporada
    for name, info in models.items():
        model = info['model']
        preds_season = model.predict(X_season)
        acc_season = accuracy_score(y_season, preds_season)
        row[name] = f'{acc_season*100:.2f}%'
    
    temporada_data.append(row)

tabela5 = pd.DataFrame(temporada_data)
print(tabela5.to_string(index=False))
tabela5.to_csv('models/tabela5_performance_temporada.csv', index=False)
print("\n✓ Salva em: models/tabela5_performance_temporada.csv")

print("\n" + "="*80)
print("TABELA 6: CLASSIFICAÇÃO POR CLASSE (DETALHADA)")
print("="*80)

from sklearn.metrics import classification_report

for name, info in models.items():
    model = info['model']
    preds = model.predict(X_test)
    
    print(f"\n{name}:")
    print("-" * 60)
    
    # Gerar classification report como dict
    report = classification_report(y_test, preds, 
                                   target_names=classes,
                                   output_dict=True,
                                   zero_division=0)
    
    # Converter para DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    # Formatar suporte como inteiro
    if 'support' in report_df.columns:
        report_df['support'] = report_df['support'].astype(int)
    
    print(report_df.round(4).to_string())
    
    # Salvar
    report_df.to_csv(f'models/tabela6_classificacao_{name.lower()}.csv')
    print(f"\n✓ Salva em: models/tabela6_classificacao_{name.lower()}.csv")

print("\n" + "="*80)
print("✅ TODAS AS TABELAS FORAM GERADAS COM SUCESSO!")
print("="*80)
print("\nArquivos criados na pasta 'models/':")
print("  - tabela1_resumo_dataset.csv")
print("  - tabela2_estatisticas_features.csv")
print("  - tabela3_comparacao_modelos.csv")
print("  - tabela4_cm_[modelo].csv (3 arquivos)")
print("  - tabela5_performance_temporada.csv")
print("  - tabela6_classificacao_[modelo].csv (3 arquivos)")
print("\n" + "="*80)
print("📊 PREVIEW DAS PRINCIPAIS TABELAS")
print("="*80)

print("\n📋 TABELA 1 - Resumo do Dataset:")
print(tabela1.to_string(index=False))

print("\n📊 TABELA 2 - Estatísticas das Features:")
print(tabela2.to_string(index=False))

print("\n🏆 TABELA 3 - Comparação de Modelos:")
print(tabela3.to_string(index=False))

print("\n📈 TABELA 5 - Performance por Temporada:")
print(tabela5.to_string(index=False))

print("\n" + "="*80)
print("💡 Para visualizar todas as tabelas de forma interativa:")
print("   streamlit run app.py")
print("   → Navegue até 'Análise Científica Consolidada'")
print("="*80)
