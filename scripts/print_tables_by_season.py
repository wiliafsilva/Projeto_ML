import os
import pickle
import argparse
import pandas as pd

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(description="Imprimir tabelas por temporada")
    parser.add_argument("--model-path", default="models/trained_models.pkl")
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


args = parse_args()
output_dir = args.output_dir
if not os.path.isabs(output_dir):
    output_dir = os.path.join(base, output_dir)

model_path = args.model_path
if not os.path.isabs(model_path):
    model_path = os.path.join(base, model_path)

csv_path = os.path.join(output_dir, 'baseline_comparison.csv')
pkl_path = model_path

print('\n== Verificando arquivos:')
print(' baseline CSV:', csv_path, '->', os.path.exists(csv_path))
print(' trained pickle:', pkl_path, '->', os.path.exists(pkl_path))

baseline_df = None
if os.path.exists(csv_path):
    try:
        baseline_df = pd.read_csv(csv_path, index_col=0)
    except Exception as e:
        print('Erro ao ler baseline_comparison.csv:', e)

trained = None
if os.path.exists(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            trained = pickle.load(f)
    except Exception as e:
        print('Erro ao carregar trained_models.pkl:', e)

print('\n== Chaves disponíveis no pickle:')
if isinstance(trained, dict):
    print(list(trained.keys()))
else:
    print('trained_models.pkl não é um dict (tipo: {})'.format(type(trained)))

# localizar resultados sazonais e resultados globais
seasonal = None
for key in ['seasonal_results', 'seasonal', 'per_season', 'season_results']:
    if isinstance(trained, dict) and key in trained:
        seasonal = trained[key]
        print('Encontrado seasonal key:', key)
        break

global_results = None
for key in ['results', 'global_results', 'all_results', 'overall_results', 'metrics', 'global_metrics']:
    if isinstance(trained, dict) and key in trained:
        global_results = trained[key]
        print('Encontrado global key:', key)
        break

seasons = ['2014-2015', '2015-2016']
for s in seasons:
    print('\n== TEMPORADA:', s)
    if seasonal and s in seasonal:
        # seasonal[s] expected to be dict(model -> metrics dict)
        try:
            df = pd.DataFrame.from_dict(seasonal[s], orient='index')
            print(df)
        except Exception as e:
            print('Erro ao construir DataFrame para', s, e)
    else:
        print('Nenhum resultado sazonal salvo para', s)

print('\n== TEMPORADA: All (agregado)')
if baseline_df is not None:
    print(f"\n- Valores em {csv_path}:")
    print(baseline_df)
else:
    print('- baseline_comparison.csv não encontrado.')

if global_results is not None:
    try:
        dfg = pd.DataFrame.from_dict(global_results, orient='index')
        print('\n- Valores em pickle (global):')
        print(dfg)
    except Exception as e:
        print('Erro ao mostrar global_results:', e)

# Verificar presença de Brier / ROC / AUC nas fontes
print('\n== Checando presença de Brier/ROC/AUC por fonte')
cols_check = ['brier', 'brier_score', 'brier_score_loss', 'roc_auc', 'auc', 'roc']

print('\n- Em baseline_comparison.csv (colunas):')
if baseline_df is not None:
    print(list(baseline_df.columns))
else:
    print(' baseline CSV ausente')

print('\n- Em seasonal_results (exemplo de chaves de métricas por modelo):')
if seasonal:
    for s in seasons:
        if s in seasonal:
            print('\n  Temporada', s)
            for model, metrics in seasonal[s].items():
                keys = list(metrics.keys()) if isinstance(metrics, dict) else []
                found = [c for c in cols_check if c in keys]
                print('   ', model, '-> chaves:', keys)
                if found:
                    print('     Encontrado:', found)
                else:
                    print('     Brier/ROC/AUC NÃO encontrados nas chaves desta entrada')
                break
        else:
            print('  Nenhum dado para', s)
else:
    print(' seasonal_results ausente')

print('\n- Em global_results (exemplo de chaves de métricas por modelo):')
if global_results is not None:
    for model, metrics in (list(global_results.items())[:5] if isinstance(global_results, dict) else []):
        keys = list(metrics.keys()) if isinstance(metrics, dict) else []
        found = [c for c in cols_check if c in keys]
        print(' ', model, '-> chaves:', keys)
        if found:
            print('   Encontrado:', found)
        else:
            print('   Brier/ROC/AUC NÃO encontrados nesta entrada')
        break
else:
    print(' global_results ausente')

print('\n== Conclusão rápida:')
print(' Se Brier/ROC/AUC aparecem como ausentes, é porque não foram computados/salvos durante o treinamento.\n Para obter estes valores, é preciso calcular:\n  - Brier: usar previsões probabilísticas (predict_proba) e sklearn.metrics.brier_score_loss\n  - ROC AUC: usar sklearn.metrics.roc_auc_score com abordagem adequada (one-vs-rest/macro para multi-classe)\n')

print('Fim do relatório.')
