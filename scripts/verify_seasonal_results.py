"""
Verificar Resultados Salvos Por Temporada
==========================================

Verifica se o arquivo trained_models.pkl contém resultados
separados por temporada (2023-2024, 2024-2025, All).

Autor: Projeto_ML
Data: Março 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from pathlib import Path

print("="*80)
print("VERIFICAÇÃO DE RESULTADOS POR TEMPORADA")
print("="*80)
print()

# Verificar se o arquivo existe
pkl_file = Path('models/trained_models.pkl')

if not pkl_file.exists():
    print("❌ Arquivo models/trained_models.pkl não encontrado!")
    print("\n💡 Execute 'python main.py' para treinar os modelos primeiro.")
    sys.exit(1)

print("✅ Arquivo encontrado: models/trained_models.pkl")
print()

# Carregar resultados
print("📂 Carregando resultados...")
results_metadata = joblib.load('models/trained_models.pkl')

# Verificar estrutura
print("\n" + "="*80)
print("ESTRUTURA DO ARQUIVO")
print("="*80)
print(f"\nChaves disponíveis: {list(results_metadata.keys())}")
print()

# Verificar se seasonal_results existe
if 'seasonal_results' in results_metadata:
    print("✅ seasonal_results encontrado!")
    
    seasonal_results = results_metadata['seasonal_results']
    
    print(f"\n📊 Temporadas disponíveis: {list(seasonal_results.keys())}")
    
    # Mostrar resultados por temporada
    print("\n" + "="*80)
    print("RESULTADOS POR TEMPORADA")
    print("="*80)
    
    for season_name in ['2023-2024', '2024-2025', 'All']:
        if season_name in seasonal_results:
            print(f"\n{'='*60}")
            print(f"TEMPORADA: {season_name}")
            print(f"{'='*60}")
            
            season_data = seasonal_results[season_name]
            
            # Criar DataFrame para visualização
            rows = []
            for model_name, metrics in season_data.items():
                rows.append({
                    'Modelo': model_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1']:.4f}",
                    'RPS': f"{metrics['rps']:.4f}",
                    'N_Samples': metrics['n_samples']
                })
            
            df = pd.DataFrame(rows)
            print(f"\n{df.to_string(index=False)}")
    
    # Comparação entre temporadas
    print("\n" + "="*80)
    print("COMPARAÇÃO: ACCURACY POR TEMPORADA")
    print("="*80)
    print()
    
    # Criar tabela comparativa
    comparison_data = []
    model_names = list(seasonal_results['All'].keys())
    
    for model_name in model_names:
        row = {'Modelo': model_name}
        for season_name in ['2023-2024', '2024-2025', 'All']:
            if season_name in seasonal_results and model_name in seasonal_results[season_name]:
                acc = seasonal_results[season_name][model_name]['accuracy']
                row[season_name] = f"{acc*100:.2f}%"
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Comparação RPS
    print("\n" + "="*80)
    print("COMPARAÇÃO: RPS POR TEMPORADA")
    print("="*80)
    print()
    
    comparison_rps = []
    for model_name in model_names:
        row = {'Modelo': model_name}
        for season_name in ['2023-2024', '2024-2025', 'All']:
            if season_name in seasonal_results and model_name in seasonal_results[season_name]:
                rps = seasonal_results[season_name][model_name]['rps']
                row[season_name] = f"{rps:.4f}"
        comparison_rps.append(row)
    
    df_rps = pd.DataFrame(comparison_rps)
    print(df_rps.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ SUCESSO! Resultados por temporada estão salvos corretamente.")
    print("="*80)
    
else:
    print("❌ seasonal_results NÃO encontrado!")
    print("\n💡 O arquivo precisa ser atualizado. Execute 'python main.py' novamente.")
    
    # Mostrar o que está disponível
    if 'models' in results_metadata:
        print("\n📊 Modelos disponíveis:")
        for model_name in results_metadata['models'].keys():
            model_info = results_metadata['models'][model_name]
            print(f"   - {model_name}")
            if 'accuracy' in model_info:
                print(f"     Accuracy: {model_info['accuracy']:.4f}")

print()
print("="*80)
print("VERIFICAÇÃO CONCLUÍDA!")
print("="*80)
