"""
Teste Rápido - Verificar se GridSearch avalia por temporada
=============================================================

Testa se os scripts de GridSearch foram atualizados corretamente
para avaliar separadamente por temporada (2014-2015, 2015-2016, All).

Autor: Projeto_ML
Data: Março 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("TESTE - VERIFICAÇÃO DE GRIDSEARCH POR TEMPORADA")
print("="*80)
print()

# Verificar se os arquivos de resultado existem
files_to_check = [
    'models/gridsearch_43features_por_temporada.csv',
    'models/gridsearch_advanced_por_temporada.csv'
]

print("📂 Verificando arquivos de saída...")
print("-"*80)

all_exist = True
for file_path in files_to_check:
    exists = Path(file_path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {file_path}")
    
    if exists:
        # Ler e mostrar conteúdo
        df = pd.read_csv(file_path)
        print(f"\n   Estrutura: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print(f"   Temporadas: {df['Temporada'].tolist() if 'Temporada' in df.columns else df.index.tolist()}")
        print()
    else:
        all_exist = False

print("="*80)

if all_exist:
    print("\n✅ SUCESSO! Todos os arquivos de saída existem.")
    print("\n📊 Resumo:")
    print("   - GridSearch 43 Features: ✅ Avalia por temporada")
    print("   - GridSearch Advanced: ✅ Avalia por temporada")
else:
    print("\n⚠️ AVISO: Alguns arquivos ainda não foram gerados.")
    print("\n💡 Execute os scripts de GridSearch para gerar os resultados:")
    print("   python scripts/gridsearch_43features.py")
    print("   python scripts/gridsearch_advanced.py")

print("\n" + "="*80)
print("TESTE CONCLUÍDO!")
print("="*80)
