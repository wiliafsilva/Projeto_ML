#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script para gerar TODAS as tabelas e figuras de uma vez"""

import sys
import subprocess
from pathlib import Path

# Forçar UTF-8 no Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

print("="*80)
print("🚀 GERAÇÃO COMPLETA DE TABELAS E FIGURAS")
print("="*80)
print("\nEste script irá gerar:")
print("  • 10 tabelas consolidadas (CSV)")
print("  • 6 visualizações de alta qualidade (PNG 300 DPI)")
print("\nTempo estimado: 30-60 segundos")
print("="*80)

# Gerar tabelas
print("\n[1/2] Gerando tabelas consolidadas...")
print("-" * 80)
try:
    result = subprocess.run(
        [sys.executable, "scripts/generate_tables.py"],
        check=True,
        capture_output=False,
        text=True
    )
    print("\n✅ Tabelas geradas com sucesso!")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Erro ao gerar tabelas: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Erro inesperado: {e}")
    sys.exit(1)

# Gerar figuras
print("\n" + "="*80)
print("[2/2] Gerando visualizações (300 DPI)...")
print("-" * 80)
try:
    result = subprocess.run(
        [sys.executable, "scripts/generate_figures.py"],
        check=True,
        capture_output=False,
        text=True
    )
    print("\n✅ Figuras geradas com sucesso!")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Erro ao gerar figuras: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Erro inesperado: {e}")
    sys.exit(1)

# Resumo final
print("\n" + "="*80)
print("🎉 GERAÇÃO COMPLETA FINALIZADA!")
print("="*80)
print("\n📁 Arquivos gerados:")
print("   Tabelas: models/*.csv (10 arquivos)")
print("   Figuras: models/figures/*.png (6 arquivos)")
print("\n💡 Próximos passos:")
print("   1. Visualizar no Streamlit:")
print("      streamlit run app.py")
print("      → Navegue até 'Análise Científica Consolidada'")
print("\n   2. Ou abrir os arquivos diretamente:")
print("      - CSVs: Excel, Google Sheets")
print("      - PNGs: Visualizador de imagens")
print("="*80)
