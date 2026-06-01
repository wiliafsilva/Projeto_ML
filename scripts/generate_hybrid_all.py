"""
Generate All - DECODER HYBRID (Reconstructed-only 43D)
=====================================================

Script central que executa todos os scripts de geração de tabelas e figuras
para o pipeline Decoder Hybrid em sequência. Observação: o Hybrid agora treina
apenas com as 43 features reconstruídas (decoder output).

Execução: python scripts/generate_hybrid_all.py

Autor: Projeto_ML
Data: Maio 2026
"""

import subprocess
import sys
import os

print("="*80)
print("GERANDO TODAS AS TABELAS E FIGURAS - DECODER HYBRID (RECONSTRUCTED 43D)")
print("="*80)
print()

scripts = [
    ("1. Tabela 3 - Comparação Completa", "update_hybrid_tabela3.py"),
    ("2. Tabela 4 - Confusion Matrices", "update_hybrid_tabela4.py"),
    ("3. Tabelas 5 e 6 - Performance Temporal", "update_hybrid_tabelas_5_6.py"),
    ("4. Baseline Comparison", "hybrid_baseline_comparison.py"),
    ("5. Correlation Heatmap", "hybrid_correlation_heatmap.py"),
    ("6. Radar Charts", "hybrid_radar_chart.py"),
    ("7. Feature Importance", "hybrid_feature_importance.py"),
    ("8. Confidence Intervals (Bootstrap)", "hybrid_confidence_intervals.py"),
    ("9. Visualizações Adicionais", "hybrid_additional_visualizations.py"),
]

print("Scripts a executar:")
for desc, script in scripts:
    print(f"   {desc}")
    print(f"      → scripts/{script}")
print()

# Executar cada script
for desc, script in scripts:
    print("\n" + "="*80)
    print(desc)
    print("="*80)
    
    script_path = os.path.join("scripts", script)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.getcwd(),
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n✅ {desc} - SUCESSO")
        else:
            print(f"\n❌ {desc} - ERRO (código: {result.returncode})")
    except Exception as e:
        print(f"\n❌ {desc} - EXCEÇÃO: {e}")

print("\n" + "="*80)
print("GERAÇÃO COMPLETA - DECODER HYBRID")
print("="*80)
print()

# Listar arquivos gerados
output_dir = "models/autoencoder_decoder_hybrid"
if os.path.exists(output_dir):
    print(f"📁 Arquivos gerados em: {output_dir}/")
    print()
    
    # CSVs
    csvs = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    if csvs:
        print("📊 Tabelas (CSV):")
        for csv in sorted(csvs):
            print(f"   ✓ {csv}")
    
    # Figuras
    figures_dir = os.path.join(output_dir, 'figures')
    if os.path.exists(figures_dir):
        pngs = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        if pngs:
            print("\n📈 Figuras (PNG):")
            for png in sorted(pngs):
                print(f"   ✓ {png}")
    
    print()

print("✅ PIPELINE COMPLETO FINALIZADO!")
print()
