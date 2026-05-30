"""
Update Tabela 3 - Comparação Completa de Modelos (DECODER HYBRID)
==================================================================

Regenera Tabela 3 usando resultados do pipeline Decoder Hybrid.
Compara: Baseline vs Latent Space vs Hybrid Decoder

Autor: Projeto_ML
Data: Maio 2026
"""

import argparse
import os
import pandas as pd
import joblib
import numpy as np

print("="*80)
print("TABELA 3: COMPARAÇÃO COMPLETA - DECODER HYBRID vs BASELINE vs LATENT")
print("="*80)
print()


def parse_args():
    parser = argparse.ArgumentParser(description="Atualizar tabela 3 hybrid")
    parser.add_argument("--hybrid-path", default="models/autoencoder_decoder_hybrid/trained_models_hybrid.pkl")
    parser.add_argument("--baseline-path", default="models/baseline_comparison.csv")
    parser.add_argument("--latent-path", default="models/autoencoder_latent/trained_models_latent.pkl")
    parser.add_argument("--output-dir", default="models/autoencoder_decoder_hybrid")
    return parser.parse_args()


args = parse_args()
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

print("📂 Carregando dados...")

# Carregar baseline_comparison.csv (para dados base)
if os.path.exists(args.baseline_path):
    baseline_df = pd.read_csv(args.baseline_path)
    print(f"   ✓ Baseline: {len(baseline_df)} modelos")
else:
    print(f"   ⚠️  Baseline não encontrado em {args.baseline_path}")
    baseline_df = None

# Carregar Hybrid
try:
    hybrid_meta = joblib.load(args.hybrid_path)
    print(f"   ✓ Hybrid: {len(hybrid_meta['models'])} modelos")
except Exception as e:
    print(f"   ❌ Erro ao carregar Hybrid: {e}")
    hybrid_meta = None

# Carregar Latent (para comparação)
try:
    latent_meta = joblib.load(args.latent_path)
    print(f"   ✓ Latent: {len(latent_meta['models'])} modelos")
except Exception as e:
    print(f"   ⚠️  Latent não encontrado: {e}")
    latent_meta = None

print()

# Construir Tabela 3
print("🔧 Construindo Tabela 3...")
table3_data = []

# ============================================================================
# BASELINE
# ============================================================================
if baseline_df is not None:
    baseline_candidates = baseline_df[baseline_df['Tipo'].str.lower() == 'baseline']
    if not baseline_candidates.empty:
        baseline_row = baseline_candidates.iloc[0]
        table3_data.append({
            'Modelo': 'Baseline (Majoritário)',
            'Pipeline': 'Baseline',
            'Accuracy': f"{baseline_row['Accuracy']:.4f}",
            'Precision': '-',
            'Recall': '-',
            'F1': '-',
            'RPS': '-',
        })

# ============================================================================
# MODELOS ML
# ============================================================================
ml_models = ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']

# Adicionar dados do Hybrid
if hybrid_meta is not None:
    for model_name in ml_models:
        if model_name in hybrid_meta['models']:
            info = hybrid_meta['models'][model_name]
            table3_data.append({
                'Modelo': model_name,
                'Pipeline': 'Hybrid (Features 50D)',
                'Accuracy': f"{info['accuracy']:.4f}",
                'Precision': '-',
                'Recall': '-',
                'F1': f"{info['f1']:.4f}",
                'RPS': f"{info['rps']:.4f}",
            })

# Adicionar dados do Latent para comparação
if latent_meta is not None:
    for model_name in ml_models:
        if model_name in latent_meta['models']:
            info = latent_meta['models'][model_name]
            table3_data.append({
                'Modelo': model_name,
                'Pipeline': 'Latent (Features 8D)',
                'Accuracy': f"{info['accuracy']:.4f}",
                'Precision': '-',
                'Recall': '-',
                'F1': f"{info['f1']:.4f}",
                'RPS': f"{info['rps']:.4f}",
            })

# Converter para DataFrame
table3_df = pd.DataFrame(table3_data)

# Salvar
output_path = os.path.join(output_dir, 'tabela3_hybrid_comparacao.csv')
table3_df.to_csv(output_path, index=False)

print(f"\n📊 TABELA 3 - COMPARAÇÃO COMPLETA")
print("="*80)
print(table3_df.to_string(index=False))
print(f"\n✓ Salvo em: {output_path}")
print()

# Estatísticas
print("="*80)
print("RESUMO COMPARATIVO")
print("="*80)
print()

if hybrid_meta is not None:
    print("DECODER HYBRID - Métricas Gerais:")
    print(f"   Anomalias detectadas: {hybrid_meta.get('anomalies_detected', 'N/A')} "
          f"({hybrid_meta.get('anomaly_ratio', 'N/A'):.2f}%)")
    print(f"   Dados de treino limpos: {hybrid_meta.get('train_size_clean', 'N/A')} amostras")
    print(f"   Features híbridas: {hybrid_meta.get('hybrid_features_count', 'N/A')}D")
    print(f"   Reconstruction threshold: {hybrid_meta.get('reconstruction_threshold', 'N/A'):.6f}")
    print()

if hybrid_meta is not None and latent_meta is not None:
    print("COMPARAÇÃO: Hybrid vs Latent")
    print("-" * 80)
    print(f"{'Modelo':<20} {'Hybrid RPS':>15} {'Latent RPS':>15} {'Diferença':>15}")
    print("-" * 80)
    
    for model_name in ml_models:
        if model_name in hybrid_meta['models'] and model_name in latent_meta['models']:
            hybrid_rps = hybrid_meta['models'][model_name]['rps']
            latent_rps = latent_meta['models'][model_name]['rps']
            diff = hybrid_rps - latent_rps
            symbol = "✓" if hybrid_rps < latent_rps else "✗"
            print(f"{model_name:<20} {hybrid_rps:>15.4f} {latent_rps:>15.4f} "
                  f"{symbol} {diff:>13.4f}")

print("\n" + "="*80)
