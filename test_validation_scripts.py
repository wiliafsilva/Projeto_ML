#!/usr/bin/env python
"""
Teste rápido dos scripts de validação do pipeline.

Executa cada script de avaliação para confirmar que funcionam
com as 59 features (43 originais + 16 latentes do autoencoder).
"""

import subprocess
import sys
from pathlib import Path

scripts_to_test = [
    ("baseline_comparison.py", "Comparação com Baseline"),
    ("confidence_intervals_fast.py", "Intervalos de Confiança"),
    ("feature_importance_simple.py", "Importância de Features"),
    ("shap_analysis.py", "Análise SHAP"),
]

print("=" * 80)
print("TESTE DOS SCRIPTS DE AVALIAÇÃO")
print("=" * 80)

results = []

for script_name, description in scripts_to_test:
    script_path = Path("scripts") / script_name
    
    print(f"\n[{len(results)+1}] Testando: {description}")
    print(f"    Script: {script_name}")
    print("-" * 80)
    
    if not script_path.exists():
        print(f"    ✗ Script não encontrado: {script_path}")
        results.append((script_name, "NOT_FOUND"))
        continue
    
    try:
        # Executar script com timeout de 30 segundos
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Verificar se houve erro
        if result.returncode == 0:
            print(f"    ✓ Sucesso!")
            results.append((script_name, "SUCCESS"))
            
            # Mostrar últimas linhas de output
            lines = result.stdout.strip().split('\n')
            if len(lines) > 3:
                print(f"    Saída final:")
                for line in lines[-3:]:
                    if line.strip():
                        print(f"      {line}")
        else:
            print(f"    ✗ Erro (código {result.returncode})")
            results.append((script_name, "ERROR"))
            
            # Mostrar erro
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                print(f"    Erro:")
                for line in error_lines[-5:]:
                    if line.strip():
                        print(f"      {line}")
    
    except subprocess.TimeoutExpired:
        print(f"    ⚠ Timeout (> 30 segundos)")
        results.append((script_name, "TIMEOUT"))
    except Exception as e:
        print(f"    ✗ Exceção: {e}")
        results.append((script_name, "EXCEPTION"))

# ── RESUMO ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RESUMO DOS TESTES")
print("=" * 80)

for script_name, status in results:
    symbol = {"SUCCESS": "✓", "ERROR": "✗", "TIMEOUT": "⚠", "EXCEPTION": "✗", "NOT_FOUND": "✗"}.get(status, "?")
    print(f"{symbol} {script_name:35s} → {status}")

# Contadores
successes = sum(1 for _, s in results if s == "SUCCESS")
total = len(results)

print(f"\nTotal: {successes}/{total} scripts executados com sucesso")

if successes == total:
    print("\n✓ TODOS OS SCRIPTS FUNCIONANDO CORRETAMENTE!")
else:
    print(f"\n⚠ {total - successes} script(s) com problemas")

print("=" * 80)
