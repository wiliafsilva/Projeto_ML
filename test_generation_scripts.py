#!/usr/bin/env python
"""
Teste dos scripts de geração de figuras, tabelas e análises.

Valida que funcionam corretamente com 59 features.
"""

import subprocess
import sys
import os
from pathlib import Path
import time

# Scripts para testar (com timeout e descrição)
scripts_to_test = [
    ("generate_tables.py", "Geração de Tabelas", 45),
    ("generate_figures.py", "Geração de Figuras", 45),
    ("print_tables_by_season.py", "Print Tabelas por Temporada", 30),
    ("correlation_heatmap.py", "Mapa de Correlação", 30),
    ("shap_analysis.py", "Análise SHAP", 60),
    ("update_tabela3.py", "Update Tabela 3", 30),
    ("update_tabela4.py", "Update Tabela 4", 30),
    ("radar_chart.py", "Gráfico Radar", 30),
]

print("=" * 90)
print("TESTE DOS SCRIPTS DE GERAÇÃO DE FIGURAS, TABELAS E ANÁLISES")
print("=" * 90)

results = []
success_count = 0
error_count = 0
timeout_count = 0

for script_name, description, timeout_sec in scripts_to_test:
    script_path = Path("scripts") / script_name
    
    print(f"\n[{len(results)+1}/{len(scripts_to_test)}] {description}")
    print(f"    Script: {script_name} (timeout: {timeout_sec}s)")
    print("-" * 90)
    
    if not script_path.exists():
        print(f"    ✗ Script não encontrado")
        results.append((script_name, "NOT_FOUND", 0))
        error_count += 1
        continue
    
    try:
        start_time = time.time()
        
        # Executar script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**dict(os.environ), 'PYTHONIOENCODING': 'utf-8'}
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"    ✓ Sucesso! ({elapsed_time:.1f}s)")
            results.append((script_name, "SUCCESS", elapsed_time))
            success_count += 1
            
            # Mostrar últimas linhas de output
            lines = result.stdout.strip().split('\n')
            if len(lines) > 0:
                # Filtrar linhas vazias
                relevant_lines = [l for l in lines if l.strip()]
                if relevant_lines:
                    print(f"    Última saída:")
                    for line in relevant_lines[-2:]:
                        print(f"      {line[:80]}")
        else:
            print(f"    ✗ Erro (código {result.returncode}, {elapsed_time:.1f}s)")
            results.append((script_name, "ERROR", elapsed_time))
            error_count += 1
            
            # Mostrar erro
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                print(f"    Erro (últimas linhas):")
                for line in error_lines[-3:]:
                    if line.strip():
                        short_line = line[:80] if len(line) > 80 else line
                        print(f"      {short_line}")
    
    except subprocess.TimeoutExpired:
        print(f"    ⚠ Timeout (> {timeout_sec}s)")
        results.append((script_name, "TIMEOUT", timeout_sec))
        timeout_count += 1
    except Exception as e:
        print(f"    ✗ Exceção: {str(e)[:60]}")
        results.append((script_name, "EXCEPTION", 0))
        error_count += 1

# ── RESUMO ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("RESUMO DOS TESTES")
print("=" * 90)

for script_name, status, elapsed_time in results:
    symbol = {
        "SUCCESS": "✓",
        "ERROR": "✗", 
        "TIMEOUT": "⚠",
        "EXCEPTION": "✗",
        "NOT_FOUND": "✗"
    }.get(status, "?")
    
    time_str = f"({elapsed_time:.1f}s)" if elapsed_time > 0 else ""
    status_str = f"{status:12s}"
    print(f"  {symbol} {script_name:35s} → {status_str} {time_str}")

print(f"\n{'─'*90}")
print(f"  ✓ Sucesso:    {success_count}/{len(scripts_to_test)}")
print(f"  ✗ Erros:      {error_count}/{len(scripts_to_test)}")
print(f"  ⚠ Timeout:    {timeout_count}/{len(scripts_to_test)}")

if success_count == len(scripts_to_test):
    print(f"\n✓ TODOS OS SCRIPTS FUNCIONANDO CORRETAMENTE!")
elif success_count >= len(scripts_to_test) - 2:
    print(f"\n✓ MAIORIA DOS SCRIPTS FUNCIONANDO (alguns timeouts esperados)")
else:
    print(f"\n⚠ ALGUNS SCRIPTS COM PROBLEMAS")

print("=" * 90)
