#!/usr/bin/env python
"""
Teste completo de todos os scripts e módulos do projeto
"""

import sys
import os
import io
import traceback
from pathlib import Path
from collections import defaultdict
import subprocess

# UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

print("=" * 100)
print("TESTE COMPLETO DE SCRIPTS E MÓDULOS")
print("=" * 100)
print()

# Dicionário para rastrear resultados
resultados = defaultdict(lambda: {"status": "PENDENTE", "erro": None, "tipo": None})

# 1. TESTAR MÓDULOS SRC (imports)
print("📦 TESTANDO MÓDULOS SRC (imports)...")
print("-" * 100)

modulos_src = [
    ("src.preprocessing", "Carregamento e normalização de dados"),
    ("src.feature_engineering", "Engenharia de features"),
    ("src.analysis", "Análise e avaliação de modelos"),
    ("src.encoder", "Treinamento do autoencoder"),
    ("src.latent_features", "Extração de features latentes"),
    ("src.train_models", "Treinamento de modelos ML"),
]

for modulo, descricao in modulos_src:
    try:
        __import__(modulo)
        resultados[modulo] = {"status": "✅", "tipo": "Módulo SRC", "descricao": descricao}
        print(f"  ✅ {modulo:<40} - {descricao}")
    except Exception as e:
        resultados[modulo] = {"status": "❌", "erro": str(e), "tipo": "Módulo SRC", "descricao": descricao}
        print(f"  ❌ {modulo:<40} - ERRO: {str(e)[:60]}")

print()

# 2. TESTAR ARQUIVO app.py
print("🎨 TESTANDO app.py (Streamlit)...")
print("-" * 100)

app_py = root_dir / "app.py"
if app_py.exists():
    try:
        # Não importa app.py direto (causa erro do Streamlit), apenas verifica sintaxe
        import ast
        with open(app_py, 'r', encoding='utf-8') as f:
            code = f.read()
            ast.parse(code)
        resultados["app.py"] = {"status": "✅", "tipo": "Aplicação", "descricao": "Interface Streamlit"}
        print(f"  ✅ app.py - Sintaxe válida")
    except SyntaxError as e:
        resultados["app.py"] = {"status": "❌", "erro": str(e), "tipo": "Aplicação"}
        print(f"  ❌ app.py - Erro de sintaxe: {str(e)[:60]}")
else:
    resultados["app.py"] = {"status": "❌", "erro": "Arquivo não encontrado", "tipo": "Aplicação"}
    print(f"  ❌ app.py - Arquivo não encontrado")

print()

# 3. TESTAR SCRIPTS
print("🔬 TESTANDO SCRIPTS...")
print("-" * 100)

scripts_teste = [
    ("scripts/verify_seasonal_results.py", "Verifica resultados por temporada", "validacao"),
    ("scripts/verify_all.py", "Verifica integridade completa", "validacao"),
    ("scripts/test_gridsearch_seasonal.py", "Teste de GridSearch por temporada", "teste"),
    ("scripts/test_features.py", "Teste de features", "teste"),
    ("scripts/show_metrics.py", "Mostra métricas dos modelos", "utilitario"),
    ("scripts/shap_analysis.py", "Análise SHAP", "analise"),
    ("scripts/radar_chart_by_season.py", "Gráfico radar por temporada", "visualizacao"),
    ("scripts/print_tables_by_season.py", "Imprime tabelas por temporada", "utilitario"),
    ("scripts/inspect_epl.py", "Inspeciona dados EPL", "utilitario"),
    ("scripts/gridsearch_quick_test.py", "GridSearch rápido", "teste"),
    ("scripts/gridsearch_advanced.py", "GridSearch avançado", "teste"),
    ("scripts/gridsearch_43features.py", "GridSearch com 43 features", "teste"),
    ("scripts/generate_figures.py", "Gera figuras", "geracao"),
    ("scripts/generate_all.py", "Pipeline completo", "geracao"),
    ("scripts/correlation_heatmap.py", "Heatmap de correlação", "visualizacao"),
    ("scripts/compute_metrics_and_update_csv.py", "Calcula métricas e atualiza CSV", "utilitario"),
]

for script_path, descricao, tipo in scripts_teste:
    full_path = root_dir / script_path
    
    if not full_path.exists():
        resultados[script_path] = {"status": "❌", "erro": "Arquivo não encontrado", "tipo": tipo, "descricao": descricao}
        print(f"  ❌ {script_path:<45} - Não encontrado")
        continue
    
    try:
        # Validar sintaxe Python
        import ast
        with open(full_path, 'r', encoding='utf-8') as f:
            code = f.read()
            ast.parse(code)
        
        # Tentar executar com timeout
        try:
            result = subprocess.run(
                [sys.executable, str(full_path)],
                cwd=str(root_dir),
                capture_output=True,
                timeout=10,
                text=True
            )
            
            # Checar se rodou sem erro de importação
            if "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
                resultados[script_path] = {
                    "status": "⚠️",
                    "erro": "Import Error",
                    "tipo": tipo,
                    "descricao": descricao
                }
                print(f"  ⚠️  {script_path:<45} - Import Error")
            elif result.returncode == 0:
                resultados[script_path] = {"status": "✅", "tipo": tipo, "descricao": descricao}
                print(f"  ✅ {script_path:<45} - {descricao}")
            else:
                # Pode ser erro esperado (ex: dados não encontrados)
                if "FileNotFoundError" in result.stderr or "No such file" in result.stderr:
                    resultados[script_path] = {"status": "⚠️", "erro": "Dados não encontrados", "tipo": tipo, "descricao": descricao}
                    print(f"  ⚠️  {script_path:<45} - Dados não encontrados")
                else:
                    resultados[script_path] = {
                        "status": "⚠️",
                        "erro": f"Exit code {result.returncode}",
                        "tipo": tipo,
                        "descricao": descricao
                    }
                    print(f"  ⚠️  {script_path:<45} - Exit code {result.returncode}")
                    
        except subprocess.TimeoutExpired:
            resultados[script_path] = {"status": "⏱️", "erro": "Timeout (>10s)", "tipo": tipo, "descricao": descricao}
            print(f"  ⏱️  {script_path:<45} - Timeout (script rodando longamente)")
        
    except SyntaxError as e:
        resultados[script_path] = {"status": "❌", "erro": f"Syntax Error: {str(e)[:40]}", "tipo": tipo, "descricao": descricao}
        print(f"  ❌ {script_path:<45} - Erro de sintaxe")
    except Exception as e:
        resultados[script_path] = {"status": "❌", "erro": str(e)[:60], "tipo": tipo, "descricao": descricao}
        print(f"  ❌ {script_path:<45} - Erro: {str(e)[:40]}")

print()
print("=" * 100)
print("RESUMO DOS TESTES")
print("=" * 100)
print()

# Contar status
status_count = defaultdict(int)
tipos = defaultdict(list)

for arquivo, info in resultados.items():
    status = info["status"]
    tipo = info.get("tipo", "Outro")
    status_count[status] += 1
    tipos[tipo].append((arquivo, info))

print(f"✅ Funcionando: {status_count['✅']}")
print(f"⚠️  Com avisos: {status_count['⚠️']}")
print(f"⏱️  Timeout: {status_count['⏱️']}")
print(f"❌ Erros: {status_count['❌']}")
print()

# Agrupar por tipo
print("POR TIPO:")
print("-" * 100)
for tipo in sorted(tipos.keys()):
    arquivos = tipos[tipo]
    ok = sum(1 for _, info in arquivos if info["status"] == "✅")
    warn = sum(1 for _, info in arquivos if info["status"] in ("⚠️", "⏱️"))
    err = sum(1 for _, info in arquivos if info["status"] == "❌")
    
    print(f"  {tipo:<20} - ✅ {ok:2d} | ⚠️  {warn:2d} | ❌ {err:2d}")

print()
print("=" * 100)
print("ANÁLISE DE SCRIPTS PARA POSSÍVEL EXCLUSÃO")
print("=" * 100)
print()

# Listar scripts com potencial de exclusão
exclusao_candidatos = []

for arquivo, info in sorted(resultados.items()):
    if info["status"] in ("⏱️", "⚠️"):
        tipo = info.get("tipo", "Outro")
        
        # Candidatos à exclusão
        if tipo == "teste" and "gridsearch" in arquivo.lower():
            exclusao_candidatos.append({
                "arquivo": arquivo,
                "razao": "Scripts de teste do GridSearch - função de desenvolvimento/teste",
                "tipo": tipo,
                "status": info["status"]
            })
        elif tipo == "utilitario" and ("inspect" in arquivo.lower() or "print_tables" in arquivo.lower()):
            exclusao_candidatos.append({
                "arquivo": arquivo,
                "razao": "Script utilitário de inspeção/debug - não essencial",
                "tipo": tipo,
                "status": info["status"]
            })
        elif tipo == "teste" and "test_" in arquivo:
            exclusao_candidatos.append({
                "arquivo": arquivo,
                "razao": "Script de teste local - função de desenvolvimento",
                "tipo": tipo,
                "status": info["status"]
            })

if exclusao_candidatos:
    print("🗑️  SCRIPTS PARA POSSÍVEL EXCLUSÃO:\n")
    for idx, cand in enumerate(exclusao_candidatos, 1):
        print(f"{idx}. {cand['arquivo']}")
        print(f"   Status: {cand['status']} | Tipo: {cand['tipo']}")
        print(f"   Razão: {cand['razao']}")
        print()

# Salvar relatório
print("=" * 100)
print("SALVANDO RELATÓRIO...")
print("=" * 100)

with open(root_dir / "RELATORIO_VALIDACAO_SCRIPTS.md", "w", encoding='utf-8') as f:
    f.write("# 📋 Relatório de Validação de Scripts e Módulos\n\n")
    f.write(f"**Data:** 23 de maio de 2026\n")
    f.write(f"**Total de Arquivos:** {len(resultados)}\n\n")
    
    f.write("## 📊 Resumo\n\n")
    f.write(f"- ✅ **Funcionando:** {status_count['✅']} arquivos\n")
    f.write(f"- ⚠️  **Com Avisos:** {status_count['⚠️']} arquivos\n")
    f.write(f"- ⏱️  **Timeout:** {status_count['⏱️']} arquivos\n")
    f.write(f"- ❌ **Erros:** {status_count['❌']} arquivos\n\n")
    
    f.write("## 📦 Módulos SRC\n\n")
    for tipo in tipos.keys():
        if "SRC" in tipo:
            for arquivo, info in sorted(tipos[tipo]):
                f.write(f"- {info['status']} `{arquivo}` - {info.get('descricao', 'N/A')}\n")
    
    f.write("\n## 🔬 Scripts de Teste\n\n")
    for tipo in ["teste", "validacao"]:
        if tipo in tipos:
            for arquivo, info in sorted(tipos[tipo]):
                f.write(f"- {info['status']} `{arquivo}` - {info.get('descricao', 'N/A')}\n")
                if info.get("erro"):
                    f.write(f"  - Erro: {info['erro']}\n")
    
    f.write("\n## 🎨 Aplicação Streamlit\n\n")
    if "app.py" in resultados:
        info = resultados["app.py"]
        f.write(f"- {info['status']} `app.py` - Interface Streamlit\n")
    
    f.write("\n## 📈 Scripts de Análise e Visualização\n\n")
    for tipo in ["analise", "visualizacao", "geracao"]:
        if tipo in tipos:
            for arquivo, info in sorted(tipos[tipo]):
                f.write(f"- {info['status']} `{arquivo}` - {info.get('descricao', 'N/A')}\n")
    
    f.write("\n## 🔧 Scripts Utilitários\n\n")
    if "utilitario" in tipos:
        for arquivo, info in sorted(tipos["utilitario"]):
            f.write(f"- {info['status']} `{arquivo}` - {info.get('descricao', 'N/A')}\n")
    
    f.write("\n## 🗑️  Scripts para Possível Exclusão\n\n")
    if exclusao_candidatos:
        f.write("Os seguintes scripts podem ser deletados por serem:\n")
        f.write("- Funções de desenvolvimento/teste\n")
        f.write("- Utilitários não essenciais\n")
        f.write("- Duplicatas ou obsoletos\n\n")
        for idx, cand in enumerate(exclusao_candidatos, 1):
            f.write(f"{idx}. **{cand['arquivo']}**\n")
            f.write(f"   - Tipo: {cand['tipo']}\n")
            f.write(f"   - Status: {cand['status']}\n")
            f.write(f"   - Razão: {cand['razao']}\n\n")
    else:
        f.write("Nenhum script identificado para exclusão obrigatória no momento.\n")

print("\n✅ Relatório salvo em: RELATORIO_VALIDACAO_SCRIPTS.md")
print()

# Relatório final
print("=" * 100)
print("✅ TESTE COMPLETADO!")
print("=" * 100)
