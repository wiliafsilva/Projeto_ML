import joblib
import pandas as pd

# Carregar modelos retreinados
results_metadata = joblib.load("models/trained_models.pkl")

# Extrair informaes
models = results_metadata.get('models', results_metadata)  # Compatibilidade com verso antiga
train_size = results_metadata.get('train_size', 'N/A')
test_size = results_metadata.get('test_size', 'N/A')
train_period = results_metadata.get('train_period', 'N/A')
test_period = results_metadata.get('test_period', 'N/A')

print("=" * 80)
print("MTRICAS DOS MODELOS RETREINADOS")
print("=" * 80)
print()

print(f" Perodo de Treinamento: {train_period} ({train_size} partidas)")
print(f" Perodo de Teste: {test_period} ({test_size} partidas)")
print()

data = []
for name, info in models.items():
    print(f" {name}:")
    print(f"   Acurcia: {info['accuracy']:.4f}")
    print(f"   F1-Score (macro): {info['f1']:.4f}")
    print(f"   RPS (Ranked Probability Score): {info['rps']:.4f}")
    print()
    data.append({
        'Modelo': name,
        'Acurcia': f"{info['accuracy']:.4f}",
        'F1-Score': f"{info['f1']:.4f}",
        'RPS': f"{info['rps']:.4f}"
    })

print("=" * 80)
print("TABELA RESUMIDA")
print("=" * 80)
df = pd.DataFrame(data)
print(df.to_string(index=False))
print()
print(" Modelos prontos para uso!")
print(" Execute 'python -m streamlit run app.py' para visualizar no Streamlit")
