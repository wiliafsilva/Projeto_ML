
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.train_models import train_models
import os

def main():
    """
    Pipeline principal seguindo o split definido no workspace:
    - Dados de treinamento: 2011-2023 (temporadas para treino)
    - Dados de teste: 2023-2025 (temporadas para teste)
    """
    print("="*80)
    print("REPLICA CIENTÍFICA - PREDIÇÃO DE RESULTADOS DA PREMIER LEAGUE")
    print("="*80)
    print("\nMetodologia do Artigo:")
    print("  - Dados de Treinamento: 2011-2023 (treino)")
    print("  - Dados de Teste: 2023-2025 (teste)")
    print("="*80)
    
    # Caminhos das pastas conforme novo split de dados
    train_dir = "data/data_2011_2023"
    test_dir = "data/data_2023_2025"
    
    # Verificar se os diretórios existem
    if not os.path.exists(train_dir):
        raise ValueError(f"Diretório de treinamento não encontrado: {train_dir}")
    if not os.path.exists(test_dir):
        raise ValueError(f"Diretório de teste não encontrado: {test_dir}")
    
    # Carregar dados de treinamento (2011-2023)
    print("\n" + "="*80)
    print("ETAPA 1: CARREGAMENTO DOS DADOS")
    print("="*80)
    df_train = load_multiple_seasons(train_dir)
    
    # Carregar dados de teste (2023-2025)
    df_test = load_multiple_seasons(test_dir)
    
    # Calcular features para dados de treinamento
    print("\n" + "="*80)
    print("ETAPA 2: ENGENHARIA DE FEATURES")
    print("="*80)
    print("\nCalculando features para dados de TREINAMENTO...")
    features_train = calculate_team_stats(df_train)
    
    # Calcular features para dados de teste
    print("\nCalculando features para dados de TESTE...")
    features_test = calculate_team_stats(df_test)
    
    # Treinar e avaliar modelos
    print("\n" + "="*80)
    print("ETAPA 3: TREINAMENTO E AVALIAÇÃO DOS MODELOS")
    print("="*80)
    train_models(features_train, features_test)
    
    print("\n" + "="*80)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*80)

if __name__ == "__main__":
    main()
