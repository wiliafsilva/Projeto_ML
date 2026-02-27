
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.train_models import train_models
import os

def main():
    """
    Pipeline principal seguindo a metodologia do artigo científico:
    - Dados de treinamento: 2005-2014 (9 temporadas)
    - Dados de teste: 2014-2016 (2 temporadas)
    """
    print("="*80)
    print("REPLICA CIENTÍFICA - PREDIÇÃO DE RESULTADOS DA PREMIER LEAGUE")
    print("="*80)
    print("\nMetodologia do Artigo:")
    print("  - Dados de Treinamento: 2005-2014 (9 temporadas)")
    print("  - Dados de Teste: 2014-2016 (2 temporadas)")
    print("="*80)
    
    # Caminhos das pastas conforme estrutura do artigo
    train_dir = "data/data_2005_2014"
    test_dir = "data/data_2014_2016"
    
    # Verificar se os diretórios existem
    if not os.path.exists(train_dir):
        raise ValueError(f"Diretório de treinamento não encontrado: {train_dir}")
    if not os.path.exists(test_dir):
        raise ValueError(f"Diretório de teste não encontrado: {test_dir}")
    
    # Carregar dados de treinamento (2005-2014)
    print("\n" + "="*80)
    print("ETAPA 1: CARREGAMENTO DOS DADOS")
    print("="*80)
    df_train = load_multiple_seasons(train_dir)
    
    # Carregar dados de teste (2014-2016)
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
