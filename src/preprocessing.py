
import pandas as pd
import os
import glob

def load_data(path):
    df = pd.read_csv(path)
    # normalize column names from the CSV to what downstream code expects
    rename_map = {}
    if 'Home' in df.columns:
        rename_map['Home'] = 'HomeTeam'
    if 'Away' in df.columns:
        rename_map['Away'] = 'AwayTeam'
    if 'HomeGoals' in df.columns:
        rename_map['HomeGoals'] = 'FTHG'
    if 'AwayGoals' in df.columns:
        rename_map['AwayGoals'] = 'FTAG'
    if rename_map:
        df = df.rename(columns=rename_map)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date')
    # Use Season_End_Year as Season to keep full season stats together
    # (a season like 1992-1993 should not be split by calendar year)
    df['Season'] = df['Season_End_Year']
    # map full-time result to numeric codes used by feature engineering
    if 'FTR' in df.columns:
        df['Result'] = df['FTR'].map({'H':0,'D':1,'A':2})
    return df

def load_multiple_seasons(directory_path):
    """
    Carrega múltiplos arquivos CSV de um diretório e combina em um único DataFrame.
    Segue a metodologia do artigo científico para separação de dados de treino/teste.
    
    Args:
        directory_path: Caminho do diretório contendo os arquivos CSV das temporadas
    
    Returns:
        DataFrame combinado com todas as temporadas do diretório
    """
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    csv_files.sort()  # Ordenar para garantir ordem cronológica
    
    if not csv_files:
        raise ValueError(f"Nenhum arquivo CSV encontrado em {directory_path}")
    
    print(f"\nCarregando dados de: {directory_path}")
    print(f"Arquivos encontrados: {len(csv_files)}")
    
    dataframes = []
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
        df_temp = pd.read_csv(csv_file)
        
        # Extrair o ano final da temporada do nome do arquivo
        # Formato esperado: Season_YYYY_YYYY.csv (ex: Season_2005_2006.csv)
        filename = os.path.basename(csv_file)
        if 'Season_' in filename:
            # Extrair os anos do nome do arquivo
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 3:
                season_end_year = int(parts[-1])  # Último ano é o final da temporada
                df_temp['Season'] = season_end_year
        
        # Normalizar nomes de colunas
        rename_map = {}
        if 'Home' in df_temp.columns:
            rename_map['Home'] = 'HomeTeam'
        if 'Away' in df_temp.columns:
            rename_map['Away'] = 'AwayTeam'
        if 'HomeGoals' in df_temp.columns:
            rename_map['HomeGoals'] = 'FTHG'
        if 'AwayGoals' in df_temp.columns:
            rename_map['AwayGoals'] = 'FTAG'
        if rename_map:
            df_temp = df_temp.rename(columns=rename_map)
        
        # Processar datas e temporadas (formato DD/MM/YYYY)
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
        
        # Mapear resultado
        if 'FTR' in df_temp.columns:
            df_temp['Result'] = df_temp['FTR'].map({'H':0,'D':1,'A':2})
        
        dataframes.append(df_temp)
    
    # Combinar todos os DataFrames
    df_combined = pd.concat(dataframes, ignore_index=True)
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)
    
    # Remover linhas com valores nulos em colunas essenciais
    essential_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result', 'Season']
    df_combined = df_combined.dropna(subset=essential_cols)
    
    print(f"Total de partidas carregadas: {len(df_combined)}")
    if 'Season' in df_combined.columns:
        seasons = df_combined['Season'].unique()
        print(f"Temporadas: {sorted(seasons)}")
    
    return df_combined

def load_all_data():
    """
    Carrega todos os dados (treino + teste) combinados.
    Útil para análises e visualizações que precisam de todos os dados.
    
    Returns:
        DataFrame combinado com todas as temporadas (2005-2016)
    """
    train_dir = "data/data_2005_2014"
    test_dir = "data/data_2014_2016"
    
    print("\nCarregando TODOS os dados (Treino + Teste)...")
    df_train = load_multiple_seasons(train_dir)
    df_test = load_multiple_seasons(test_dir)
    
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all = df_all.sort_values('Date').reset_index(drop=True)
    
    print(f"\n[OK] Total geral: {len(df_all)} partidas ({df_train.shape[0]} treino + {df_test.shape[0]} teste)")
    
    return df_all
