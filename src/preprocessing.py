
import pandas as pd
import os
import glob

def load_ratings():
    """
    Carrega ratings FIFA de data/fifa_ratings.csv.
    
    Returns:
        DataFrame com colunas: Team, Season, Overall, Attack, Midfield, Defense
    """
    ratings_path = 'data/fifa_ratings.csv'
    
    try:
        df_ratings = pd.read_csv(ratings_path)
        print(f"\n[OK] Ratings FIFA carregados: {len(df_ratings)} entradas")
        print(f"   Temporadas: {sorted(df_ratings['Season'].unique())}")
        print(f"   Times únicos: {df_ratings['Team'].nunique()}")
        return df_ratings
    except FileNotFoundError:
        print(f"\n[AVISO] AVISO: {ratings_path} não encontrado!")
        print("   Features de Ratings não serão adicionadas.")
        return pd.DataFrame()

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
    
    # ========== ADICIONAR RATINGS FIFA ==========
    df_ratings = load_ratings()
    
    if not df_ratings.empty:
        print("\n[Pipeline] Integrando Ratings FIFA...")
        
        # Merge para time casa
        df_combined = df_combined.merge(
            df_ratings.rename(columns={
                'Team': 'HomeTeam',
                'Overall': 'HOverall',
                'Attack': 'HAttack',
                'Midfield': 'HMidfield',
                'Defense': 'HDefense'
            }),
            on=['HomeTeam', 'Season'],
            how='left'
        )
        
        # Merge para time visitante
        df_combined = df_combined.merge(
            df_ratings.rename(columns={
                'Team': 'AwayTeam',
                'Overall': 'AOverall',
                'Attack': 'AAttack',
                'Midfield': 'AMidfield',
                'Defense': 'ADefense'
            }),
            on=['AwayTeam', 'Season'],
            how='left'
        )
        
        # Calcular diferenciais (Class B features)
        df_combined['overall_diff'] = df_combined['HOverall'] - df_combined['AOverall']
        df_combined['attack_diff'] = df_combined['HAttack'] - df_combined['AAttack']
        df_combined['midfield_diff'] = df_combined['HMidfield'] - df_combined['AMidfield']
        df_combined['defense_diff'] = df_combined['HDefense'] - df_combined['ADefense']
        
        # Verificar missings
        missing_count = df_combined[['HOverall', 'AOverall']].isna().sum().sum()
        if missing_count > 0:
            print(f"[AVISO] AVISO: {missing_count} ratings ausentes (times não encontrados no CSV)")
            # Listar alguns times com ratings ausentes
            missing_teams = df_combined[df_combined['HOverall'].isna()][['HomeTeam', 'Season']].drop_duplicates().head(5)
            if not missing_teams.empty:
                print("   Exemplos de times sem ratings:")
                for _, row in missing_teams.iterrows():
                    print(f"     - {row['HomeTeam']} (temporada {row['Season']})")
            
            # Preencher valores ausentes com mediana (time mediano)
            print("\n[Pipeline] Preenchendo ratings ausentes com mediana...")
            rating_columns = ['HOverall', 'HAttack', 'HMidfield', 'HDefense',
                            'AOverall', 'AAttack', 'AMidfield', 'ADefense']
            
            for col in rating_columns:
                if col in df_combined.columns:
                    median_value = df_combined[col].median()
                    df_combined[col] = df_combined[col].fillna(median_value)
                    print(f"   {col}: preenchido com {median_value:.1f}")
            
            # Recalcular diferenciais
            df_combined['overall_diff'] = df_combined['HOverall'] - df_combined['AOverall']
            df_combined['attack_diff'] = df_combined['HAttack'] - df_combined['AAttack']
            df_combined['midfield_diff'] = df_combined['HMidfield'] - df_combined['AMidfield']
            df_combined['defense_diff'] = df_combined['HDefense'] - df_combined['ADefense']
            
            print("[OK] Ratings ausentes preenchidos com sucesso!")
        else:
            print("[OK] Todas as partidas têm ratings!")
    
    # ========== ADICIONAR ODDS FEATURES ==========
    odds_columns = ['B365H', 'B365D', 'B365A']
    
    # Verificar se colunas de odds existem
    if all(col in df_combined.columns for col in odds_columns):
        print("\n[Pipeline] Integrando Odds Features (Bet365)...")
        
        # Verificar valores faltantes
        missing_odds = df_combined[odds_columns].isna().sum().sum()
        if missing_odds > 0:
            print(f"[AVISO] AVISO: {missing_odds} valores de odds ausentes")
            # Preencher com odds neutras (valores que refletem 33.3% de probabilidade para cada resultado)
            # Odd neutra ≈ 3.0 para cada resultado
            for col in odds_columns:
                median_odd = df_combined[col].median()
                df_combined[col] = df_combined[col].fillna(median_odd)
                print(f"   {col}: preenchido com mediana {median_odd:.2f}")
        
        # Criar features derivadas das odds (probabilidades implícitas)
        # P(evento) = 1 / Odd (antes da margem da casa de apostas)
        df_combined['prob_home'] = 1 / df_combined['B365H']
        df_combined['prob_draw'] = 1 / df_combined['B365D']
        df_combined['prob_away'] = 1 / df_combined['B365A']
        
        # Normalizar probabilidades (remover margem da casa - overround)
        prob_sum = df_combined['prob_home'] + df_combined['prob_draw'] + df_combined['prob_away']
        df_combined['prob_home_norm'] = df_combined['prob_home'] / prob_sum
        df_combined['prob_draw_norm'] = df_combined['prob_draw'] / prob_sum
        df_combined['prob_away_norm'] = df_combined['prob_away'] / prob_sum
        
        print("[OK] Odds Features criadas:")
        print("   - B365H, B365D, B365A (odds brutas)")
        print("   - prob_home/draw/away (probabilidades implícitas)")
        print("   - prob_home/draw/away_norm (probabilidades normalizadas)")
        print(f"   Total: 9 features de odds")
    else:
        missing_cols = [col for col in odds_columns if col not in df_combined.columns]
        print(f"\n[AVISO] AVISO: Colunas de odds não encontradas: {missing_cols}")
        print("   Features de Odds não serão adicionadas.")
    
    # Remover linhas com valores nulos em colunas essenciais
    essential_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Result', 'Season']
    df_combined = df_combined.dropna(subset=essential_cols)
    
    print(f"\nTotal de partidas carregadas: {len(df_combined)}")
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
