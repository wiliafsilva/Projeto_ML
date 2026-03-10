
import pandas as pd
import numpy as np

def calculate_form_feature(df, gamma=0.33):
    """
    Implementa sistema Form (ELO-style) conforme artigo científico.
    
    Citação do artigo (Baboota & Kaur, 2018, pg. 4-5):
    "The Form value of each team is initialized to one at the beginning 
    of each season and then updated after each match according to the 
    result of the match."
    
    Fórmulas:
    - Time α vence β:
        ξᵅⱼ = ξᵅ(j-1) + γ · ξᵝ(j-1)
        ξᵝⱼ = ξᵝ(j-1) - γ · ξᵝ(j-1)
    
    - Empate:
        ξᵅⱼ = ξᵅ(j-1) - γ(ξᵅ(j-1) - ξᵝ(j-1))
        ξᵝⱼ = ξᵝ(j-1) - γ(ξᵝ(j-1) - ξᵅ(j-1))
    
    Args:
        df: DataFrame com colunas HomeTeam, AwayTeam, Result, Season
        gamma: Fração de "roubo" (padrão 0.33 conforme artigo)
    
    Returns:
        DataFrame com colunas:
        - form_diff: Diferença Form casa - fora (Class B)
        - home_form: Form casa individual (Class A para Naive Bayes)
        - away_form: Form fora individual (Class A para Naive Bayes)
    
    Exemplo (Tabela 1 do artigo):
        Situação: ξᵅ = 1.4, ξᵝ = 0.8, γ = 0.33
        
        Se α vencer:
            ξᵅ = 1.4 + 0.33 × 0.8 = 1.664 ✓
            ξᵝ = 0.8 - 0.33 × 0.8 = 0.536 ✓
        
        Se empatar:
            ξᵅ = 1.4 - 0.33 × (1.4 - 0.8) = 1.202 ✓
            ξᵝ = 0.8 - 0.33 × (0.8 - 1.4) = 0.998 ✓
    """
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    # Inicializar Form = 1.0 para todos os times
    form_dict = {team: 1.0 for team in teams}
    current_season = None
    
    features = []
    
    for idx, row in df.iterrows():
        # Reset Form = 1.0 ao início de cada temporada
        if current_season is None:
            current_season = row['Season']
        elif row['Season'] != current_season:
            form_dict = {team: 1.0 for team in teams}
            current_season = row['Season']
            print(f"[Form] Resetando para temporada {current_season}")
        
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Capturar Form ANTES do jogo (evita data leakage)
        home_form_before = form_dict[home]
        away_form_before = form_dict[away]
        
        features.append({
            'form_diff': home_form_before - away_form_before,  # Class B
            'home_form': home_form_before,                     # Class A
            'away_form': away_form_before                      # Class A
        })
        
        # Atualizar Form APÓS criar features
        result = row['Result']  # 0=H, 1=D, 2=A
        
        if result == 0:  # Home vence
            form_dict[home] = form_dict[home] + gamma * form_dict[away]
            form_dict[away] = form_dict[away] - gamma * form_dict[away]
        
        elif result == 2:  # Away vence
            form_dict[away] = form_dict[away] + gamma * form_dict[home]
            form_dict[home] = form_dict[home] - gamma * form_dict[home]
        
        else:  # Empate (result == 1)
            form_dict[home] = form_dict[home] - gamma * (form_dict[home] - form_dict[away])
            form_dict[away] = form_dict[away] - gamma * (form_dict[away] - form_dict[home])
    
    print(f"[Form] Calculado para {len(features)} partidas")
    print(f"[Form] Exemplo Form final: {list(form_dict.items())[:3]}")
    
    return pd.DataFrame(features)


def calculate_mu_features(df, k=6):
    """
    Calcula μₖ features: médias móveis dos últimos k jogos (padrão k=6).
    
    Implementa features do artigo Baboota & Kaur (2018):
    - Corners (HC, AC) → corners_diff = μₖ_home_corners - μₖ_away_corners
    - Shots on Target (HST, AST) → shotsontarget_diff
    - Total Shots (HS, AS) → shots_diff  
    - Goals (FTHG, FTAG) → goals_avg_diff
    
    Anti-leakage: usa apenas dados ANTERIORES ao jogo atual.
    Reset sazonal: estatísticas resetam a cada temporada.
    
    Args:
        df: DataFrame com colunas HC, AC, HST, AST, HS, AS, FTHG, FTAG, HomeTeam, AwayTeam, Season
        k: janela de jogos para média móvel (padrão 6)
        
    Returns:
        DataFrame com colunas: corners_diff, shotsontarget_diff, shots_diff, goals_avg_diff
    """
    print(f"\n[μₖ Features] Iniciando cálculo com k={k} jogos...")
    
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    # Inicializar histórico para cada time
    team_history = {
        team: {
            'corners': [],
            'shots_on_target': [],
            'total_shots': [],
            'goals': []
        } for team in teams
    }
    
    current_season = None
    features = []
    
    for idx, row in df.iterrows():
        # Reset sazonal
        if current_season is None:
            current_season = row['Season']
        elif row['Season'] != current_season:
            print(f"[μₖ] Reset sazonal: {current_season} → {row['Season']}")
            team_history = {
                team: {
                    'corners': [],
                    'shots_on_target': [],
                    'total_shots': [],
                    'goals': []
                } for team in teams
            }
            current_season = row['Season']
        
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Calcular μₖ (média dos últimos k jogos) - ANTES do jogo atual
        def calculate_mu(history_list, k_games):
            """Média dos últimos k jogos, ou média de todos se < k jogos"""
            if not history_list:
                return 0.0
            recent = history_list[-k_games:]
            return np.mean(recent)
        
        # μₖ para Home team
        mu_home_corners = calculate_mu(team_history[home]['corners'], k)
        mu_home_sot = calculate_mu(team_history[home]['shots_on_target'], k)
        mu_home_shots = calculate_mu(team_history[home]['total_shots'], k)
        mu_home_goals = calculate_mu(team_history[home]['goals'], k)
        
        # μₖ para Away team
        mu_away_corners = calculate_mu(team_history[away]['corners'], k)
        mu_away_sot = calculate_mu(team_history[away]['shots_on_target'], k)
        mu_away_shots = calculate_mu(team_history[away]['total_shots'], k)
        mu_away_goals = calculate_mu(team_history[away]['goals'], k)
        
        # Diferenças (Home - Away)
        feature_row = {
            'corners_diff': mu_home_corners - mu_away_corners,
            'shotsontarget_diff': mu_home_sot - mu_away_sot,
            'shots_diff': mu_home_shots - mu_away_shots,
            'goals_avg_diff': mu_home_goals - mu_away_goals
        }
        
        features.append(feature_row)
        
        # *** ATUALIZAR HISTÓRICO APÓS FEATURE CREATION (anti-leakage) ***
        team_history[home]['corners'].append(row['HC'])
        team_history[away]['corners'].append(row['AC'])
        
        team_history[home]['shots_on_target'].append(row['HST'])
        team_history[away]['shots_on_target'].append(row['AST'])
        
        team_history[home]['total_shots'].append(row['HS'])
        team_history[away]['total_shots'].append(row['AS'])
        
        team_history[home]['goals'].append(row['FTHG'])
        team_history[away]['goals'].append(row['FTAG'])
    
    df_mu = pd.DataFrame(features)
    
    print(f"[μₖ] ✓ 4 features calculadas: {df_mu.columns.tolist()}")
    print(f"[μₖ] Shape: {df_mu.shape}")
    print(f"[μₖ] Exemplo corners_diff (primeiras 5): {df_mu['corners_diff'].head().tolist()}")
    print(f"[μₖ] Estatísticas goals_avg_diff:")
    print(f"      Min: {df_mu['goals_avg_diff'].min():.3f}")
    
    return df_mu


def calculate_head_to_head_features(df, window=5):
    """
    Calcula features de confronto direto (Head-to-Head) entre os times.
    
    Para cada jogo, calcula estatísticas dos últimos 'window' confrontos diretos:
    - h2h_home_wins: Vitórias do time da casa nos últimos confrontos
    - h2h_draws: Empates nos últimos confrontos
    - h2h_away_wins: Vitórias do visitante nos últimos confrontos
    - h2h_home_goals_avg: Média de gols do mandante em confrontos anteriores
    - h2h_away_goals_avg: Média de gols do visitante em confrontos anteriores
    - h2h_games: Número de confrontos anteriores considerados
    
    Anti-leakage: Usa apenas jogos ANTERIORES ao jogo atual.
    Reset sazonal: Mantém histórico entre temporadas (confrontos históricos).
    
    Args:
        df: DataFrame com HomeTeam, AwayTeam, Result, FTHG, FTAG
        window: Número máximo de confrontos anteriores a considerar (padrão 5)
    
    Returns:
        DataFrame com features h2h_*
    """
    print(f"\n[H2H Features] Iniciando cálculo com window={window} jogos...")
    
    # Armazenar histórico de confrontos entre cada par de times
    # Chave: (home_team, away_team) → lista de resultados passados
    h2h_history = {}
    
    features = []
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Chave para o confronto (sempre na ordem home, away)
        h2h_key = (home, away)
        
        # Buscar histórico de confrontos ANTERIORES a este jogo
        if h2h_key not in h2h_history:
            h2h_history[h2h_key] = []
        
        past_games = h2h_history[h2h_key][-window:] if h2h_history[h2h_key] else []
        
        # Calcular estatísticas dos confrontos anteriores
        if past_games:
            h2h_home_wins = sum(1 for g in past_games if g['result'] == 0)  # Home wins
            h2h_draws = sum(1 for g in past_games if g['result'] == 1)      # Draws
            h2h_away_wins = sum(1 for g in past_games if g['result'] == 2)  # Away wins
            h2h_home_goals_avg = np.mean([g['home_goals'] for g in past_games])
            h2h_away_goals_avg = np.mean([g['away_goals'] for g in past_games])
            h2h_games = len(past_games)
        else:
            # Sem histórico: valores neutros
            h2h_home_wins = 0
            h2h_draws = 0
            h2h_away_wins = 0
            h2h_home_goals_avg = 0.0
            h2h_away_goals_avg = 0.0
            h2h_games = 0
        
        feature_row = {
            'h2h_home_wins': h2h_home_wins,
            'h2h_draws': h2h_draws,
            'h2h_away_wins': h2h_away_wins,
            'h2h_home_goals_avg': h2h_home_goals_avg,
            'h2h_away_goals_avg': h2h_away_goals_avg,
            'h2h_games': h2h_games
        }
        
        features.append(feature_row)
        
        # *** ATUALIZAR HISTÓRICO APÓS FEATURE CREATION (anti-leakage) ***
        h2h_history[h2h_key].append({
            'result': row['Result'],
            'home_goals': row['FTHG'],
            'away_goals': row['FTAG']
        })
    
    df_h2h = pd.DataFrame(features)
    
    print(f"[H2H] ✓ 6 features calculadas: {df_h2h.columns.tolist()}")
    print(f"[H2H] Shape: {df_h2h.shape}")
    print(f"[H2H] Jogos com histórico H2H: {(df_h2h['h2h_games'] > 0).sum()} ({(df_h2h['h2h_games'] > 0).sum() / len(df_h2h) * 100:.1f}%)")
    print(f"[H2H] Média de confrontos considerados: {df_h2h['h2h_games'].mean():.2f}")
    
    return df_h2h


def calculate_league_position_features(df):
    """
    Calcula features de posição na tabela (league standings) para cada time.
    
    Simula a tabela de classificação ao longo da temporada:
    - home_position: Posição do mandante na tabela (1 = líder, 20 = lanterna)
    - away_position: Posição do visitante na tabela
    - position_diff: Diferença de posição (home - away, negativo = mandante melhor)
    - home_points: Pontos acumulados do mandante
    - away_points: Pontos acumulados do visitante
    - points_diff: Diferença de pontos (home - away)
    
    Anti-leakage: Usa apenas jogos ANTERIORES ao jogo atual.
    Reset sazonal: Tabela reseta a cada temporada (todos começam com 0 pontos).
    
    Args:
        df: DataFrame com HomeTeam, AwayTeam, Result, Season
    
    Returns:
        DataFrame com features de posição/classificação
    """
    print(f"\n[League Position] Iniciando cálculo...")
    
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    # Armazenar pontos de cada time na temporada atual
    standings = {team: {'points': 0, 'gd': 0, 'gf': 0} for team in teams}
    current_season = None
    
    features = []
    
    for idx, row in df.iterrows():
        # Reset sazonal: zerar pontos no início de cada temporada
        if current_season is None:
            current_season = row['Season']
        elif row['Season'] != current_season:
            print(f"[Position] Reset sazonal: {current_season} → {row['Season']}")
            standings = {team: {'points': 0, 'gd': 0, 'gf': 0} for team in teams}
            current_season = row['Season']
        
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Calcular posições ANTES do jogo (anti-leakage)
        # Ordenar times por: pontos (desc), saldo de gols (desc), gols favor (desc)
        sorted_teams = sorted(
            standings.items(),
            key=lambda x: (x[1]['points'], x[1]['gd'], x[1]['gf']),
            reverse=True
        )
        
        # Atribuir posições (1 = melhor, 20 = pior)
        team_positions = {team: pos + 1 for pos, (team, _) in enumerate(sorted_teams)}
        
        home_position = team_positions[home]
        away_position = team_positions[away]
        home_points = standings[home]['points']
        away_points = standings[away]['points']
        
        feature_row = {
            'home_position': home_position,
            'away_position': away_position,
            'position_diff': home_position - away_position,  # Negativo = mandante melhor colocado
            'home_points': home_points,
            'away_points': away_points,
            'points_diff': home_points - away_points
        }
        
        features.append(feature_row)
        
        # *** ATUALIZAR STANDINGS APÓS FEATURE CREATION (anti-leakage) ***
        result = row['Result']  # 0=H, 1=D, 2=A
        home_goals = row['FTHG']
        away_goals = row['FTAG']
        
        # Atualizar pontos
        if result == 0:  # Home vence
            standings[home]['points'] += 3
        elif result == 2:  # Away vence
            standings[away]['points'] += 3
        else:  # Empate
            standings[home]['points'] += 1
            standings[away]['points'] += 1
        
        # Atualizar saldo de gols e gols a favor
        standings[home]['gd'] += (home_goals - away_goals)
        standings[away]['gd'] += (away_goals - home_goals)
        standings[home]['gf'] += home_goals
        standings[away]['gf'] += away_goals
    
    df_position = pd.DataFrame(features)
    
    print(f"[Position] ✓ 6 features calculadas: {df_position.columns.tolist()}")
    print(f"[Position] Shape: {df_position.shape}")
    print(f"[Position] Home position range: {df_position['home_position'].min():.0f} - {df_position['home_position'].max():.0f}")
    print(f"[Position] Away position range: {df_position['away_position'].min():.0f} - {df_position['away_position'].max():.0f}")
    print(f"[Position] Position diff (mean): {df_position['position_diff'].mean():.2f}")
    
    return df_position


def calculate_interaction_features(df_features):
    """
    Calcula interaction features baseadas nos insights do DIA 8 Feature Importance.
    
    DIA 9: Interaction Features Engineering
    
    Motivação (DIA 8 Analysis):
    - Top 3 features: h2h_games (6.91%), away_position (6.36%), Season (5.41%)
    - away_position >> home_position (+32% mais importante)
    - Form features explicam 18.6% da importância
    
    Interaction features criadas:
    1. h2h_confidence: Normalização de h2h_games (0-1 scale)
       - Indica "confiança" nas predições baseadas em H2H
       - h2h_games=5 → confidence=1.0 (histórico completo)
       
    2. away_advantage: Relação posicional favorável ao visitante
       - away_position / (home_position + 0.01) evita divisão por zero
       - Valores baixos (<1) → visitante melhor posicionado
       - Valores altos (>1) → mandante melhor posicionado
       
    3. season_trend: Tendência temporal normalizada
       - (Season - min) / (max - min)
       - Captura evolução temporal sem dependência de valores absolutos
       
    4. position_form_home: Interação posição × forma mandante
       - home_position * (1 / (home_form + 0.01))
       - Valores altos → time fraco (posição ruim) E forma baixa
       
    5. position_form_away: Interação posição × forma visitante
       - away_position * (1 / (away_form + 0.01))
       - Complementa position_form_home para capturar assimetria
       
    6. strength_balance: Desequilíbrio geral de força
       - |position_form_home - position_form_away|
       - Captura disparidade total entre times
    
    Args:
        df_features: DataFrame com features calculadas (incluindo h2h, position, form)
    
    Returns:
        DataFrame com 6 novas interaction features
    """
    interactions = pd.DataFrame(index=df_features.index)
    
    # 1. H2H Confidence (normalizado 0-1)
    if 'h2h_games' in df_features.columns:
        interactions['h2h_confidence'] = df_features['h2h_games'] / 5.0
        interactions['h2h_confidence'] = interactions['h2h_confidence'].clip(0, 1)
    else:
        interactions['h2h_confidence'] = 0.0
    
    # 2. Away Advantage (relação de posições)
    if 'away_position' in df_features.columns and 'home_position' in df_features.columns:
        interactions['away_advantage'] = df_features['away_position'] / (df_features['home_position'] + 0.01)
    else:
        interactions['away_advantage'] = 1.0
    
    # 3. Season Trend (normalizado)
    if 'Season' in df_features.columns:
        min_season = df_features['Season'].min()
        max_season = df_features['Season'].max()
        if max_season > min_season:
            interactions['season_trend'] = (df_features['Season'] - min_season) / (max_season - min_season)
        else:
            interactions['season_trend'] = 0.0
    else:
        interactions['season_trend'] = 0.0
    
    # 4. Position × Form (Home)
    if 'home_position' in df_features.columns and 'home_form' in df_features.columns:
        # Posição alta (ruim) × forma baixa = valor alto (time fraco)
        interactions['position_form_home'] = df_features['home_position'] * (1.0 / (df_features['home_form'] + 0.01))
    else:
        interactions['position_form_home'] = 0.0
    
    # 5. Position × Form (Away)
    if 'away_position' in df_features.columns and 'away_form' in df_features.columns:
        interactions['position_form_away'] = df_features['away_position'] * (1.0 / (df_features['away_form'] + 0.01))
    else:
        interactions['position_form_away'] = 0.0
    
    # 6. Strength Balance (disparidade total)
    interactions['strength_balance'] = np.abs(
        interactions['position_form_home'] - interactions['position_form_away']
    )
    
    print(f"\n[Interactions] ✓ 6 interaction features calculadas:")
    print(f"[Interactions]   - h2h_confidence (mean: {interactions['h2h_confidence'].mean():.3f})")
    print(f"[Interactions]   - away_advantage (mean: {interactions['away_advantage'].mean():.3f})")
    print(f"[Interactions]   - season_trend (mean: {interactions['season_trend'].mean():.3f})")
    print(f"[Interactions]   - position_form_home (mean: {interactions['position_form_home'].mean():.3f})")
    print(f"[Interactions]   - position_form_away (mean: {interactions['position_form_away'].mean():.3f})")
    print(f"[Interactions]   - strength_balance (mean: {interactions['strength_balance'].mean():.3f})")
    print(f"[Interactions] Shape: {interactions.shape}")
    
    return interactions


def calculate_team_stats(df):

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_data = {team: {'gd':0, 'history':[]} for team in teams}
    current_season = None

    features = []

    for _, row in df.iterrows():
        # reset team stats at season boundary to avoid accumulation across seasons
        if current_season is None:
            current_season = row['Season']
        elif row['Season'] != current_season:
            team_data = {team: {'gd':0, 'history':[]} for team in teams}
            current_season = row['Season']
        home = row['HomeTeam']
        away = row['AwayTeam']

        home_hist = team_data[home]['history'][-5:]
        away_hist = team_data[away]['history'][-5:]

        def streak(hist):
            return sum(hist)/ (3*len(hist)) if hist else 0

        def weighted(hist):
            weights = list(range(1,len(hist)+1))
            return sum(w*h for w,h in zip(weights,hist))/(3*sum(weights)) if hist else 0

        home_streak = streak(home_hist)
        away_streak = streak(away_hist)

        home_weight = weighted(home_hist)
        away_weight = weighted(away_hist)

        feature_row = {
            'gd_diff': team_data[home]['gd'] - team_data[away]['gd'],
            'streak_diff': home_streak - away_streak,
            'weighted_diff': home_weight - away_weight,
            'Result': row['Result'],
            'Season': row['Season']
        }

        features.append(feature_row)

        # update after feature creation (prevents leakage)
        home_goals = row['FTHG']
        away_goals = row['FTAG']

        team_data[home]['gd'] += (home_goals - away_goals)
        team_data[away]['gd'] += (away_goals - home_goals)

        result_map = {0:3,1:1,2:0}
        reverse_map = {0:0,1:1,2:3}

        team_data[home]['history'].append(result_map[row['Result']])
        team_data[away]['history'].append(reverse_map[row['Result']])

    df_features = pd.DataFrame(features)
    
    # ======== ADICIONAR FORM FEATURES (Class A e Class B) ========
    print("\n[Pipeline] Calculando Form features...")
    df_form = calculate_form_feature(df)
    
    # Adicionar form_diff (Class B) e também home_form, away_form (Class A)
    df_features = pd.concat([
        df_features, 
        df_form[['form_diff', 'home_form', 'away_form']]
    ], axis=1)
    
    # ======== ADICIONAR μₖ FEATURES ========
    print("[Pipeline] Calculando μₖ features (k=6)...")
    df_mu = calculate_mu_features(df, k=6)
    
    # Adicionar: corners_diff, shotsontarget_diff, shots_diff, goals_avg_diff
    df_features = pd.concat([df_features, df_mu], axis=1)
    
    # ======== ADICIONAR RATINGS SE EXISTIREM ========
    if 'overall_diff' in df.columns:
        print("[Pipeline] Adicionando Ratings features...")
        rating_cols = ['overall_diff', 'attack_diff', 'midfield_diff', 'defense_diff']
        
        # Verificar quais colunas existem
        existing_rating_cols = [col for col in rating_cols if col in df.columns]
        
        if existing_rating_cols:
            df_features = pd.concat([df_features, df[existing_rating_cols]], axis=1)
            print(f"   ✓ {len(existing_rating_cols)} features de ratings adicionadas")
    
    # ======== ADICIONAR HEAD-TO-HEAD FEATURES ========
    print("[Pipeline] Calculando Head-to-Head features...")
    df_h2h = calculate_head_to_head_features(df, window=5)
    df_features = pd.concat([df_features, df_h2h], axis=1)
    
    # ======== ADICIONAR LEAGUE POSITION FEATURES ========
    print("[Pipeline] Calculando League Position features...")
    df_position = calculate_league_position_features(df)
    df_features = pd.concat([df_features, df_position], axis=1)
    
    # ======== ADICIONAR ODDS FEATURES SE EXISTIREM ========
    odds_cols_raw = ['B365H', 'B365D', 'B365A']
    odds_cols_prob = ['prob_home', 'prob_draw', 'prob_away']
    odds_cols_norm = ['prob_home_norm', 'prob_draw_norm', 'prob_away_norm']
    all_odds_cols = odds_cols_raw + odds_cols_prob + odds_cols_norm
    
    if any(col in df.columns for col in all_odds_cols):
        print("[Pipeline] Adicionando Odds features...")
        
        # Verificar quais colunas de odds existem
        existing_odds_cols = [col for col in all_odds_cols if col in df.columns]
        
        if existing_odds_cols:
            df_features = pd.concat([df_features, df[existing_odds_cols]], axis=1)
            print(f"   ✓ {len(existing_odds_cols)} features de odds adicionadas")
            print(f"     - Odds brutas: {len([c for c in odds_cols_raw if c in existing_odds_cols])}")
            print(f"     - Probabilidades: {len([c for c in odds_cols_prob if c in existing_odds_cols])}")
            print(f"     - Probabilidades normalizadas: {len([c for c in odds_cols_norm if c in existing_odds_cols])}")
    
    # ======== ADICIONAR INTERACTION FEATURES (DIA 9) ========
    print("[Pipeline] Calculando Interaction features (DIA 9)...")
    df_interactions = calculate_interaction_features(df_features)
    df_features = pd.concat([df_features, df_interactions], axis=1)
    
    print(f"[Pipeline] Features finais: {df_features.columns.tolist()}")
    print(f"[Pipeline] Shape: {df_features.shape}")
    
    return df_features
