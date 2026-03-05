
import pandas as pd
import numpy as np

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

    return pd.DataFrame(features)
