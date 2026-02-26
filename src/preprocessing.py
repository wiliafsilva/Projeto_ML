
import pandas as pd

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

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    # Use Season_End_Year as Season to keep full season stats together
    # (a season like 1992-1993 should not be split by calendar year)
    df['Season'] = df['Season_End_Year']
    # map full-time result to numeric codes used by feature engineering
    if 'FTR' in df.columns:
        df['Result'] = df['FTR'].map({'H':0,'D':1,'A':2})
    return df
