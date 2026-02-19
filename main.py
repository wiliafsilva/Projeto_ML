
from src.preprocessing import load_data
from src.feature_engineering import calculate_team_stats
from src.train_models import train_models

def main():
    df = load_data("data/epl.csv")
    features = calculate_team_stats(df)
    train_models(features)

if __name__ == "__main__":
    main()
