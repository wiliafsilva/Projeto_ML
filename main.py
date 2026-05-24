
import sys
import os

# Forçar UTF-8 no Windows para evitar erros de caractere Unicode
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.encoder import train_encoder, extract_latent_features, save_encoder_models
from src.latent_features import combine_original_and_latent, save_combined_features
from src.train_models import train_models

# Features Class B usadas como entrada do encoder (diferenciais)
CLASS_B_FEATURES = [
    'gd_diff', 'streak_diff', 'weighted_diff',
    'form_diff',
    'corners_diff', 'shotsontarget_diff', 'shots_diff', 'goals_avg_diff',
    'overall_diff', 'attack_diff', 'midfield_diff', 'defense_diff',
    'position_diff', 'points_diff',
    'h2h_confidence', 'away_advantage', 'season_trend',
    'position_form_home', 'position_form_away', 'strength_balance',
    'B365H', 'B365D', 'B365A',
    'prob_home', 'prob_draw', 'prob_away',
    'prob_home_norm', 'prob_draw_norm', 'prob_away_norm',
]


def get_encoder_feature_matrix(df, feature_list):
    """Retorna a matriz X com apenas as features disponíveis para o encoder."""
    available = [f for f in feature_list if f in df.columns]
    return df[available].fillna(0).values, available


def main():
    """
    Pipeline principal seguindo a metodologia do artigo científico:
    - Dados de treinamento: 2005-2014 (9 temporadas)
    - Dados de teste: 2014-2016 (2 temporadas)
    - [NOVO] Encoder 43 → 16 dims (features latentes)
    - [NOVO] 59 features totais (43 originais + 16 latentes)
    """
    print("="*80)
    print("REPLICA CIENTÍFICA - PREDIÇÃO DE RESULTADOS DA PREMIER LEAGUE")
    print("="*80)
    print("\nMetodologia do Artigo (com Autoencoder):")
    print("  - Dados de Treinamento: 2005-2014 (9 temporadas)")
    print("  - Dados de Teste: 2014-2016 (2 temporadas)")
    print("  - [NOVO] Encoder: 43 features → 16 dimensões latentes")
    print("  - [NOVO] 59 features totais (43 originais + 16 latentes)")
    print("="*80)

    # Caminhos das pastas conforme estrutura do artigo
    train_dir = "data/data_2005_2014"
    test_dir  = "data/data_2014_2016"

    # Verificar se os diretórios existem
    if not os.path.exists(train_dir):
        raise ValueError(f"Diretório de treinamento não encontrado: {train_dir}")
    if not os.path.exists(test_dir):
        raise ValueError(f"Diretório de teste não encontrado: {test_dir}")

    # ── ETAPA 1: Carregar dados ────────────────────────────────────────────────
    print("\n" + "="*80)
    print("ETAPA 1: CARREGAMENTO DOS DADOS")
    print("="*80)
    df_train = load_multiple_seasons(train_dir)
    df_test  = load_multiple_seasons(test_dir)

    # ── ETAPA 2: Engenharia de features ───────────────────────────────────────
    print("\n" + "="*80)
    print("ETAPA 2: ENGENHARIA DE FEATURES")
    print("="*80)
    print("\nCalculando features para dados de TREINAMENTO...")
    features_train = calculate_team_stats(df_train, add_latent=False)

    print("\nCalculando features para dados de TESTE...")
    features_test = calculate_team_stats(df_test, add_latent=False)

    # ── ETAPA 3: Treinar Encoder (43 → 16 dims) ────────────────────────────────
    print("\n" + "="*80)
    print("ETAPA 3: TREINAMENTO DO ENCODER (43 → 16 DIMS)")
    print("="*80)

    X_train_enc, used_features = get_encoder_feature_matrix(features_train, CLASS_B_FEATURES)
    input_dim = X_train_enc.shape[1]
    print(f"\nFeatures de entrada do encoder: {input_dim}")
    print(f"Dimensões latentes: 16")

    encoder, scaler = train_encoder(
        X_train_enc,
        input_dim=input_dim,
        latent_dim=16,
        epochs=100,
        batch_size=32,
        verbose=1,
    )

    # Salvar encoder e scaler
    save_encoder_models(encoder, scaler, path="models")

    # ── ETAPA 4: Extrair features latentes ────────────────────────────────────
    print("\n" + "="*80)
    print("ETAPA 4: EXTRAÇÃO DE 16 DIMENSÕES LATENTES")
    print("="*80)

    X_test_enc, _ = get_encoder_feature_matrix(features_test, used_features)

    print("\nExtraindo features latentes de TREINO...")
    latent_train = extract_latent_features(X_train_enc, encoder, scaler)

    print("Extraindo features latentes de TESTE...")
    latent_test = extract_latent_features(X_test_enc, encoder, scaler)

    # ── ETAPA 5: Combinar 43 originais + 16 latentes = 59 features ───────────
    print("\n" + "="*80)
    print("ETAPA 5: COMBINAÇÃO (43 ORIGINAIS + 16 LATENTES = 59 FEATURES)")
    print("="*80)

    features_combined_train = combine_original_and_latent(features_train, latent_train)
    features_combined_test  = combine_original_and_latent(features_test,  latent_test)

    # Salvar CSVs combinados
    save_combined_features(features_combined_train, "data/features_59combined_train.csv")
    save_combined_features(features_combined_test,  "data/features_59combined_test.csv")

    # ── ETAPA 6: Treinar e avaliar modelos com 59 features ────────────────────
    print("\n" + "="*80)
    print("ETAPA 6: TREINAMENTO E AVALIAÇÃO DOS MODELOS (59 FEATURES)")
    print("="*80)
    train_models(features_combined_train, features_combined_test)

    print("\n" + "="*80)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("  → Encoder salvo:    models/encoder_16dims.keras")
    print("  → Scaler salvo:     models/scaler_43features.pkl")
    print("  → Features treino:  data/features_59combined_train.csv")
    print("  → Features teste:   data/features_59combined_test.csv")
    print("="*80)


if __name__ == "__main__":
    main()
