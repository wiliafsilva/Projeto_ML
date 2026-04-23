
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.train_models import train_models
import os
import pandas as pd

try:
    from src.gans import TabularGAN
    GAN_AVAILABLE = True
except Exception:
    GAN_AVAILABLE = False
    TabularGAN = None

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
    
    # ======= Opção: geração de dados sintéticos via GAN tabular =======
    # Disponível apenas se SDV estiver instalado e a flag estiver ligada.
    if os.environ.get("AUGMENT_WITH_GAN", "0").lower() in {"1", "true", "yes", "on"} and GAN_AVAILABLE:
        try:
            print("\n[AUGMENT] Iniciando geração de dados sintéticos com TabularGAN (por Season)...")
            # Definir as colunas de features usadas pelo modelo (excluir 'Result' e 'Season')
            feature_cols = [c for c in features_train.columns if c not in {"Result", "Season"}]
            gan = TabularGAN(feature_cols=feature_cols)
            gan.fit(features_train)

            # Distribuição de Seasons para geração
            season_counts = features_train['Season'].value_counts()
            augment_ratio = float(os.environ.get("GAN_AUGMENT_RATIO", 0.5))
            synthetic_parts = []
            for season, count in season_counts.items():
                n_samples = max(1, int(count * augment_ratio))
                X_synth = gan.generate(n_samples, season=str(season))
                if not X_synth.empty:
                    X_synth = X_synth.copy()
                    X_synth['Season'] = season
                    synthetic_parts.append(X_synth)

            if synthetic_parts:
                df_synth_features = pd.concat(synthetic_parts, ignore_index=True)
                synthetic_path = os.path.join("data", "synthetic_features_gan.csv")
                os.makedirs(os.path.dirname(synthetic_path), exist_ok=True)
                df_synth_features.to_csv(synthetic_path, index=False)
                print(f"[AUGMENT] Dados sintéticos salvos em: {synthetic_path} (linhas={len(df_synth_features)})")
            else:
                print("[AUGMENT] Sem amostras sintéticas geradas para as Seasons encontradas.")
        except Exception as e:
            print(f"[AUGMENT] Aviso: falha na geração de dados sintéticos: {e}")
    
    print("\n" + "="*80)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*80)

if __name__ == "__main__":
    main()
