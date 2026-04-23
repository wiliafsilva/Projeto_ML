"""GANS tabulares simples para dados de engenharia de features.

Este módulo oferece um wrapper leve em torno do CTGAN (SDV) para treinar
modelos por temporada (Season) e gerar amostras sintéticas de features
tabulares. A abordagem por temporada evita dependência direta de condições
no modelo, mantendo a compatibilidade com datasets puramente numéricos.

Observações:
- Requer a dependência 'sdv' (SDV) para CTGAN. Caso não esteja disponível,
  a importação falhará e o usuário será avisado no momento da chamada.
- O gerador é treinado separadamente para cada Season existente no conjunto de
  dados. A geração pode ser feita porSeason ou para todas as Seasons disponíveis.
- Este wrapper gera apenas as features (sem 'Result' nem 'Season'); a agregação
  de rótulos (labels) deve ser feita pelo usuário se for aplicar augmentação de
  treino supervisionado.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


class TabularGAN:
    def __init__(self, feature_cols: Optional[List[str]] = None, random_state: Optional[int] = 42):
        # Colunas de features a serem usadas pelo GAN (exclui 'Result' e 'Season')
        self.feature_cols: List[str] = feature_cols or []
        self.random_state = random_state
        # Mapeamento de Season -> CTGAN model
        self.models_by_season: Dict[str, object] = {}
        self.seasons: List[str] = []
        self._built = False

    def _ensure_sdv_available(self):
        try:
            from sdv.tabular import CTGAN  # type: ignore
        except Exception as e:
            raise ImportError(
                "SDV (sdv) não está instalado ou não acessível. Instale com: pip install sdv"
            ) from e
        return CTGAN  # type: ignore

    def fit(self, df: pd.DataFrame) -> None:
        """Treina CTGANs separados por Season a partir de df.

        df: DataFrame contendo pelo menos as colunas de features (definidas em
            self.feature_cols) e a coluna 'Season' para particionar os dados.
            Opcionalmente pode conter 'Result', que não será usado pelo GAN.
        """
        CTGAN = self._ensure_sdv_available()

        if not self.feature_cols:
            # Inferir a partir do DataFrame, exceto 'Season' e 'Result'
            self.feature_cols = [c for c in df.columns if c not in {"Season", "Result"}]

        # Garantir que todas as colunas existem
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas de features não encontradas no DataFrame: {missing}")

        # Agrupar por Season e treinar um CTGAN por temporada
        self.models_by_season = {}
        self.seasons = sorted(df['Season'].dropna().unique())

        for season in self.seasons:
            subdf = df[df['Season'] == season]

            # Dados de treino para esta temporada
            X = subdf[self.feature_cols].copy()
            if X.empty:
                continue

            # Construir esquema simples: todas as features são numéricas
            schema = {col: {'type': 'numerical'} for col in self.feature_cols}

            model = CTGAN(epochs=300, batch_size=512, random_state=self.random_state, verbose=False)
            # Treina o CTGAN com o dataset desta temporada
            model.fit(X, schema=schema)  # type: ignore[arg-type]
            self.models_by_season[str(season)] = model

        self._built = True

    def generate(self, n_samples: int, season: Optional[str] = None) -> pd.DataFrame:
        """Gera amostras sintéticas.

        Se season for informado, gera amostras apenas para aquela temporada.
        Caso contrário, gera amostras igualmente distribuídas entre as temporadas
        disponíveis. Retorna um DataFrame com as mesmas colunas de features (sem
        'Season' nem 'Result').
        """
        if not self._built:
            raise RuntimeError("TabularGAN não foi treinado. Chame fit(df) antes de generate().")

        import numpy as np

        if season is not None:
            model = self.models_by_season.get(str(season))
            if model is None:
                raise ValueError(f"Season não encontrada ou não treinada: {season}")
            samples = model.sample(n_samples)  # type: ignore[attr-defined]
            return samples[self.feature_cols].copy()

        # Gera por temporada igualmente distribuídas
        results: List[pd.DataFrame] = []
        if not self.seasons:
            return pd.DataFrame(columns=self.feature_cols)

        per_season = max(1, int(np.ceil(n_samples / len(self.seasons))))
        for s in self.seasons:
            model = self.models_by_season.get(str(s))
            if model is None:
                continue
            samples = model.sample(min(per_season, n_samples))  # type: ignore[attr-defined]
            results.append(samples[self.feature_cols].copy())

        if not results:
            return pd.DataFrame(columns=self.feature_cols)

        return pd.concat(results, ignore_index=True).head(n_samples)

    def save(self, path: str) -> None:
        """Salva o estado do(s) modelo(s) em disco."""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "TabularGAN":
        import joblib
        return joblib.load(path)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"TabularGAN(seasons={self.seasons}, features={self.feature_cols})"
