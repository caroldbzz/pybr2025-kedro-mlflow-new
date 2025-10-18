from __future__ import annotations

from typing import Dict, Sequence, Tuple

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ========= Helpers internos =========

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas: minúsculas, trim e '_' no lugar de espaços."""
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _ensure_columns_exist(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Valida se todas as colunas exigidas existem; lança erro informativo se faltar algo."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        preview = ", ".join(df.columns[:10])
        raise ValueError(
            f"Colunas ausentes: {missing}. Verifique nomes após normalização. "
            f"Algumas colunas disponíveis: {preview}..."
        )


def _make_model(model_type: str) -> RegressorMixin:
    """Cria e retorna o estimador de regressão suportado."""
    if model_type == "LinearRegression":
        return LinearRegression()
    raise ValueError(f"model_type '{model_type}' não suportado.")


# ========= Nodes (expostos ao Kedro) =========

def clean_data(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
) -> pd.DataFrame:
    """
    Limpa e reduz o dataframe às colunas relevantes (features + target).

    - Normaliza nomes de colunas.
    - Valida presença de target e, ao menos, uma feature.
    - Seleciona apenas features + target (nessa ordem).
    - Remove linhas com target nulo.

    Parameters
    ----------
    df : pd.DataFrame
        Dados brutos.
    features : Sequence[str]
        Lista de nomes de features (já no formato normalizado).
    target : str
        Nome da coluna alvo (normalizado).

    Returns
    -------
    pd.DataFrame
        DataFrame contendo apenas features + target, sem nulos no target.
    """
    df = _normalize_columns(df)

    # Garantir existência de colunas
    all_required = list(features) + [target]
    _ensure_columns_exist(df, all_required)

    # Seleção e limpeza mínima
    df = df[all_required].copy()
    df = df.dropna(subset=[target])

    return df



def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separa em treino e teste e garante que y_* sejam DataFrames
    (necessário para salvar via ParquetDataSet).

    Parameters
    ----------
    df : pd.DataFrame
        Dados limpos contendo features + target.
    target : str
        Nome da coluna alvo.
    test_size : float
        Proporção do conjunto de teste (0 < test_size < 1).
    random_state : int
        Semente para reprodutibilidade.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        X_train, X_test, y_train (DF), y_test (DF)
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    y_train = y_train.to_frame(name=target)
    y_test = y_test.to_frame(name=target)

    return X_train, X_test, y_train, y_test


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "LinearRegression",
) -> RegressorMixin:
    """
    Treina o regressor especificado em `model_type`.
    """
    model = _make_model(model_type)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Calcula métricas MAE e R² no conjunto de teste.
    """
    preds = model.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)),
        "n_test": float(len(y_test)),
    }
