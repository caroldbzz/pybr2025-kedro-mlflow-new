from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression

def _make_model(model_type: str) -> RegressorMixin:
    """Cria e retorna o estimador de regressão suportado."""
    if model_type == "LinearRegression":
        return LinearRegression()
    raise ValueError(f"model_type '{model_type}' não suportado.")

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
