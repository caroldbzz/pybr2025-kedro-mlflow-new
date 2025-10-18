from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series | pd.DataFrame,
) -> Dict[str, dict]:
    if isinstance(y_test, pd.DataFrame):
        if y_test.shape[1] != 1:
            raise ValueError("y_test deve ter exatamente 1 coluna.")
        y_true = y_test.iloc[:, 0]
    else:
        y_true = y_test

    preds = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_true, preds)),
        "R2": float(r2_score(y_true, preds)),
        "n_test": float(len(y_true)),
    }

    metrics = {k: float(v) for k, v in metrics.items() if np.isfinite(v)}

    # 🔑 esquema esperado pelo MlflowMetricsHistoryDataset
    logged = {name: {"value": val, "step": 0} for name, val in metrics.items()}

    return logged
