from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


def predict(
    model: RegressorMixin,
    X_infer: pd.DataFrame,
    output_column: str | None = None,
) -> pd.DataFrame:
    """
    Gera predições a partir de um modelo scikit-learn.
    """
    preds = model.predict(X_infer)

    if isinstance(preds, pd.DataFrame):
        out = preds.copy()
        if output_column:
            if out.shape[1] == 1:
                out.columns = [output_column]
            else:
                out.columns = [f"{output_column}_{i}" for i in range(out.shape[1])]
        out.index = X_infer.index
        return out

    if isinstance(preds, pd.Series):
        col = output_column or "prediction"
        out = preds.to_frame(name=col)
        out.index = X_infer.index
        return out

    # Converte para ndarray e trata formas (n,), (n,1) e (n,k)
    arr = np.asarray(preds)
    col = output_column or "prediction"

    if arr.ndim == 1:
        out = pd.DataFrame({col: arr}, index=X_infer.index)
    elif arr.ndim == 2:
        if arr.shape[1] == 1:
            out = pd.DataFrame({col: arr.ravel()}, index=X_infer.index)
        else:
            cols = [f"{col}_{i}" for i in range(arr.shape[1])]
            out = pd.DataFrame(arr, columns=cols, index=X_infer.index)
    else:
        raise ValueError(
            f"Forma de predição não suportada: {arr.shape} (ndim={arr.ndim})."
        )

    return out
