from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline: limpeza → split → treino → avaliação."""
    return pipeline(
        [
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model",
            )
        ]
    )
