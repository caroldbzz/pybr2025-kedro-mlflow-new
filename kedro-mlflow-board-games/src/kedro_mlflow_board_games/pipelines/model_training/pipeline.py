from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_regressor


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline: limpeza → split → treino → avaliação."""
    return pipeline(
        [
            node(
                func=train_regressor,
                inputs=dict(
                    X_train="X_train",
                    y_train="y_train",
                    model_type="params:model.type",
                ),
                outputs="regressor",
                name="train_regressor",
            )
        ]
    )
