from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=predict,
                inputs=dict(
                    model="inference_model",
                    X_infer="X_test",
                    output_column="params:inference.output_column",
                ),
                outputs="y_pred",
                name="predict",
            ),
        ]
    )
