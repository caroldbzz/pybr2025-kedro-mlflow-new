from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data, split_data

def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline: limpeza → split dos dados."""
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=dict(
                    df="raw_board_games",
                    features="params:features",
                    target="params:feature_target",
                ),
                outputs="clean_board_games",
                name="clean_data",
            ),
            node(
                func=split_data,
                inputs=dict(
                    df="clean_board_games",
                    target="params:feature_target",
                    test_size="params:split.test_size",
                    random_state="params:split.random_state",
                ),
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            )
        ]
    )
