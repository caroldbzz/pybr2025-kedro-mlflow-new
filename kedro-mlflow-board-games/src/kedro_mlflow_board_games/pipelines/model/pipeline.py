from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data, split_data, train_regressor, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline: limpeza → split → treino → avaliação."""
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
            ),
            node(
                func=train_regressor,
                inputs=dict(
                    X_train="X_train",
                    y_train="y_train",
                    model_type="params:model.type",
                ),
                outputs="regressor",
                name="train_regressor",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model",
            ),
        ]
    )
