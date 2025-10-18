# src/kedro_mlflow_board_games/pipeline_registry.py
from __future__ import annotations
from kedro.pipeline import Pipeline
from kedro_mlflow_board_games.pipelines.pre_processing import create_pipeline as create_pre_processing
from kedro_mlflow_board_games.pipelines.model_training import create_pipeline as create_model_training
from kedro_mlflow_board_games.pipelines.evaluation import create_pipeline as create_evaluation

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "train": create_model_training(),
        "pre_processing": create_pre_processing(),
        "evaluation": create_evaluation(),
        "__default__": create_pre_processing() + create_model_training() + create_evaluation(),
    }
