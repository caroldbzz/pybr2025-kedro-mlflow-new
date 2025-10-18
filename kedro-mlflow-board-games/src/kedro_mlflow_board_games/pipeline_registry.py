# src/kedro_mlflow_board_games/pipeline_registry.py
from __future__ import annotations
from kedro.pipeline import Pipeline
from kedro_mlflow_board_games.pipelines.model import create_pipeline as create_model_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    model = create_model_pipeline()
    return {
        "model": model,
        "__default__": model,
    }
