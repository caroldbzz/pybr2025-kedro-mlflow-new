from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import inference


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline: limpeza → split → treino → avaliação."""
    return pipeline(
        [
            node(
                func=inference,
                inputs=dict(
                    
                ),
                outputs="clean_board_games",
                name="inference",
            ),
        ]
    )
