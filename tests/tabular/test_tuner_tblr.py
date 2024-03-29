import os

import pytest

from theseus.ml.callbacks.optuna_callbacks import OptunaCallbacks
from theseus.ml.pipeline import MLPipeline


@pytest.mark.order(1)
def test_train_tblr_tune(override_tuner_config, override_tuner_tuner):
    override_tuner_tuner.tune(
        config=override_tuner_config,
        pipeline_class=MLPipeline,
        optuna_callback=OptunaCallbacks,
        trial_user_attrs={
            "best_key": "bl_acc",
            "model_name": override_tuner_config["model"]["args"]["model_name"],
        },
    )

    leaderboard_df = override_tuner_tuner.leaderboard()
    os.makedirs("runs/optuna/tablr/overview", exist_ok=True)
    leaderboard_df.to_json(
        "runs/optuna/tablr/overview/leaderboard.json", orient="records"
    )

    figs = override_tuner_tuner.visualize("all")
    for fig_type, fig in figs:
        fig.write_image(f"runs/optuna/tablr/overview/{fig_type}.png")
