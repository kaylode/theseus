import os

import pytest

from theseus.cv.classification.pipeline import ClassificationPipeline


@pytest.mark.order(1)
def test_train_clf_tune(override_tuner_config, override_tuner_tuner):
    override_tuner_tuner.tune(
        config=override_tuner_config,
        pipeline_class=ClassificationPipeline,
        trial_user_attrs={
            "best_key": "bl_acc",
            "model_name": override_tuner_config["model"]["args"]["model_name"],
        },
    )

    leaderboard_df = override_tuner_tuner.leaderboard()
    os.makedirs("runs/optuna/clf/overview", exist_ok=True)
    # leaderboard_df.to_csv("runs/optuna/clf/overview/leaderboard.csv", index=False)
    leaderboard_df.to_json(
        "runs/optuna/clf/overview/leaderboard.json", orient="records"
    )

    # figs = override_tuner_tuner.visualize("all")
    # for fig_type, fig in figs:
    #     fig.write_image(f"runs/optuna/clf/overview/{fig_type}.png")
