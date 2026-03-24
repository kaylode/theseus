import os

import pytest

from theseus.ml.tradml import TradMLTuner


@pytest.mark.order(3)
def test_tuner_xgboost(titanic_data):
    """Run TradMLTuner with XGBoost on Titanic data (2 trials, no wandb)."""
    save_dir = "runs/optuna/tablr_test"
    os.makedirs(save_dir, exist_ok=True)
    storage_path = os.path.join(save_dir, "test_tuner.log")

    tuner = TradMLTuner(
        storage=storage_path,
        study_name="pytest_xgboost_tune",
        n_trials=2,
        direction="maximize",
        save_dir=save_dir,
        method="xgboost",
        wandb_kwargs=None,
        feature_names=titanic_data["feature_names"],
        classnames=titanic_data["classnames"],
    )

    best_model = tuner.tune(
        X_train=titanic_data["X_train"],
        X_val=titanic_data["X_val"],
        y_train=titanic_data["y_train"],
        y_val=titanic_data["y_val"],
        is_classification=True,
    )

    assert best_model is not None

    # Check leaderboard
    df = tuner.leaderboard()
    assert len(df) >= 2
    print(f"Tuner leaderboard:\n{df}")

    # Check best config was saved
    assert os.path.exists(os.path.join(save_dir, "best_config.json"))
