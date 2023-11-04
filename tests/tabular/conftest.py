import os

import pytest
from hydra import compose, initialize
from optuna.storages import JournalFileStorage, JournalStorage

from theseus.base.utilities.optuna_tuner import OptunaWrapper

MODELS = ["xgboost"]  # , "catboost", 'lightgbm']
TUNER_MODELS = ["xgboost_tune"]  # , "catboost_tune"] #, 'lightgbm_tune']


@pytest.fixture(scope="session", params=MODELS)
def override_config(request):
    with initialize(config_path="configs"):
        config = compose(
            config_name=f"{request.param}",
            overrides=[
                "global.exp_name=pytest_tablr",
                "global.exist_ok=True",
                "global.save_dir=runs",
            ],
        )

    return config


@pytest.fixture(scope="function", params=TUNER_MODELS)
def override_tuner_config(request):
    with initialize(config_path="configs/optuna"):
        config = compose(
            config_name=f"{request.param}",
            overrides=[
                "global.exp_name=pytest_tablr_optuna",
                "global.exist_ok=True",
                "global.save_dir=runs",
            ],
        )

    return config


@pytest.fixture(scope="session")
def override_tuner_tuner():
    os.makedirs("runs/optuna/tablr", exist_ok=True)
    database = JournalStorage(
        JournalFileStorage("runs/optuna/tablr/pytest_tablr_optuna.log")
    )

    tuner = OptunaWrapper(
        storage=database,
        study_name="pytest_tablr_optuna",
        n_trials=3,
        direction="maximize",
        save_dir="runs/optuna/tablr",
    )

    return tuner
