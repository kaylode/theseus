import os

import pytest
from optuna.storages import JournalFileStorage, JournalStorage

from theseus.base.utilities.optuna_tuner import OptunaWrapper
from theseus.opt import Config

MODELS = ["xgboost"]  # , "catboost", 'lightgbm']
TUNER_MODELS = ["xgboost_tune"]  # , "catboost_tune"] #, 'lightgbm_tune']


@pytest.fixture(scope="session", params=MODELS)
def override_config(request):
    config = Config(f"./configs/tabular/{request.param}.yaml")
    config["global"]["exp_name"] = "pytest_tablr"
    config["global"]["exist_ok"] = True
    config["global"]["save_dir"] = "runs"
    config["global"]["device"] = "cpu"
    return config


@pytest.fixture(scope="function", params=TUNER_MODELS)
def override_tuner_config(request):
    config = Config(f"./configs/tabular/optuna/{request.param}.yaml")
    config["global"]["exp_name"] = "pytest_tablr_optuna"
    config["global"]["exist_ok"] = True
    config["global"]["save_dir"] = "runs"
    config["global"]["device"] = "cpu"
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
