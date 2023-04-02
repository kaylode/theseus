import os

import optuna
import pytest
from optuna.storages import JournalFileStorage, JournalStorage

from theseus.base.utilities.optuna_tuner import OptunaWrapper
from theseus.opt import Config


@pytest.fixture(scope="session")
def override_config():
    config = Config("./configs/classification/pipeline.yaml")
    config["global"]["exp_name"] = "pytest_clf"
    config["global"]["exist_ok"] = True
    config["global"]["save_dir"] = "runs"
    config["global"]["device"] = "cpu"
    config["trainer"]["args"]["use_fp16"] = False
    config["trainer"]["args"]["num_iterations"] = 10
    config["data"]["dataloader"]["train"]["args"]["batch_size"] = 1
    config["data"]["dataloader"]["val"]["args"]["batch_size"] = 1
    return config


@pytest.fixture(scope="session")
def override_test_config():
    config = Config("./configs/classification/test.yaml")
    config["global"]["exp_name"] = "pytest_clf"
    config["global"]["exist_ok"] = True
    config["global"]["save_dir"] = "runs"
    config["global"]["device"] = "cpu"
    config["data"]["dataloader"]["args"]["batch_size"] = 1
    return config


@pytest.fixture(scope="session")
def override_tuner_config():
    config = Config(f"./configs/classification/optuna/pipeline.yaml")
    config["global"]["exp_name"] = "pytest_clf_optuna"
    config["global"]["exist_ok"] = True
    config["global"]["save_dir"] = "runs"
    config["global"]["device"] = "cpu"
    return config


@pytest.fixture(scope="session")
def override_tuner_tuner():

    os.makedirs("runs/optuna/clf", exist_ok=True)
    database = JournalStorage(
        JournalFileStorage("runs/optuna/clf/pytest_clf_optuna.log")
    )

    tuner = OptunaWrapper(
        storage=database,
        study_name="pytest_clf_optuna",
        n_trials=2,
        direction="maximize",
        save_dir="runs/optuna/clf/",
        sampler=optuna.samplers.RandomSampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    return tuner
