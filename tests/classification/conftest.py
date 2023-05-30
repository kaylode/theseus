import os

import optuna
import pytest
from optuna.storages import JournalFileStorage, JournalStorage

from theseus.base.utilities.optuna_tuner import OptunaWrapper
from omegaconf import OmegaConf
from hydra import compose, initialize, initialize_config_module

@pytest.fixture(scope="session")
def override_config():
    with initialize(config_path="configs"):
        config = compose(
            config_name="pipeline",
            overrides=[
                "global.exp_name=pytest_clf",
                "global.exist_ok=True",
                "global.save_dir=runs",
                "trainer.args.max_epochs=5",
                "trainer.args.precision=32",
                "trainer.args.accelerator=cpu",
                "trainer.args.devices=1",
                "data.dataloader.train.args.batch_size=1",
                "data.dataloader.val.args.batch_size=1",
            ],
        )
    
    return config


@pytest.fixture(scope="session")
def override_test_config():
    with initialize(config_path="configs"):
        config = compose(
            config_name="test",
            overrides=[
                "global.exp_name=pytest_clf",
                "global.exist_ok=True",
                "global.save_dir=runs",
                "trainer.args.precision=32",
                "trainer.args.accelerator=cpu",
                "trainer.args.devices=1",
                "data.dataloader.args.batch_size=1",
            ],
        )

    return config


@pytest.fixture(scope="session")
def override_tuner_config():

    with initialize(config_path="configs"):
        config = compose(
            config_name="optuna",
            overrides=[
                "global.exp_name=pytest_clf",
                "global.exist_ok=True",
                "global.save_dir=runs",
                "trainer.args.precision=32",
                "trainer.args.accelerator=cpu",
                "trainer.args.devices=1",
            ],
        )

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
