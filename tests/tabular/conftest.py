import pytest

from theseus.opt import Config

MODELS = ["xgboost"]  # , "catboost", 'lightgbm']
TUNER_MODELS = ["xgboost_tune"]  # , "catboost_tune"]


@pytest.fixture(scope="session", params=MODELS)
def override_config(request):
    config = Config(f"./configs/tabular/{request.param}.yaml")
    config["global"]["exp_name"] = "pytest_tablr"
    config["global"]["exist_ok"] = True
    config["global"]["save_dir"] = "runs"
    config["global"]["device"] = "cpu"
    return config


@pytest.fixture(scope="session", params=TUNER_MODELS)
def override_tuner_config(request):
    config = Config(f"./configs/tabular/optuna/{request.param}.yaml")
    config["global"]["exp_name"] = "pytest_tablr_optuna"
    config["global"]["exist_ok"] = True
    config["global"]["save_dir"] = "runs"
    config["global"]["device"] = "cpu"
    return config
