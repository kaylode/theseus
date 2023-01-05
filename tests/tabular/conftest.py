import pytest

from theseus.opt import Config

MODELS = ["xgboost", "catboost"]


@pytest.fixture(scope="session", params=MODELS)
def override_config(request):
    config = Config(f"./configs/tabular/{request.param}.yaml")
    config["global"]["exp_name"] = "pytest_tablr"
    config["global"]["exist_ok"] = True
    config["global"]["save_dir"] = "runs"
    config["global"]["device"] = "cpu"
    return config
