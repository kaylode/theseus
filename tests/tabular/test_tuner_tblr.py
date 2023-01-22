import pytest

from theseus.base.utilities.optuna_tuner import OptunaWrapper

# from configs.tabular.infer import TestPipeline
from theseus.tabular.classification.pipeline import TabularPipeline


@pytest.mark.order(1)
def test_train_tblr_tune(override_tuner_config):
    tuner = OptunaWrapper()
    tuner.tune(
        config=override_tuner_config,
        pipeline_class=TabularPipeline,
        best_key="bl_acc",
        n_trials=2,
        direction="maximize",
        save_dir="runs/optuna/",
    )
