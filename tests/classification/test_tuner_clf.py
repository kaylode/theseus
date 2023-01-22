import optuna
import pytest

from theseus.base.utilities.optuna_tuner import OptunaWrapper
from theseus.cv.classification.pipeline import ClassificationPipeline


@pytest.mark.order(1)
def test_train_clf_tune(override_tuner_config):
    tuner = OptunaWrapper()
    tuner.tune(
        config=override_tuner_config,
        pipeline_class=ClassificationPipeline,
        best_key="bl_acc",
        n_trials=2,
        direction="maximize",
        save_dir="runs/optuna/",
        sampler=optuna.samplers.RandomSampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
