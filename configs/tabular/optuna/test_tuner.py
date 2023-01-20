import pytest

from theseus.base.utilities.optuna_tuner import DETRCustomBackbone

# from configs.tabular.infer import TestPipeline
from theseus.tabular.classification.pipeline import TabularPipeline


@pytest.mark.order(1)
def test_train_clf_tune(override_tuner_config):
    tuner = DETRCustomBackbone()
    tuner.tune(
        config=override_tuner_config,
        pipeline_class=TabularPipeline,
        best_key="bl_acc",
        n_trials=2,
        direction="maximize",
        save_dir="runs/optuna/",
    )


# @pytest.mark.order(2)
# def test_eval_clf(override_config):
#     override_config["global"]["pretrained"] = "runs/pytest_tablr/checkpoints/best.pth"
#     val_pipeline = TabularPipeline(override_config)
#     val_pipeline.evaluate()


# @pytest.mark.order(2)
# def test_infer_clf(override_test_config):
#     override_test_config["global"]["weights"] = "runs/pytest_segm/checkpoints/best.pth"
#     test_pipeline = TestPipeline(override_test_config)
#     test_pipeline.inference()
