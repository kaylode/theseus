import pytest

from tests.classification.inference import TestPipeline
from theseus.cv.classification.pipeline import ClassificationPipeline


@pytest.mark.order(1)
def test_train_clf(override_config):
    train_pipeline = ClassificationPipeline(override_config)
    train_pipeline.fit()


@pytest.mark.order(2)
def test_eval_clf(override_config):
    override_config["global"]["resume"] = "runs/pytest_clf/checkpoints/best.ckpt"
    val_pipeline = ClassificationPipeline(override_config)
    val_pipeline.evaluate()


@pytest.mark.order(2)
def test_infer_clf(override_test_config):
    override_test_config["global"]["weights"] = "runs/pytest_clf/checkpoints/best.ckpt"
    test_pipeline = TestPipeline(override_test_config)
    test_pipeline.inference()
