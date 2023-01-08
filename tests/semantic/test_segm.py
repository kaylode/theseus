import pytest

from configs.semantic.infer import TestPipeline
from theseus.cv.semantic.pipeline import SemanticPipeline


@pytest.mark.order(1)
def test_train_clf(override_config):
    train_pipeline = SemanticPipeline(override_config)
    train_pipeline.fit()


@pytest.mark.order(2)
def test_eval_clf(override_config):
    override_config["global"]["pretrained"] = "runs/pytest_segm/checkpoints/best.pth"
    override_config["global"]["cfg_transform"] = "runs/pytest_segm/transform.yaml"
    val_pipeline = SemanticPipeline(override_config)
    val_pipeline.evaluate()


@pytest.mark.order(2)
def test_infer_clf(override_test_config):
    override_test_config["global"]["weights"] = "runs/pytest_segm/checkpoints/best.pth"
    override_test_config["global"]["cfg_transform"] = "runs/pytest_segm/transform.yaml"
    test_pipeline = TestPipeline(override_test_config)
    test_pipeline.inference()
