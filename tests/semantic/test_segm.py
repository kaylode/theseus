import pytest

from tests.semantic.inference import TestPipeline
from theseus.cv.semantic.pipeline import SemanticPipeline


@pytest.mark.order(1)
def test_train_clf(override_config):
    train_pipeline = SemanticPipeline(override_config)
    train_pipeline.fit()


@pytest.mark.order(2)
def test_eval_clf(override_config):
    override_config["global"]["resume"] = "runs/pytest_segm/checkpoints/best.ckpt"
    val_pipeline = SemanticPipeline(override_config)
    val_pipeline.evaluate()


@pytest.mark.order(2)
def test_infer_clf(override_test_config):
    override_test_config["global"][
        "pretrained"
    ] = "runs/pytest_segm/checkpoints/best.ckpt"
    test_pipeline = TestPipeline(override_test_config)
    test_pipeline.inference()
