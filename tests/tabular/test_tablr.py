import pytest

# from configs.tabular.infer import TestPipeline
from theseus.tabular.classification.pipeline import TabularPipeline


@pytest.mark.order(1)
def test_train_tblr(override_config):
    train_pipeline = TabularPipeline(override_config)
    train_pipeline.fit()


@pytest.mark.order(2)
def test_eval_tblr(override_config):
    override_config["global"]["pretrained"] = "runs/pytest_tablr/checkpoints/last"
    val_pipeline = TabularPipeline(override_config)
    val_pipeline.evaluate()


# @pytest.mark.order(2)
# def test_infer_clf(override_test_config):
#     override_test_config["global"]["weights"] = "runs/pytest_segm/checkpoints/best.pth"
#     test_pipeline = TestPipeline(override_test_config)
#     test_pipeline.inference()
