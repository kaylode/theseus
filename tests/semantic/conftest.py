import pytest
from hydra import compose, initialize


@pytest.fixture(scope="session")
def override_config():
    with initialize(config_path="configs"):
        config = compose(
            config_name="pipeline",
            overrides=[
                "global.exp_name=pytest_segm",
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
                "global.exp_name=pytest_segm",
                "global.exist_ok=True",
                "global.save_dir=runs",
                "trainer.args.precision=32",
                "trainer.args.accelerator=cpu",
                "trainer.args.devices=1",
                "data.dataloader.args.batch_size=1",
            ],
        )

    return config
