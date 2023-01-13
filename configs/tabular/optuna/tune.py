from theseus.opt import Config
from theseus.tabular.classification.pipeline import TabularPipeline
from theseus.theseus.tabular.base.utilities.optuna_tuner import OptunaWrapper

if __name__ == "__main__":
    config = Config("configs/tabular/optuna.yaml")
    tuner = OptunaWrapper()

    tuner.tune(
        config=config,
        pipeline_class=TabularPipeline,
        best_key="bl_acc",
        n_trials=5,
        direction="maximize",
        save_dir="runs/optuna/",
    )
