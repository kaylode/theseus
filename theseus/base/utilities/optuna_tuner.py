import os
import os.path as osp
from copy import deepcopy

import optuna

from theseus.base.pipeline import BasePipeline
from theseus.base.utilities.loggers import LoggerObserver
from theseus.opt import Config

LOGGER = LoggerObserver.getLogger("main")


class OptunaWrapper:
    def __init__(self, storage=None) -> None:
        self.storage = storage

    def tune(
        self,
        config: Config,
        pipeline_class: BasePipeline,
        study_name: str = None,
        best_key: str = None,
        n_trials: int = 100,
        direction: str = "maximize",
        save_dir: str = None,
    ):

        if "optuna" not in config.keys():
            LOGGER.text(
                "Optuna key not found in config. Exit optuna",
                level=LoggerObserver.CRITICAL,
            )
            raise ValueError()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        wrapped_objective = lambda trial: self.objective(
            trial, config, pipeline_class, best_key
        )

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=self.storage,
            load_if_exists=True,
        )

        self.study.optimize(wrapped_objective, n_trials=n_trials)

        # LOGGER.log(
        #     [
        #         {
        #             "tag": "Optuna Optimization History",
        #             "value": optuna.visualization.plot_optimization_history(self.study),
        #             "type": LoggerObserver.FIGURE,
        #             "kwargs": {"step": 0},
        #         }
        #     ]
        # )

        # LOGGER.log(
        #     [
        #         {
        #             "tag": "Optuna Parameter Importances",
        #             "value": optuna.visualization.plot_param_importances(self.study),
        #             "type": LoggerObserver.FIGURE,
        #             "kwargs": {"step": 0},
        #         }
        #     ]
        # )

        best_trial = self.study.best_trial
        self.save_best_config(save_dir, config, best_trial.params)
        return best_trial

    def save_best_config(self, save_dir: str, config: Config, best_params: dict):
        for param_str, param_val in best_params.items():
            here = config
            keys = param_str.split(".")
            for key in keys[:-1]:
                here = here.setdefault(key, {})
            here[keys[-1]] = param_val
        config.save_yaml(osp.join(save_dir, "best_pipeline.yaml"))
        LOGGER.text(
            f"Best configuration saved at {save_dir}", level=LoggerObserver.INFO
        )

    def _override_dict_with_optuna(
        self, trial, config: Config, param_str: str, variable_type: str
    ):
        """
        Override config with optuna suggested params
        """
        # Start off pointing at the original dictionary that was passed in.
        here = config

        # Turn the string of key names into a list of strings.
        keys = param_str.split(".")

        # For every key *before* the last one, we concentrate on navigating through the dictionary.
        for key in keys[:-1]:
            # Try to find here[key]. If it doesn't exist, create it with an empty dictionary. Then,
            # update our `here` pointer to refer to the thing we just found (or created).
            here = here.setdefault(key, {})

        # Finally, set the final key to the given value
        old_value = here[keys[-1]]

        if variable_type == "int":
            low_value, high_value = old_value
            here[keys[-1]] = trial.suggest_int(param_str, low_value, high_value)
        elif variable_type == "loguniform":
            low_value, high_value = old_value
            here[keys[-1]] = trial.suggest_loguniform(param_str, low_value, high_value)

        elif variable_type == "float":
            low_value, high_value = old_value
            here[keys[-1]] = trial.suggest_float(param_str, low_value, high_value)

        elif variable_type == "categorical":
            here[keys[-1]] = trial.suggest_categorical(param_str, old_value)

        else:
            LOGGER.text(
                f"{variable_type} is not supported by Optuna",
                level=LoggerObserver.ERROR,
            )
            raise ValueError()

    def objective(
        self, trial, config: Config, pipeline_class: BasePipeline, best_key: str = None
    ):
        """Define the objective function"""

        tmp_config = deepcopy(config)
        optuna_params = tmp_config["optuna"]
        for variable_type in optuna_params.keys():
            for param_str in optuna_params[variable_type]:
                self._override_dict_with_optuna(
                    trial, tmp_config, param_str, variable_type
                )
        pipeline = pipeline_class(tmp_config)
        pipeline.fit()
        score_dict = pipeline.evaluate()
        del tmp_config
        if best_key is not None:
            return score_dict[best_key]
        return score_dict
