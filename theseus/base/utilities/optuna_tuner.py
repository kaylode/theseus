import os
import os.path as osp
from copy import deepcopy

import optuna
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)

from theseus.base.callbacks.optuna_callbacks import OptunaCallbacks
from theseus.base.pipeline import BasePipeline
from theseus.base.utilities.loggers import LoggerObserver
from omegaconf import DictConfig


class OptunaWrapper:
    def __init__(
        self,
        storage=None,
        study_name: str = None,
        n_trials: int = 100,
        direction: str = "maximize",
        pruner=None,
        sampler=None,
        save_dir: str = None,
    ) -> None:

        self.logger = LoggerObserver.getLogger("main")
        self.storage = storage
        self.study_name = study_name
        self.n_trials = n_trials
        self.direction = direction
        self.pruner = pruner
        self.sampler = sampler
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=self.storage,
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler,
        )

    def tune(
        self,
        config: DictConfig,
        pipeline_class: BasePipeline,
        trial_user_attrs: dict = {},
    ):

        if "optuna" not in config.keys():
            self.logger.text(
                "Optuna key not found in config. Exit optuna",
                level=LoggerObserver.CRITICAL,
            )
            raise ValueError()

        wrapped_objective = lambda trial: self.objective(
            trial, config, pipeline_class, trial_user_attrs
        )

        self.study.optimize(wrapped_objective, n_trials=self.n_trials)
        best_trial = self.study.best_trial
        self.save_best_config(self.save_dir, config, best_trial.params)
        self._rename_params()
        return best_trial

    def save_best_config(self, save_dir: str, config: DictConfig, best_params: dict):
        for param_str, param_val in best_params.items():
            here = config
            keys = param_str.split(".")
            for key in keys[:-1]:
                here = here.setdefault(key, {})
            here[keys[-1]] = param_val
        save_dir = osp.join(save_dir, "best_configs")
        os.makedirs(save_dir, exist_ok=True)
        config.save_yaml(osp.join(save_dir, "best_pipeline.yaml"))
        self.logger.text(
            f"Best configuration saved at {save_dir}", level=LoggerObserver.INFO
        )

    def _override_dict_with_optuna(
        self, trial, config: DictConfig, param_str: str, variable_type: str
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
            self.logger.text(
                f"{variable_type} is not supported by Optuna",
                level=LoggerObserver.ERROR,
            )
            raise ValueError()

    def objective(
        self,
        trial: optuna.Trial,
        config: DictConfig,
        pipeline_class: BasePipeline,
        trial_user_attrs: dict = {},
    ):
        """Define the objective function"""

        # Override config with optuna trials values
        tmp_config = deepcopy(config)
        optuna_params = tmp_config["optuna"]
        for variable_type in optuna_params.keys():
            for param_str in optuna_params[variable_type]:
                self._override_dict_with_optuna(
                    trial, tmp_config, param_str, variable_type
                )

        # Set fixed run's config
        for key, value in trial_user_attrs.items():
            trial.set_user_attr(key, value)

        if tmp_config["global"]["exp_name"] is not None:
            tmp_config["global"]["exp_name"] += f"_{trial.number}"
        tmp_config["global"]["save_dir"] = self.save_dir

        # Hook a callback inside pipeline
        pipeline = pipeline_class(tmp_config)
        pipeline.init_trainer = self.callback_hook(
            trial=trial, init_trainer_function=pipeline.init_trainer
        )

        # Start training and evaluation
        pipeline.fit()
        score_dict = pipeline.evaluate()
        del tmp_config

        best_key = trial_user_attrs.get("best_key", None)
        if best_key is not None:
            return score_dict[best_key]
        return score_dict

    def callback_hook(self, trial, init_trainer_function):
        callback = OptunaCallbacks(trial=trial)

        def hook_optuna_callback(callbacks):
            callbacks.append(callback)
            init_trainer_function(callbacks)

        return hook_optuna_callback

    def _rename_params(self):
        trials = self.study.get_trials(deepcopy=False)
        for trial in trials:
            trial_param_names = list(trial.params.keys())
            if len(trial_param_names) == 1:
                break
            common_prefix = osp.commonprefix(trial_param_names)
            if common_prefix != "":
                for trial_param_name in trial_param_names:
                    new_param_name = trial_param_name.replace(common_prefix, "")
                    trial.params.update(
                        {new_param_name: trial.params[trial_param_name]}
                    )
                    trial.distributions.update(
                        {new_param_name: trial.distributions[trial_param_name]}
                    )
                    del trial.params[trial_param_name]
                    del trial.distributions[trial_param_name]

    def leaderboard(self):
        """Print leaderboard of all trials"""
        df = self.study.trials_dataframe()
        df.columns = [col.replace("user_attrs_", "") for col in df.columns]
        return df

    def visualize(self, plot: str, plot_params: dict = {}):
        """Visualize everything"""

        allow_plot_types = [
            "history",
            "contour",
            "edf",
            "intermediate_values",
            "parallel_coordinate",
            "param_importances",
            "slice",
        ]
        assert plot in ["all", *allow_plot_types], f"{plot} is not supported by Optuna"

        if plot == "history":
            fig = plot_optimization_history(self.study, **plot_params)
        elif plot == "contour":
            fig = plot_contour(self.study, **plot_params)
        elif plot == "edf":
            fig = plot_edf(self.study, **plot_params)
        elif plot == "intermediate_values":
            fig = plot_intermediate_values(self.study, **plot_params)
        elif plot == "parallel_coordinate":
            fig = plot_parallel_coordinate(self.study, **plot_params)
        elif plot == "param_importances":
            fig = plot_param_importances(self.study, **plot_params)
        elif plot == "slice":
            fig = plot_slice(self.study, **plot_params)
        elif plot == "all":
            fig = []
            for plot_type in allow_plot_types:
                one_fig = self.visualize(plot_type, plot_params)
                fig.append((plot_type, one_fig))
        else:
            self.logger.text(
                f"{plot} is not supported by Optuna", level=LoggerObserver.ERROR
            )
            raise ValueError()

        return fig
