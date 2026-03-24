import json
import os
import os.path as osp

import lightgbm
import numpy as np
import optuna
import wandb
import xgboost as xgb
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from theseus import LoggerObserver
from theseus.ml.visualize import visualize_shap

LOGGER = LoggerObserver.getLogger("main")

# wandb might cause an error without this.
os.environ["WANDB_START_METHOD"] = "thread"

__all__ = [
    "fit_xgboost",
    "fit_catboost",
    "fit_lightgbm",
    "fit_rf",
    "fit_svm",
    "fit_logistics",
    "fit_adaboost",
    "fit_knn",
    "fit_mlp",
    "objective",
    "TradMLTuner",
]


def fit_xgboost(params, X_train, X_val, y_train, y_val, is_classification=True):
    if is_classification:
        num_classes = len(np.unique(y_train))
        objective_fn = "multi:softprob" if num_classes > 2 else "binary:logistic"
        params.update(
            {"eval_metric": ["auc"], "objective": objective_fn, "early_stopping_rounds": 100}
        )
        model = xgb.XGBClassifier(**params)
    else:
        params.update(
            {
                "eval_metric": ["rmse"],
                "objective": "reg:squarederror",
            }
        )
        model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    return model, params


def fit_catboost(
    params, X_train, X_val, y_train, y_val, is_classification: bool = True, cat_features=None
):
    from catboost import CatBoostClassifier, CatBoostRegressor

    params.update(
        {
            "verbose": 100,
        }
    )
    if not is_classification:
        params.update(
            {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
            }
        )
        clf = CatBoostRegressor(**params)
    else:
        clf = CatBoostClassifier(**params)

    clf.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], cat_features=None, early_stopping_rounds=50
    )
    return clf, params


def fit_lightgbm(
    params, X_train, X_val, y_train, y_val, is_classification: bool = True, cat_features=None
):
    from lightgbm import LGBMClassifier, LGBMRegressor

    params.update({"verbose": 100})

    if not is_classification:
        params.update(
            {
                "objective": "regression",
                "metric": "rmse",
            }
        )
        clf = LGBMRegressor(**params)
    else:
        clf = LGBMClassifier(**params)

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_features,
        callbacks=[
            lightgbm.early_stopping(stopping_rounds=50, verbose=False),
            lightgbm.log_evaluation(period=2000),
        ],
    )
    return clf, params


def fit_rf(params, X_train, X_val, y_train, y_val, is_classification: bool = True):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    params.update({"verbose": 1})
    if not is_classification:
        params.update({"n_jobs": -1, "criterion": "squared_error"})
        clf = RandomForestRegressor(**params)
    else:
        clf = RandomForestClassifier(**params)
    clf.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))
    return clf, params


def fit_svm(params, X_train, X_val, y_train, y_val):
    from sklearn.svm import SVC

    params.update({"kernel": "linear", "probability": True, "verbose": True})
    clf = SVC(**params)
    clf.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))
    return clf, params


def fit_logistics(params, X_train, X_val, y_train, y_val, is_classification=True):
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if is_classification:
        params.update(
            {
                "verbose": 1,
                "penalty": "l2",
            }
        )
        clf = LogisticRegression(**params)
    else:
        params.pop("C", None)
        clf = LinearRegression(**params)
    clf.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))
    return clf, params


def fit_adaboost(params, X_train, X_val, y_train, y_val):
    from sklearn.ensemble import AdaBoostClassifier

    params.update({"random_state": 0})
    clf = AdaBoostClassifier(**params)
    clf.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))
    return clf, params


def fit_knn(params, X_train, X_val, y_train, y_val, is_classification=True, weight_fn=None):
    from scipy.spatial import distance
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    if weight_fn is not None:

        def my_weight_fn(x, y):
            return distance.minkowski(x, y, p=2, w=weight_fn)

        params.update(dict(metric=my_weight_fn))

    if is_classification:
        # from ehrret import get_retrieval_weight
        # def my_distance(weights):
        #     weights = get_retrieval_weight('feattype', 'eicu', 'READMISSION', feature_names=feature_names)
        #     return weights
        clf = KNeighborsClassifier(**params)
    else:
        clf = KNeighborsRegressor(**params)
    clf.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))
    return clf, params


def fit_mlp(params, X_train, X_val, y_train, y_val, is_classification=True):
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    hidden_layer_sizes = params.pop("type", "2-layer")
    if hidden_layer_sizes == "2-layer":
        params["hidden_layer_sizes"] = (X_train.shape[1], 64)
    elif hidden_layer_sizes == "3-layer":
        params["hidden_layer_sizes"] = (X_train.shape[1], 128, 64)

    params.update(
        {
            "early_stopping": True,
            "validation_fraction": 0.2,
        }
    )
    if is_classification:
        clf = MLPClassifier(**params)
    else:
        clf = MLPRegressor(**params)

    clf.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))
    return clf, params


def objective(
    trial,
    X_train,
    X_val,
    y_train,
    y_val,
    is_classification=True,
    method="xgboost",
    cat_features=None,
    extra_params: dict | None = None,
):
    if extra_params is None:
        extra_params = {}
    if wandb.run is not None:
        wandb.run.config.update(
            {
                "MODEL": {
                    "MODEL_NAME": method,
                },
                "MODEL_NAME": method,
                "FOLD": os.environ.get("FOLD", None),
                "TASK_NAME": os.environ.get("TASK_NAME", None),
            }
        )

    len(np.unique(y_train))

    assert method in [
        "xgboost",
        "rf",
        "svm",
        "logistics",
        "adaboost",
        "catboost",
        "lightgbm",
        "knn",
        "mlp",
    ], f"{method} is not supported"

    if method == "xgboost":
        model, params = fit_xgboost(
            {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 300
                ),  # The number of sequential trees to be modeled
                "max_depth": trial.suggest_int(
                    "max_depth", 1, 9
                ),  # The maximum depth of a tree.higher depth will allow model to learn relations very specific to a particular sample. Should be tuned
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 1.0
                ),  # impact of each tree on the final outcome
                "gamma": trial.suggest_float(
                    "gamma", 0.001, 1.0
                ),  # This will anyways be tuned later.
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 0.001, 1.0
                ),  # This will anyways be tuned later.
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 0.001, 1.0
                ),  # This will anyways be tuned later.
            },
            X_train,
            X_val,
            y_train,
            y_val,
            is_classification=is_classification,
        )

    elif method == "catboost":
        model, params = fit_catboost(
            {
                "iterations": trial.suggest_int("iterations", 100, 300),
                "depth": trial.suggest_int("depth", 1, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.001, 1.0),
            },
            X_train,
            X_val,
            y_train,
            y_val,
            is_classification=is_classification,
            cat_features=cat_features,
        )

    elif method == "lightgbm":
        model, params = fit_lightgbm(
            {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 300
                ),  # The number of sequential trees to be modeled
                "max_depth": trial.suggest_int(
                    "max_depth", 1, 9
                ),  # The maximum depth of a tree.higher depth will allow model to learn relations very specific to a particular sample. Should be tuned
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 1.0
                ),  # impact of each tree on the final outcome
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 0.001, 1.0
                ),  # This will anyways be tuned later.
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 0.001, 1.0
                ),  # This will anyways be tuned later.
            },
            X_train,
            X_val,
            y_train,
            y_val,
            is_classification=is_classification,
            cat_features=cat_features,
        )

    elif method == "rf":
        model, params = fit_rf(
            {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 300
                ),  # The number of sequential trees to be modeled
                "max_depth": trial.suggest_int(
                    "max_depth", 1, 9
                ),  # The maximum depth of a tree.higher depth will allow model to learn relations very specific to a particular sample. Should be tuned
                # 'criterion': trial.suggest_categorical("criterion", ["gini", "log_loss", "entropy"]),
                "min_samples_split": trial.suggest_int("min_samples_split", 10, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 10),
            },
            X_train,
            X_val,
            y_train,
            y_val,
            is_classification=is_classification,
        )

    elif method == "svm":
        model, params = fit_svm(
            {
                "gamma": trial.suggest_float("gamma", 0.001, 1.0),
                "C": trial.suggest_float("C", 0.001, 1.0),
            },
            X_train,
            X_val,
            y_train,
            y_val,
            is_classification=is_classification,
        )

    elif method == "logistics":
        model, params = fit_logistics(
            {
                "C": trial.suggest_float("C", 0.001, 1.0),
            },
            X_train,
            X_val,
            y_train,
            y_val,
            is_classification=is_classification,
        )

    elif method == "adaboost":
        model, params = fit_adaboost(
            {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 300
                ),  # The number of sequential trees to be modeled
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 1.0
                ),  # impact of each tree on the final outcome
            },
            X_train,
            X_val,
            y_train,
            y_val,
            is_classification=is_classification,
        )

    elif method == "knn":
        model, params = fit_knn(
            {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_int("p", 1, 2),  # 1=manhattan, 2=euclidean
            },
            X_train,
            X_val,
            y_train,
            y_val,
            is_classification=is_classification,
            weight_fn=extra_params.get("weight_fn"),
        )

    elif method == "mlp":
        model, params = fit_mlp(
            {
                "type": trial.suggest_categorical("type", ["2-layer", "3-layer"]),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", ["constant", "adaptive"]
                ),
            },
            X_train,
            X_val,
            y_train,
            y_val,
            is_classification=is_classification,
        )

    else:
        raise NotImplementedError()

    # Validate the model
    if is_classification:
        preds = model.predict_proba(X_val)
        pred_labels = np.argmax(preds, axis=1)
    else:
        preds = model.predict(X_val)
        pred_labels = preds

    if wandb.run is not None:
        wandb.run.config.update(params)

    # Evaluate the model
    if not is_classification:
        rmse = np.sqrt(np.mean((y_val - pred_labels) ** 2))
        # Log the metrics to wandb
        if wandb.run is not None:
            wandb.run.log(
                {
                    "Validation/RMSE": rmse,
                }
            )
        return float(rmse)
    else:
        # is_binary = len(np.unique(y_val)) == 2
        # if is_binary:
        #     f1score = f1_score(y_val, pred_labels, average='binary', pos_label=1)
        #     precision = precision_score(y_val, pred_labels, average='binary', pos_label=1)
        #     recall = recall_score(y_val, pred_labels, average='binary', pos_label=1)
        # else:
        f1score = f1_score(y_val, pred_labels, average="macro")
        precision = precision_score(y_val, pred_labels, average="macro")
        recall = recall_score(y_val, pred_labels, average="macro")
        mcc_score = matthews_corrcoef(y_val, pred_labels)
        accuracy = accuracy_score(y_val, pred_labels)

        # Log the metrics to wandb
        if wandb.run is not None:
            wandb.run.log(
                {
                    "Validation/Accuracy": accuracy,
                    "Validation/F1": f1score,
                    "Validation/MCC": mcc_score,
                    "Validation/Precision": precision,
                    "Validation/Recall": recall,
                }
            )
        return float(f1score)


class TradMLTuner:
    def __init__(
        self,
        storage: str = None,
        study_name: str = None,
        n_trials: int = 100,
        direction: str = "maximize",
        pruner=None,
        sampler=None,
        save_dir: str = None,
        method: str = "xgboost",
        use_best_params: bool = False,
        wandb_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        self.storage = None
        if storage is not None and storage.endswith(".log"):
            self.storage = JournalStorage(JournalFileStorage(storage))

        self.save_dir = save_dir
        self.study_name = study_name
        self.n_trials = n_trials
        self.direction = direction
        self.pruner = pruner
        self.sampler = sampler
        self.save_dir = save_dir
        self.method = method
        self.use_best_params = use_best_params
        self.feature_names = kwargs.get("feature_names")
        self.classnames = kwargs.get("classnames")
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

        if wandb_kwargs is not None:
            # Initialise wandb callback
            self.wandb_kwargs = {
                "entity": wandb_kwargs.get("entity", "kaylode"),
                "project": wandb_kwargs.get("project", "tabpfn-ehr"),
                "reinit": True,
                "group": "tabpfn-ehr-optuna",
                "job_type": "optuna",
                "tags": [self.method, "optuna"] + wandb_kwargs.get("tags", []),
                "resume": "allow",
                "name": f"{self.method}",
                "dir": self.save_dir,
            }
            self.WANDB_CALLBACK = WeightsAndBiasesCallback(
                metric_name="Validation/F1", wandb_kwargs=self.wandb_kwargs, as_multirun=True
            )
        else:
            self.wandb_kwargs = None
            self.WANDB_CALLBACK = None

    def tune(
        self,
        X_train,
        X_val,
        y_train,
        y_val,
        is_classification: bool = True,
        cat_features=None,
        extra_params: dict | None = None,
    ):
        if extra_params is None:
            extra_params = {}
        def wrapped_objective(trial):
            return objective(
                    trial,
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    is_classification=is_classification,
                    method=self.method,
                    cat_features=cat_features,
                    extra_params=extra_params,
                )

        callbacks = None
        if self.WANDB_CALLBACK is not None:
            decorator = self.WANDB_CALLBACK.track_in_wandb()
            wrapped_objective = decorator(wrapped_objective)
            callbacks = []
            callbacks.append(self.WANDB_CALLBACK)

        if not self.use_best_params:
            try:
                self.study.optimize(wrapped_objective, n_trials=self.n_trials, callbacks=callbacks)
            except KeyboardInterrupt:
                LOGGER.text("KeyboardInterrupt", level=LoggerObserver.ERROR)
        best_trial = self.study.best_trial
        self.save_best_config(best_trial.params)

        if self.wandb_kwargs is not None:
            wandb_kwargs = {
                "name": f"{self.method}",
                "entity": self.wandb_kwargs.get("entity", "kaylode"),
                "project": self.wandb_kwargs.get("project", "tabpfn-ehr"),
                "reinit": True,
                "group": "tabpfn-ehr-finetune",
                "job_type": "train",
                "tags": list(
                    set([self.method, "optuna", "best"] + self.wandb_kwargs.get("tags", []))
                ),
                "resume": "allow",
                "dir": self.save_dir,
            }
            wandb.init(**wandb_kwargs)

        if self.save_dir is not None:
            leaderboard_df = self.leaderboard()
            leaderboard_df.to_csv(osp.join(self.save_dir, "leaderboard.csv"))
            leaderboard_df.to_json(osp.join(self.save_dir, "leaderboard.json"), orient="records")
            LOGGER.text(
                f"Leaderboard saved to {self.save_dir}/leaderboard.csv", level=LoggerObserver.INFO
            )
            figs = self.visualize("all")
            os.makedirs(osp.join(self.save_dir, "figures"), exist_ok=True)
            for fig_name, fig in figs:
                try:
                    fig.write_image(osp.join(self.save_dir, "figures", f"{fig_name}.png"))
                    LOGGER.text(
                        f"{fig_name} plot saved to {self.save_dir}/{fig_name}.png",
                        level=LoggerObserver.INFO,
                    )
                except Exception:
                    pass

        # Log the best trial to wandb
        if self.wandb_kwargs is not None:
            wandb.run.config.update(
                {
                    "MODEL": {
                        "MODEL_NAME": self.method,
                    },
                    "MODEL_NAME": self.method,
                    "FOLD": os.environ.get("FOLD", None),
                    "TASK_NAME": os.environ.get("TASK_NAME", None),
                }
            )

        if self.method == "xgboost":
            best_model, params = fit_xgboost(
                best_trial.params,
                X_train,
                X_val,
                y_train,
                y_val,
                is_classification=is_classification,
            )
        elif self.method == "catboost":
            best_model, params = fit_catboost(
                best_trial.params,
                X_train,
                X_val,
                y_train,
                y_val,
                is_classification=is_classification,
            )
        elif self.method == "lightgbm":
            best_model, params = fit_lightgbm(
                best_trial.params,
                X_train,
                X_val,
                y_train,
                y_val,
                is_classification=is_classification,
            )
        elif self.method == "rf":
            best_model, params = fit_rf(
                best_trial.params,
                X_train,
                X_val,
                y_train,
                y_val,
                is_classification=is_classification,
            )
        elif self.method == "svm":
            best_model, params = fit_svm(
                best_trial.params,
                X_train,
                X_val,
                y_train,
                y_val,
                is_classification=is_classification,
            )
        elif self.method == "logistics":
            best_model, params = fit_logistics(
                best_trial.params,
                X_train,
                X_val,
                y_train,
                y_val,
                is_classification=is_classification,
            )
        elif self.method == "adaboost":
            best_model, params = fit_adaboost(
                best_trial.params,
                X_train,
                X_val,
                y_train,
                y_val,
                is_classification=is_classification,
            )
        elif self.method == "knn":
            best_model, params = fit_knn(
                best_trial.params,
                X_train,
                X_val,
                y_train,
                y_val,
                is_classification=is_classification,
                weight_fn=extra_params.get("weight_fn"),
            )
        elif self.method == "mlp":
            best_model, params = fit_mlp(
                best_trial.params,
                X_train,
                X_val,
                y_train,
                y_val,
                is_classification=is_classification,
            )
        else:
            raise NotImplementedError()

        if self.wandb_kwargs is not None:
            wandb.run.config.update(params)

        # Evaluate the model
        if not is_classification:
            preds = best_model.predict(X_val)
            pred_labels = preds
        else:
            preds = best_model.predict_proba(X_val)
            pred_labels = np.argmax(preds, axis=1)

        if is_classification:
            # is_binary = len(np.unique(y_val)) == 2
            # if is_binary:
            #     f1score = f1_score(y_val, pred_labels, average='binary', pos_label=1)
            #     precision = precision_score(y_val, pred_labels, average='binary', pos_label=1)
            #     recall = recall_score(y_val, pred_labels, average='binary', pos_label=1)
            # else:
            f1score = f1_score(y_val, pred_labels, average="macro")
            precision = precision_score(y_val, pred_labels, average="macro")
            recall = recall_score(y_val, pred_labels, average="macro")
            mcc_score = matthews_corrcoef(y_val, pred_labels)
            accuracy = accuracy_score(y_val, pred_labels)
            # Log the metrics to wandb
            if self.wandb_kwargs is not None:
                wandb.run.log(
                    {
                        "Validation/Accuracy": accuracy,
                        "Validation/F1": f1score,
                        "Validation/MCC": mcc_score,
                        "Validation/Precision": precision,
                        "Validation/Recall": recall,
                    }
                )
        else:
            rmse = np.sqrt(np.mean((y_val - pred_labels) ** 2))
            # Log the metrics to wandb
            if self.wandb_kwargs is not None:
                wandb.run.log(
                    {
                        "Validation/RMSE": rmse,
                    }
                )
        LOGGER.text(
            f"Best trial: {best_trial.number} with value: {best_trial.value}",
            level=LoggerObserver.INFO,
        )

        if self.method == "logistics":
            # Save coefficient and intercept in json
            coef = best_model.coef_
            intercept = best_model.intercept_
            with open(osp.join(self.save_dir, "logistics_params.json"), "w") as f:
                json.dump(
                    {
                        "coef": coef.tolist(),
                        "intercept": intercept.tolist(),
                        "feature_names": self.feature_names,
                    },
                    f,
                    indent=4,
                )

        if is_classification and self.feature_names is not None and self.classnames is not None:
            try:
                fig = visualize_shap(
                    best_model,
                    X_val,
                    feature_names=self.feature_names,
                    classnames=self.classnames,
                    plot_type="dot",
                    plot_size=(12, 6),
                    cross_validation=False,
                )
                fig.savefig(osp.join(self.save_dir, "shap_summary.png"), bbox_inches="tight")
                if self.wandb_kwargs is not None:
                    wandb.log({"Shap Summary": wandb.Image(fig)})
                LOGGER.text(
                    f"Shap Summary plot saved to {self.save_dir}/shap_summary.png",
                    level=LoggerObserver.INFO,
                )
            except Exception:
                pass
        else:
            LOGGER.text(
                "Shap values are not supported for regression tasks.", level=LoggerObserver.WARN
            )

        if self.wandb_kwargs is not None:
            wandb.finish()

        return best_model

    def save_best_config(self, best_params: dict):
        with open(os.path.join(self.save_dir, "best_config.json"), "w") as f:
            json.dump(best_params, f, indent=4)

        LOGGER.text(
            f"Best config saved to {self.save_dir}/best_config.json", level=LoggerObserver.INFO
        )

    def leaderboard(self):
        """Print leaderboard of all trials"""
        df = self.study.trials_dataframe()
        df.columns = [col.replace("user_attrs_", "") for col in df.columns]
        return df

    def visualize(self, plot: str, plot_params: dict = None):
        """Visualize everything"""

        if plot_params is None:
            plot_params = {}
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

        if plot == "all":
            fig = []
            for plot_type in allow_plot_types:
                one_fig = self.visualize(plot_type, plot_params)
                if one_fig is not None:
                    fig.append((plot_type, one_fig))
        else:
            try:
                if plot == "history":
                    fig = plot_optimization_history(self.study, **plot_params)
                elif plot == "contour":
                    fig = plot_contour(self.study, **plot_params)
                elif plot == "edf":
                    fig = plot_edf(self.study, **plot_params)
                elif plot == "intermediate_values":
                    fig = plot_intermediate_values(self.study)
                elif plot == "parallel_coordinate":
                    fig = plot_parallel_coordinate(self.study, **plot_params)
                elif plot == "param_importances":
                    fig = plot_param_importances(self.study, **plot_params)
                elif plot == "slice":
                    fig = plot_slice(self.study, **plot_params)
                else:
                    LOGGER.text(f"{plot} is not supported by Optuna", level=LoggerObserver.ERROR)
                    raise ValueError()

                if plot != "all" and self.wandb_kwargs is not None:
                    wandb.log({f"Plot/{plot}": fig})

            except Exception as e:
                LOGGER.text(f"Plotting error: {e}", level=LoggerObserver.ERROR)
                return None

        return fig
