import catboost as cb
import lightgbm as lgb
import xgboost as xgb

from theseus.base.utilities.loggers.observer import LoggerObserver
from omegaconf import DictConfig, OmegaConf
LOGGER = LoggerObserver.getLogger("main")


class GBClassifiers:
    def __init__(
        self, model_name, num_classes, model_config:DictConfig={}, training_params={}, **kwargs
    ):
        OmegaConf.set_struct(model_config, False)
        self.training_params = training_params
        self.model_name = model_name
        self.num_classes = num_classes
        if model_name == "catboost":
            self.model = cb.CatBoostClassifier(**model_config)
        elif model_name == "lightgbm":
            model_config.update({"num_class": num_classes})
            self.model = lgb.LGBMClassifier(**model_config)
        elif model_name == "xgboost":
            model_config.update({"num_class": num_classes})
            self.model = xgb.XGBClassifier(**model_config)
        else:
            LOGGER.text("Model not supported", level=LoggerObserver.ERROR)

    def get_model(self):
        return self.model

    def fit(self, trainset, valset, **kwargs):
        X, y = trainset
        self.model.fit(
            X,
            y,
            eval_set=[trainset, valset],
            # eval_set=[(trainset, 'train'), (valset, 'validation')],
            **self.training_params,
        )

    def save_model(self, savepath):
        if self.model_name == "xgboost":
            self.model.save_model(savepath)
        elif self.model_name == "lightgbm":
            # LightGBM models should be saved as .txt files
            self.model.booster_.save_model(savepath)
        elif self.model_name == "xgboost":
            self.model.save(savepath)

        LOGGER.text(f"Model saved at {savepath}", level=LoggerObserver.INFO)

    def load_model(self, checkpoint_path):
        self.model.load_model(checkpoint_path)
        LOGGER.text(
            f"Loaded checkpoint at {checkpoint_path}",
            level=LoggerObserver.INFO,
        )

    def predict(self, X, return_probs=False):
        if return_probs:
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)
