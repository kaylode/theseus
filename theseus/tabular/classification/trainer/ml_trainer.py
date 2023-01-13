from theseus.base.callbacks import CallbacksList
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")


class MLTrainer:
    def __init__(
        self, model, trainset, valset, metrics, callbacks=None, **kwargs
    ) -> None:

        if callbacks is not None and not isinstance(callbacks, CallbacksList):
            callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
            callbacks = CallbacksList(callbacks)
            callbacks.set_params({"trainer": self})
        self.callbacks = callbacks

        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.metrics = metrics

    def fit(self):

        # On start callbacks
        self.callbacks.run("on_start")
        self.callbacks.run("on_train_epoch_start")
        self.model.fit(
            (self.trainset["inputs"], self.trainset["targets"]),
            (self.valset["inputs"], self.valset["targets"]),
        )
        self.callbacks.run(
            "on_train_epoch_end",
            {"trainset": self.trainset, "valset": self.valset},
        )

        self.callbacks.run("on_val_epoch_start")
        metric_dict = self.evaluate_epoch()
        self.callbacks.run(
            "on_val_epoch_end",
            {
                "iters": 0,
                "trainset": self.trainset,
                "valset": self.valset,
                "metric_dict": metric_dict,
            },
        )
        self.callbacks.run("on_finish")

    def evaluate_epoch(self):
        """
        Perform validation one epoch
        """

        X_test, y_test = self.valset["inputs"], self.valset["targets"]
        y_pred = self.model.predict(X_test, return_probs=True)
        score_dict = {}

        if self.metrics is not None:
            for metric in self.metrics:
                score_dict.update(
                    metric.value(
                        {"outputs": y_pred}, {"inputs": X_test, "targets": y_test}
                    )
                )
        return score_dict
