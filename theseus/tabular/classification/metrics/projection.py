import os
from typing import Any, Dict

import numpy as np

from theseus.base.metrics.metric_template import Metric
from theseus.base.utilities.loggers import LoggerObserver
from scipy.special import softmax

LOGGER = LoggerObserver.getLogger("main")

class SKLEmbeddingProjection(Metric):
    """
    Visualize embedding project for classification
    """

    def __init__(self, classnames=None, save_dir=".cache", has_labels=True, **kwargs):
        super().__init__(**kwargs)
        self.has_labels = has_labels
        self.save_dir = save_dir
        self.classnames = classnames
        os.makedirs(self.save_dir, exist_ok=True)

    def value(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """

        embeddings = batch["inputs"]
        targets = batch["targets"]
        probs = softmax(outputs["outputs"], axis=-1)
        predictions = np.argmax(probs, axis=-1)

        ## Metadata, in column style
        if self.has_labels:
            if self.classnames is not None:
                metadata = [(self.classnames[int(a)], self.classnames[int(b)]) for a,b in zip(targets, predictions)]
            else:
                metadata = [a for a in zip(targets, predictions)]
            metadata_header = ["ground truth", "prediction"]
        else:
            if self.classnames is not None:
                metadata = [[self.classnames[int(a)] for a in predictions]]
            else:
                metadata = [predictions]
            metadata_header = ["prediction"]

        LOGGER.log(
            [
                {
                    "tag": f"Projection",
                    "value": embeddings,
                    "type": LoggerObserver.EMBED,
                    "kwargs": {
                        "step": 0,
                        "metadata": metadata,
                        "metadata_header": metadata_header,
                    },
                }
            ]
        )

        return {"projection": "Embedding projection generated"}
