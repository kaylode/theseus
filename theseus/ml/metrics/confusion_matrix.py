from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from theseus.base.metrics.metric_template import Metric


def plot_cfm(cm, ax, labels: List):
    """
    Make confusion matrix figure
    labels: `Optional[List]`
        classnames for visualization
    """

    ax = sns.heatmap(cm, annot=False, fmt="", cmap="Blues", ax=ax)

    ax.set_xlabel("\nActual")
    ax.set_ylabel("Predicted ")

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels, rotation=0)


def make_cm_fig(cms, labels: Optional[List] = None):

    if cms.shape[0] > 1:  # multilabel
        num_classes = cms.shape[0]
    else:
        num_classes = cms.shape[1]

    ## Ticket labels - List must be in alphabetical order
    if not labels:
        labels = [str(i) for i in range(num_classes)]

    ##
    num_cfms = cms.shape[0]
    nrow = int(np.ceil(np.sqrt(num_cfms)))

    # Clear figures first to prevent memory-consuming
    plt.cla()
    plt.clf()
    plt.close()

    fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))

    if num_cfms > 1:
        for ax, cfs_matrix, label in zip(axes.flatten(), cms, labels):
            ax.set_title(f"{label}\n\n")
            plot_cfm(cfs_matrix, ax, labels=["N", "Y"])
    else:
        plot_cfm(cms[0], axes, labels=labels)

    fig.tight_layout()
    return fig


class SKLConfusionMatrix(Metric):
    """
    Confusion Matrix metric for classification
    """

    def __init__(self, classnames=None, label_type: str = "multiclass", **kwargs):
        super().__init__(**kwargs)
        self.type = label_type
        self.classnames = classnames
        self.num_classes = (
            [i for i in range(len(self.classnames))] if classnames is not None else None
        )

    def value(self, output: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        output = output["outputs"]
        target = batch["targets"]

        probs = softmax(output, axis=-1)
        predictions = np.argmax(probs, axis=-1)

        if self.type == "multiclass":
            values = confusion_matrix(
                predictions,
                target,
                labels=self.num_classes,
                normalize="pred",
            )
            values = values[np.newaxis, :, :]
        else:
            values = multilabel_confusion_matrix(
                predictions, target, labels=self.num_classes
            )

        fig = make_cm_fig(values, self.classnames)
        return {"cfm": fig}
