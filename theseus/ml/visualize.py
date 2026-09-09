
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from lime import lime_tabular
from sklearn.inspection import permutation_importance


def plot_cfm(cm, ax, labels: list):
    """
    Make confusion matrix figure
    labels: `Optional[List]`
        classnames for visualization
    """
    ax = sns.heatmap(cm, annot=False, fmt="", cmap="Blues", ax=ax)
    ax.set_xlabel("\nActual")
    ax.set_ylabel("Predicted ")
    ax.xaxis.set_ticklabels(labels, rotation=90)
    ax.yaxis.set_ticklabels(labels, rotation=0)


def make_cm_fig(cms, labels: list | None = None):
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


def visualize_shap(
    model,
    inputs,
    feature_names,
    classnames,
    plot_type="bar",
    plot_size="auto",
    cross_validation=False,
):
    import shap

    if cross_validation:
        assert isinstance(model, list), "model must be a list of models"
        all_shap_values = []

        if isinstance(inputs, list):
            for _, (m, input) in enumerate(zip(model, inputs)):
                explainer = shap.TreeExplainer(m)
                shap_values = explainer.shap_values(input, check_additivity=False)
                all_shap_values.append(shap_values)
        else:
            for _, m in enumerate(model):
                explainer = shap.TreeExplainer(m)
                shap_values = explainer.shap_values(inputs, check_additivity=False)
                all_shap_values.append(shap_values)

        if isinstance(feature_names[0], list):  # each fold have different order of features
            feat_shap_dict = {}
            for feature_list, shap_values in zip(feature_names, all_shap_values):
                shap_values = np.array(shap_values).T
                for feat, shap_val in zip(feature_list, shap_values):
                    if feat not in feat_shap_dict:
                        feat_shap_dict[feat] = []
                    feat_shap_dict[feat].append(shap_val)

            feat_shap_dict = {k: np.mean(v, axis=0) for k, v in feat_shap_dict.items()}
            feat_shap_dict.pop("none", None)
            feature_names = sorted(list(feat_shap_dict.keys()))
            average_shap_values = np.array([feat_shap_dict[feat] for feat in feature_names]).T
        else:
            all_shap_values = np.array(all_shap_values)
            average_shap_values = np.mean(all_shap_values, axis=0)
        # std_shap_values = np.std(all_shap_values, axis=0)
        # range_shap_values = np.max(all_shap_values, axis=0) - np.min(all_shap_values, axis=0)
        plt.clf()

        shap.summary_plot(
            average_shap_values,
            inputs,
            plot_type=plot_type,
            feature_names=feature_names,
            class_names=classnames,
            show=False,
            plot_size=plot_size,
        )
        fig = plt.gcf()
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(inputs, check_additivity=False)
        plt.clf()
        shap.summary_plot(
            shap_values,
            inputs,
            plot_type=plot_type,
            feature_names=feature_names,
            class_names=classnames,
            show=False,
            plot_size=plot_size,
        )
        fig = plt.gcf()
    return fig


def visualize_feature_importance(model, inputs, targets, feature_names):
    perm_importance = permutation_importance(model, inputs, targets)
    sorted_idx = perm_importance.importances_mean.argsort()

    fig = go.Figure(
        go.Bar(
            x=perm_importance.importances_mean[sorted_idx],
            y=[feature_names[i] for i in sorted_idx],
            orientation="h",
        )
    )

    fig.update_layout(title=go.layout.Title(text="Permutation Importance", x=0))
    return fig


def visualize_lime_instance(training_data, proba_func, item, feature_names=None, class_names=None):
    """
    Get explaination for a single instance
    """
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=False,
    )

    fig = explainer.explain_instance(data_row=item, predict_fn=proba_func)
    return fig
