import numpy as np
import pytest

from theseus.ml.tradml import fit_xgboost


@pytest.mark.order(1)
def test_train_xgboost(titanic_data):
    """Train an XGBoost classifier on Titanic data."""
    model, params = fit_xgboost(
        params={
            "n_estimators": 50,
            "max_depth": 5,
            "learning_rate": 0.1,
            "gamma": 0.1,
            "reg_alpha": 0,
            "reg_lambda": 1,
        },
        X_train=titanic_data["X_train"],
        X_val=titanic_data["X_val"],
        y_train=titanic_data["y_train"],
        y_val=titanic_data["y_val"],
        is_classification=True,
    )
    assert model is not None
    assert "objective" in params

    # Evaluate
    preds = model.predict_proba(titanic_data["X_val"])
    pred_labels = np.argmax(preds, axis=1)
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(titanic_data["y_val"], pred_labels)
    print(f"XGBoost accuracy: {acc:.4f}")
    assert acc > 0.5, f"Expected accuracy > 0.5, got {acc}"


@pytest.mark.order(2)
def test_eval_xgboost(titanic_data):
    """Train and evaluate XGBoost — verify metrics are reasonable."""
    model, _ = fit_xgboost(
        params={
            "n_estimators": 50,
            "max_depth": 5,
            "learning_rate": 0.1,
            "gamma": 0.1,
            "reg_alpha": 0,
            "reg_lambda": 1,
        },
        X_train=titanic_data["X_train"],
        X_val=titanic_data["X_val"],
        y_train=titanic_data["y_train"],
        y_val=titanic_data["y_val"],
        is_classification=True,
    )

    from sklearn.metrics import f1_score, matthews_corrcoef

    preds = model.predict_proba(titanic_data["X_val"])
    pred_labels = np.argmax(preds, axis=1)
    f1 = f1_score(titanic_data["y_val"], pred_labels, average="macro")
    mcc = matthews_corrcoef(titanic_data["y_val"], pred_labels)
    print(f"F1: {f1:.4f}, MCC: {mcc:.4f}")
    assert f1 > 0.0
