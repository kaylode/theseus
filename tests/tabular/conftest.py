import pandas as pd
import pytest

from theseus.ml.preprocessors import FillNaN, LabelEncode, PreprocessCompose, Standardize


# --- Data Loading Helper ---
def load_titanic_data(split: str):
    """Load and preprocess Titanic data for a given split."""
    data_path = f"samples/titanic/{split}.csv"
    classnames_path = "samples/titanic/classnames.txt"
    target_column = "Survived"

    df = pd.read_csv(data_path)

    # Apply preprocessing (same pipeline as the old YAML config)
    transform = PreprocessCompose(
        preproc_list=[
            FillNaN(column_names=["Embarked", "Cabin"], fill_with="None"),
            FillNaN(column_names=["Age"], fill_with=0),
            LabelEncode(),
            Standardize(method="minmax", column_names=["*"], exclude_columns=[target_column]),
        ]
    )
    df = transform.run(df)

    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    feature_names = list(df.drop(target_column, axis=1).columns)
    classnames = open(classnames_path).read().splitlines()

    return X, y, feature_names, classnames


# --- Fixtures ---
@pytest.fixture(scope="session")
def titanic_data():
    """Load train and val splits of Titanic dataset."""
    X_train, y_train, feature_names, classnames = load_titanic_data("train")
    X_val, y_val, _, _ = load_titanic_data("val")
    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "feature_names": feature_names,
        "classnames": classnames,
    }
