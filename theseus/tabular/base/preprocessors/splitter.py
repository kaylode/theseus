import os
import os.path as osp
import random

from sklearn.model_selection import StratifiedKFold, train_test_split

from theseus.base.utilities.loggers.observer import LoggerObserver

from .base import Preprocessor

LOGGER = LoggerObserver.getLogger("main")


class Splitter(Preprocessor):
    def __init__(
        self,
        splitter_type="default",
        save_folder=None,
        ratio=None,
        label_column=None,
        seed=0,
        n_splits=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert splitter_type in [
            "default",
            "stratified",
            "stratifiedkfold",
            "unique",
        ], "splitter type not supported"

        self.splitter_type = splitter_type
        self.save_folder = save_folder
        self.label_column = label_column
        self.seed = seed

        if self.save_folder is not None:
            os.makedirs(self.save_folder, exist_ok=True)

        if self.splitter_type == "stratified":
            assert label_column is not None, "Label column should be specified"
            self.splitter = train_test_split
            self.ratio = ratio
        elif self.splitter_type == "stratifiedkfold":
            assert label_column is not None, "Label column should be specified"
            assert n_splits is not None, "number of splits should be specified"
            self.splitter = StratifiedKFold(
                n_splits=n_splits, random_state=self.seed, shuffle=True
            )
        elif self.splitter_type == "default":
            assert ratio is not None, "should specify ratio"
            self.ratio = ratio
        elif self.splitter_type == "unique":
            assert ratio is not None, "should specify ratio"
            self.splitter = random.sample
            self.ratio = ratio

    def run(self, df):
        num_samples, num_features = df.shape
        if self.splitter_type == "default":
            train_df = df.sample(frac=self.ratio, random_state=self.seed)
            val_df = df.drop(train_df.index)
            train_df.to_csv(osp.join(self.save_folder, "train.csv"), index=False)
            val_df.to_csv(osp.join(self.save_folder, "val.csv"), index=False)
        elif self.splitter_type == "stratified":
            train_df, val_df = self.splitter(
                df, stratify=df[[self.label_column]], random_state=self.seed,
                train_size=self.ratio,
            )
            train_df.to_csv(osp.join(self.save_folder, "train.csv"), index=False)
            val_df.to_csv(osp.join(self.save_folder, "val.csv"), index=False)
        elif self.splitter_type == "unique":
            unique_values = df[self.label_column].unique().tolist()
            num_unique_samples = len(unique_values)
            train_idx = self.splitter(
                unique_values, int(num_unique_samples * self.ratio)
            )
            train_df = df[df[self.label_column].isin(train_idx)]
            val_df = df[~df[self.label_column].isin(train_idx)]
            train_df.to_csv(osp.join(self.save_folder, "train.csv"), index=False)
            val_df.to_csv(osp.join(self.save_folder, "val.csv"), index=False)
        else:
            x, y = (
                df.drop(self.label_column, axis=1).values,
                df[self.label_column].values,
            )
            splits = self.splitter.split(x, y)
            for fold_id, (train_ids, val_ids) in enumerate(splits):
                train_df = df.iloc[train_ids]
                val_df = df.iloc[val_ids]
                train_df.to_csv(
                    osp.join(self.save_folder, f"train_fold{fold_id}.csv"),
                    index=False,
                )
                val_df.to_csv(
                    osp.join(self.save_folder, f"val_fold{fold_id}.csv"),
                    index=False,
                )

        self.log(
            f"Splitted using {self.splitter_type}: {len(train_df)} train, {len(val_df)} validation"
        )

        return df
