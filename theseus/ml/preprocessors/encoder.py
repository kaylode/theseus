import json
import os
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from theseus.base.utilities.loggers.observer import LoggerObserver

from .base import Preprocessor

LOGGER = LoggerObserver.getLogger("main")


class LabelEncode(Preprocessor):
    def __init__(
        self, encoder_type="le", pickle_path=None, engine: str = "pandas", **kwargs
    ):
        super().__init__(**kwargs)

        assert encoder_type in [
            "le",
            "onehot",
            "ordinal",
        ], "Encoder type not supported"

        self.encoder_type = encoder_type
        self.pickle_path = pickle_path
        self.engine = engine
        if self.engine == "polars":
            import polars as pl

        if self.pickle_path is not None:
            with open(self.pickle_path, "rb") as fb:
                config = pickle.load(fb)
            self.column_names = config["column_names"]
            self.encoder_type = config["encoder_type"]
            self.engine = config["engine"]
            self.encoders = config["encoders"]
            self.log(f"Loaded mapping dict from {self.pickle_path}")
        else:
            self.encoders = {}
            if self.encoder_type == "le":
                encoder = LabelEncoder()
            elif self.encoder_type == "onehot":
                encoder = OneHotEncoder()
            else:
                encoder = OrdinalEncoder()

            for column in self.column_names:
                self.encoders[column] = encoder

    @classmethod
    def from_pickle(cls, pickle_path: str):
        return cls(pickle_path=pickle_path)

    def save_pickle(self, pickle_path: str):
        with open(pickle_path, "wb") as fb:
            pickle.dump(
                {
                    "column_names": self.column_names,
                    "encoder_type": self.encoder_type,
                    "engine": self.engine,
                    "encoders": self.encoders,
                },
                fb,
            )
        self.log(f"Saved encoder to {pickle_path}")

    def save_json(self, json_path: str):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        mapping_dict = {}
        for column_name in self.column_names:
            class_mapping = dict(
                zip(
                    self.encoders[column_name].classes_,
                    [
                        int(i)
                        for i in self.encoders[column_name].transform(
                            self.encoders[column_name].classes_
                        )
                    ],
                )
            )
            mapping_dict[column_name] = class_mapping
        with open(json_path, "w") as fb:
            json.dump(mapping_dict, fb, indent=4)
        self.log(f"Saved mapping dict to {json_path}")

    def encode_corpus(self, df):
        for column_name in self.column_names:
            encoder = self.encoders[column_name]
            if self.engine == "pandas":
                df[column_name] = encoder.fit_transform(df[column_name].values).copy()
            elif self.engine == "polars":
                import polars as pl

                encoder.fit_transform(df[column_name].to_numpy())
                le_name_mapping = dict(
                    zip(
                        encoder.classes_,
                        [int(i) for i in encoder.transform(encoder.classes_)],
                    )
                )
                df = df.with_columns(
                    pl.col(column_name).replace_strict(
                        le_name_mapping, return_dtype=pl.Int32, default=None
                    )
                )

        return df

    def encode_query(self, df):
        for column_name in self.column_names:
            encoder = self.encoders[column_name]
            if self.engine == "pandas":
                df[column_name] = encoder.transform(df[column_name].values).copy()
            elif self.engine == "polars":
                import polars as pl

                le_name_mapping = dict(
                    zip(
                        encoder.classes_,
                        [int(i) for i in encoder.transform(encoder.classes_)],
                    )
                )
                df = df.with_columns(
                    pl.col(column_name).replace_strict(
                        le_name_mapping, return_dtype=pl.Int32, default=None
                    )
                )
        return df

    def run(self, df):
        self.prerun(df)

        if self.column_names is None:
            self.log(
                "Column names not specified. Automatically label encode columns with non-defined types",
                level=LoggerObserver.WARN,
            )
            self.column_names = [col for col, dt in df.dtypes.items() if dt == object]
        df = self.encode_corpus(df)

        self.log(f"Label-encoded columns: {self.column_names}")
        return df
