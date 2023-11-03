import json
import os
import os.path as osp

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from theseus.base.utilities.loggers.observer import LoggerObserver

from .base import Preprocessor

LOGGER = LoggerObserver.getLogger("main")


class LabelEncode(Preprocessor):
    def __init__(self, encoder_type="le", save_folder=None, **kwargs):
        super().__init__(**kwargs)

        assert encoder_type in [
            "le",
            "onehot",
            "ordinal",
        ], "Encoder type not supported"

        self.encoder_type = encoder_type
        self.save_folder = save_folder
        self.mapping_dict = {}

        if self.encoder_type == "le":
            self.encoder = LabelEncoder()
        elif self.encoder_type == "onehot":
            self.encoder = OneHotEncoder()
        else:
            self.encoder = OrdinalEncoder()

    @classmethod
    def from_json(cls, json_path: str):
        return cls(json_path=json_path, encoder_type="json_mapping")

    def create_mapping_dict(self, column_name):
        le_name_mapping = dict(
            zip(
                self.encoder.classes_,
                [int(i) for i in self.encoder.transform(self.encoder.classes_)],
            )
        )
        if self.save_folder is not None:
            os.makedirs(self.save_folder, exist_ok=True)
            json.dump(
                le_name_mapping,
                open(osp.join(self.save_folder, column_name + ".json"), "w"),
                indent=4,
            )
        return le_name_mapping

    def encode_corpus(self, df):
        for column_name in self.column_names:
            df[column_name] = self.encoder.fit_transform(df[column_name].values).copy()
            mapping_dict = self.create_mapping_dict(column_name)
            self.mapping_dict[column_name] = mapping_dict
        return df

    def encode_query(self, df):
        for column_name in self.column_names:
            df[column_name] = self.apply(
                df[column_name], lambda x: self.mapping_dict[column_name].get(x, -1)
            ).copy()
        return df

    def run(self, df):
        self.prerun(df)

        if self.column_names is None:
            self.log(
                "Column names not specified. Automatically label encode columns with non-defined types",
                level=LoggerObserver.WARN,
            )
            self.column_names = [col for col, dt in df.dtypes.items() if dt == object]
        self.encode_corpus(df)

        self.log(f"Label-encoded columns: {self.column_names}")
        return df
