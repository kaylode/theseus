from .base import Preprocessor


def fill_with_mean(df, column_name):
    return df[column_name].fillna(value=df[column_name].mean(), inplace=False)


def fill_with_interpolation(df, column_name, method_name="linear"):
    return df[column_name].interpolate(method=method_name, inplace=False)


def fill_with_value(df, column_name, value):
    return df[column_name].fillna(value=value, inplace=False)


class FillNaN(Preprocessor):
    def __init__(self, fill_with="mean", **kwargs):
        super().__init__(**kwargs)
        self.fill_with = fill_with

        if self.fill_with == "interpolate":
            self.interpolate_method_name = kwargs.get("interpolate_method", "linear")

    def run(self, df):

        self.prerun(df)
        if self.column_names is None:
            self.column_names = [k for k, i in df.isna().mean().items() if i > 0]

        for column_name in self.column_names:
            if self.fill_with == "mean":
                df[column_name] = fill_with_mean(df, column_name)
            if self.fill_with == "interpolate":
                df[column_name] = fill_with_interpolation(
                    df, column_name, self.interpolate_method_name
                )
            else:
                df[column_name] = fill_with_value(df, column_name, value=self.fill_with)

        self.log(f"Filled NaN with {self.fill_with}: {self.column_names}")

        return df
