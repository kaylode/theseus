from .compose import PreprocessCompose
from .fill_nan import FillNaN
from .csv_saver import CSVSaver
from .datetime import ToDatetime, DateDecompose
from .drop_col import (
    DropColumns, DropDuplicatedRows, 
    DropSingleValuedColumns, DropEmptyColumns,
)
from .categorize import Categorize
from .encoder import LabelEncode
from .splitter import Splitter
from .standardize import Standardize
from .aggregation import Aggregate

from theseus.base.augmentations import TRANSFORM_REGISTRY

TRANSFORM_REGISTRY.register(PreprocessCompose)
TRANSFORM_REGISTRY.register(DateDecompose)
TRANSFORM_REGISTRY.register(FillNaN)
TRANSFORM_REGISTRY.register(CSVSaver)
TRANSFORM_REGISTRY.register(ToDatetime)
TRANSFORM_REGISTRY.register(DropColumns)
TRANSFORM_REGISTRY.register(DropDuplicatedRows)
TRANSFORM_REGISTRY.register(DropSingleValuedColumns)
TRANSFORM_REGISTRY.register(DropEmptyColumns)
TRANSFORM_REGISTRY.register(Categorize)
TRANSFORM_REGISTRY.register(LabelEncode)
TRANSFORM_REGISTRY.register(Splitter)
TRANSFORM_REGISTRY.register(Standardize)
TRANSFORM_REGISTRY.register(Aggregate)