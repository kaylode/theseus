from theseus.base.augmentations import TRANSFORM_REGISTRY

from .aggregation import Aggregate
from .base import Preprocessor
from .categorize import Categorize, EnforceType
from .compose import PreprocessCompose
from .csv_saver import CSVSaver
from .datetime import DateDecompose, ToDatetime
from .drop_col import (
    DropColumns,
    DropDuplicatedRows,
    DropEmptyColumns,
    DropSingleValuedColumns,
    LambdaDropRows,
)
from .encoder import LabelEncode
from .fill_nan import FillNaN
from .mapping import MapScreenToBinary
from .new_col import LambdaCreateColumn
from .sort import SortBy
from .splitter import Splitter
from .standardize import Standardize

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
TRANSFORM_REGISTRY.register(EnforceType)
TRANSFORM_REGISTRY.register(LambdaDropRows)
TRANSFORM_REGISTRY.register(LambdaCreateColumn)
TRANSFORM_REGISTRY.register(SortBy)
TRANSFORM_REGISTRY.register(MapScreenToBinary)
