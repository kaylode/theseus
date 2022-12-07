from .compose import PreprocessCompose
from .fill_nan import FillNaN
from .csv_saver import CSVSaver
from .datetime import ToDatetime, DateDecompose
from .drop_col import (
    DropColumns, DropDuplicatedRows, 
    DropSingleValuedColumns, DropEmptyColumns,
    DropColumnsWithNameFiltered
)
from .categorize import Categorize
from theseus.registry import Registry
from .encoder import LabelEncode


TRANSFORM_REGISTRY = Registry('TRANSFORM')

TRANSFORM_REGISTRY.register(PreprocessCompose)
TRANSFORM_REGISTRY.register(DateDecompose)
TRANSFORM_REGISTRY.register(FillNaN)
TRANSFORM_REGISTRY.register(CSVSaver)
TRANSFORM_REGISTRY.register(ToDatetime)
TRANSFORM_REGISTRY.register(DropColumns)
TRANSFORM_REGISTRY.register(DropDuplicatedRows)
TRANSFORM_REGISTRY.register(DropSingleValuedColumns)
TRANSFORM_REGISTRY.register(DropColumnsWithNameFiltered)
TRANSFORM_REGISTRY.register(DropEmptyColumns)
TRANSFORM_REGISTRY.register(Categorize)
TRANSFORM_REGISTRY.register(LabelEncode)