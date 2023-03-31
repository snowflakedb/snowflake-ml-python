from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd


# TODO: Better typing, string now to avoid yaml object serde.
# TODO: Handle numpy dtypes more.
class DataType:
    int = "int32"
    long = "int64"
    float = "float32"
    double = "float64"


# A registry of all the type spec for dispatch
TYPE_SPEC_REGISTRY: Dict[str, Type["TypeSpec"]] = dict()


def register_spec(cls: Type["TypeSpec"]) -> Type["TypeSpec"]:
    TYPE_SPEC_REGISTRY[cls.TYPE] = cls
    return cls


# TODO(halu): Better typing
class TypeSpec(ABC):
    """Specification of column container data types for input and output schema.(e.g., numpy.ndarray, pd.DataFrame)
    for native Snowflake model packaging format.

    The spec will be captured as part of native model metadata and will provides means to auto-generate data type
    conversion code.(e.g., from dataframe to ndarray expected by model)

    """

    TYPE: str = "_base"

    def __init__(self, cols: Optional[List[str]] = None) -> None:
        self.cols = cols

    @abstractmethod
    def py_type(self) -> Any:
        ...

    @classmethod
    def from_dict(cls, type_dict: Dict[str, Any]) -> "TypeSpec":
        spec_type = type_dict["_type"]
        type_dict.pop("_type")
        return TYPE_SPEC_REGISTRY[spec_type](**type_dict)

    def to_dict(self) -> Dict[str, Any]:
        res = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        res.update({"_type": self.TYPE})
        return res

    @abstractmethod
    def types(self) -> List[str]:
        ...


@register_spec
class NumpyNdarray(TypeSpec):
    TYPE = "np.ndarray"

    def __init__(self, dtype: str, dim: int, cols: Optional[List[str]] = None) -> None:
        self.dtype = dtype
        self.dim = dim
        super().__init__(cols)

    def py_type(self) -> Any:
        return np.ndarray

    def types(self) -> List[str]:
        return [self.dtype] * self.dim


@register_spec
class PandasDataFrame(TypeSpec):
    TYPE = "pd.DataFrame"

    def __init__(self, dtypes: Union[str, List[str]], dim: int, cols: Optional[List[str]] = None) -> None:
        self.dtypes = dtypes if isinstance(dtypes, list) else [dtypes]
        self.dim = dim
        super().__init__(cols)

    def py_type(self) -> Any:
        return pd.DataFrame

    def types(self) -> List[str]:
        return self.dtypes


@register_spec
class PandasSeries(TypeSpec):
    TYPE = "pd.Series"

    def __init__(self, dtype: str, cols: Optional[List[str]] = None) -> None:
        self.dtype = dtype
        super().__init__(cols)

    def py_type(self) -> Any:
        return pd.Series

    def types(self) -> List[str]:
        return [self.dtype]


def infer_spec(data: Any) -> TypeSpec:
    if isinstance(data, np.ndarray):
        shape = data.shape
        assert len(shape) <= 2, "Only 1d or 2d-array supported."
        dim = 1 if len(shape) == 1 else shape[1]
        dtype = data.dtype
        return NumpyNdarray(dtype.name, dim)
    elif isinstance(data, pd.Series):
        col = getattr(data, "name", None)
        dtype = data.dtype
        return PandasSeries(dtype, col)
    elif isinstance(data, pd.DataFrame):
        dtypes = [data[col].dtype for col in data.columns]
        cols = data.columns if data.columns.dtype == np.dtype("O") else None
        return PandasDataFrame(dtypes, len(data.columns), cols)
    else:
        raise TypeError("Unknown type for type spec inference.")


def to_pd_series(spec: TypeSpec, y: Any) -> pd.Series:
    if isinstance(spec, NumpyNdarray):
        return pd.Series(y)
    else:
        raise NotImplementedError


def from_pd_dataframe(spec: TypeSpec, X: pd.DataFrame) -> Any:
    if isinstance(spec, NumpyNdarray):
        return X.to_numpy()
