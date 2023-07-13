import json
import textwrap
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    final,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import TypeGuard

import snowflake.snowpark
import snowflake.snowpark.types as spt
from snowflake.ml._internal import type_utils
from snowflake.ml._internal.utils import formatting, identifier
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._deploy_client.warehouse import infer_template

if TYPE_CHECKING:
    import tensorflow
    import torch


class DataType(Enum):
    def __init__(self, value: str, snowpark_type: Type[spt.DataType], numpy_type: npt.DTypeLike) -> None:
        self._value = value
        self._snowpark_type = snowpark_type
        self._numpy_type = numpy_type

    INT8 = ("int8", spt.ByteType, np.int8)
    INT16 = ("int16", spt.ShortType, np.int16)
    INT32 = ("int32", spt.IntegerType, np.int32)
    INT64 = ("int64", spt.LongType, np.int64)

    FLOAT = ("float", spt.FloatType, np.float32)
    DOUBLE = ("double", spt.DoubleType, np.float64)

    UINT8 = ("uint8", spt.ByteType, np.uint8)
    UINT16 = ("uint16", spt.ShortType, np.uint16)
    UINT32 = ("uint32", spt.IntegerType, np.uint32)
    UINT64 = ("uint64", spt.LongType, np.uint64)

    BOOL = ("bool", spt.BooleanType, np.bool_)
    STRING = ("string", spt.StringType, np.str_)
    BYTES = ("bytes", spt.BinaryType, np.bytes_)

    def as_snowpark_type(self) -> spt.DataType:
        """Convert to corresponding Snowpark Type.

        Returns:
            A Snowpark type.
        """
        return self._snowpark_type()

    def __repr__(self) -> str:
        return f"DataType.{self.name}"

    @classmethod
    def from_numpy_type(cls, np_type: npt.DTypeLike) -> "DataType":
        """Translate numpy dtype to DataType for signature definition.

        Args:
            np_type: The numpy dtype.

        Raises:
            NotImplementedError: Raised when the given numpy type is not supported.

        Returns:
            Corresponding DataType.
        """
        np_to_snowml_type_mapping = {i._numpy_type: i for i in DataType}
        for potential_type in np_to_snowml_type_mapping.keys():
            if np.can_cast(np_type, potential_type, casting="no"):
                # This is used since the same dtype might represented in different ways.
                return np_to_snowml_type_mapping[potential_type]
        raise NotImplementedError(f"Type {np_type} is not supported as a DataType.")

    @classmethod
    def from_torch_type(cls, torch_type: "torch.dtype") -> "DataType":
        import torch

        """Translate torch dtype to DataType for signature definition.

        Args:
            torch_type: The torch dtype.

        Returns:
            Corresponding DataType.
        """
        torch_dtype_to_numpy_dtype_mapping = {
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.bool: np.bool_,
        }
        return cls.from_numpy_type(torch_dtype_to_numpy_dtype_mapping[torch_type])

    @classmethod
    def from_snowpark_type(cls, snowpark_type: spt.DataType) -> "DataType":
        """Translate snowpark type to DataType for signature definition.

        Args:
            snowpark_type: The snowpark type.

        Raises:
            NotImplementedError: Raised when the given numpy type is not supported.

        Returns:
            Corresponding DataType.
        """
        if isinstance(snowpark_type, spt.ArrayType):
            actual_sp_type = snowpark_type.element_type
        else:
            actual_sp_type = snowpark_type

        snowpark_to_snowml_type_mapping: Dict[Type[spt.DataType], DataType] = {
            i._snowpark_type: i
            for i in DataType
            # We by default infer as signed integer.
            if i not in [DataType.UINT8, DataType.UINT16, DataType.UINT32, DataType.UINT64]
        }
        for potential_type in snowpark_to_snowml_type_mapping.keys():
            if isinstance(actual_sp_type, potential_type):
                return snowpark_to_snowml_type_mapping[potential_type]
        # Fallback for decimal type.
        if isinstance(snowpark_type, spt.DecimalType):
            if snowpark_type.scale == 0:
                return DataType.INT64
        raise NotImplementedError(f"Type {snowpark_type} is not supported as a DataType.")

    def is_same_snowpark_type(self, incoming_snowpark_type: spt.DataType) -> bool:
        """Check if provided snowpark type is the same as Data Type.

        Args:
            incoming_snowpark_type: The snowpark type.

        Raises:
            NotImplementedError: Raised when the given numpy type is not supported.

        Returns:
            If the provided snowpark type is the same as the DataType.
        """
        # Special handle for Decimal Type.
        if isinstance(incoming_snowpark_type, spt.DecimalType):
            if incoming_snowpark_type.scale == 0:
                return self == DataType.INT64 or self == DataType.UINT64
            raise NotImplementedError(f"Type {incoming_snowpark_type} is not supported as a DataType.")

        return isinstance(incoming_snowpark_type, self._snowpark_type)


class BaseFeatureSpec(ABC):
    """Abstract Class for specification of a feature."""

    def __init__(self, name: str) -> None:
        self._name = name

    @final
    @property
    def name(self) -> str:
        """Name of the feature."""
        return self._name

    @abstractmethod
    def as_snowpark_type(self) -> spt.DataType:
        """Convert to corresponding Snowpark Type."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialization"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(self, input_dict: Dict[str, Any]) -> "BaseFeatureSpec":
        """Deserialization"""
        pass


class FeatureSpec(BaseFeatureSpec):
    """Specification of a feature in Snowflake native model packaging."""

    def __init__(
        self,
        name: str,
        dtype: DataType,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Initialize a feature.

        Args:
            name: Name of the feature.
            dtype: Type of the elements in the feature.
            shape: Used to represent scalar feature, 1-d feature list or n-d tensor.
                -1 is used to represent variable length.Defaults to None.

                E.g.
                None: scalar
                (2,): 1d list with fixed len of 2.
                (-1,): 1d list with variable length. Used for ragged tensor representation.
                (d1, d2, d3): 3d tensor.

        Raises:
            TypeError: Raised when the dtype input type is incorrect.
            TypeError: Raised when the shape input type is incorrect.
        """
        super().__init__(name=name)

        if not isinstance(dtype, DataType):
            raise TypeError("dtype should be a model signature datatype.")
        self._dtype = dtype

        if shape and not isinstance(shape, tuple):
            raise TypeError("Shape should be a tuple if presented.")
        self._shape = shape

    def as_snowpark_type(self) -> spt.DataType:
        result_type = self._dtype.as_snowpark_type()
        if not self._shape:
            return result_type
        for _ in range(len(self._shape)):
            result_type = spt.ArrayType(result_type)
        return result_type

    def as_dtype(self) -> npt.DTypeLike:
        """Convert to corresponding local Type."""
        if not self._shape:
            return self._dtype._numpy_type
        return np.object_

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FeatureSpec):
            return self._name == other._name and self._dtype == other._dtype and self._shape == other._shape
        else:
            return False

    def __repr__(self) -> str:
        shape_str = f", shape={repr(self._shape)}" if self._shape else ""
        return f"FeatureSpec(dtype={repr(self._dtype)}, name={repr(self._name)}{shape_str})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the feature group into a dict.

        Returns:
            A dict that serializes the feature group.
        """
        base_dict: Dict[str, Any] = {
            "type": self._dtype.name,
            "name": self._name,
        }
        if self._shape is not None:
            base_dict["shape"] = self._shape
        return base_dict

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any]) -> "FeatureSpec":
        """Deserialize the feature specification from a dict.

        Args:
            input_dict: The dict containing information of the feature specification.

        Returns:
            A feature specification instance deserialized and created from the dict.
        """
        name = input_dict["name"]
        shape = input_dict.get("shape", None)
        if shape:
            shape = tuple(shape)
        type = DataType[input_dict["type"]]
        return FeatureSpec(name=name, dtype=type, shape=shape)


class FeatureGroupSpec(BaseFeatureSpec):
    """Specification of a group of features in Snowflake native model packaging."""

    def __init__(self, name: str, specs: List[FeatureSpec]) -> None:
        """Initialize a feature group.

        Args:
            name: Name of the feature group.
            specs: A list of feature specifications that composes the group. All children feature specs have to have
                name. And all of them should have the same type.
        """
        super().__init__(name=name)
        self._specs = specs
        self._validate()

    def _validate(self) -> None:
        if len(self._specs) == 0:
            raise ValueError("No children feature specs.")
        # each has to have name, and same type
        if not all(s._name is not None for s in self._specs):
            raise ValueError("All children feature specs have to have name.")
        if not (all(s._shape is None for s in self._specs) or all(s._shape is not None for s in self._specs)):
            raise ValueError("All children feature specs have to have same shape.")
        first_type = self._specs[0]._dtype
        if not all(s._dtype == first_type for s in self._specs):
            raise ValueError("All children feature specs have to have same type.")

    def as_snowpark_type(self) -> spt.DataType:
        first_type = self._specs[0].as_snowpark_type()
        return spt.MapType(spt.StringType(), first_type)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FeatureGroupSpec):
            return self._specs == other._specs
        else:
            return False

    def __repr__(self) -> str:
        spec_strs = ",\n\t\t".join(repr(spec) for spec in self._specs)
        return textwrap.dedent(
            f"""FeatureGroupSpec(
                name={repr(self._name)},
                specs=[
                    {spec_strs}
                ]
            )
            """
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the feature group into a dict.

        Returns:
            A dict that serializes the feature group.
        """
        return {"feature_group": {"name": self._name, "specs": [s.to_dict() for s in self._specs]}}

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any]) -> "FeatureGroupSpec":
        """Deserialize the feature group from a dict.

        Args:
            input_dict: The dict containing information of the feature group.

        Returns:
            A feature group instance deserialized and created from the dict.
        """
        specs = []
        for e in input_dict["feature_group"]["specs"]:
            spec = FeatureSpec.from_dict(e)
            specs.append(spec)
        return FeatureGroupSpec(name=input_dict["feature_group"]["name"], specs=specs)


class ModelSignature:
    """Signature of a model that specifies the input and output of a model."""

    def __init__(self, inputs: Sequence[BaseFeatureSpec], outputs: Sequence[BaseFeatureSpec]) -> None:
        """Initialize a model signature

        Args:
            inputs: A sequence of feature specifications and feature group specifications that will compose the
                input of the model.
            outputs: A sequence of feature specifications and feature group specifications that will compose the
                output of the model.
        """
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self) -> Sequence[BaseFeatureSpec]:
        """Inputs of the model, containing a sequence of feature specifications and feature group specifications."""
        return self._inputs

    @property
    def outputs(self) -> Sequence[BaseFeatureSpec]:
        """Outputs of the model, containing a sequence of feature specifications and feature group specifications."""
        return self._outputs

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModelSignature):
            return self._inputs == other._inputs and self._outputs == other._outputs
        else:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Generate a dict to represent the whole signature.

        Returns:
            A dict that serializes the signature.
        """

        return {
            "inputs": [spec.to_dict() for spec in self._inputs],
            "outputs": [spec.to_dict() for spec in self._outputs],
        }

    @classmethod
    def from_dict(cls, loaded: Dict[str, Any]) -> "ModelSignature":
        """Create a signature given the dict containing specifications of children features and feature groups.

        Args:
            loaded: The dict to be deserialized.

        Returns:
            A signature deserialized and created from the dict.
        """
        sig_outs = loaded["outputs"]
        sig_inputs = loaded["inputs"]

        deserialize_spec: Callable[[Dict[str, Any]], BaseFeatureSpec] = (
            lambda sig_spec: FeatureGroupSpec.from_dict(sig_spec)
            if "feature_group" in sig_spec
            else FeatureSpec.from_dict(sig_spec)
        )

        return ModelSignature(
            inputs=[deserialize_spec(s) for s in sig_inputs], outputs=[deserialize_spec(s) for s in sig_outs]
        )

    def __repr__(self) -> str:
        inputs_spec_strs = ",\n\t\t".join(repr(spec) for spec in self._inputs)
        outputs_spec_strs = ",\n\t\t".join(repr(spec) for spec in self._outputs)
        return textwrap.dedent(
            f"""ModelSignature(
                    inputs=[
                        {inputs_spec_strs}
                    ],
                    outputs=[
                        {outputs_spec_strs}
                    ]
                )"""
        )


class _BaseDataHandler(ABC, Generic[model_types._DataType]):
    FEATURE_PREFIX: Final[str] = "feature"
    INPUT_PREFIX: Final[str] = "input"
    OUTPUT_PREFIX: Final[str] = "output"
    SIG_INFER_ROWS_COUNT_LIMIT: Final[int] = 10

    @staticmethod
    @abstractmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[model_types._DataType]:
        ...

    @staticmethod
    @abstractmethod
    def count(data: model_types._DataType) -> int:
        ...

    @staticmethod
    @abstractmethod
    def truncate(data: model_types._DataType) -> model_types._DataType:
        ...

    @staticmethod
    @abstractmethod
    def validate(data: model_types._DataType) -> None:
        ...

    @staticmethod
    @abstractmethod
    def infer_signature(data: model_types._DataType, role: Literal["input", "output"]) -> Sequence[BaseFeatureSpec]:
        ...

    @staticmethod
    @abstractmethod
    def convert_to_df(data: model_types._DataType, ensure_serializable: bool = True) -> pd.DataFrame:
        ...


class _PandasDataFrameHandler(_BaseDataHandler[pd.DataFrame]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[pd.DataFrame]:
        return isinstance(data, pd.DataFrame)

    @staticmethod
    def count(data: pd.DataFrame) -> int:
        return len(data.index)

    @staticmethod
    def truncate(data: pd.DataFrame) -> pd.DataFrame:
        return data.head(min(_PandasDataFrameHandler.count(data), _PandasDataFrameHandler.SIG_INFER_ROWS_COUNT_LIMIT))

    @staticmethod
    def validate(data: pd.DataFrame) -> None:
        df_cols = data.columns

        if df_cols.has_duplicates:  # Rule out categorical index with duplicates
            raise ValueError("Data Validation Error: Duplicate column index is found.")

        assert all(hasattr(data[col], "dtype") for col in data.columns), f"Unknown column confronted in {data}"

        if len(df_cols) == 0:
            raise ValueError("Data Validation Error: Empty data is found.")

        if df_cols.dtype not in [
            np.int64,
            np.uint64,
            np.float64,
            np.object_,
        ]:  # To keep compatibility with Pandas 2.x and 1.x
            raise ValueError("Data Validation Error: Unsupported column index type is found.")

        df_col_dtypes = [data[col].dtype for col in data.columns]
        for df_col, df_col_dtype in zip(df_cols, df_col_dtypes):
            if df_col_dtype == np.dtype("O"):
                # Check if all objects have the same type
                if not all(isinstance(data_row, type(data[df_col][0])) for data_row in data[df_col]):
                    raise ValueError(
                        f"Data Validation Error: Inconsistent type of object found in column data {data[df_col]}."
                    )

                if isinstance(data[df_col][0], list):
                    arr = _convert_list_to_ndarray(data[df_col][0])
                    arr_dtype = DataType.from_numpy_type(arr.dtype)

                    converted_data_list = [_convert_list_to_ndarray(data_row) for data_row in data[df_col]]

                    if not all(
                        DataType.from_numpy_type(converted_data.dtype) == arr_dtype
                        for converted_data in converted_data_list
                    ):
                        raise ValueError(
                            "Data Validation Error: "
                            + f"Inconsistent type of element in object found in column data {data[df_col]}."
                        )

                elif isinstance(data[df_col][0], np.ndarray):
                    arr_dtype = DataType.from_numpy_type(data[df_col][0].dtype)

                    if not all(DataType.from_numpy_type(data_row.dtype) == arr_dtype for data_row in data[df_col]):
                        raise ValueError(
                            "Data Validation Error: "
                            + f"Inconsistent type of element in object found in column data {data[df_col]}."
                        )
                elif not isinstance(data[df_col][0], (str, bytes)):
                    raise ValueError(f"Data Validation Error: Unsupported type confronted in {data[df_col]}")

    @staticmethod
    def infer_signature(data: pd.DataFrame, role: Literal["input", "output"]) -> Sequence[BaseFeatureSpec]:
        feature_prefix = f"{_PandasDataFrameHandler.FEATURE_PREFIX}_"
        df_cols = data.columns
        role_prefix = (
            _PandasDataFrameHandler.INPUT_PREFIX if role == "input" else _PandasDataFrameHandler.OUTPUT_PREFIX
        ) + "_"
        if df_cols.dtype in [np.int64, np.uint64, np.float64]:
            ft_names = [f"{role_prefix}{feature_prefix}{i}" for i in df_cols]
        else:
            ft_names = list(map(str, data.columns.to_list()))

        df_col_dtypes = [data[col].dtype for col in data.columns]

        specs = []
        for df_col, df_col_dtype, ft_name in zip(df_cols, df_col_dtypes, ft_names):
            if df_col_dtype == np.dtype("O"):
                if isinstance(data[df_col][0], list):
                    arr = _convert_list_to_ndarray(data[df_col][0])
                    arr_dtype = DataType.from_numpy_type(arr.dtype)
                    ft_shape = np.shape(data[df_col][0])

                    converted_data_list = [_convert_list_to_ndarray(data_row) for data_row in data[df_col]]

                    if not all(np.shape(converted_data) == ft_shape for converted_data in converted_data_list):
                        ft_shape = (-1,)

                    specs.append(FeatureSpec(dtype=arr_dtype, name=ft_name, shape=ft_shape))
                elif isinstance(data[df_col][0], np.ndarray):
                    arr_dtype = DataType.from_numpy_type(data[df_col][0].dtype)
                    ft_shape = np.shape(data[df_col][0])

                    if not all(np.shape(data_row) == ft_shape for data_row in data[df_col]):
                        ft_shape = (-1,)

                    specs.append(FeatureSpec(dtype=arr_dtype, name=ft_name, shape=ft_shape))
                elif isinstance(data[df_col][0], str):
                    specs.append(FeatureSpec(dtype=DataType.STRING, name=ft_name))
                elif isinstance(data[df_col][0], bytes):
                    specs.append(FeatureSpec(dtype=DataType.BYTES, name=ft_name))
            else:
                specs.append(FeatureSpec(dtype=DataType.from_numpy_type(df_col_dtype), name=ft_name))
        return specs

    @staticmethod
    def convert_to_df(data: pd.DataFrame, ensure_serializable: bool = True) -> pd.DataFrame:
        if not ensure_serializable:
            return data
        # This convert is necessary since numpy dataframe cannot be correctly handled when provided as an element of
        # a list when creating Snowpark Dataframe.
        df_cols = data.columns
        df_col_dtypes = [data[col].dtype for col in data.columns]
        for df_col, df_col_dtype in zip(df_cols, df_col_dtypes):
            if df_col_dtype == np.dtype("O"):
                if isinstance(data[df_col][0], np.ndarray):
                    data[df_col] = data[df_col].map(np.ndarray.tolist)
        return data


class _NumpyArrayHandler(_BaseDataHandler[model_types._SupportedNumpyArray]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[model_types._SupportedNumpyArray]:
        return isinstance(data, np.ndarray)

    @staticmethod
    def count(data: model_types._SupportedNumpyArray) -> int:
        return data.shape[0]

    @staticmethod
    def truncate(data: model_types._SupportedNumpyArray) -> model_types._SupportedNumpyArray:
        return data[: min(_NumpyArrayHandler.count(data), _NumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT)]

    @staticmethod
    def validate(data: model_types._SupportedNumpyArray) -> None:
        if data.shape == (0,):
            # Empty array
            raise ValueError("Data Validation Error: Empty data is found.")

        if data.shape == ():
            # scalar
            raise ValueError("Data Validation Error: Scalar data is found.")

    @staticmethod
    def infer_signature(
        data: model_types._SupportedNumpyArray, role: Literal["input", "output"]
    ) -> Sequence[BaseFeatureSpec]:
        feature_prefix = f"{_NumpyArrayHandler.FEATURE_PREFIX}_"
        dtype = DataType.from_numpy_type(data.dtype)
        role_prefix = (_NumpyArrayHandler.INPUT_PREFIX if role == "input" else _NumpyArrayHandler.OUTPUT_PREFIX) + "_"
        if len(data.shape) == 1:
            return [FeatureSpec(dtype=dtype, name=f"{role_prefix}{feature_prefix}0")]
        else:
            # For high-dimension array, 0-axis is for batch, 1-axis is for column, further more is details of columns.
            features = []
            n_cols = data.shape[1]
            ft_names = [f"{role_prefix}{feature_prefix}{i}" for i in range(n_cols)]
            for col_data, ft_name in zip(data[0], ft_names):
                if isinstance(col_data, np.ndarray):
                    ft_shape = np.shape(col_data)
                    features.append(FeatureSpec(dtype=dtype, name=ft_name, shape=ft_shape))
                else:
                    features.append(FeatureSpec(dtype=dtype, name=ft_name))
            return features

    @staticmethod
    def convert_to_df(data: model_types._SupportedNumpyArray, ensure_serializable: bool = True) -> pd.DataFrame:
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)
        n_cols = data.shape[1]
        if len(data.shape) == 2:
            return pd.DataFrame(data)
        else:
            n_rows = data.shape[0]
            if ensure_serializable:
                return pd.DataFrame(data={i: [data[k, i].tolist() for k in range(n_rows)] for i in range(n_cols)})
            return pd.DataFrame(data={i: [list(data[k, i]) for k in range(n_rows)] for i in range(n_cols)})


class _SeqOfNumpyArrayHandler(_BaseDataHandler[Sequence[model_types._SupportedNumpyArray]]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[Sequence[model_types._SupportedNumpyArray]]:
        if not isinstance(data, list):
            return False
        if len(data) == 0:
            return False
        if isinstance(data[0], np.ndarray):
            return all(isinstance(data_col, np.ndarray) for data_col in data)
        return False

    @staticmethod
    def count(data: Sequence[model_types._SupportedNumpyArray]) -> int:
        return min(_NumpyArrayHandler.count(data_col) for data_col in data)

    @staticmethod
    def truncate(data: Sequence[model_types._SupportedNumpyArray]) -> Sequence[model_types._SupportedNumpyArray]:
        return [
            data_col[: min(_SeqOfNumpyArrayHandler.count(data), _SeqOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT)]
            for data_col in data
        ]

    @staticmethod
    def validate(data: Sequence[model_types._SupportedNumpyArray]) -> None:
        for data_col in data:
            _NumpyArrayHandler.validate(data_col)

    @staticmethod
    def infer_signature(
        data: Sequence[model_types._SupportedNumpyArray], role: Literal["input", "output"]
    ) -> Sequence[BaseFeatureSpec]:
        feature_prefix = f"{_SeqOfNumpyArrayHandler.FEATURE_PREFIX}_"
        features: List[BaseFeatureSpec] = []
        role_prefix = (
            _SeqOfNumpyArrayHandler.INPUT_PREFIX if role == "input" else _SeqOfNumpyArrayHandler.OUTPUT_PREFIX
        ) + "_"

        for i, data_col in enumerate(data):
            dtype = DataType.from_numpy_type(data_col.dtype)
            ft_name = f"{role_prefix}{feature_prefix}{i}"
            if len(data_col.shape) == 1:
                features.append(FeatureSpec(dtype=dtype, name=ft_name))
            else:
                ft_shape = tuple(data_col.shape[1:])
                features.append(FeatureSpec(dtype=dtype, name=ft_name, shape=ft_shape))
        return features

    @staticmethod
    def convert_to_df(
        data: Sequence[model_types._SupportedNumpyArray], ensure_serializable: bool = True
    ) -> pd.DataFrame:
        if ensure_serializable:
            return pd.DataFrame(data={i: data_col.tolist() for i, data_col in enumerate(data)})
        return pd.DataFrame(data={i: list(data_col) for i, data_col in enumerate(data)})


class _SeqOfPyTorchTensorHandler(_BaseDataHandler[Sequence["torch.Tensor"]]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[Sequence["torch.Tensor"]]:
        if not isinstance(data, list):
            return False
        if len(data) == 0:
            return False
        if type_utils.LazyType("torch.Tensor").isinstance(data[0]):
            return all(type_utils.LazyType("torch.Tensor").isinstance(data_col) for data_col in data)
        return False

    @staticmethod
    def count(data: Sequence["torch.Tensor"]) -> int:
        return min(data_col.shape[0] for data_col in data)

    @staticmethod
    def truncate(data: Sequence["torch.Tensor"]) -> Sequence["torch.Tensor"]:
        return [
            data_col[
                : min(_SeqOfPyTorchTensorHandler.count(data), _SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)
            ]
            for data_col in data
        ]

    @staticmethod
    def validate(data: Sequence["torch.Tensor"]) -> None:
        import torch

        for data_col in data:
            if data_col.shape == torch.Size([0]):
                # Empty array
                raise ValueError("Data Validation Error: Empty data is found.")

            if data_col.shape == torch.Size([1]):
                # scalar
                raise ValueError("Data Validation Error: Scalar data is found.")

    @staticmethod
    def infer_signature(data: Sequence["torch.Tensor"], role: Literal["input", "output"]) -> Sequence[BaseFeatureSpec]:
        feature_prefix = f"{_SeqOfPyTorchTensorHandler.FEATURE_PREFIX}_"
        features: List[BaseFeatureSpec] = []
        role_prefix = (
            _SeqOfPyTorchTensorHandler.INPUT_PREFIX if role == "input" else _SeqOfPyTorchTensorHandler.OUTPUT_PREFIX
        ) + "_"

        for i, data_col in enumerate(data):
            dtype = DataType.from_torch_type(data_col.dtype)
            ft_name = f"{role_prefix}{feature_prefix}{i}"
            if len(data_col.shape) == 1:
                features.append(FeatureSpec(dtype=dtype, name=ft_name))
            else:
                ft_shape = tuple(data_col.shape[1:])
                features.append(FeatureSpec(dtype=dtype, name=ft_name, shape=ft_shape))
        return features

    @staticmethod
    def convert_to_df(data: Sequence["torch.Tensor"], ensure_serializable: bool = True) -> pd.DataFrame:
        # Use list(...) instead of .tolist() to ensure that
        # the content is still numpy array so that the type could be preserved.
        # But that would not serializable and cannot use as UDF input and output.
        if ensure_serializable:
            return pd.DataFrame({i: data_col.detach().to("cpu").numpy().tolist() for i, data_col in enumerate(data)})
        return pd.DataFrame({i: list(data_col.detach().to("cpu").numpy()) for i, data_col in enumerate(data)})

    @staticmethod
    def convert_from_df(
        df: pd.DataFrame, features: Optional[Sequence[BaseFeatureSpec]] = None
    ) -> Sequence["torch.Tensor"]:
        import torch

        res = []
        if features:
            for feature in features:
                if isinstance(feature, FeatureGroupSpec):
                    raise NotImplementedError("FeatureGroupSpec is not supported.")
                assert isinstance(feature, FeatureSpec), "Invalid feature kind."
                res.append(torch.from_numpy(np.stack(df[feature.name].to_numpy()).astype(feature._dtype._numpy_type)))
            return res
        return [torch.from_numpy(np.stack(df[col].to_numpy())) for col in df]


class _SeqOfTensorflowTensorHandler(_BaseDataHandler[Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]]):
    @staticmethod
    def can_handle(
        data: model_types.SupportedDataType,
    ) -> TypeGuard[Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]]:
        if not isinstance(data, list):
            return False
        if len(data) == 0:
            return False
        if type_utils.LazyType("tensorflow.Tensor").isinstance(data[0]) or type_utils.LazyType(
            "tensorflow.Variable"
        ).isinstance(data[0]):
            return all(
                type_utils.LazyType("tensorflow.Tensor").isinstance(data_col)
                or type_utils.LazyType("tensorflow.Variable").isinstance(data_col)
                for data_col in data
            )
        return False

    @staticmethod
    def count(data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]) -> int:
        import tensorflow as tf

        rows = []
        for data_col in data:
            shapes = data_col.shape.as_list()
            if data_col.shape == tf.TensorShape(None) or (not shapes) or (shapes[0] is None):
                # Unknown shape array
                raise ValueError("Data Validation Error: Unknown shape data is found.")
            # Make mypy happy
            assert isinstance(shapes[0], int)

            rows.append(shapes[0])

        return min(rows)

    @staticmethod
    def truncate(
        data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]
    ) -> Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]:
        return [
            data_col[
                : min(
                    _SeqOfTensorflowTensorHandler.count(data), _SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT
                )
            ]
            for data_col in data
        ]

    @staticmethod
    def validate(data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]) -> None:
        import tensorflow as tf

        for data_col in data:
            if data_col.shape == tf.TensorShape(None) or any(dim is None for dim in data_col.shape.as_list()):
                # Unknown shape array
                raise ValueError("Data Validation Error: Unknown shape data is found.")

            if data_col.shape == tf.TensorShape([0]):
                # Empty array
                raise ValueError("Data Validation Error: Empty data is found.")

            if data_col.shape == tf.TensorShape([1]) or data_col.shape == tf.TensorShape([]):
                # scalar
                raise ValueError("Data Validation Error: Scalar data is found.")

    @staticmethod
    def infer_signature(
        data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]], role: Literal["input", "output"]
    ) -> Sequence[BaseFeatureSpec]:
        feature_prefix = f"{_SeqOfTensorflowTensorHandler.FEATURE_PREFIX}_"
        features: List[BaseFeatureSpec] = []
        role_prefix = (
            _SeqOfTensorflowTensorHandler.INPUT_PREFIX
            if role == "input"
            else _SeqOfTensorflowTensorHandler.OUTPUT_PREFIX
        ) + "_"

        for i, data_col in enumerate(data):
            dtype = DataType.from_numpy_type(data_col.dtype.as_numpy_dtype)
            ft_name = f"{role_prefix}{feature_prefix}{i}"
            if len(data_col.shape) == 1:
                features.append(FeatureSpec(dtype=dtype, name=ft_name))
            else:
                ft_shape = tuple(data_col.shape[1:])
                features.append(FeatureSpec(dtype=dtype, name=ft_name, shape=ft_shape))
        return features

    @staticmethod
    def convert_to_df(
        data: Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]], ensure_serializable: bool = True
    ) -> pd.DataFrame:
        if ensure_serializable:
            return pd.DataFrame({i: data_col.numpy().tolist() for i, data_col in enumerate(iterable=data)})
        return pd.DataFrame({i: list(data_col.numpy()) for i, data_col in enumerate(iterable=data)})

    @staticmethod
    def convert_from_df(
        df: pd.DataFrame, features: Optional[Sequence[BaseFeatureSpec]] = None
    ) -> Sequence[Union["tensorflow.Tensor", "tensorflow.Variable"]]:
        import tensorflow as tf

        res = []
        if features:
            for feature in features:
                if isinstance(feature, FeatureGroupSpec):
                    raise NotImplementedError("FeatureGroupSpec is not supported.")
                assert isinstance(feature, FeatureSpec), "Invalid feature kind."
                res.append(
                    tf.convert_to_tensor(np.stack(df[feature.name].to_numpy()).astype(feature._dtype._numpy_type))
                )
            return res
        return [tf.convert_to_tensor(np.stack(df[col].to_numpy())) for col in df]


class _ListOfBuiltinHandler(_BaseDataHandler[model_types._SupportedBuiltinsList]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[model_types._SupportedBuiltinsList]:
        return (
            isinstance(data, list)
            and len(data) > 0
            and all(isinstance(data_col, (int, float, bool, str, bytes, list)) for data_col in data)
        )

    @staticmethod
    def count(data: model_types._SupportedBuiltinsList) -> int:
        return len(data)

    @staticmethod
    def truncate(data: model_types._SupportedBuiltinsList) -> model_types._SupportedBuiltinsList:
        return data[: min(_ListOfBuiltinHandler.count(data), _ListOfBuiltinHandler.SIG_INFER_ROWS_COUNT_LIMIT)]

    @staticmethod
    def validate(data: model_types._SupportedBuiltinsList) -> None:
        if not all(isinstance(data_row, type(data[0])) for data_row in data):
            raise ValueError(f"Data Validation Error: Inconsistent type of object found in data {data}.")
        df = pd.DataFrame(data)
        if df.isnull().values.any():
            raise ValueError(f"Data Validation Error: Ill-shaped list data {data} confronted.")

    @staticmethod
    def infer_signature(
        data: model_types._SupportedBuiltinsList, role: Literal["input", "output"]
    ) -> Sequence[BaseFeatureSpec]:
        return _PandasDataFrameHandler.infer_signature(pd.DataFrame(data), role)

    @staticmethod
    def convert_to_df(
        data: model_types._SupportedBuiltinsList,
        ensure_serializable: bool = True,
    ) -> pd.DataFrame:
        return pd.DataFrame(data)


class _SnowparkDataFrameHandler(_BaseDataHandler[snowflake.snowpark.DataFrame]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[snowflake.snowpark.DataFrame]:
        return isinstance(data, snowflake.snowpark.DataFrame)

    @staticmethod
    def count(data: snowflake.snowpark.DataFrame) -> int:
        return data.count()

    @staticmethod
    def truncate(data: snowflake.snowpark.DataFrame) -> snowflake.snowpark.DataFrame:
        return cast(snowflake.snowpark.DataFrame, data.limit(_SnowparkDataFrameHandler.SIG_INFER_ROWS_COUNT_LIMIT))

    @staticmethod
    def validate(data: snowflake.snowpark.DataFrame) -> None:
        schema = data.schema
        for field in schema.fields:
            data_type = field.datatype
            if isinstance(data_type, spt.ArrayType):
                actual_data_type = data_type.element_type
            else:
                actual_data_type = data_type
            if not any(type.is_same_snowpark_type(actual_data_type) for type in DataType):
                raise ValueError(
                    f"Data Validation Error: Unsupported data type {field.datatype} in column {field.name}."
                )

    @staticmethod
    def infer_signature(
        data: snowflake.snowpark.DataFrame, role: Literal["input", "output"]
    ) -> Sequence[BaseFeatureSpec]:
        features: List[BaseFeatureSpec] = []
        schema = data.schema
        for field in schema.fields:
            name = identifier.get_unescaped_names(field.name)
            if isinstance(field.datatype, spt.ArrayType):
                raise NotImplementedError("Cannot infer model signature from Snowpark DataFrame with Array Type.")
            else:
                features.append(FeatureSpec(name=name, dtype=DataType.from_snowpark_type(field.datatype)))
        return features

    @staticmethod
    def convert_to_df(
        data: snowflake.snowpark.DataFrame,
        ensure_serializable: bool = True,
        features: Optional[Sequence[BaseFeatureSpec]] = None,
    ) -> pd.DataFrame:
        # This method do things on top of to_pandas, to make sure the local dataframe got is in correct shape.
        dtype_map = {}
        if features:
            for feature in features:
                if isinstance(feature, FeatureGroupSpec):
                    raise NotImplementedError("FeatureGroupSpec is not supported.")
                assert isinstance(feature, FeatureSpec), "Invalid feature kind."
                dtype_map[feature.name] = feature.as_dtype()
        df_local = data.to_pandas()
        # This is because Array will become string (Even though the correct schema is set)
        # and object will become variant type and requires an additional loads
        # to get correct data otherwise it would be string.
        for field in data.schema.fields:
            if isinstance(field.datatype, spt.ArrayType):
                df_local[identifier.get_unescaped_names(field.name)] = df_local[
                    identifier.get_unescaped_names(field.name)
                ].map(json.loads)
        # Only when the feature is not from inference, we are confident to do the type casting.
        # Otherwise, dtype_map will be empty
        df_local = df_local.astype(dtype=dtype_map)
        return df_local

    @staticmethod
    def convert_from_df(
        session: snowflake.snowpark.Session, df: pd.DataFrame, keep_order: bool = True
    ) -> snowflake.snowpark.DataFrame:
        # This method is necessary to create the Snowpark Dataframe in correct schema.
        # Snowpark ignore the schema argument when providing a pandas DataFrame.
        # However, in this case, if a cell of the original Dataframe is some array type,
        # they will be inferred as VARIANT.
        # To make sure Snowpark get the correct schema, we have to provide in a list of records.
        # However, in this case, the order could not be preserved. Thus, a _ID column has to be added,
        # if keep_order is True.
        # Although in this case, the column with array type can get correct ARRAY type, however, the element
        # type is not preserved, and will become string type. This affect the implementation of convert_from_df.
        df = _PandasDataFrameHandler.convert_to_df(df)
        df_cols = df.columns
        if df_cols.dtype != np.object_:
            raise ValueError("Cannot convert a Pandas DataFrame whose column index is not a string")
        features = _PandasDataFrameHandler.infer_signature(df, role="input")
        # Role will be no effect on the column index. That is to say, the feature name is the actual column name.
        schema_list = []
        for feature in features:
            if isinstance(feature, FeatureGroupSpec):
                raise NotImplementedError("FeatureGroupSpec is not supported.")
            assert isinstance(feature, FeatureSpec), "Invalid feature kind."
            schema_list.append(
                spt.StructField(
                    identifier.get_inferred_name(feature.name),
                    feature.as_snowpark_type(),
                    nullable=df[feature.name].isnull().any(),
                )
            )

        data = df.rename(columns=identifier.get_inferred_name).to_dict("records")
        if keep_order:
            for idx, data_item in enumerate(data):
                data_item[infer_template._KEEP_ORDER_COL_NAME] = idx
            schema_list.append(spt.StructField(infer_template._KEEP_ORDER_COL_NAME, spt.LongType(), nullable=False))
        sp_df = session.create_dataframe(
            data,  # To make sure the schema can be used, otherwise, array will become variant.
            spt.StructType(schema_list),
        )
        return sp_df


_LOCAL_DATA_HANDLERS: List[Type[_BaseDataHandler[Any]]] = [
    _PandasDataFrameHandler,
    _NumpyArrayHandler,
    _ListOfBuiltinHandler,
    _SeqOfNumpyArrayHandler,
    _SeqOfPyTorchTensorHandler,
    _SeqOfTensorflowTensorHandler,
]
_ALL_DATA_HANDLERS = _LOCAL_DATA_HANDLERS + [_SnowparkDataFrameHandler]


def _truncate_data(data: model_types.SupportedDataType) -> model_types.SupportedDataType:
    for handler in _ALL_DATA_HANDLERS:
        if handler.can_handle(data):
            row_count = handler.count(data)
            if row_count <= handler.SIG_INFER_ROWS_COUNT_LIMIT:
                return data

            warnings.warn(
                formatting.unwrap(
                    f"""
                    The sample input has {row_count} rows, thus a truncation happened before inferring signature.
                    This might cause inaccurate signature inference.
                    If that happens, consider specifying signature manually.
                    """
                ),
                category=UserWarning,
            )
            return handler.truncate(data)
    raise NotImplementedError(
        f"Unable to infer model signature: Un-supported type provided {type(data)} for data truncate."
    )


def _infer_signature(
    data: model_types.SupportedLocalDataType, role: Literal["input", "output"]
) -> Sequence[BaseFeatureSpec]:
    """Infer the inputs/outputs signature given a data that could be dataframe, numpy array or list.
        Dispatching is used to separate logic for different types.
        (Not using Python's singledispatch for unsupported feature of union dispatching in 3.8)

    Args:
        data: The data that we want to infer signature from.
        role: a flag indicating that if this is to infer an input or output feature.

    Raises:
        NotImplementedError: Raised when an unsupported data type is provided.

    Returns:
        A sequence of feature specifications and feature group specifications.
    """
    for handler in _ALL_DATA_HANDLERS:
        if handler.can_handle(data):
            handler.validate(data)
            return handler.infer_signature(data, role)
    raise NotImplementedError(
        f"Unable to infer model signature: Un-supported type provided {type(data)} for X type inference."
    )


def _convert_list_to_ndarray(data: List[Any]) -> npt.NDArray[Any]:
    """Create a numpy array from list or nested list. Avoid ragged list and unaligned types.

    Args:
        data: List or nested list.

    Raises:
        ValueError: Raised when ragged nested list or list containing non-basic type confronted.
        ValueError: Raised when ragged nested list or list containing non-basic type confronted.

    Returns:
        The converted numpy array.
    """
    warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
    try:
        arr = np.array(data)
    except np.VisibleDeprecationWarning:
        # In recent version of numpy, this warning should be raised when bad list provided.
        raise ValueError(
            f"Unable to construct signature: Ragged nested or Unsupported list-like data {data} confronted."
        )
    warnings.filterwarnings("default", category=np.VisibleDeprecationWarning)
    if arr.dtype == object:
        # If not raised, then a array of object would be created.
        raise ValueError(
            f"Unable to construct signature: Ragged nested or Unsupported list-like data {data} confronted."
        )
    return arr


def _rename_features(
    features: Sequence[BaseFeatureSpec], feature_names: Optional[List[str]] = None
) -> Sequence[BaseFeatureSpec]:
    """It renames the feature in features provided optional feature names.

    Args:
        features: A sequence of feature specifications and feature group specifications.
        feature_names: A list of names to assign to features and feature groups. Defaults to None.

    Raises:
        ValueError: Raised when provided feature_names does not match the data shape.

    Returns:
        A sequence of feature specifications and feature group specifications being renamed if names provided.
    """
    if feature_names:
        if len(feature_names) == len(features):
            for ft, ft_name in zip(features, feature_names):
                ft._name = ft_name
        else:
            raise ValueError(
                f"{len(feature_names)} feature names are provided, while there are {len(features)} features."
            )
    return features


def _rename_pandas_df(data: pd.DataFrame, features: Sequence[BaseFeatureSpec]) -> pd.DataFrame:
    """It renames pandas dataframe that has non-object column index with provided features.

    Args:
        data: A pandas dataframe to be renamed.
        features: A sequence of feature specifications and feature group specifications to rename the dataframe.

    Raises:
        ValueError: Raised when the data does not have the same number of features as signature.

    Returns:
        A pandas dataframe with columns renamed.
    """
    df_cols = data.columns
    if df_cols.dtype in [np.int64, np.uint64, np.float64]:
        if len(features) != len(data.columns):
            raise ValueError(
                "Data does not have the same number of features as signature. "
                + f"Signature requires {len(features)} features, but have {len(data.columns)} in input data."
            )
        data.columns = pd.Index([feature.name for feature in features])
    return data


def _validate_pandas_df(data: pd.DataFrame, features: Sequence[BaseFeatureSpec]) -> None:
    """It validates pandas dataframe with provided features.

    Args:
        data: A pandas dataframe to be validated.
        features: A sequence of feature specifications and feature group specifications, where the dataframe should fit.

    Raises:
        NotImplementedError: FeatureGroupSpec is not supported.
        ValueError: Raised when a feature cannot be found.
        ValueError: Raised when feature is scalar but confront list element.
        ValueError: Raised when feature type is not aligned in list element.
        ValueError: Raised when feature shape is not aligned in list element.
        ValueError: Raised when feature is scalar but confront array element.
        ValueError: Raised when feature type is not aligned in numpy array element.
        ValueError: Raised when feature shape is not aligned in numpy array element.
        ValueError: Raised when feature type is not aligned in string element.
        ValueError: Raised when feature type is not aligned in bytes element.
    """
    for feature in features:
        ft_name = feature.name
        try:
            data_col = data[ft_name]
        except KeyError:
            raise ValueError(f"Data Validation Error: feature {ft_name} does not exist in data.")

        df_col_dtype = data_col.dtype
        if isinstance(feature, FeatureGroupSpec):
            raise NotImplementedError("FeatureGroupSpec is not supported.")

        assert isinstance(feature, FeatureSpec), "Invalid feature kind."
        ft_type = feature._dtype
        ft_shape = feature._shape
        if df_col_dtype != np.dtype("O"):
            if ft_type != DataType.from_numpy_type(df_col_dtype):
                raise ValueError(
                    f"Data Validation Error in feature {ft_name}: "
                    + f"Feature type {ft_type} is not met by all elements in {data_col}."
                )
            elif ft_shape is not None:
                raise ValueError(
                    f"Data Validation Error in feature {ft_name}: "
                    + "Feature is a array type feature while scalar data is provided."
                )
        else:
            if isinstance(data_col[0], list):
                if not ft_shape:
                    raise ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + "Feature is a scalar feature while list data is provided."
                    )

                converted_data_list = [_convert_list_to_ndarray(data_row) for data_row in data_col]

                if not all(
                    DataType.from_numpy_type(converted_data.dtype) == ft_type for converted_data in converted_data_list
                ):
                    raise ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + f"Feature type {ft_type} is not met by all elements in {data_col}."
                    )

                if ft_shape and ft_shape != (-1,):
                    if not all(np.shape(converted_data) == ft_shape for converted_data in converted_data_list):
                        raise ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + f"Feature shape {ft_shape} is not met by all elements in {data_col}."
                        )
            elif isinstance(data_col[0], np.ndarray):
                if not ft_shape:
                    raise ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + "Feature is a scalar feature while array data is provided."
                    )

                if not all(DataType.from_numpy_type(data_row.dtype) == ft_type for data_row in data_col):
                    raise ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + f"Feature type {ft_type} is not met by all elements in {data_col}."
                    )

                ft_shape = feature._shape
                if ft_shape and ft_shape != (-1,):
                    if not all(np.shape(data_row) == ft_shape for data_row in data_col):
                        ft_shape = (-1,)
                        raise ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + f"Feature shape {ft_shape} is not met by all elements in {data_col}."
                        )
            elif isinstance(data_col[0], str):
                if ft_shape is not None:
                    raise ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + "Feature is a array type feature while scalar data is provided."
                    )
                if ft_type != DataType.STRING:
                    raise ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + f"Feature type {ft_type} is not met by all elements in {data_col}."
                    )
            elif isinstance(data_col[0], bytes):
                if ft_shape is not None:
                    raise ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + "Feature is a array type feature while scalar data is provided."
                    )
                if ft_type != DataType.BYTES:
                    raise ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + f"Feature type {ft_type} is not met by all elements in {data_col}."
                    )


def _validate_snowpark_data(data: snowflake.snowpark.DataFrame, features: Sequence[BaseFeatureSpec]) -> None:
    """Validate Snowpark DataFrame as input

    Args:
        data: A snowpark dataframe to be validated.
        features: A sequence of feature specifications and feature group specifications, where the dataframe should fit.

    Raises:
        NotImplementedError: FeatureGroupSpec is not supported.
        ValueError: Raised when confronting invalid feature.
        ValueError: Raised when a feature cannot be found.
    """
    schema = data.schema
    for feature in features:
        ft_name = feature.name
        found = False
        for field in schema.fields:
            name = identifier.get_unescaped_names(field.name)
            if name == ft_name:
                found = True
                if field.nullable:
                    warnings.warn(
                        f"Warn in feature {ft_name}: Nullable column {field.name} provided,"
                        + " inference might fail if there is null value.",
                        category=RuntimeWarning,
                    )
                if isinstance(feature, FeatureGroupSpec):
                    raise NotImplementedError("FeatureGroupSpec is not supported.")
                assert isinstance(feature, FeatureSpec), "Invalid feature kind."
                ft_type = feature._dtype
                field_data_type = field.datatype
                if isinstance(field_data_type, spt.ArrayType):
                    if feature._shape is None:
                        raise ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + f"Feature is a array feature, while {field.name} is not."
                        )
                    warnings.warn(
                        f"Warn in feature {ft_name}: Feature is a array feature," + " type validation cannot happen.",
                        category=RuntimeWarning,
                    )
                else:
                    if feature._shape:
                        raise ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + f"Feature is a scalar feature, while {field.name} is not."
                        )
                    if not ft_type.is_same_snowpark_type(field_data_type):
                        raise ValueError(
                            f"Data Validation Error in feature {ft_name}: "
                            + f"Feature type {ft_type} is not met by column {field.name}."
                        )
        if not found:
            raise ValueError(f"Data Validation Error: feature {ft_name} does not exist in data.")


def _convert_local_data_to_df(data: model_types.SupportedLocalDataType) -> pd.DataFrame:
    """Convert local data to pandas DataFrame or Snowpark DataFrame

    Args:
        data: The provided data.

    Raises:
        ValueError: Raised when data cannot be handled by any data handler.

    Returns:
        The converted dataframe with renamed column index.
    """
    df = None
    for handler in _LOCAL_DATA_HANDLERS:
        if handler.can_handle(data):
            handler.validate(data)
            df = handler.convert_to_df(data, ensure_serializable=False)
            break
    if df is None:
        raise ValueError(f"Data Validation Error: Un-supported type {type(data)} provided.")
    return df


def _convert_and_validate_local_data(
    data: model_types.SupportedLocalDataType, features: Sequence[BaseFeatureSpec]
) -> pd.DataFrame:
    """Validate the data with features in model signature and convert to DataFrame

    Args:
        features: A list of feature specs that the data should follow.
        data: The provided data.

    Returns:
        The converted dataframe with renamed column index.
    """
    df = _convert_local_data_to_df(data)
    df = _rename_pandas_df(df, features)
    _validate_pandas_df(df, features)
    df = _PandasDataFrameHandler.convert_to_df(df, ensure_serializable=True)

    return df


def infer_signature(
    input_data: model_types.SupportedLocalDataType,
    output_data: model_types.SupportedLocalDataType,
    input_feature_names: Optional[List[str]] = None,
    output_feature_names: Optional[List[str]] = None,
) -> ModelSignature:
    """Infer model signature from given input and output sample data.

    Currently, we support infer the model signature from example input/output data in the following cases:
        - Pandas data frame whose column could have types of supported data types,
            list (including list of supported data types, list of numpy array of supported data types, and nested list),
            and numpy array of supported data types.
            - Does not support DataFrame with CategoricalIndex column index.
            - Does not support DataFrame with column of variant length list or numpy array.
        - Numpy array of supported data types.
        - List of Numpy array of supported data types.
        - List of supported data types, or nested list of supported data types.
            - Does not support list of list of variant length list.

    When a ValueError is raised when inferring the signature, it indicates that the data is ill and it is impossible to
    create a signature reflecting that.
    When a NotImplementedError is raised, it indicates that it might be possible to create a signature reflecting the
    provided data, however, we could not infer it.

    Args:
        input_data: Sample input data for the model.
        output_data: Sample output data for the model.
        input_feature_names: Name for input features. Defaults to None.
        output_feature_names: Name for output features. Defaults to None.

    Returns:
        A model signature.
    """
    inputs = _infer_signature(input_data, role="input")
    inputs = _rename_features(inputs, input_feature_names)
    outputs = _infer_signature(output_data, role="output")
    outputs = _rename_features(outputs, output_feature_names)
    return ModelSignature(inputs, outputs)
