import textwrap
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
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
from snowflake.ml._internal.utils import formatting, identifier
from snowflake.ml.model import type_hints as model_types


class DataType(Enum):
    def __init__(self, value: str, snowpark_type: Type[spt.DataType], numpy_type: npt.DTypeLike) -> None:
        self._value = value
        self._snowpark_type = snowpark_type
        self._numpy_type = numpy_type

    INT8 = ("int8", spt.IntegerType, np.int8)
    INT16 = ("int16", spt.IntegerType, np.int16)
    INT32 = ("int32", spt.IntegerType, np.int32)
    INT64 = ("int64", spt.IntegerType, np.int64)

    FLOAT = ("float", spt.FloatType, np.float32)
    DOUBLE = ("double", spt.DoubleType, np.float64)

    UINT8 = ("uint8", spt.IntegerType, np.uint8)
    UINT16 = ("uint16", spt.IntegerType, np.uint16)
    UINT32 = ("uint32", spt.IntegerType, np.uint32)
    UINT64 = ("uint64", spt.IntegerType, np.uint64)

    BOOL = ("bool", spt.BooleanType, np.bool8)
    STRING = ("string", spt.StringType, np.str0)
    BYTES = ("bytes", spt.BinaryType, np.bytes0)

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
    def from_snowpark_type(cls, snowpark_type: spt.DataType) -> "DataType":
        """Translate snowpark type to DataType for signature definition.

        Args:
            snowpark_type: The snowpark type.

        Raises:
            NotImplementedError: Raised when the given numpy type is not supported.

        Returns:
            Corresponding DataType.
        """
        snowpark_to_snowml_type_mapping: Dict[Type[spt.DataType], DataType] = {
            spt._IntegralType: DataType.INT64,
            **{i._snowpark_type: i for i in DataType if i._snowpark_type != spt.IntegerType},
        }
        for potential_type in snowpark_to_snowml_type_mapping.keys():
            if isinstance(snowpark_type, potential_type):
                return snowpark_to_snowml_type_mapping[potential_type]
        raise NotImplementedError(f"Type {snowpark_type} is not supported as a DataType.")

    def is_same_snowpark_type(self, incoming_snowpark_type: spt.DataType) -> bool:
        """Check if provided snowpark type is the same as Data Type.
            Since for Snowflake all integer types are same, thus when datatype is a integer type, the incoming snowpark
            type can be any type inherit from _IntegralType.

        Args:
            incoming_snowpark_type: The snowpark type.

        Returns:
            If the provided snowpark type is the same as the DataType.
        """
        if self._snowpark_type == spt.IntegerType:
            return isinstance(incoming_snowpark_type, spt._IntegralType)
        else:
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
        """
        super().__init__(name=name)
        self._dtype = dtype
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
        return np.object0

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
    def convert_to_df(data: model_types._DataType) -> Union[pd.DataFrame, snowflake.snowpark.DataFrame]:
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
            np.object0,
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
    def convert_to_df(data: pd.DataFrame) -> pd.DataFrame:
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
        feature_prefix = f"{_PandasDataFrameHandler.FEATURE_PREFIX}_"
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
    def convert_to_df(data: model_types._SupportedNumpyArray) -> pd.DataFrame:
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)
        n_cols = data.shape[1]
        if len(data.shape) == 2:
            return pd.DataFrame(data={i: data[:, i] for i in range(n_cols)})
        else:
            n_rows = data.shape[0]
            return pd.DataFrame(data={i: [np.array(data[k, i]) for k in range(n_rows)] for i in range(n_cols)})


class _ListOfNumpyArrayHandler(_BaseDataHandler[List[model_types._SupportedNumpyArray]]):
    @staticmethod
    def can_handle(data: model_types.SupportedDataType) -> TypeGuard[List[model_types._SupportedNumpyArray]]:
        return (
            isinstance(data, list)
            and len(data) > 0
            and all(_NumpyArrayHandler.can_handle(data_col) for data_col in data)
        )

    @staticmethod
    def count(data: List[model_types._SupportedNumpyArray]) -> int:
        return min(_NumpyArrayHandler.count(data_col) for data_col in data)

    @staticmethod
    def truncate(data: List[model_types._SupportedNumpyArray]) -> List[model_types._SupportedNumpyArray]:
        return [
            data_col[: min(_ListOfNumpyArrayHandler.count(data), _ListOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT)]
            for data_col in data
        ]

    @staticmethod
    def validate(data: List[model_types._SupportedNumpyArray]) -> None:
        for data_col in data:
            _NumpyArrayHandler.validate(data_col)

    @staticmethod
    def infer_signature(
        data: List[model_types._SupportedNumpyArray], role: Literal["input", "output"]
    ) -> Sequence[BaseFeatureSpec]:
        features: List[BaseFeatureSpec] = []
        role_prefix = (
            _ListOfNumpyArrayHandler.INPUT_PREFIX if role == "input" else _ListOfNumpyArrayHandler.OUTPUT_PREFIX
        ) + "_"

        for i, data_col in enumerate(data):
            inferred_res = _NumpyArrayHandler.infer_signature(data_col, role)
            for ft in inferred_res:
                ft._name = f"{role_prefix}{i}_{ft._name[len(role_prefix):]}"
            features.extend(inferred_res)
        return features

    @staticmethod
    def convert_to_df(data: List[model_types._SupportedNumpyArray]) -> pd.DataFrame:
        l_data = []
        for data_col in data:
            if len(data_col.shape) == 1:
                l_data.append(np.expand_dims(data_col, axis=1))
            else:
                l_data.append(data_col)
        arr = np.concatenate(l_data, axis=1)
        return _NumpyArrayHandler.convert_to_df(arr)


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
    def convert_to_df(data: model_types._SupportedBuiltinsList) -> pd.DataFrame:
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
            if not any(type.is_same_snowpark_type(field.datatype) for type in DataType):
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
            features.append(FeatureSpec(name=name, dtype=DataType.from_snowpark_type(field.datatype)))
        return features

    @staticmethod
    def convert_to_df(data: snowflake.snowpark.DataFrame) -> snowflake.snowpark.DataFrame:
        return data


_LOCAL_DATA_HANDLERS: List[Type[_BaseDataHandler[Any]]] = [
    _PandasDataFrameHandler,
    _NumpyArrayHandler,
    _ListOfNumpyArrayHandler,
    _ListOfBuiltinHandler,
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
                if not ft_type.is_same_snowpark_type(field.datatype):
                    raise ValueError(
                        f"Data Validation Error in feature {ft_name}: "
                        + f"Feature type {ft_type} is not met by column {field.name}."
                    )
        if not found:
            raise ValueError(f"Data Validation Error: feature {ft_name} does not exist in data.")


def _convert_and_validate_local_data(
    data: model_types.SupportedDataType, features: Sequence[BaseFeatureSpec]
) -> pd.DataFrame:
    """Validate the data with features in model signature and convert to DataFrame

    Args:
        features: A list of feature specs that the data should follow.
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
            df = handler.convert_to_df(data)
            break
    if df is None:
        raise ValueError(f"Data Validation Error: Un-supported type {type(data)} provided.")
    assert isinstance(df, pd.DataFrame)
    df = _rename_pandas_df(df, features)
    _validate_pandas_df(df, features)

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
