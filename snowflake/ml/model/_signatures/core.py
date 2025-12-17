import textwrap
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Sequence,
    Union,
    final,
    get_args,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

import snowflake.snowpark.types as spt
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)

if TYPE_CHECKING:
    import mlflow
    import torch

PandasExtensionTypes = Union[
    pd.Int8Dtype,
    pd.Int16Dtype,
    pd.Int32Dtype,
    pd.Int64Dtype,
    pd.UInt8Dtype,
    pd.UInt16Dtype,
    pd.UInt32Dtype,
    pd.UInt64Dtype,
    pd.Float32Dtype,
    pd.Float64Dtype,
    pd.BooleanDtype,
    pd.StringDtype,
]


class DataType(Enum):
    def __init__(self, value: str, snowpark_type: type[spt.DataType], numpy_type: npt.DTypeLike) -> None:
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

    TIMESTAMP_NTZ = ("datetime64[ns]", spt.TimestampType, "datetime64[ns]")

    def as_snowpark_type(self) -> spt.DataType:
        """Convert to corresponding Snowpark Type.

        Returns:
            A Snowpark type.
        """
        return self._snowpark_type()

    def __repr__(self) -> str:
        return f"DataType.{self.name}"

    @classmethod
    def from_numpy_type(cls, input_type: Union[npt.DTypeLike, PandasExtensionTypes]) -> "DataType":
        """Translate numpy dtype to DataType for signature definition.

        Args:
            input_type: The numpy dtype or Pandas Extension Dtype

        Raises:
            SnowflakeMLException: NotImplementedError: Raised when the given numpy type is not supported.

        Returns:
            Corresponding DataType.
        """
        # To support pandas extension dtype
        if isinstance(input_type, get_args(PandasExtensionTypes)):
            input_type = input_type.type

        np_to_snowml_type_mapping = {i._numpy_type: i for i in DataType}

        # Add datetime types:
        datetime_res = ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"]

        for res in datetime_res:
            np_to_snowml_type_mapping[f"datetime64[{res}]"] = DataType.TIMESTAMP_NTZ

        for potential_type in np_to_snowml_type_mapping.keys():
            if np.can_cast(input_type, potential_type, casting="no"):
                # This is used since the same dtype might represented in different ways.
                return np_to_snowml_type_mapping[potential_type]
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_IMPLEMENTED,
            original_exception=NotImplementedError(f"Type {input_type} is not supported as a DataType."),
        )

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
            SnowflakeMLException: NotImplementedError: Raised when the given numpy type is not supported.

        Returns:
            Corresponding DataType.
        """
        if isinstance(snowpark_type, spt.ArrayType):
            actual_sp_type = snowpark_type.element_type
        else:
            actual_sp_type = snowpark_type

        snowpark_to_snowml_type_mapping: dict[type[spt.DataType], DataType] = {
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
                warnings.warn(
                    f"Warning: Type {snowpark_type}"
                    " is being automatically converted to INT64 in the Snowpark DataFrame. "
                    "This automatic conversion may lead to potential precision loss and rounding errors. "
                    "If you wish to prevent this conversion, you should manually perform "
                    "the necessary data type conversion.",
                    stacklevel=2,
                )
                return DataType.INT64
            else:
                warnings.warn(
                    f"Warning: Type {snowpark_type}"
                    " is being automatically converted to DOUBLE in the Snowpark DataFrame. "
                    "This automatic conversion may lead to potential precision loss and rounding errors. "
                    "If you wish to prevent this conversion, you should manually perform "
                    "the necessary data type conversion.",
                    stacklevel=2,
                )
                return DataType.DOUBLE
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_IMPLEMENTED,
            original_exception=NotImplementedError(f"Type {snowpark_type} is not supported as a DataType."),
        )


class BaseFeatureSpec(ABC):
    """Abstract Class for specification of a feature."""

    def __init__(self, name: str, shape: Optional[tuple[int, ...]]) -> None:
        self._name = name

        if shape and not isinstance(shape, tuple):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_TYPE,
                original_exception=TypeError("Shape should be a tuple if presented."),
            )
        self._shape = shape

    @final
    @property
    def name(self) -> str:
        """Name of the feature."""
        return self._name

    @abstractmethod
    def as_snowpark_type(self) -> spt.DataType:
        """Convert to corresponding Snowpark Type."""

    @abstractmethod
    def as_dtype(self, force_numpy_dtype: bool = False) -> Union[npt.DTypeLike, str, PandasExtensionTypes]:
        """Convert to corresponding local Type."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialization"""

    @classmethod
    @abstractmethod
    def from_dict(self, input_dict: dict[str, Any]) -> "BaseFeatureSpec":
        """Deserialization"""


class FeatureSpec(BaseFeatureSpec):
    """Specification of a feature in Snowflake native model packaging."""

    def __init__(
        self,
        name: str,
        dtype: DataType,
        shape: Optional[tuple[int, ...]] = None,
        nullable: bool = True,
    ) -> None:
        """
        Initialize a feature.

        Args:
            name: Name of the feature.
            dtype: Type of the elements in the feature.
            nullable: Whether the feature is nullable. Defaults to True.
            shape: Used to represent scalar feature, 1-d feature list,
                or n-d tensor. Use -1 to represent variable length. Defaults to None.

                Examples:
                    - None: scalar
                    - (2,): 1d list with a fixed length of 2.
                    - (-1,): 1d list with variable length, used for ragged tensor representation.
                    - (d1, d2, d3): 3d tensor.
            nullable: Whether the feature is nullable. Defaults to True.

        Raises:
            SnowflakeMLException: TypeError: When the dtype input type is incorrect.
            SnowflakeMLException: TypeError: When the shape input type is incorrect.
        """
        super().__init__(name=name, shape=shape)

        if not isinstance(dtype, DataType):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_TYPE,
                original_exception=TypeError("dtype should be a model signature datatype."),
            )
        self._dtype = dtype

        self._nullable = nullable

    def as_snowpark_type(self) -> spt.DataType:
        result_type = self._dtype.as_snowpark_type()
        if not self._shape:
            return result_type
        for _ in range(len(self._shape)):
            result_type = spt.ArrayType(result_type)
        return result_type

    def as_dtype(self, force_numpy_dtype: bool = False) -> Union[npt.DTypeLike, str, PandasExtensionTypes]:
        """Convert to corresponding local Type."""

        if not self._shape:
            # scalar dtype: use keys from `np.sctypeDict` to prevent unit-less dtype 'datetime64'
            if "datetime64" in self._dtype._value:
                return self._dtype._value

            np_type = self._dtype._numpy_type
            if self._nullable and not force_numpy_dtype:
                np_to_pd_dtype_mapping = {
                    np.int8: pd.Int8Dtype(),
                    np.int16: pd.Int16Dtype(),
                    np.int32: pd.Int32Dtype(),
                    np.int64: pd.Int64Dtype(),
                    np.uint8: pd.UInt8Dtype(),
                    np.uint16: pd.UInt16Dtype(),
                    np.uint32: pd.UInt32Dtype(),
                    np.uint64: pd.UInt64Dtype(),
                    np.float32: pd.Float32Dtype(),
                    np.float64: pd.Float64Dtype(),
                    np.bool_: pd.BooleanDtype(),
                    np.str_: pd.StringDtype(),
                }

                return np_to_pd_dtype_mapping.get(np_type, np_type)  # type: ignore[arg-type]

            return np_type
        return np.object_

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FeatureSpec):
            return self._name == other._name and self._dtype == other._dtype and self._shape == other._shape
        else:
            return False

    def __repr__(self) -> str:
        shape_str = f", shape={repr(self._shape)}" if self._shape else ""
        return (
            f"FeatureSpec(dtype={repr(self._dtype)}, "
            f"name={repr(self._name)}{shape_str}, nullable={repr(self._nullable)})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the feature group into a dict.

        Returns:
            A dict that serializes the feature group.
        """
        base_dict: dict[str, Any] = {"type": self._dtype.name, "name": self._name, "nullable": self._nullable}
        if self._shape is not None:
            base_dict["shape"] = self._shape
        return base_dict

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> "FeatureSpec":
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
        # If nullable is not provided, default to False for backward compatibility.
        nullable = input_dict.get("nullable", False)
        return FeatureSpec(name=name, dtype=type, shape=shape, nullable=nullable)

    @classmethod
    def from_mlflow_spec(
        cls, input_spec: Union["mlflow.types.ColSpec", "mlflow.types.TensorSpec"], feature_name: str
    ) -> "FeatureSpec":
        import mlflow

        if isinstance(input_spec, mlflow.types.ColSpec):
            name = input_spec.name
            if name is None:
                name = feature_name
            return FeatureSpec(name=name, dtype=DataType.from_numpy_type(input_spec.type.to_numpy()))
        elif isinstance(input_spec, mlflow.types.TensorSpec):
            if len(input_spec.shape) == 1:
                shape = None
            else:
                shape = tuple(input_spec.shape[1:])

            name = input_spec.name
            if name is None:
                name = feature_name
            return FeatureSpec(name=name, dtype=DataType.from_numpy_type(input_spec.type), shape=shape)
        else:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_IMPLEMENTED,
                original_exception=NotImplementedError(f"MLFlow schema type {type(input_spec)} is not supported."),
            )


class FeatureGroupSpec(BaseFeatureSpec):
    """Specification of a group of features in Snowflake native model packaging."""

    def __init__(self, name: str, specs: list[BaseFeatureSpec], shape: Optional[tuple[int, ...]] = None) -> None:
        """Initialize a feature group.

        Args:
            name: Name of the feature group.
            specs: A list of feature specifications that composes the group. All children feature specs have to have
                name. And all of them should have the same type.
            shape: Used to represent scalar feature, 1-d feature list,
                or n-d tensor. Use -1 to represent variable length. Defaults to None.

                Examples:
                    - None: scalar
                    - (2,): 1d list with a fixed length of 2.
                    - (-1,): 1d list with variable length, used for ragged tensor representation.
                    - (d1, d2, d3): 3d tensor.
        """
        super().__init__(name=name, shape=shape)
        self._specs = specs
        self._validate()

    def _validate(self) -> None:
        if len(self._specs) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT, original_exception=ValueError("No children feature specs.")
            )
        # each has to have name, and same type
        if not all(s._name is not None for s in self._specs):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError("All children feature specs have to have name."),
            )

    def as_snowpark_type(self) -> spt.DataType:
        spt_type = spt.StructType(
            fields=[
                spt.StructField(
                    s._name, datatype=s.as_snowpark_type(), nullable=s._nullable if isinstance(s, FeatureSpec) else True
                )
                for s in self._specs
            ]
        )
        if not self._shape:
            return spt_type
        return spt.ArrayType(spt_type)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FeatureGroupSpec):
            return self._name == other._name and self._specs == other._specs and self._shape == other._shape
        else:
            return False

    def __repr__(self) -> str:
        spec_strs = ",\n\t\t".join(repr(spec) for spec in self._specs)
        shape_str = f", shape={repr(self._shape)}" if self._shape else ""
        return textwrap.dedent(
            f"""FeatureGroupSpec(
                name={repr(self._name)},
                specs=[
                    {spec_strs}
                ]{shape_str}
            )
            """
        )

    def as_dtype(self, force_numpy_dtype: bool = False) -> Union[npt.DTypeLike, str, PandasExtensionTypes]:
        return np.object_

    def to_dict(self) -> dict[str, Any]:
        """Serialize the feature group into a dict.

        Returns:
            A dict that serializes the feature group.
        """
        base_dict: dict[str, Any] = {"name": self._name, "specs": [s.to_dict() for s in self._specs]}
        if self._shape is not None:
            base_dict["shape"] = self._shape
        return base_dict

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> "FeatureGroupSpec":
        """Deserialize the feature group from a dict.

        Args:
            input_dict: The dict containing information of the feature group.

        Returns:
            A feature group instance deserialized and created from the dict.
        """
        specs = []
        for e in input_dict["specs"]:
            spec = FeatureGroupSpec.from_dict(e) if "specs" in e else FeatureSpec.from_dict(e)
            specs.append(spec)
        shape = input_dict.get("shape", None)
        if shape:
            shape = tuple(shape)
        return FeatureGroupSpec(name=input_dict["name"], specs=specs, shape=shape)


class ModelSignature:
    """Signature of a model that specifies the input and output of a model."""

    def __init__(self, inputs: Sequence[BaseFeatureSpec], outputs: Sequence[BaseFeatureSpec]) -> None:
        """Initialize a model signature.

        Args:
            inputs: A sequence of feature specifications and feature group specifications that will compose
                the input of the model.
            outputs: A sequence of feature specifications and feature group specifications that will compose
                the output of the model.
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

    def to_dict(self) -> dict[str, Any]:
        """Generate a dict to represent the whole signature.

        Returns:
            A dict that serializes the signature.
        """

        return {
            "inputs": [spec.to_dict() for spec in self._inputs],
            "outputs": [spec.to_dict() for spec in self._outputs],
        }

    @classmethod
    def from_dict(cls, loaded: dict[str, Any]) -> "ModelSignature":
        """Create a signature given the dict containing specifications of children features and feature groups.

        Args:
            loaded: The dict to be deserialized.

        Returns:
            A signature deserialized and created from the dict.
        """
        sig_outs = loaded["outputs"]
        sig_inputs = loaded["inputs"]

        deserialize_spec: Callable[[dict[str, Any]], BaseFeatureSpec] = lambda sig_spec: (
            FeatureGroupSpec.from_dict(sig_spec) if "specs" in sig_spec else FeatureSpec.from_dict(sig_spec)
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

    def _repr_html_(self) -> str:
        """Generate an HTML representation of the model signature.

        Returns:
            str: HTML string containing formatted signature details.
        """
        from snowflake.ml.utils import html_utils

        # Create collapsible sections for inputs and outputs
        inputs_content = html_utils.create_features_html(self.inputs, "Input")
        outputs_content = html_utils.create_features_html(self.outputs, "Output")

        inputs_section = html_utils.create_collapsible_section("Inputs", inputs_content, open_by_default=True)
        outputs_section = html_utils.create_collapsible_section("Outputs", outputs_content, open_by_default=True)

        content = f"""
        <div style="margin-top: 10px;">
            {inputs_section}
            {outputs_section}
        </div>
        """

        return html_utils.create_base_container("Model Signature", content)

    @classmethod
    def from_mlflow_sig(cls, mlflow_sig: "mlflow.models.ModelSignature") -> "ModelSignature":
        return ModelSignature(
            inputs=[
                FeatureSpec.from_mlflow_spec(spec, f"input_feature_{idx}") for idx, spec in enumerate(mlflow_sig.inputs)
            ],
            outputs=[
                FeatureSpec.from_mlflow_spec(spec, f"output_feature_{idx}")
                for idx, spec in enumerate(mlflow_sig.outputs)
            ],
        )
