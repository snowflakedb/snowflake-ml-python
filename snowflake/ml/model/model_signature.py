import textwrap
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

import snowflake.snowpark.types as spt
from snowflake.ml.model import type_hints as model_types


class DataType(Enum):
    def __init__(self, value: str, sql_type: str, snowpark_type: spt.DataType, numpy_type: npt.DTypeLike) -> None:
        self._value = value
        self._sql_type = sql_type
        self._snowpark_type = snowpark_type
        self._numpy_type = numpy_type

    INT16 = ("int16", "INTEGER", spt.IntegerType(), np.int16)
    INT32 = ("int32", "INTEGER", spt.IntegerType(), np.int32)
    INT64 = ("int64", "INTEGER", spt.IntegerType(), np.int64)

    FLOAT = ("float", "FLOAT", spt.FloatType(), np.float32)
    DOUBLE = ("double", "DOUBLE", spt.DoubleType(), np.float64)

    UINT16 = ("uint16", "INTEGER", spt.IntegerType(), np.uint16)
    UINT32 = ("uint32", "INTEGER", spt.IntegerType(), np.uint32)
    UINT64 = ("uint64", "INTEGER", spt.IntegerType(), np.uint64)

    BOOL = ("bool", "BOOLEAN", spt.BooleanType(), np.bool8)
    STRING = ("string", "VARCHAR", spt.StringType(), np.str0)
    BYTES = ("bytes", "VARBINARY", spt.BinaryType(), np.bytes0)

    def as_sql_type(self) -> str:
        """Convert to corresponding Snowflake Logic Type.

        Returns:
            A Snowflake Logic Type.
        """
        return self._sql_type

    def as_snowpark_type(self) -> spt.DataType:
        """Convert to corresponding Snowpark Type.

        Returns:
            A Snowpark type.
        """
        return self._snowpark_type

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


class BaseFeatureSpec(ABC):
    """Abstract Class for specification of a feature."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Name of the feature."""
        return self._name

    @abstractmethod
    def as_snowpark_type(self) -> spt.DataType:
        """Convert to corresponding Snowpark Type."""
        pass

    @abstractmethod
    def to_dict(self, as_sql_type: Optional[bool] = False) -> Dict[str, Any]:
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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FeatureSpec):
            return self._name == other._name and self._dtype == other._dtype and self._shape == other._shape
        else:
            return False

    def __repr__(self) -> str:
        shape_str = f", shape={repr(self._shape)}" if self._shape else ""
        return f"FeatureSpec(dtype={repr(self._dtype)}, name={repr(self._name)}{shape_str})"

    def to_dict(self, as_sql_type: Optional[bool] = False) -> Dict[str, Any]:
        """Serialize the feature group into a dict.

        Args:
            as_sql_type: Whether to use Snowflake Logic Types.

        Returns:
            A dict that serializes the feature group.
        """
        base_dict: Dict[str, Any] = {
            "type": self._dtype.as_sql_type() if as_sql_type else self._dtype.name,
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
            raise ValueError("All children feature specs have to have same type.")
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

    def to_dict(self, as_sql_type: Optional[bool] = False) -> Dict[str, Any]:
        """Serialize the feature group into a dict.

        Args:
            as_sql_type: Whether to use Snowflake Logic Types.

        Returns:
            A dict that serializes the feature group.
        """
        return {
            "feature_group": {"name": self._name, "specs": [s.to_dict(as_sql_type=as_sql_type) for s in self._specs]}
        }

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

    def to_dict(self, as_sql_type: Optional[bool] = False) -> Dict[str, Any]:
        """Generate a dict to represent the whole signature.

        Args:
            as_sql_type: Whether to use Snowflake Logic Types.

        Returns:
            A dict that serializes the signature.
        """

        return {
            "inputs": [spec.to_dict(as_sql_type=as_sql_type) for spec in self._inputs],
            "outputs": [spec.to_dict(as_sql_type=as_sql_type) for spec in self._outputs],
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


def _infer_signature(
    data: model_types.SupportedDataType,
    feature_prefix: str = "feature_",
    output_prefix: str = "output_",
    is_output: bool = False,
) -> Sequence[FeatureSpec]:
    """Infer the inputs/outputs signature given a data that could be dataframe, numpy array or list.
        Dispatching is used to separate logic for different types.
        (Not using Python's singledispatch for unsupported feature of union dispatching in 3.8)

    Args:
        data: The data that we want to infer signature from.
        feature_prefix: a prefix string to added before the column name to distinguish them as a fallback.
            Defaults to "feature_".
        output_prefix: a prefix string to added in multi-output case before the column name to distinguish them as a
            fallback. Defaults to "output_".
        is_output: a flag indicating that if this is to infer an output feature.

    Raises:
        NotImplementedError: Raised when an unsupported data type is provided.

    Returns:
        A sequence of feature specifications and feature group specifications.
    """
    if isinstance(data, pd.DataFrame):
        return _infer_signature_pd_DataFrame(data, feature_prefix=feature_prefix)
    if isinstance(data, np.ndarray):
        return _infer_signature_np_ndarray(data, feature_prefix=feature_prefix)
    if isinstance(data, list) and len(data) > 0:
        if is_output and all(isinstance(data_col, np.ndarray) for data_col in data):
            # Added because mypy still claiming that data has a wider type than
            # Sequence[model_types._SupportedNumpyArray] since we don't have pandas stubs.
            data = cast(Sequence[model_types._SupportedNumpyArray], data)
            return _infer_signature_list_multioutput(data, feature_prefix=feature_prefix, output_prefix=output_prefix)
        else:
            return _infer_signature_list_builtins(data, feature_prefix=feature_prefix)
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


def _infer_signature_pd_DataFrame(data: pd.DataFrame, feature_prefix: str = "feature_") -> Sequence[FeatureSpec]:
    """If a dataframe is provided, its index will be used to name these features. Children features specifications are
    are created according to dataframe type. If it is simple type, then scalar feature specification would be created.
    If it is Python list, then a feature specification with shape are created correspondingly.

    Args:
        data: The data that we want to infer signature from.
        feature_prefix: a prefix string to added before the column name to distinguish them as a fallback.
            Defaults to "feature_".

    Raises:
        NotImplementedError: Raised when an unsupported column is provided.
        ValueError: Raised when an empty data is provided.
        NotImplementedError: Raised when an unsupported column index type is provided.

        ValueError: Raised when an object column have different Python object types.

        ValueError: Raised when an column of list have different element types.
        NotImplementedError: Raised when an column of list have different variant shapes.

        ValueError: Raised when an column of array have different element types.
        NotImplementedError: Raised when an column of array have different variant shapes.

        NotImplementedError: Raised when an unsupported data type is provided.

    Returns:
        A sequence of feature specifications and feature group specifications.
    """
    df_cols = data.columns
    if not all(hasattr(data[col], "dtype") for col in data.columns):
        raise NotImplementedError(f"Unable to infer signature: Unsupported column confronted in {data}.")

    if len(df_cols) == 0:
        raise ValueError("Unable to construct signature: Empty data is found.")

    df_col_dtypes = [data[col].to_numpy().dtype for col in data.columns]
    if df_cols.dtype == np.dtype("O"):  # List of String index
        ft_names = list(map(str, data.columns.to_list()))
    elif isinstance(df_cols, pd.RangeIndex):
        ft_names = [f"{feature_prefix}{i}" for i in df_cols]
    elif isinstance(df_cols, pd.CategoricalIndex):
        raise NotImplementedError(
            f"Unable to infer model signature: Unsupported column index type confronted in {df_cols}."
        )
    else:
        ft_names = [str(x) for x in df_cols.to_list()]

    specs = []
    for df_col, df_col_dtype, ft_name in zip(df_cols, df_col_dtypes, ft_names):
        if df_col_dtype == np.dtype("O"):
            # Check if all objects have the same type
            if not all(isinstance(data_row, type(data[df_col][0])) for data_row in data[df_col]):
                raise ValueError(
                    "Unable to construct model signature: "
                    + f"Inconsistent type of object found in column data {data[df_col]}."
                )

            if isinstance(data[df_col][0], list):
                arr = _convert_list_to_ndarray(data[df_col][0])
                arr_dtype = DataType.from_numpy_type(arr.dtype)
                ft_shape = np.shape(data[df_col][0])

                converted_data_list = [_convert_list_to_ndarray(data_row) for data_row in data[df_col]]

                if not all(
                    DataType.from_numpy_type(converted_data.dtype) == arr_dtype
                    for converted_data in converted_data_list
                ):
                    raise ValueError(
                        "Unable to construct model signature: "
                        + f"Inconsistent type of object found in column data {data[df_col]}."
                    )

                if not all(np.shape(converted_data) == ft_shape for converted_data in converted_data_list):
                    raise NotImplementedError(
                        "Unable to infer model signature: "
                        + f"Inconsistent shape of element found in column data {data[df_col]}. "
                        + "Model signature infer for variant length feature is not currently supported. "
                        + "Consider specify the model signature manually."
                    )

                specs.append(FeatureSpec(dtype=arr_dtype, name=ft_name, shape=ft_shape))
            elif isinstance(data[df_col][0], np.ndarray):
                arr_dtype = DataType.from_numpy_type(data[df_col][0].dtype)
                ft_shape = np.shape(data[df_col][0])

                if not all(DataType.from_numpy_type(data_row.dtype) == arr_dtype for data_row in data[df_col]):
                    raise ValueError(
                        "Unable to construct model signature: "
                        + f"Inconsistent type of object found in column data {data[df_col]}."
                    )

                if not all(np.shape(data_row) == ft_shape for data_row in data[df_col]):
                    raise NotImplementedError(
                        "Unable to infer model signature: "
                        + f"Inconsistent shape of element found in column data {data[df_col]}. "
                        + "Model signature infer for variant length feature is not currently supported. "
                        + "Consider specify the model signature manually."
                    )

                specs.append(FeatureSpec(dtype=arr_dtype, name=ft_name, shape=ft_shape))
            elif isinstance(data[df_col][0], str):
                specs.append(FeatureSpec(dtype=DataType.STRING, name=ft_name))
            elif isinstance(data[df_col][0], bytes):
                specs.append(FeatureSpec(dtype=DataType.BYTES, name=ft_name))
            else:
                raise NotImplementedError(f"Unsupported type confronted in {data[df_col]}")
        else:
            specs.append(FeatureSpec(dtype=DataType.from_numpy_type(df_col_dtype), name=ft_name))
    return specs


def _infer_signature_np_ndarray(
    data: model_types._SupportedNumpyArray, feature_prefix: str = "feature_"
) -> Sequence[FeatureSpec]:
    """If a numpy array is provided, `feature_name` if provided is used to name features, otherwise, name like
    `feature_0` `feature_1` will be generated and assigned.

    Args:
        data: The data that we want to infer signature from.
        feature_prefix: a prefix string to added before the column name to distinguish them as a fallback.
            Defaults to "feature_".

    Raises:
        ValueError: Raised when an empty data is provided.
        ValueError: Raised when a scalar data is provided.

    Returns:
        A sequence of feature specifications and feature group specifications.
    """
    if data.shape == (0,):
        # Empty array
        raise ValueError("Unable to construct signature: Empty data is found. Unable to infer signature.")

    if data.shape == ():
        # scalar
        raise ValueError("Unable to construct signature: Scalar data is found. Unable to infer signature.")
    dtype = DataType.from_numpy_type(data.dtype)
    if len(data.shape) == 1:
        return [FeatureSpec(dtype=dtype, name=f"{feature_prefix}0")]
    else:
        # For high-dimension array, 0-axis is for batch, 1-axis is for column, further more is details of columns.
        features = []
        n_cols = data.shape[1]
        ft_names = [f"{feature_prefix}{i}" for i in range(n_cols)]
        for col_data, ft_name in zip(data[0], ft_names):
            if isinstance(col_data, np.ndarray):
                ft_shape = np.shape(col_data)
                features.append(FeatureSpec(dtype=dtype, name=ft_name, shape=ft_shape))
            else:
                features.append(FeatureSpec(dtype=dtype, name=ft_name))
        return features


def _infer_signature_list_builtins(
    data: Union[Sequence[model_types._SupportedNumpyArray], model_types._SupportedBuiltinsList],
    feature_prefix: str = "feature_",
) -> Sequence[FeatureSpec]:
    """If a list or a nested list of built-in types are provided, we treat them as a pd.DataFrame.
        Before that we check if all elements have the same type.
        After converting to dataframe, if the original data has ill shape, there would be nan or None.

    Args:
        data: The data that we want to infer signature from.
        feature_prefix: a prefix string to added before the column name to distinguish them as a fallback.
            Defaults to "feature_".

    Raises:
        ValueError: Raised when the list have different Python object types.
        ValueError: Raised when converted dataframe has nan or None, meaning that the original data is ill-shaped.

    Returns:
        A sequence of feature specifications and feature group specifications.
    """
    if not all(isinstance(data_row, type(data[0])) for data_row in data):
        raise ValueError(f"Unable to construct model signature: Inconsistent type of object found in data {data}.")
    df = pd.DataFrame(data)
    if df.isnull().values.any():
        raise ValueError(f"Unable to construct model signature: Ill-shaped list data {data} confronted.")
    return _infer_signature_pd_DataFrame(df, feature_prefix=feature_prefix)


def _infer_signature_list_multioutput(
    data: Sequence[model_types._SupportedNumpyArray], feature_prefix: str = "feature_", output_prefix: str = "output_"
) -> Sequence[FeatureSpec]:
    """If a Python list is provided, which will happen if user packs a multi-output model. In this case,
    _infer_signature is called for every element of the list, and a output prefix like `output_0_` `output_1_` would be
    added to the name. All children feature specifications would be flatten.

    Args:
        data: The data that we want to infer signature from.
        feature_prefix: a prefix string to added before the column name to distinguish them as a fallback.
            Defaults to "feature_".
        output_prefix: a prefix string to added in multi-output case before the column name to distinguish them as a
            fallback. Defaults to "output_".

    Returns:
        A sequence of feature specifications and feature group specifications.
    """
    # If the estimator is a multi-output estimator, output will be a list of ndarrays.
    features: List[FeatureSpec] = []

    for i, d in enumerate(data):
        inferred_res = _infer_signature(d, feature_prefix=feature_prefix, output_prefix=output_prefix)
        for ft in inferred_res:
            ft._name = f"{output_prefix}{i}_{ft._name}"
        features.extend(inferred_res)
    return features


def _rename_features(
    features: Sequence[FeatureSpec], feature_names: Optional[List[str]] = None
) -> Sequence[FeatureSpec]:
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


def _validate_data_with_features_and_convert_to_df(features: Sequence[BaseFeatureSpec], data: Any) -> pd.DataFrame:
    """Validate the data with features in model signature and convert to DataFrame

    Args:
        features: A list of feature specs that the data should follow.
        data: The provided data.

    Raises:
        ValueError: Raised when input data is empty dataframe.
        ValueError: Raised when input data is empty array.
        ValueError: Raised when input data is scalar.
        ValueError: Raised when input data is list with different types.
        ValueError: Raised when input data is ill-shaped list.
        NotImplementedError: Raised when input data has unsupported types.
        ValueError: Raised when input data has different number of features as the features required.

    Returns:
        The converted dataframe with renamed column index.
    """
    keep_columns = False
    if isinstance(data, pd.DataFrame):
        df_cols = data.columns

        if len(df_cols) == 0:
            raise ValueError("Empty dataframe is invalid input data.")
        if df_cols.dtype == np.dtype("O"):
            # List of String index, users should take care about names
            keep_columns = True
        df = data
    elif isinstance(data, np.ndarray):
        if data.shape == (0,):
            # Empty array
            raise ValueError("Empty array is invalid input data.")

        if data.shape == ():
            # scalar
            raise ValueError("Scalar is invalid input data.")

        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)
        df = pd.DataFrame(data)
    elif isinstance(data, list) and len(data) > 0:
        if not all(isinstance(data_row, type(data[0])) for data_row in data):
            raise ValueError("List of data with different types is invalid input data.")
        df = pd.DataFrame(data)
        if df.isnull().values.any():
            raise ValueError("Ill-shaped list is invalid input data.")
    else:
        raise NotImplementedError(f"Unable to validate data: Un-supported type provided {type(data)} as X.")

    # Rename if that data may have name inferred if provided to infer signature
    if not keep_columns:
        if len(features) != len(df.columns):
            raise ValueError(
                "Input data does not have the same number of features as signature. "
                + f"Signature requires {len(features)} features, but have {len(df.columns)} in input data."
            )
        df.columns = pd.Index([feature.name for feature in features])
    return df


def infer_signature(
    input_data: model_types.SupportedDataType,
    output_data: model_types.SupportedDataType,
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
    inputs = _infer_signature(input_data)
    inputs = _rename_features(inputs, input_feature_names)
    outputs = _infer_signature(output_data, is_output=True)
    outputs = _rename_features(outputs, output_feature_names)
    return ModelSignature(inputs, outputs)
