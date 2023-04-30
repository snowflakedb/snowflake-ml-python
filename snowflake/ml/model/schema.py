from abc import ABC  # , abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class DataType(Enum):
    """Type of data that is supported in Snowflake native model packaging."""

    def __init__(self, value: str) -> None:
        self._value = value

    FLOAT = "float"
    DOUBLE = "double"
    INT32 = "int32"
    INT64 = "int64"
    # TODO(SNOW-786515): support other data types listed in the doc

    def __repr__(self) -> str:
        return str(self.name)


class BaseColSpec(ABC):
    """Abstract Class for specification of a column."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Name of the column."""
        return self._name


class ColSpec(BaseColSpec):
    """Specification of a column in Snowflake native model packaging."""

    def __init__(
        self,
        name: str,
        dtype: DataType,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Initialize a column.

        Args:
            name: Name of the column.
            dtype: Type of the elements in the column.
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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ColSpec):
            return self._name == other._name and self._dtype == other._dtype and self._shape == other._shape
        else:
            return False

    def __repr__(self) -> str:
        rs = "ColSpec("
        rs += f"type={repr(self._dtype)}"
        rs += f", name={self._name}"
        if self._shape:
            rs += f", shape={repr(self._shape)}"
        rs += ")"
        return rs


class ColGroupSpec(BaseColSpec):
    """Specification of a group of columns in Snowflake native model packaging."""

    def __init__(self, name: str, specs: List[ColSpec]) -> None:
        """Initialize a column group.

        Args:
            name: Name of the column group.
            specs: A list of column specifications that composes the group. All children column specs have to have name.
                And all of them should have the same type.
        """
        super().__init__(name=name)
        self._specs = specs
        self._validate()

    def _validate(self) -> None:
        if len(self._specs) == 0:
            raise TypeError("No children col specs.")
        # each has to have name, and same type
        if not all(s._name is not None for s in self._specs):
            raise TypeError("All children col specs have to have name.")
        first_type = self._specs[0]._dtype
        if not all(s._dtype == first_type for s in self._specs):
            raise TypeError("All children col specs have to have same type.")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ColGroupSpec):
            return self._specs == other._specs
        else:
            return False

    def __repr__(self) -> str:
        rs = "ColGroupSpec(\n"
        rs += f"\tname={self._name}\n"
        rs += "\tspecs=[\n\t\t"
        rs += "\n\t\t".join(repr(spec) for spec in self._specs)
        rs += "\n\t])"
        return rs


class Schema:
    """Schema of a model that specifies the input and output of a model."""

    def __init__(self, inputs: Sequence[BaseColSpec], outputs: Sequence[BaseColSpec]) -> None:
        """Initialize a model schema

        Args:
            inputs: A sequence of column specifications and column group specifications that will compose the
                input of the model.
            outputs: A sequence of column specifications and column group specifications that will compose the
                output of the model.
        """
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self) -> Sequence[BaseColSpec]:
        """Inputs of the model, containing a sequence of column specifications and column group specifications."""
        return self._inputs

    @property
    def outputs(self) -> Sequence[BaseColSpec]:
        """Outputs of the model, containing a sequence of column specifications and column group specifications."""
        return self._outputs

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Schema):
            return self._inputs == other._inputs and self._outputs == other._outputs
        else:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Generate a dict to represent the whole schema.

        Returns:
            A dict that serializes the schema.
        """

        def serialize_spec(spec: BaseColSpec) -> Dict[str, Any]:
            if isinstance(spec, ColSpec):
                base_dict: Dict[str, Any] = dict()
                base_dict["type"] = spec._dtype.name
                if spec._name is not None:
                    base_dict["name"] = spec._name
                if spec._shape is not None:
                    base_dict["shape"] = spec._shape
                return base_dict
            elif isinstance(spec, ColGroupSpec):
                return {"column_group": {"name": spec._name, "specs": [serialize_spec(s) for s in spec._specs]}}
            else:
                raise TypeError("Not supported col spec.")

        return {
            "inputs": [serialize_spec(spec) for spec in self._inputs],
            "outputs": [serialize_spec(spec) for spec in self._outputs],
        }

    @classmethod
    def from_dict(cls, loaded: Dict[str, Any]) -> "Schema":
        """Create a schema given the dict containing specifications of children columns and column groups.

        Args:
            loaded: The dict to be deserialized.

        Returns:
            A schema deserialized and created from the dict.
        """
        souts = loaded["outputs"]
        sinputs = loaded["inputs"]

        def deserialize_spec(sspec: Dict[str, Any]) -> BaseColSpec:
            if "column_group" in sspec:
                specs = []
                for e in sspec["column_group"]["specs"]:
                    spec = deserialize_spec(e)
                    assert isinstance(spec, ColSpec)
                    specs.append(spec)
                return ColGroupSpec(name=sspec["column_group"]["name"], specs=specs)
            else:
                name = sspec["name"]
                shape = sspec.get("shape", None)
                type = DataType[sspec["type"]]
                return ColSpec(name=name, dtype=type, shape=shape)

        return Schema(inputs=[deserialize_spec(s) for s in sinputs], outputs=[deserialize_spec(s) for s in souts])

    def __repr__(self) -> str:
        rs = "ModelSchema(\n"
        rs += "\tinputs=[\n\t\t"
        rs += "\n\t\t".join(repr(spec) for spec in self._inputs)
        rs += "\n\t]\n"
        rs += "\toutputs=[\n\t\t"
        rs += "\n\t\t".join(repr(spec) for spec in self._outputs)
        rs += "\n\t]"
        rs += "\n)"
        return rs


# TODO(SNOW-786515): Rework this function.
def from_np(t: Any) -> DataType:
    """Translate numpy dtype to DataType for schema definition.

    Args:
        t: The numpy dtype.

    Returns:
        Corresponding DataType.
    """
    # TODO(halu): Proper mapping
    if t == np.dtype("float32"):
        return DataType.FLOAT
    elif t == np.dtype("float64"):
        return DataType.DOUBLE
    return DataType.INT64


def _infer_schema(data: Any, col_names: Optional[List[str]] = None) -> Sequence[ColSpec]:
    """Infer the inputs/outputs schema given a data that could be dataframe, numpy array or list.

    If a dataframe is provided, its index will be used to name these columns. Children columns specifications are
    are created according to dataframe type. If it is simple type, then scalar column specification would be created.
    If it is Python list, then a column specification with shape are created correspondingly.

    If a numpy array is provided, `col_name` if provided is used to name columns, otherwise, name like `col_1` `col_2`
    will be generated and assigned. Only support 1d or 2d array are currently supported, thus all column specifications
    are created as scalar.

    If a Python list is provided, which will happen if user packs a multi-output model. In this case, _infer_schema is
    called for every element of the list, and a prefix like `output_1_` `output_2_` would be added to the name. All
    children column specifications would be flatten.

    Args:
        data: The data that we want to infer schema from.
        col_names: A list of names to assign to columns and column groups. Defaults to None.

    Raises:
        TypeError: Raised when input is pd.DataFrame but does not have a string index,
        TypeError: Raised when a numpy array that is not 1d or 2d array is provided.
        TypeError: Raised when provided col_names does not match the data shape.
        TypeError: Raised when an unsupported data type is provided.

    Returns:
        A sequence of column specifications and column group specifications.
    """
    if isinstance(data, pd.DataFrame):
        # TODO(SNOW-786515): Support tensor(list or list of list..) and check if uniformed.
        dtypes = [data[col].dtype for col in data.columns]
        if data.columns.dtype != np.dtype("O"):
            # TODO(SNOW-786515): Use col_names if provided in this case
            raise TypeError("pd.DataFrame has to have string columns.")
        cols = data.columns
        specs = []
        # TODO(SNOW-786515): Handle ndarray
        for i in range(len(cols)):
            if data[cols[i]].dtype == np.dtype("O") and isinstance(data[cols[i]][0], list):
                dim = len(data[cols[i]][0])
                specs.append(ColSpec(dtype=from_np(dtypes[i]), name=cols[i], shape=(dim,)))
            else:
                specs.append(ColSpec(dtype=from_np(dtypes[i]), name=cols[i]))
        return specs
    elif isinstance(data, np.ndarray):
        if len(data.shape) > 2:
            raise TypeError(f"Only support 1d or 2d array, not of shape {data.shape}")
        n_cols = 1 if len(data.shape) == 1 else data.shape[1]
        if col_names is None:
            col_names = [f"col_{i}" for i in range(n_cols)]
        if n_cols != len(col_names):
            raise TypeError(f"Col names are {col_names}, data shape not matching.")
        dtype = from_np(data.dtype)
        return [ColSpec(dtype=dtype, name=col_names[i]) for i in range(len(col_names))]
    elif isinstance(data, list) and len(data) > 0 and (isinstance(data[0], np.ndarray)):
        # If the estimator is a multioutput estimator, output will be a list of ndarrays.
        # TODO(SNOW-786515): Respect col_names arguments if provided.
        output_cols: List[ColSpec] = []
        for i, d in enumerate(data):
            output_cols.extend(map(lambda x: ColSpec(dtype=x._dtype, name=f"output_{i}_{x._name}"), _infer_schema(d)))
        return output_cols
    else:
        raise TypeError("Un-supported type {type(label)} for X type inference.")


# TODO(halu): Check status of named arg.
def infer_schema(
    input_data: Any,
    output_data: Any,
    input_col_names: Optional[List[str]] = None,
    output_col_names: Optional[List[str]] = None,
) -> Schema:
    """Infer model schema from given input and output sample data.

    Args:
        input_data: Sample input data for the model.
        output_data: Sample output data for the model.
        input_col_names: Name for input columns. Defaults to None.
        output_col_names: Name for output columns. Defaults to None.

    Returns:
        A model schema.
    """
    inputs = _infer_schema(input_data, input_col_names)
    outputs = _infer_schema(output_data, output_col_names)
    return Schema(inputs, outputs)
