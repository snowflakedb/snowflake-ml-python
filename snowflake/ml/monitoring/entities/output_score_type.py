from __future__ import annotations

from enum import Enum
from typing import List, Mapping

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import type_hints
from snowflake.snowpark import types

# Accepted data types for each OutputScoreType.
REGRESSION_DATA_TYPES = (
    types.ByteType,
    types.ShortType,
    types.IntegerType,
    types.LongType,
    types.FloatType,
    types.DoubleType,
    types.DecimalType,
)
CLASSIFICATION_DATA_TYPES = (
    types.ByteType,
    types.ShortType,
    types.IntegerType,
    types.BooleanType,
    types.BinaryType,
)
PROBITS_DATA_TYPES = (
    types.ByteType,
    types.ShortType,
    types.IntegerType,
    types.LongType,
    types.FloatType,
    types.DoubleType,
    types.DecimalType,
)


# OutputScoreType enum
class OutputScoreType(Enum):
    UNKNOWN = "UNKNOWN"
    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"
    PROBITS = "PROBITS"

    @classmethod
    def deduce_score_type(
        cls,
        table_schema: Mapping[str, types.DataType],
        prediction_columns: List[sql_identifier.SqlIdentifier],
        task: type_hints.Task,
    ) -> OutputScoreType:
        """Find the score type for monitoring given a table schema and the task.

        Args:
            table_schema: Dictionary of column names and types in the source table.
            prediction_columns: List of prediction columns.
            task: Enum value for the task of the model.

        Returns:
            Enum value for the score type, informing monitoring table set up.

        Raises:
            ValueError: If prediction type fails to align with task.
        """
        # Already validated we have just one prediction column type
        prediction_column_type = {table_schema[column_name] for column_name in prediction_columns}.pop()

        if task == type_hints.Task.TABULAR_REGRESSION:
            if isinstance(prediction_column_type, REGRESSION_DATA_TYPES):
                return OutputScoreType.REGRESSION
            else:
                raise ValueError(
                    f"Expected prediction column type to be one of {REGRESSION_DATA_TYPES} "
                    f"for REGRESSION task. Found: {prediction_column_type}."
                )

        elif task == type_hints.Task.TABULAR_BINARY_CLASSIFICATION:
            if isinstance(prediction_column_type, CLASSIFICATION_DATA_TYPES):
                return OutputScoreType.CLASSIFICATION
            elif isinstance(prediction_column_type, PROBITS_DATA_TYPES):
                return OutputScoreType.PROBITS
            else:
                raise ValueError(
                    f"Expected prediction column type to be one of {CLASSIFICATION_DATA_TYPES} "
                    f"or one of {PROBITS_DATA_TYPES} for CLASSIFICATION task. "
                    f"Found: {prediction_column_type}."
                )

        else:
            raise ValueError(f"Received unsupported task for model monitoring: {task}.")
