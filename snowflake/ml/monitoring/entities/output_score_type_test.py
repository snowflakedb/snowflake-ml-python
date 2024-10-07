import re
from typing import List, Mapping, Tuple

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import type_hints
from snowflake.ml.monitoring.entities import output_score_type
from snowflake.snowpark import types

DEDUCE_SCORE_TYPE_ACCEPTED_COMBINATIONS: List[
    Tuple[
        Mapping[str, types.DataType],
        List[sql_identifier.SqlIdentifier],
        type_hints.Task,
        output_score_type.OutputScoreType,
    ]
] = [
    (
        {"PREDICTION1": types.FloatType()},
        [sql_identifier.SqlIdentifier("PREDICTION1")],
        type_hints.Task.TABULAR_REGRESSION,
        output_score_type.OutputScoreType.REGRESSION,
    ),
    (
        {"PREDICTION1": types.DecimalType(38, 1)},
        [sql_identifier.SqlIdentifier("PREDICTION1")],
        type_hints.Task.TABULAR_BINARY_CLASSIFICATION,
        output_score_type.OutputScoreType.PROBITS,
    ),
    (
        {"PREDICTION1": types.BinaryType()},
        [sql_identifier.SqlIdentifier("PREDICTION1")],
        type_hints.Task.TABULAR_BINARY_CLASSIFICATION,
        output_score_type.OutputScoreType.CLASSIFICATION,
    ),
]


DEDUCE_SCORE_TYPE_FAILURE_COMBINATIONS: List[
    Tuple[Mapping[str, types.DataType], List[sql_identifier.SqlIdentifier], type_hints.Task, str]
] = [
    (
        {"PREDICTION1": types.BinaryType()},
        [sql_identifier.SqlIdentifier("PREDICTION1")],
        type_hints.Task.TABULAR_REGRESSION,
        f"Expected prediction column type to be one of {output_score_type.REGRESSION_DATA_TYPES} "
        f"for REGRESSION task. Found: {types.BinaryType()}.",
    ),
    (
        {"PREDICTION1": types.StringType()},
        [sql_identifier.SqlIdentifier("PREDICTION1")],
        type_hints.Task.TABULAR_BINARY_CLASSIFICATION,
        f"Expected prediction column type to be one of {output_score_type.CLASSIFICATION_DATA_TYPES} "
        f"or one of {output_score_type.PROBITS_DATA_TYPES} for CLASSIFICATION task. "
        f"Found: {types.StringType()}.",
    ),
    (
        {"PREDICTION1": types.BinaryType()},
        [sql_identifier.SqlIdentifier("PREDICTION1")],
        type_hints.Task.UNKNOWN,
        f"Received unsupported task for model monitoring: {type_hints.Task.UNKNOWN}.",
    ),
]


class OutputScoreTypeTest(absltest.TestCase):
    def test_deduce_score_type(self) -> None:
        # Success cases
        for (
            table_schema,
            prediction_column_names,
            task,
            expected_score_type,
        ) in DEDUCE_SCORE_TYPE_ACCEPTED_COMBINATIONS:
            actual_score_type = output_score_type.OutputScoreType.deduce_score_type(
                table_schema, prediction_column_names, task
            )
            self.assertEqual(actual_score_type, expected_score_type)

        # Failure cases
        for (
            table_schema,
            prediction_column_names,
            task,
            expected_error,
        ) in DEDUCE_SCORE_TYPE_FAILURE_COMBINATIONS:
            with self.assertRaisesRegex(ValueError, re.escape(expected_error)):
                output_score_type.OutputScoreType.deduce_score_type(table_schema, prediction_column_names, task)


if __name__ == "__main__":
    absltest.main()
