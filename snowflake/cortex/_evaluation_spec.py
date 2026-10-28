import dataclasses
import re
from typing import Any

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import sql_identifier, sql_literal
from snowflake.ml.dataset import dataset


@dataclasses.dataclass(frozen=True)
class EvaluationMetric:
    name: str
    _: dataclasses.KW_ONLY
    judge_model: str | None = None
    custom_udf: str | None = None

    def to_spec_dict(self) -> dict[str, str]:
        if not self.name:
            raise invalid_argument("Evaluation metric name must not be empty.")
        if self.name == "custom" and not self.custom_udf:
            raise invalid_argument("The custom evaluation metric requires a custom UDF name.")
        if self.custom_udf is not None and self.name != "custom":
            raise invalid_argument("A custom UDF can only be used with the custom evaluation metric.")
        if self.judge_model is not None and self.name != "llm_judge":
            raise invalid_argument("A judge model can only be used with the llm_judge evaluation metric.")

        metric_specification = {"name": self.name}
        if self.judge_model is not None:
            metric_specification["judge_model"] = self.judge_model
        if self.custom_udf is not None:
            metric_specification["custom_udf"] = self.custom_udf
        return metric_specification


@dataclasses.dataclass(frozen=True)
class EvaluationSpecification:
    model: str
    prompt: str
    inputs: list[str]
    dataset: dataset.Dataset
    label_col: str
    metrics: list[EvaluationMetric]

    def to_spec_dict(self) -> dict[str, Any]:
        if len(self.metrics) != 1:
            raise invalid_argument(f"AI function evaluation requires exactly one metric; received {len(self.metrics)}.")
        if not self.label_col:
            raise invalid_argument("AI function evaluation requires a ground-truth column.")
        if not self.inputs:
            raise invalid_argument("AI function evaluation requires at least one input column.")

        dataset_name, dataset_version = resolve_dataset(self.dataset)
        return {
            # AI_COMPLETE returns a VARIANT; without the cast, metrics compare JSON-quoted predictions to the label.
            "query_text": f"{build_ai_complete_query(self.model, self.prompt, self.inputs)}::VARCHAR",
            "metrics": [self.metrics[0].to_spec_dict()],
            "dataset": {
                "name": dataset_name,
                "version": dataset_version,
                "ground_truth": resolve_column_identifier(self.label_col, context="evaluation", role="ground-truth"),
            },
        }


def resolve_dataset(data: dataset.Dataset, *, context: str = "evaluation") -> tuple[str, str]:
    if not isinstance(data, dataset.Dataset):
        raise invalid_argument(
            f"AI function {context} requires a snowflake.ml.Dataset; table, view, and DataFrame inputs are not "
            "supported."
        )

    selected_version = data.selected_version
    if selected_version is None:
        raise invalid_argument(f"AI function {context} requires a Dataset with a selected version.")
    return data.fully_qualified_name, selected_version.name


def build_ai_complete_query(
    model: str,
    prompt: str,
    inputs: list[str],
    *,
    context: str = "evaluation",
) -> str:
    model_expression = sql_literal.string_literal(model)
    prompt_expression = sql_literal.string_literal(prompt)
    if inputs:
        input_expressions = ", ".join(resolve_column_identifier(name, context=context, role="input") for name in inputs)
        prompt_expression = f"PROMPT({prompt_expression}, {input_expressions})"
    placeholders = sorted({int(index) for index in re.findall(r"\{(\d+)\}", prompt)})
    if placeholders != list(range(len(inputs))):
        raise invalid_argument(
            f"AI function {context} prompt placeholders {placeholders} do not match its {len(inputs)} input "
            "column(s); use {0} for the first input column, {1} for the second, and so on."
        )
    return f"AI_COMPLETE({model_expression}, {prompt_expression})"


def resolve_column_identifier(name: str, *, context: str, role: str) -> str:
    try:
        return sql_identifier.SqlIdentifier(name).identifier()
    except ValueError as exception:
        raise invalid_argument(f"AI function {context} received an invalid {role} column identifier.") from exception


def invalid_argument(message: str) -> snowml_exceptions.SnowflakeMLException:
    return snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.INVALID_ARGUMENT,
        original_exception=ValueError(message),
    )
