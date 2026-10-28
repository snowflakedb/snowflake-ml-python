import dataclasses
from typing import Any

from snowflake.cortex import _evaluation_spec
from snowflake.ml.dataset import dataset

_VALID_BUDGETS = frozenset({"auto", "ultra-light", "light", "medium", "heavy"})
_VALID_STRATEGIES = frozenset({"quality_first", "balanced", "cost_first"})
_VALID_ALGORITHMS = frozenset({"gepa", "cocoevolve"})


@dataclasses.dataclass(frozen=True)
class OptimizationSpecification:
    model: str
    prompt: str
    inputs: list[str]
    trainset: dataset.Dataset
    label_col: str
    metric: _evaluation_spec.EvaluationMetric
    holdout_data: dataset.Dataset | None = None
    models: list[str] | None = None
    budget: str | None = None
    reflection_model: str | None = None
    strategy: str | None = None
    algorithm: str | None = None

    def to_spec_dict(self) -> dict[str, Any]:
        if not self.inputs:
            raise _evaluation_spec.invalid_argument("AI function optimization requires at least one input column.")
        if not self.label_col:
            raise _evaluation_spec.invalid_argument("AI function optimization requires a ground-truth column.")
        if self.models is not None and not self.models:
            raise _evaluation_spec.invalid_argument("AI function optimization requires at least one candidate model.")
        if self.budget is not None and self.budget not in _VALID_BUDGETS:
            raise _evaluation_spec.invalid_argument(
                f"Unsupported optimization budget {self.budget!r}; expected one of {sorted(_VALID_BUDGETS)}."
            )
        if self.reflection_model is not None and not self.reflection_model:
            raise _evaluation_spec.invalid_argument("Optimization reflection model must not be empty.")
        if self.strategy is not None and self.strategy not in _VALID_STRATEGIES:
            raise _evaluation_spec.invalid_argument(
                f"Unsupported optimization strategy {self.strategy!r}; expected one of {sorted(_VALID_STRATEGIES)}."
            )
        if self.algorithm is not None and self.algorithm.lower() not in _VALID_ALGORITHMS:
            raise _evaluation_spec.invalid_argument(
                f"Unsupported optimization algorithm {self.algorithm!r}; expected one of {sorted(_VALID_ALGORITHMS)}."
            )

        dataset_name, dataset_version = _evaluation_spec.resolve_dataset(self.trainset, context="optimization")
        normalized_inputs = [
            _evaluation_spec.resolve_column_identifier(name, context="optimization", role="input")
            for name in self.inputs
        ]
        dataset_specification: dict[str, Any] = {
            "name": dataset_name,
            "version": dataset_version,
            "column_mapping": {
                "argument_mapping": {
                    f"${position}": column for position, column in enumerate(normalized_inputs, start=1)
                },
                "ground_truth": _evaluation_spec.resolve_column_identifier(
                    self.label_col, context="optimization", role="ground-truth"
                ),
            },
        }
        if self.holdout_data is not None:
            holdout_name, holdout_version = _evaluation_spec.resolve_dataset(self.holdout_data, context="optimization")
            dataset_specification["holdout_data"] = holdout_name
            dataset_specification["holdout_version"] = holdout_version
        optimization_specification: dict[str, Any] = {
            "models": self.models if self.models is not None else [self.model],
        }
        if self.budget is not None:
            optimization_specification["budget"] = self.budget
        if self.reflection_model is not None:
            optimization_specification["reflection_model"] = self.reflection_model
        if self.strategy is not None:
            optimization_specification["strategy"] = self.strategy
        if self.algorithm is not None:
            optimization_specification["algorithm"] = self.algorithm
        return {
            "function": {
                "function_body": _evaluation_spec.build_ai_complete_query(
                    self.model, self.prompt, self.inputs, context="optimization"
                ),
            },
            "metrics": [self.metric.to_spec_dict()],
            "dataset": dataset_specification,
            "optimization": optimization_specification,
        }
