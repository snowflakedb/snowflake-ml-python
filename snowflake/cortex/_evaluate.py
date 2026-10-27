import yaml

from snowflake import snowpark
from snowflake.cortex import (
    _constants,
    _evaluation_job,
    _evaluation_spec,
    _experiment_job,
)
from snowflake.cortex._util import CORTEX_FUNCTIONS_TELEMETRY_PROJECT
from snowflake.ml._internal import telemetry
from snowflake.ml.dataset import dataset
from snowflake.ml.experiment._entities import experiment as experiment_entity
from snowflake.snowpark import context


@telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT, subproject="Evaluate")
def evaluate(
    *,
    data: dataset.Dataset,
    scorers: list[str | _evaluation_spec.EvaluationMetric],
    model: str,
    prompt: str,
    inputs: list[str],
    label_col: str,
    session: snowpark.Session | None = None,
    experiment: str | None = None,
) -> _evaluation_job.EvaluationJob:
    """Evaluate a single AI_COMPLETE call asynchronously.

    If the experiment is created but execution does not start, a ``RuntimeError`` is raised and the experiment is
    left in place for inspection or cleanup.

    Args:
        data: Version-selected Dataset containing the evaluation rows.
        scorers: Evaluation metrics to apply. The initial release supports exactly one metric.
        model: Model used by AI_COMPLETE.
        prompt: Prompt passed to AI_COMPLETE.
        inputs: Dataset columns interpolated into the prompt.
        label_col: Dataset column containing the expected response.
        session: Snowpark session. Uses the active session when omitted.
        experiment: Experiment name. A name is generated when omitted.

    Returns:
        A job that tracks the asynchronous evaluation.
    """
    resolved_session = session or context.get_active_session()
    metrics = [
        _evaluation_spec.EvaluationMetric(name=scorer) if isinstance(scorer, str) else scorer for scorer in scorers
    ]
    specification = _evaluation_spec.EvaluationSpecification(
        model=model,
        prompt=prompt,
        inputs=inputs,
        dataset=data,
        label_col=label_col,
        metrics=metrics,
    )
    experiment_fqn = _experiment_job.create_and_execute_experiment(
        resolved_session,
        experiment,
        name_prefix=_constants.EVALUATION_EXPERIMENT_NAME_PREFIX,
        experiment_type=experiment_entity.ExperimentType.AI_FUNCTION_EVALUATION,
        specification=yaml.safe_dump(specification.to_spec_dict(), sort_keys=False),
    )
    return _evaluation_job.EvaluationJob(resolved_session, experiment_fqn)
