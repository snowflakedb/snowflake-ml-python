import yaml

from snowflake import snowpark
from snowflake.cortex import (
    _constants,
    _evaluation_spec,
    _experiment_job,
    _optimization_job,
    _optimization_spec,
)
from snowflake.cortex._util import CORTEX_FUNCTIONS_TELEMETRY_PROJECT
from snowflake.ml._internal import telemetry
from snowflake.ml.dataset import dataset
from snowflake.ml.experiment._entities import experiment as experiment_entity
from snowflake.snowpark import context


@telemetry.send_api_usage_telemetry(project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT, subproject="OptimizePrompt")
def optimize_prompt(
    *,
    trainset: dataset.Dataset,
    metric: str | _evaluation_spec.EvaluationMetric,
    model: str,
    prompt: str,
    inputs: list[str],
    label_col: str,
    holdout_data: dataset.Dataset | None = None,
    models: list[str] | None = None,
    budget: str | None = None,
    reflection_model: str | None = None,
    strategy: str | None = None,
    algorithm: str | None = None,
    session: snowpark.Session | None = None,
    experiment: str | None = None,
) -> _optimization_job.OptimizationJob:
    """Optimize an AI_COMPLETE prompt asynchronously.

    If the experiment is created but execution does not start, a ``RuntimeError`` is raised and the experiment is
    left in place for inspection or cleanup.

    Args:
        trainset: Version-selected Dataset containing the optimization rows.
        metric: Evaluation metric used to compare candidates.
        model: Model used by the seed AI_COMPLETE call.
        prompt: Seed prompt passed to AI_COMPLETE.
        inputs: Dataset columns interpolated into the prompt.
        label_col: Dataset column containing the expected response.
        holdout_data: Optional version-selected Dataset used to evaluate optimized candidates.
        models: Models the optimizer may use for candidate AI_COMPLETE calls.
        budget: Optimization budget.
        reflection_model: Model used to reflect on and improve candidates.
        strategy: Cost-quality optimization strategy.
        algorithm: Optimization algorithm.
        session: Snowpark session. Uses the active session when omitted.
        experiment: Experiment name. A name is generated when omitted.

    Returns:
        A job that tracks the asynchronous optimization.
    """
    resolved_session = session or context.get_active_session()
    resolved_metric = _evaluation_spec.EvaluationMetric(name=metric) if isinstance(metric, str) else metric
    specification = _optimization_spec.OptimizationSpecification(
        model=model,
        prompt=prompt,
        inputs=inputs,
        trainset=trainset,
        label_col=label_col,
        metric=resolved_metric,
        holdout_data=holdout_data,
        models=models,
        budget=budget,
        reflection_model=reflection_model,
        strategy=strategy,
        algorithm=algorithm,
    )
    experiment_fqn = _experiment_job.create_and_execute_experiment(
        resolved_session,
        experiment,
        name_prefix=_constants.OPTIMIZATION_EXPERIMENT_NAME_PREFIX,
        experiment_type=experiment_entity.ExperimentType.AI_FUNCTION_OPTIMIZATION,
        specification=yaml.safe_dump(specification.to_spec_dict(), sort_keys=False),
    )
    return _optimization_job.OptimizationJob(resolved_session, experiment_fqn)
