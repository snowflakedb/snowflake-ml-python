import inspect
from typing import Optional, Union

import cloudpickle
import numpy as np
import numpy.typing as npt
import sklearn
from packaging import version
from sklearn import metrics

import snowflake.snowpark._internal.utils as snowpark_utils
from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import result
from snowflake.ml.modeling.metrics import metrics_utils
from snowflake.snowpark import functions as F

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Metrics"
_MULTIOUTPUT_UNIFORM_AVG = "uniform_average"
_MULTIOUTPUT_RAW_VALUES = "raw_values"


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def d2_absolute_error_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    sample_weight_col_name: Optional[str] = None,
    multioutput: Union[str, npt.ArrayLike] = "uniform_average",
) -> Union[float, npt.NDArray[np.float64]]:
    """
    :math:`D^2` regression score function, \
    fraction of absolute error explained.

    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A model that always uses the empirical median of `y_true`
    as constant prediction, disregarding the input features,
    gets a :math:`D^2` score of 0.0.

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
        multioutput: {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.
            'raw_values':
                Returns a full set of errors in case of multioutput input.
            'uniform_average':
                Errors of all outputs are averaged with uniform weight.

    Returns:
        score: float or ndarray of floats
            The :math:`D^2` score with an absolute error deviance
            or ndarray of scores if 'multioutput' is 'raw_values'.
    """
    metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)

    session = df._session
    assert session is not None
    sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)
    sklearn_release = version.parse(sklearn.__version__).release
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)

    cols = metrics_utils.flatten_cols([y_true_col_names, y_pred_col_names, sample_weight_col_name])
    queries = df[cols].queries["queries"]

    pickled_snowflake_result = cloudpickle.dumps(result)

    @F.sproc(  # type: ignore[misc]
        is_permanent=False,
        session=session,
        name=sproc_name,
        replace=True,
        packages=[
            f"cloudpickle=={cloudpickle.__version__}",
            f"scikit-learn=={sklearn_release[0]}.{sklearn_release[1]}.*",
            f"numpy=={np.__version__}",
            "snowflake-snowpark-python",
        ],
        statement_params=statement_params,
        anonymous=True,
    )
    def d2_absolute_error_score_anon_sproc(session: snowpark.Session) -> bytes:
        for query in queries[:-1]:
            _ = session.sql(query).collect(statement_params=statement_params)
        sp_df = session.sql(queries[-1])
        df = sp_df.to_pandas(statement_params=statement_params)
        df.columns = sp_df.columns

        y_true = df[y_true_col_names]
        y_pred = df[y_pred_col_names]
        sample_weight = df[sample_weight_col_name] if sample_weight_col_name else None

        score = metrics.d2_absolute_error_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )
        result_module = cloudpickle.loads(pickled_snowflake_result)
        return result_module.serialize(session, score)  # type: ignore[no-any-return]

    kwargs = telemetry.get_sproc_statement_params_kwargs(d2_absolute_error_score_anon_sproc, statement_params)
    result_object = result.deserialize(session, d2_absolute_error_score_anon_sproc(session, **kwargs))
    score: Union[float, npt.NDArray[np.float64]] = result_object
    return score


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def d2_pinball_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    sample_weight_col_name: Optional[str] = None,
    alpha: float = 0.5,
    multioutput: Union[str, npt.ArrayLike] = "uniform_average",
) -> Union[float, npt.NDArray[np.float64]]:
    """
    :math:`D^2` regression score function, fraction of pinball loss explained.

    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A model that always uses the empirical alpha-quantile of
    `y_true` as constant prediction, disregarding the input features,
    gets a :math:`D^2` score of 0.0.

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
        alpha: float, default=0.5
            Slope of the pinball deviance. It determines the quantile level
            alpha for which the pinball deviance and also D2 are optimal.
            The default `alpha=0.5` is equivalent to `d2_absolute_error_score`.
        multioutput: {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.
            'raw_values':
                Returns a full set of errors in case of multioutput input.
            'uniform_average':
                Scores of all outputs are averaged with uniform weight.

    Returns:
        score: float or ndarray of floats
            The :math:`D^2` score with a pinball deviance
            or ndarray of scores if `multioutput='raw_values'`.
    """
    metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)

    session = df._session
    assert session is not None
    sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)
    sklearn_release = version.parse(sklearn.__version__).release
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)

    cols = metrics_utils.flatten_cols([y_true_col_names, y_pred_col_names, sample_weight_col_name])
    queries = df[cols].queries["queries"]

    pickled_result_module = cloudpickle.dumps(result)

    @F.sproc(  # type: ignore[misc]
        is_permanent=False,
        session=session,
        name=sproc_name,
        replace=True,
        packages=[
            f"cloudpickle=={cloudpickle.__version__}",
            f"scikit-learn=={sklearn_release[0]}.{sklearn_release[1]}.*",
            f"numpy=={np.__version__}",
            "snowflake-snowpark-python",
        ],
        statement_params=statement_params,
        anonymous=True,
    )
    def d2_pinball_score_anon_sproc(session: snowpark.Session) -> bytes:
        for query in queries[:-1]:
            _ = session.sql(query).collect(statement_params=statement_params)
        sp_df = session.sql(queries[-1])
        df = sp_df.to_pandas(statement_params=statement_params)
        df.columns = sp_df.columns

        y_true = df[y_true_col_names]
        y_pred = df[y_pred_col_names]
        sample_weight = df[sample_weight_col_name] if sample_weight_col_name else None

        score = metrics.d2_pinball_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            alpha=alpha,
            multioutput=multioutput,
        )
        result_module = cloudpickle.loads(pickled_result_module)
        return result_module.serialize(session, score)  # type: ignore[no-any-return]

    kwargs = telemetry.get_sproc_statement_params_kwargs(d2_pinball_score_anon_sproc, statement_params)
    result_object = result.deserialize(session, d2_pinball_score_anon_sproc(session, **kwargs))

    score: Union[float, npt.NDArray[np.float64]] = result_object
    return score


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def explained_variance_score(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    sample_weight_col_name: Optional[str] = None,
    multioutput: Union[str, npt.ArrayLike] = "uniform_average",
    force_finite: bool = True,
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Explained variance regression score function.

    Best possible score is 1.0, lower values are worse.

    In the particular case when ``y_true`` is constant, the explained variance
    score is not finite: it is either ``NaN`` (perfect predictions) or
    ``-Inf`` (imperfect predictions). To prevent such non-finite numbers to
    pollute higher-level experiments such as a grid search cross-validation,
    by default these cases are replaced with 1.0 (perfect predictions) or 0.0
    (imperfect predictions) respectively. If ``force_finite``
    is set to ``False``, this score falls back on the original :math:`R^2`
    definition.

    Note:
       The Explained Variance score is similar to the
       :func:`R^2 score <r2_score>`, with the notable difference that it
       does not account for systematic offsets in the prediction. Most often
       the :func:`R^2 score <r2_score>` should be preferred.

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
        multioutput: {'raw_values', 'uniform_average', 'variance_weighted'} or \
            array-like of shape (n_outputs,), default='uniform_average'
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.
            'raw_values':
                Returns a full set of scores in case of multioutput input.
            'uniform_average':
                Scores of all outputs are averaged with uniform weight.
            'variance_weighted':
                Scores of all outputs are averaged, weighted by the variances
                of each individual output.
        force_finite: boolean, default=True
            Flag indicating if ``NaN`` and ``-Inf`` scores resulting
            from constant data should be replaced with real numbers (``1.0`` if
            prediction is perfect, ``0.0`` otherwise). Default is ``True``, a
            convenient setting for hyperparameters' search procedures (e.g. grid
            search cross-validation).

    Returns:
        score: float or ndarray of floats
            The explained variance or ndarray if 'multioutput' is 'raw_values'.
    """
    metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)

    session = df._session
    assert session is not None
    sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)
    sklearn_release = version.parse(sklearn.__version__).release
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)

    cols = metrics_utils.flatten_cols([y_true_col_names, y_pred_col_names, sample_weight_col_name])
    queries = df[cols].queries["queries"]

    pickled_result_module = cloudpickle.dumps(result)

    @F.sproc(  # type: ignore[misc]
        is_permanent=False,
        session=session,
        name=sproc_name,
        replace=True,
        packages=[
            f"cloudpickle=={cloudpickle.__version__}",
            f"scikit-learn=={sklearn_release[0]}.{sklearn_release[1]}.*",
            f"numpy=={np.__version__}",
            "snowflake-snowpark-python",
        ],
        statement_params=statement_params,
        anonymous=True,
    )
    def explained_variance_score_anon_sproc(session: snowpark.Session) -> bytes:
        for query in queries[:-1]:
            _ = session.sql(query).collect(statement_params=statement_params)
        sp_df = session.sql(queries[-1])
        df = sp_df.to_pandas(statement_params=statement_params)
        df.columns = sp_df.columns

        y_true = df[y_true_col_names]
        y_pred = df[y_pred_col_names]
        sample_weight = df[sample_weight_col_name] if sample_weight_col_name else None

        score = metrics.explained_variance_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
            force_finite=force_finite,
        )
        result_module = cloudpickle.loads(pickled_result_module)
        return result_module.serialize(session, score)  # type: ignore[no-any-return]

    kwargs = telemetry.get_sproc_statement_params_kwargs(explained_variance_score_anon_sproc, statement_params)
    result_object = result.deserialize(session, explained_variance_score_anon_sproc(session, **kwargs))
    score: Union[float, npt.NDArray[np.float64]] = result_object
    return score


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def mean_absolute_error(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    sample_weight_col_name: Optional[str] = None,
    multioutput: Union[str, npt.ArrayLike] = "uniform_average",
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Mean absolute error regression loss.

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
        multioutput: {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.
            'raw_values':
                Returns a full set of errors in case of multioutput input.
            'uniform_average':
                Errors of all outputs are averaged with uniform weight.

    Returns:
        loss: float or ndarray of floats
            If multioutput is 'raw_values', then mean absolute error is returned
            for each output separately.
            If multioutput is 'uniform_average' or an ndarray of weights, then the
            weighted average of all output errors is returned.

            MAE output is non-negative floating point. The best value is 0.0.
    """
    metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)
    session = df._session
    assert session is not None
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)
    y_true = y_true_col_names if isinstance(y_true_col_names, list) else [y_true_col_names]
    y_pred = y_pred_col_names if isinstance(y_pred_col_names, list) else [y_pred_col_names]
    num_outputs = len(y_true)
    _validate_multioutput(multioutput, num_outputs)

    for i in range(num_outputs):
        df = df.with_column(
            "diff" + str(i),
            F.abs(df[y_pred[i]] - df[y_true[i]]),
        )
    output_errors = np.array(
        [
            metrics_utils.weighted_sum(
                df=df,
                sample_score_column=df["diff" + str(i)],
                sample_weight_column=df[sample_weight_col_name] if sample_weight_col_name else None,
                normalize=True,
                statement_params=statement_params,
            )
            for i in range(num_outputs)
        ]
    )

    if multioutput == _MULTIOUTPUT_RAW_VALUES:
        return output_errors

    weights = None if multioutput == _MULTIOUTPUT_UNIFORM_AVG else multioutput
    return float(np.average(output_errors, weights=weights))


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def mean_absolute_percentage_error(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    sample_weight_col_name: Optional[str] = None,
    multioutput: Union[str, npt.ArrayLike] = "uniform_average",
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Mean absolute percentage error (MAPE) regression loss.

    Note here that the output is not a percentage in the range [0, 100]
    and a value of 100 does not mean 100% but 1e2. Furthermore, the output
    can be arbitrarily high when `y_true` is small (which is specific to the
    metric) or when `abs(y_true - y_pred)` is large (which is common for most
    regression metrics).

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
        multioutput: {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.
            'raw_values':
                Returns a full set of errors in case of multioutput input.
            'uniform_average':
                Errors of all outputs are averaged with uniform weight.

    Returns:
        loss: float or ndarray of floats
            If multioutput is 'raw_values', then mean absolute percentage error
            is returned for each output separately.
            If multioutput is 'uniform_average' or an ndarray of weights, then the
            weighted average of all output errors is returned.

            MAPE output is non-negative floating point. The best value is 0.0.
            But note that bad predictions can lead to arbitrarily large
            MAPE values, especially if some `y_true` values are very close to zero.
            Note that we return a large value instead of `inf` when `y_true` is zero.
    """
    metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)
    session = df._session
    assert session is not None
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)
    y_true = y_true_col_names if isinstance(y_true_col_names, list) else [y_true_col_names]
    y_pred = y_pred_col_names if isinstance(y_pred_col_names, list) else [y_pred_col_names]
    num_outputs = len(y_true)
    _validate_multioutput(multioutput, num_outputs)

    epsilon = float(np.finfo(np.float64).eps)
    for i in range(num_outputs):
        df = df.with_column(
            "MAPE" + str(i),
            F.abs(df[y_pred[i]] - df[y_true[i]]) / F.iff(F.abs(df[y_true[i]]) > epsilon, F.abs(df[y_true[i]]), epsilon),
        )
    output_errors = np.array(
        [
            metrics_utils.weighted_sum(
                df=df,
                sample_score_column=df["MAPE" + str(i)],
                sample_weight_column=df[sample_weight_col_name] if sample_weight_col_name else None,
                normalize=True,
                statement_params=statement_params,
            )
            for i in range(num_outputs)
        ]
    )

    if multioutput == _MULTIOUTPUT_RAW_VALUES:
        return output_errors

    weights = None if multioutput == _MULTIOUTPUT_UNIFORM_AVG else multioutput
    return float(np.average(output_errors, weights=weights))


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def mean_squared_error(
    *,
    df: snowpark.DataFrame,
    y_true_col_names: Union[str, list[str]],
    y_pred_col_names: Union[str, list[str]],
    sample_weight_col_name: Optional[str] = None,
    multioutput: Union[str, npt.ArrayLike] = "uniform_average",
    squared: bool = True,
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Mean squared error regression loss.

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_names: string or list of strings
            Column name(s) representing actual values.
        y_pred_col_names: string or list of strings
            Column name(s) representing predicted values.
        sample_weight_col_name: string, default=None
            Column name representing sample weights.
        multioutput: {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.
            'raw_values':
                Returns a full set of errors in case of multioutput input.
            'uniform_average':
                Errors of all outputs are averaged with uniform weight.
        squared: boolean, default=True
            If True returns MSE value, if False returns RMSE value.

    Returns:
        loss: float or ndarray of floats
            A non-negative floating point value (the best value is 0.0), or an
            array of floating point values, one for each individual target.
    """
    metrics_utils.check_label_columns(y_true_col_names, y_pred_col_names)
    session = df._session
    assert session is not None
    statement_params = telemetry.get_statement_params(_PROJECT, _SUBPROJECT)
    y_true = y_true_col_names if isinstance(y_true_col_names, list) else [y_true_col_names]
    y_pred = y_pred_col_names if isinstance(y_pred_col_names, list) else [y_pred_col_names]
    num_outputs = len(y_true)
    _validate_multioutput(multioutput, num_outputs)

    for i in range(num_outputs):
        df = df.with_column(
            "diff" + str(i),
            F.pow(df[y_true[i]] - df[y_pred[i]], 2),
        )

    output_errors = np.array(
        [
            metrics_utils.weighted_sum(
                df=df,
                sample_score_column=df["diff" + str(i)],
                sample_weight_column=df[sample_weight_col_name] if sample_weight_col_name else None,
                normalize=True,
                statement_params=statement_params,
            )
            for i in range(num_outputs)
        ]
    )
    if not squared:
        output_errors = np.sqrt(output_errors)

    if multioutput == _MULTIOUTPUT_RAW_VALUES:
        return output_errors

    weights = None if multioutput == _MULTIOUTPUT_UNIFORM_AVG else multioutput
    return float(np.average(output_errors, weights=weights))


@telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
def r2_score(*, df: snowpark.DataFrame, y_true_col_name: str, y_pred_col_name: str) -> float:
    """:math:`R^2` (coefficient of determination) regression score function.
    Returns R squared metric on 2 columns in the dataframe.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). In the general case when the true y is
    non-constant, a constant model that always predicts the average y
    disregarding the input features would get a :math:`R^2` score of 0.0.

    Args:
        df: snowpark.DataFrame
            Input dataframe.
        y_true_col_name: string
            Column name representing actual values.
        y_pred_col_name: string
            Column name representing predicted values.

    Returns:
        R squared metric.
    """

    df_avg = df.select(F.avg(y_true_col_name).as_("avg_y_true"))
    df_r_square = df.join(df_avg).select(
        F.lit(1)
        - F.sum((df[y_true_col_name] - df[y_pred_col_name]) ** 2)
        / F.sum((df[y_true_col_name] - df_avg["avg_y_true"]) ** 2)
    )

    statement_params = telemetry.get_function_usage_statement_params(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), None),
    )
    return float(df_r_square.collect(statement_params=statement_params)[0][0])


def _validate_multioutput(multioutput: Union[str, npt.ArrayLike], num_outputs: int) -> None:
    """Validates multioutput parameter for MAPE calculation.

    Args:
        multioutput: Parameter specifying how to deal with multiple outputs.
        num_outputs: Integer representing number of outputs.

    Raises:
        ValueError: multioutput parameter is invalid.

    """
    allowed_str_vals = ("raw_values", "uniform_average")
    if isinstance(multioutput, str):
        if multioutput not in allowed_str_vals:
            raise ValueError(
                f"Allowed 'multioutput' string values are {allowed_str_vals}. You provided multioutput={multioutput}"
            )
    else:
        multioutput_np = np.array(multioutput)
        if num_outputs == 1:
            raise ValueError("Custom weights are useful only in multi-output cases.")
        elif num_outputs != len(multioutput_np):
            raise ValueError(
                f"There must be equally many custom weights ({len(multioutput_np)}) as outputs ({num_outputs})."
            )
