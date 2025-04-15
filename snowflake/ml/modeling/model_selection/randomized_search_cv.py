from typing import Any, Iterable, Optional, Union

import cloudpickle as cp
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
from sklearn.utils.metaestimators import available_if

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import identifier, pkg_version_utils
from snowflake.ml.model._signatures import utils as model_signature_utils
from snowflake.ml.model.model_signature import (
    BaseFeatureSpec,
    DataType,
    FeatureSpec,
    ModelSignature,
    _infer_signature,
    _rename_signature_with_snowflake_identifiers,
    _truncate_data,
)
from snowflake.ml.modeling._internal.estimator_utils import (
    gather_dependencies,
    original_estimator_has_callable,
    transform_snowml_obj_to_sklearn_obj,
    validate_sklearn_args,
)
from snowflake.ml.modeling._internal.model_trainer_builder import ModelTrainerBuilder
from snowflake.ml.modeling._internal.model_transformer_builder import (
    ModelTransformerBuilder,
)
from snowflake.ml.modeling._internal.transformer_protocols import (
    BatchInferenceKwargsTypedDict,
    ScoreKwargsTypedDict,
)
from snowflake.ml.modeling.framework.base import BaseTransformer
from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark._internal.type_utils import convert_sp_to_sf_type

_PROJECT = "ModelDevelopment"
# Derive subproject from module name by removing "sklearn"
# and converting module name from underscore to CamelCase
# e.g. sklearn.linear_model -> LinearModel.
_SUBPROJECT = "ModelSelection"
DEFAULT_UDTF_NJOBS = 3

INFER_SIGNATURE_MAX_ROWS = 100

DATAFRAME_TYPE = Union[DataFrame, pd.DataFrame]


class RandomizedSearchCV(BaseTransformer):
    r"""Randomized search on hyper parameters
    For more details on this class, see [sklearn.model_selection.RandomizedSearchCV]
    (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

    Parameters
    ----------
    estimator: estimator object
        An object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions: dict or list of dicts
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    input_cols: Optional[Union[str, List[str]]]
        A string or list of strings representing column names that contain features.
        If this parameter is not specified, all columns in the input DataFrame except
        the columns specified by label_cols and sample-weight_col parameters are
        considered input columns.

    label_cols: Optional[Union[str, List[str]]]
        A string or list of strings representing column names that contain labels.
        This is a required param for estimators, as there is no way to infer these
        columns. If this parameter is not specified, then object is fitted without
        labels(Like a transformer).

    output_cols: Optional[Union[str, List[str]]]
        A string or list of strings representing column names that will store the
        output of predict and transform operations. The length of output_cols mus
        match the expected number of output columns from the specific estimator or
        transformer class used.
        If this parameter is not specified, output column names are derived by
        adding an OUTPUT_ prefix to the label column names. These inferred output
        column names work for estimator's predict() method, but output_cols must
        be set explicitly for transformers.

    passthrough_cols: A string or a list of strings indicating column names to be excluded from any
        operations (such as train, transform, or inference). These specified column(s)
        will remain untouched throughout the process. This option is helpful in scenarios
        requiring automatic input_cols inference, but need to avoid using specific
        columns, like index columns, during training or inference.

    sample_weight_col: Optional[str]
        A string representing the column name containing the examples’ weights.
        This argument is only required when working with weighted datasets.

    drop_input_cols: Optional[bool], default=False
        If set, the response of predict(), transform() methods will not contain input columns.

    n_iter: int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring: str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's score method is used.

    n_jobs: int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    refit: bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    cv: int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    verbose: int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch: int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    random_state: int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    error_score: 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score: bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.
    """

    _ENABLE_DISTRIBUTED = True

    def __init__(  # type: ignore[no-untyped-def]
        self,
        *,
        estimator,
        param_distributions,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        label_cols: Optional[Union[str, Iterable[str]]] = None,
        passthrough_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
        sample_weight_col: Optional[str] = None,
    ) -> None:
        super().__init__()
        deps: set[str] = {
            f"numpy=={np.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"cloudpickle=={cp.__version__}",
        }
        deps = deps | gather_dependencies(estimator)
        self._deps = list(deps)
        estimator = transform_snowml_obj_to_sklearn_obj(estimator)
        init_args = {
            "estimator": (estimator, None, True),
            "param_distributions": (param_distributions, None, True),
            "n_iter": (n_iter, 10, False),
            "scoring": (scoring, None, False),
            "n_jobs": (n_jobs, None, False),
            "refit": (refit, True, False),
            "cv": (cv, None, False),
            "verbose": (verbose, 0, False),
            "pre_dispatch": (pre_dispatch, "2*n_jobs", False),
            "random_state": (random_state, None, False),
            "error_score": (error_score, np.nan, False),
            "return_train_score": (return_train_score, False, False),
        }
        cleaned_up_init_args = validate_sklearn_args(args=init_args, klass=sklearn.model_selection.RandomizedSearchCV)
        self._sklearn_object: Any = sklearn.model_selection.RandomizedSearchCV(
            **cleaned_up_init_args,
        )
        self._model_signature_dict: Optional[dict[str, ModelSignature]] = None
        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)
        self.set_label_cols(label_cols)
        self.set_drop_input_cols(drop_input_cols)
        self.set_sample_weight_col(sample_weight_col)
        self.set_passthrough_cols(passthrough_cols)

        self._autogenerated = False
        self._snowpark_cols = self.input_cols
        self._autogenerated = False
        self._class_name = RandomizedSearchCV.__class__.__name__
        self._subproject = _SUBPROJECT

    def _get_active_columns(self) -> list[str]:
        """ "Get the list of columns that are relevant to the transformer."""
        selected_cols = (
            self.input_cols + self.label_cols + ([self.sample_weight_col] if self.sample_weight_col is not None else [])
        )
        return selected_cols

    def _fit(self, dataset: Union[DataFrame, pd.DataFrame]) -> "RandomizedSearchCV":
        """Run fit with all sets of parameters
        For more details on this function, see [sklearn.model_selection.RandomizedSearchCV.fit]
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV.fit)

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.

        Returns:
            self
        """
        self._infer_input_output_cols(dataset)
        if hasattr(self._sklearn_object, "n_jobs") and self._sklearn_object.n_jobs is None:
            self._sklearn_object.n_jobs = -1
        if isinstance(dataset, DataFrame):
            session = dataset._session
            assert session is not None  # keep mypy happy
            # Validate that key package version in user workspace are supported in snowflake conda channel
            # If customer doesn't have package in conda channel, replace the ones have the closest versions
            self._deps = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
                pkg_versions=self._get_dependencies(), session=session, subproject=_SUBPROJECT
            )

            # Specify input columns so column pruning will be enforced
            selected_cols = self._get_active_columns()
            if len(selected_cols) > 0:
                dataset = dataset.select(selected_cols)

            self._snowpark_cols = dataset.select(self.input_cols).columns

        model_trainer = ModelTrainerBuilder.build(
            estimator=self._sklearn_object,
            dataset=dataset,
            input_cols=self.input_cols,
            label_cols=self.label_cols,
            sample_weight_col=self.sample_weight_col,
            autogenerated=False,
            subproject=_SUBPROJECT,
        )
        self._sklearn_object = model_trainer.train()
        self._is_fitted = True
        self._generate_model_signatures(dataset)
        return self

    def _batch_inference_validate_snowpark(
        self,
        dataset: DataFrame,
        inference_method: str,
    ) -> None:
        """Util method to run validate that batch inference can be run on a snowpark dataframe.

        Args:
            dataset: snowpark dataframe
            inference_method: the inference method such as predict, score...

        Raises:
            SnowflakeMLException: If the estimator is not fitted, raise error
            SnowflakeMLException: If the session is None, raise error

        """
        if not self._is_fitted:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.METHOD_NOT_ALLOWED,
                original_exception=RuntimeError(
                    f"Estimator {self.__class__.__name__} not fitted before calling {inference_method} method."
                ),
            )

        session = dataset._session
        if session is None:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError("Session must not specified for snowpark dataset."),
            )

    @available_if(original_estimator_has_callable("predict"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def predict(self, dataset: Union[DataFrame, pd.DataFrame]) -> Union[DataFrame, pd.DataFrame]:
        """Call predict on the estimator with the best found parameters
        For more details on this function, see [sklearn.model_selection.RandomizedSearchCV.predict]
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV.predict)

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.

        Returns:
            Transformed dataset.

        Raises:
            SnowflakeMLException: when the output column(s) doesn't exist in the model signature, raise error
        """
        super()._check_dataset_type(dataset)

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used.
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()
        inference_method = "predict"

        if isinstance(dataset, DataFrame):
            expected_type_inferred = ""
            # infer the datatype from label columns
            if "predict" in self.model_signatures:
                # Batch inference takes a single expected output column type. Use the first columns type for now.
                label_cols_signatures = [
                    row for row in self.model_signatures["predict"].outputs if row.name in self.output_cols
                ]
                if len(label_cols_signatures) == 0:
                    error_str = (
                        f"Output columns {self.output_cols} do not match"
                        f"model signatures {self.model_signatures['predict'].outputs}."
                    )
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_ATTRIBUTE,
                        original_exception=ValueError(error_str),
                    )

                expected_type_inferred = convert_sp_to_sf_type(label_cols_signatures[0].as_snowpark_type())
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()

            assert isinstance(
                dataset._session, Session
            )  # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_type=expected_type_inferred,
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated,
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=self.output_cols,
            **transform_kwargs,
        )

        return output_df

    @available_if(original_estimator_has_callable("transform"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def transform(self, dataset: Union[DataFrame, pd.DataFrame]) -> Union[DataFrame, pd.DataFrame]:
        """Call transform on the estimator with the best found parameters
        For more details on this function, see [sklearn.model_selection.RandomizedSearchCV.transform]
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV.transform)

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.

        Returns:
            Transformed dataset.
        """
        super()._check_dataset_type(dataset)

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used.
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()
        inference_method = "transform"

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()

            assert isinstance(
                dataset._session, Session
            )  # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated,
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=self.output_cols,
            **transform_kwargs,
        )
        return output_df

    @available_if(original_estimator_has_callable("predict_proba"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def predict_proba(
        self, dataset: Union[DataFrame, pd.DataFrame], output_cols_prefix: str = "predict_proba_"
    ) -> Union[DataFrame, pd.DataFrame]:
        """Call predict_proba on the estimator with the best found parameters
        For more details on this function, see [sklearn.model_selection.RandomizedSearchCV.predict_proba]
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV.predict_proba)

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.
            output_cols_prefix: Prefix for the response columns

        Returns:
            Output dataset with probability of the sample for each class in the model.
        """
        super()._check_dataset_type(dataset)

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used.
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()

        inference_method = "predict_proba"

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()

            assert isinstance(
                dataset._session, Session
            )  # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_type="float",
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated,
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=self._get_output_column_names(output_cols_prefix),
            **transform_kwargs,
        )
        return output_df

    @available_if(original_estimator_has_callable("predict_log_proba"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def predict_log_proba(
        self, dataset: Union[DataFrame, pd.DataFrame], output_cols_prefix: str = "predict_log_proba_"
    ) -> Union[DataFrame, pd.DataFrame]:
        """Call predict_proba on the estimator with the best found parameters
        For more details on this function, see [sklearn.model_selection.RandomizedSearchCV.predict_proba]
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV.predict_proba)

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.
            output_cols_prefix: str
                Prefix for the response columns

        Returns:
            Output dataset with log probability of the sample for each class in the model.
        """
        super()._check_dataset_type(dataset)

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used.
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()

        inference_method = "predict_log_proba"

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()

            assert isinstance(
                dataset._session, Session
            )  # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_type="float",
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated,
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=self._get_output_column_names(output_cols_prefix),
            **transform_kwargs,
        )
        return output_df

    @available_if(original_estimator_has_callable("decision_function"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def decision_function(
        self, dataset: Union[DataFrame, pd.DataFrame], output_cols_prefix: str = "decision_function_"
    ) -> Union[DataFrame, pd.DataFrame]:
        """Call decision_function on the estimator with the best found parameters
        For more details on this function, see [sklearn.model_selection.RandomizedSearchCV.decision_function]
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV.decision_function)

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.
            output_cols_prefix: str
                Prefix for the response columns

        Returns:
            Output dataset with results of the decision function for the samples in input dataset.
        """
        super()._check_dataset_type(dataset)

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used.
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()
        inference_method = "decision_function"

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()

            assert isinstance(
                dataset._session, Session
            )  # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_type="float",
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated,
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=self._get_output_column_names(output_cols_prefix),
            **transform_kwargs,
        )
        return output_df

    @available_if(original_estimator_has_callable("score_samples"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    def score_samples(
        self, dataset: Union[DataFrame, pd.DataFrame], output_cols_prefix: str = "score_samples_"
    ) -> Union[DataFrame, pd.DataFrame]:
        """Call score_samples on the estimator with the best found parameters.
        Only available if refit=True and the underlying estimator supports score_samples.

        Args:
            dataset (Union[DataFrame, pd.DataFrame]):
                Snowpark or Pandas DataFrame.
            output_cols_prefix (str):
                Prefix for the response columns. Defaults to "score_samples_".

        Returns:
            Union[DataFrame, pd.DataFrame]:
                Output dataset with results of the decision function for the samples in input dataset.
        """
        super()._check_dataset_type(dataset)

        inference_method = "score_samples"

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used.
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()

            assert isinstance(
                dataset._session, Session
            )  # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_type="float",
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated,
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=self._get_output_column_names(output_cols_prefix),
            **transform_kwargs,
        )
        return output_df

    @available_if(original_estimator_has_callable("score"))  # type: ignore[misc]
    def score(self, dataset: Union[DataFrame, pd.DataFrame]) -> float:
        """
        If implemented by the original estimator, return the score for the dataset.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.

        Returns:
            Score.
        """
        self._infer_input_output_cols(dataset)
        super()._check_dataset_type(dataset)

        # This dictionary contains optional kwargs for scoring. These kwargs
        # are specific to the type of dataset used.
        transform_kwargs: ScoreKwargsTypedDict = dict()

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method="score")
            self._deps = self._get_dependencies()

            selected_cols = self._get_active_columns()
            if len(selected_cols) > 0:
                dataset = dataset.select(selected_cols)

            assert isinstance(dataset._session, Session)  # keep mypy happy
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                score_sproc_imports=["sklearn"],
            )
        elif isinstance(dataset, pd.DataFrame):
            # pandas_handler.score() does not require any extra kwargs.
            transform_kwargs = dict()

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated,
        )

        output_score = transform_handlers.score(
            input_cols=identifier.get_unescaped_names(self.input_cols),
            label_cols=identifier.get_unescaped_names(self.label_cols),
            sample_weight_col=identifier.get_unescaped_names(self.sample_weight_col),
            **transform_kwargs,
        )

        return output_score

    def to_sklearn(self) -> sklearn.model_selection.RandomizedSearchCV:
        """
        Get sklearn.model_selection.RandomizedSearchCV object.
        """
        assert self._sklearn_object is not None
        return self._sklearn_object

    def _get_dependencies(self) -> list[str]:
        return self._deps

    def _generate_model_signatures(self, dataset: Union[DataFrame, pd.DataFrame]) -> None:
        self._model_signature_dict = dict()

        PROB_FUNCTIONS = ["predict_log_proba", "predict_proba", "decision_function"]

        inputs = list(
            _infer_signature(
                _truncate_data(dataset[self.input_cols], INFER_SIGNATURE_MAX_ROWS),
                "input",
                use_snowflake_identifiers=True,
            )
        )
        outputs: list[BaseFeatureSpec] = []
        if hasattr(self, "predict"):
            # keep mypy happy
            assert self._sklearn_object is not None and hasattr(self._sklearn_object, "_estimator_type")
            # For classifier, the type of predict is the same as the type of label
            if self._sklearn_object._estimator_type == "classifier":
                # label columns is the desired type for output
                outputs = list(
                    _infer_signature(
                        _truncate_data(dataset[self.label_cols], INFER_SIGNATURE_MAX_ROWS),
                        "output",
                        use_snowflake_identifiers=True,
                    )
                )
                # rename the output columns
                outputs = list(model_signature_utils.rename_features(outputs, self.output_cols))
                self._model_signature_dict["predict"] = ModelSignature(
                    inputs, ([] if self._drop_input_cols else inputs) + outputs
                )

            # For regressor, the type of predict is float64
            elif self._sklearn_object._estimator_type == "regressor":
                outputs = [FeatureSpec(dtype=DataType.DOUBLE, name=c) for c in self.output_cols]
                self._model_signature_dict["predict"] = ModelSignature(
                    inputs, ([] if self._drop_input_cols else inputs) + outputs
                )

        for prob_func in PROB_FUNCTIONS:
            if hasattr(self, prob_func):
                output_cols_prefix: str = f"{prob_func}_"
                output_column_names = self._get_output_column_names(output_cols_prefix)
                outputs = [FeatureSpec(dtype=DataType.DOUBLE, name=c) for c in output_column_names]
                self._model_signature_dict[prob_func] = ModelSignature(
                    inputs, ([] if self._drop_input_cols else inputs) + outputs
                )

        # Output signature names may still need to be renamed, since they were not created with `_infer_signature`.
        items = list(self._model_signature_dict.items())
        for method, signature in items:
            signature._outputs = _rename_signature_with_snowflake_identifiers(signature._outputs)
            self._model_signature_dict[method] = signature

    @property
    def model_signatures(self) -> dict[str, ModelSignature]:
        """Returns model signature of current class.

        Raises:
            SnowflakeMLException: If estimator is not fitted, then model signature cannot be inferred

        Returns:
            each method and its input output signature
        """
        if self._model_signature_dict is None:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=RuntimeError("Estimator not fitted before accessing property model_signatures!"),
            )
        return self._model_signature_dict
