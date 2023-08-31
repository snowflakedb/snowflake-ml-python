import copy
import inspect
import os
import posixpath
import sys
from collections import defaultdict
from math import ceil
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
from uuid import uuid4

import cachetools
import cloudpickle
import cloudpickle as cp
import numpy
import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
from sklearn.model_selection import ParameterSampler
from sklearn.utils.metaestimators import available_if
from typing_extensions import TypeGuard

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import identifier, pkg_version_utils
from snowflake.ml._internal.utils.query_result_checker import SqlResultValidator
from snowflake.ml._internal.utils.temp_file_utils import (
    cleanup_temp_files,
    get_temp_file_path,
)
from snowflake.ml.model._signatures import utils as model_signature_utils
from snowflake.ml.model.model_signature import (
    BaseFeatureSpec,
    DataType,
    FeatureSpec,
    ModelSignature,
    _infer_signature,
)
from snowflake.ml.modeling.framework._utils import to_native_format
from snowflake.ml.modeling.framework.base import BaseTransformer
from snowflake.snowpark import DataFrame, Session, functions as F
from snowflake.snowpark._internal.type_utils import convert_sp_to_sf_type
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    random_name_for_temp_object,
)
from snowflake.snowpark.functions import col, pandas_udf, sproc, udtf
from snowflake.snowpark.types import (
    BinaryType,
    PandasSeries,
    StringType,
    StructField,
    StructType,
)

_PROJECT = "ModelDevelopment"
# Derive subproject from module name by removing "sklearn"
# and converting module name from underscore to CamelCase
# e.g. sklearn.linear_model -> LinearModel.
_SUBPROJECT = "ModelSelection"


# TODO: refactor all the common logic into a shared utility module.
def _original_estimator_has_callable(attr: str) -> Callable[[Any], bool]:
    """Checks that the original estimator has callable `attr`.

    Args:
        attr: Attribute to check for.

    Returns:
        A function which checks for the existance of callable `attr` on the given object.
    """

    def check(self: BaseTransformer) -> TypeGuard[Callable[..., object]]:
        """Check for the existance of callable `attr` in self.

        Args:
            self: BaseTransformer object

        Returns:
            True of the callable `attr` exists in self, False otherwise.
        """
        return callable(getattr(self._sklearn_object, attr, None))

    return check


def _gather_dependencies(obj: Any) -> Set[str]:
    """Gethers dependencies from the SnowML Estimator and Transformer objects.

    Args:
        obj: Source object to collect dependencies from. Source object could of any type, example, lists, tuples, etc.

    Returns:
        A set of dependencies required to work with the object.
    """

    if isinstance(obj, list) or isinstance(obj, tuple):
        deps: Set[str] = set()
        for elem in obj:
            deps = deps | set(_gather_dependencies(elem))
        return deps
    elif isinstance(obj, BaseTransformer):
        return set(obj._get_dependencies())
    else:
        return set()


def _transform_snowml_obj_to_sklearn_obj(obj: Any) -> Any:
    """Converts SnowML Estimator and Transformer objects to equivalent SKLearn objects.

    Args:
        obj: Source object that needs to be converted. Source object could of any type, example, lists, tuples, etc.

    Returns:
        An equivalent object with SnowML estimators and transforms replaced with equivalent SKLearn objects.
    """

    if isinstance(obj, list):
        # Apply transform function to each element in the list
        return list(map(_transform_snowml_obj_to_sklearn_obj, obj))
    elif isinstance(obj, tuple):
        # Apply transform function to each element in the tuple
        return tuple(map(_transform_snowml_obj_to_sklearn_obj, obj))
    elif isinstance(obj, BaseTransformer):
        # Convert SnowML object to equivalent SKLearn object
        return to_native_format(obj)
    else:
        # Return all other objects as it is.
        return obj


def _validate_sklearn_args(args: Dict[str, Any], klass: type) -> Dict[str, Any]:
    """Validate if all the keyword args are supported by current version of SKLearn/XGBoost object.

    Args:
        args: Dictionary of keyword args for the wrapper init method.
        klass: Underlying SKLearn/XGBoost class object.

    Returns:
        result: sklearn arguments

    Raises:
        SnowflakeMLException: if a user specified arg is not supported by current version of sklearn/xgboost.
    """
    result = {}
    signature = inspect.signature(klass.__init__)  # type: ignore
    for k, v in args.items():
        if k not in signature.parameters.keys():  # Arg is not supported.
            if v[2] or (  # Arg doesn't have default value in the signature.
                v[0] != v[1]  # Value is not same as default.
                and not (isinstance(v[0], float) and np.isnan(v[0]) and np.isnan(v[1]))
            ):  # both are not NANs
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.DEPENDENCY_VERSION_ERROR,
                    original_exception=RuntimeError(f"Arg {k} is not supported by current version of SKLearn/XGBoost."),
                )
        else:
            result[k] = v[0]
    return result


class RandomizedSearchCV(BaseTransformer):
    r"""Randomized search on hyper parameters
    For more details on this class, see [sklearn.model_selection.RandomizedSearchCV]
    (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

    Parameters
    ----------
    estimator : estimator object
        An object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict or list of dicts
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : str, callable, list, tuple or dict, default=None
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

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    refit : bool, str, or callable, default=True
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

    cv : int, cross-validation generator or an iterable, default=None
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

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch : int, or str, default='2*n_jobs'
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

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    input_cols : Optional[Union[str, List[str]]]
        A string or list of strings representing column names that contain features.
        If this parameter is not specified, all columns in the input DataFrame except
        the columns specified by label_cols and sample-weight_col parameters are
        considered input columns.

    label_cols : Optional[Union[str, List[str]]]
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

    sample_weight_col: Optional[str]
        A string representing the column name containing the examples’ weights.
        This argument is only required when working with weighted datasets.

    drop_input_cols: Optional[bool], default=False
        If set, the response of predict(), transform() methods will not contain input columns.
    """

    def __init__(  # type: ignore
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
        drop_input_cols: Optional[bool] = False,
        sample_weight_col: Optional[str] = None,
    ) -> None:
        super().__init__()
        deps: Set[str] = {
            f"numpy=={np.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"cloudpickle=={cp.__version__}",
            f"cachetools=={cachetools.__version__}",  # type: ignore
        }
        deps = deps | _gather_dependencies(estimator)
        self._deps = list(deps)
        estimator = _transform_snowml_obj_to_sklearn_obj(estimator)
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
        cleaned_up_init_args = _validate_sklearn_args(args=init_args, klass=sklearn.model_selection.RandomizedSearchCV)
        self._sklearn_object = sklearn.model_selection.RandomizedSearchCV(
            **cleaned_up_init_args,
        )
        self._model_signature_dict: Optional[Dict[str, ModelSignature]] = None
        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)
        self.set_label_cols(label_cols)
        self.set_drop_input_cols(drop_input_cols)
        self.set_sample_weight_col(sample_weight_col)

    def _get_rand_id(self) -> str:
        """
        Generate random id to be used in sproc and stage names.

        Returns:
            Random id string usable in sproc, table, and stage names.
        """
        return str(uuid4()).replace("-", "_").upper()

    def _infer_input_output_cols(self, dataset: Union[DataFrame, pd.DataFrame]) -> None:
        """
        Infer `self.input_cols` and `self.output_cols` if they are not explicitly set.

        Args:
            dataset: Input dataset.
        """
        if not self.input_cols:
            cols = [c for c in dataset.columns if c not in self.get_label_cols() and c != self.sample_weight_col]
            self.set_input_cols(input_cols=cols)

        if not self.output_cols:
            cols = [identifier.concat_names(ids=["OUTPUT_", c]) for c in self.label_cols]
            self.set_output_cols(output_cols=cols)

    def _get_active_columns(self) -> List[str]:
        """ "Get the list of columns that are relevant to the transformer."""
        selected_cols = (
            self.input_cols + self.label_cols + ([self.sample_weight_col] if self.sample_weight_col is not None else [])
        )
        return selected_cols

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    def fit(self, dataset: Union[DataFrame, pd.DataFrame]) -> "RandomizedSearchCV":
        """Run fit with all sets of parameters
        For more details on this function, see [sklearn.model_selection.RandomizedSearchCV.fit]
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV.fit)


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.

        Returns:
            self
        """
        self._infer_input_output_cols(dataset)
        if isinstance(dataset, pd.DataFrame):
            self._fit_pandas(dataset)
        elif isinstance(dataset, DataFrame):
            self._fit_snowpark(dataset)
        else:
            raise TypeError(
                f"Unexpected dataset type: {type(dataset)}."
                "Supported dataset types: snowpark.DataFrame, pandas.DataFrame."
            )
        self._is_fitted = True
        self._get_model_signatures(dataset)
        return self

    def _fit_snowpark(self, dataset: DataFrame) -> None:
        session = dataset._session
        assert session is not None
        # Validate that key package version in user workspace are supported in snowflake conda channel
        # If customer doesn't have package in conda channel, replace the ones have the closest versions
        self._deps = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=self._get_dependencies(), session=session, subproject=_SUBPROJECT
        )

        # Create two stages - one for data and one for estimators.
        temp_stage_name = random_name_for_temp_object(TempObjectType.STAGE)
        temp_stage_creation_query = f"CREATE OR REPLACE TEMP STAGE {temp_stage_name};"
        session.sql(temp_stage_creation_query).collect()

        # Stage data.
        data_name = temp_stage_name
        selected_cols = self._get_active_columns()
        if len(selected_cols) > 0:
            dataset = dataset.select(selected_cols)
        # TODO: add index column to the staged data
        dataset.write.save_as_table(f"{data_name}", table_type="temp")

        # TODO: explore using Fileset.make()
        file_format_name = "parquet_file_format"
        file_format_query = f"CREATE OR REPLACE FILE FORMAT {file_format_name} TYPE = 'PARQUET';"
        session.sql(file_format_query).collect()

        file_format = f"FILE_FORMAT = (FORMAT_NAME = '{file_format_name}')"
        stage_load_query = f"COPY INTO @{temp_stage_name}/{data_name} FROM {data_name} {file_format} header = true"
        session.sql(stage_load_query).collect()

        imports = [f"@{row.name}" for row in session.sql(f"LIST @{temp_stage_name}").collect()]

        # Create estimators with subset of param grid.
        # TODO: Decide how to choose parallelization factor.
        parallel_factor = 16

        assert self._sklearn_object is not None
        params_to_evaluate = list(
            ParameterSampler(self._sklearn_object.param_distributions, n_iter=self._sklearn_object.n_iter)
        )
        max_params_per_estimator = ceil(len(params_to_evaluate) / parallel_factor)
        param_chunks = [
            params_to_evaluate[x : x + max_params_per_estimator]
            for x in range(0, len(params_to_evaluate), max_params_per_estimator)
        ]
        target_locations = []
        for param_chunk in param_chunks:

            param_chunk_dist: Any = defaultdict(set)
            for d in param_chunk:
                for k, v in d.items():
                    param_chunk_dist[k].add(v)
            for k, v in param_chunk_dist.items():
                param_chunk_dist[k] = list(v)

            estimator = copy.deepcopy(self._sklearn_object)
            estimator.param_distributions = param_chunk_dist

            # Create a temp file and dump the transform to that file.
            local_transform_file_name = get_temp_file_path()
            with open(local_transform_file_name, mode="w+b") as local_transform_file:
                cp.dump(estimator, local_transform_file)

            # Put locally serialized transform on stage and add it to the list of imports.
            # TODO: Add statement params.
            put_result = session.file.put(
                local_transform_file_name,
                temp_stage_name,
                auto_compress=False,
                overwrite=True,
            )
            target_location = put_result[0].target
            target_locations.append(target_location)
            imports.append(f"@{temp_stage_name}/{target_location}")

        input_cols = copy.deepcopy(self.input_cols)
        label_cols = copy.deepcopy(self.label_cols)

        @cachetools.cached(cache={})
        def _load_data_into_udtf() -> Tuple[pd.DataFrame, pd.DataFrame]:
            data_files = [
                filename
                for filename in os.listdir(sys._xoptions["snowflake_import_directory"])
                if filename.startswith(data_name)
            ]
            partial_df = [
                pd.read_parquet(os.path.join(sys._xoptions["snowflake_import_directory"], file_name))
                for file_name in data_files
            ]
            df = pd.concat(partial_df, ignore_index=True)

            for column in df:
                df[column] = pd.to_numeric(df[column])

            dfx = df[input_cols]
            dfy = df[label_cols]

            return dfx, dfy

        # TODO: set refit = False, and fit after retrieving the resultf from udtf
        @udtf(
            output_schema=StructType(
                [
                    StructField("ESTIMATOR_LOCATION", StringType()),
                    StructField("BEST_SCORE", StringType()),
                    StructField("ESTIMATOR", BinaryType()),
                ]
            ),
            input_types=[StringType()],
            name="hyperparameter_tuning",
            packages=["snowflake-snowpark-python"] + self._get_dependencies(),
            replace=True,
            imports=imports,
        )
        class SearchCV:
            def __init__(self) -> None:
                dfx, dfy = _load_data_into_udtf()
                self._dfx = dfx
                self._dfy = dfy

            def process(self, estimator_location):
                local_transform_file_path = os.path.join(
                    sys._xoptions["snowflake_import_directory"], f"{estimator_location}"
                )
                with open(local_transform_file_path, mode="rb") as local_transform_file_obj:
                    estimator = cp.load(local_transform_file_obj)

                # TODO: handle sample weights.
                fit_estimator = estimator.fit(self._dfx, self._dfy)

                # TODO: handle the case of estimator size > maximum column size or to just serialize and return score.
                yield (estimator_location, fit_estimator.best_score_, cloudpickle.dumps(fit_estimator))

            def end_partition(self) -> None:
                ...

        # TODO: Check partitioning to ensure that partitions are uniformly distributed over UDTF intances
        # Set parallelism to 16 and ensure that one partion goes to one instance of UDTF
        HP_TUNING = F.table_function("hyperparameter_tuning")

        df = session.create_dataframe(target_locations, schema=["estimator_location"])
        results = df.select(HP_TUNING(df["estimator_location"]).over(partition_by=df["estimator_location"])).sort(
            col("BEST_SCORE").desc()
        )
        # TODO: check cv_results
        best_estimator = cloudpickle.loads(results.select("ESTIMATOR").first()[0])
        self._sklearn_object = best_estimator

    def _fit_pandas(self, dataset: pd.DataFrame) -> None:
        assert self._sklearn_object is not None
        argspec = inspect.getfullargspec(self._sklearn_object.fit)
        args = {"X": dataset[self.input_cols]}
        if self.label_cols:
            label_arg_name = "Y" if "Y" in argspec.args else "y"
            args[label_arg_name] = dataset[self.label_cols].squeeze()

        if self.sample_weight_col is not None and "sample_weight" in argspec.args:
            args["sample_weight"] = dataset[self.sample_weight_col].squeeze()

        self._sklearn_object.fit(**args)

    def _get_pass_through_columns(self, dataset: DataFrame) -> List[str]:
        if self._drop_input_cols:
            return []
        else:
            return list(set(dataset.columns) - set(self.output_cols))

    def _batch_inference(
        self,
        dataset: DataFrame,
        inference_method: str,
        expected_output_cols_list: List[str],
        expected_output_cols_type: str = "",
    ) -> DataFrame:
        """Util method to create UDF and run batch inference."""
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
        # Validate that key package version in user workspace are supported in snowflake conda channel
        pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=self._get_dependencies(), session=session, subproject=_SUBPROJECT
        )

        # Register vectorized UDF for batch inference
        batch_inference_udf_name = random_name_for_temp_object(TempObjectType.FUNCTION)

        # Need to do this since if we use self._sklearn_object directly in the UDF, Snowpark
        # will try to pickle all of self which fails.
        estimator = self._sklearn_object

        # Input columns for UDF are sorted by column names.
        # We need actual order of input cols to reorder dataframe before calling inference methods.
        input_cols = self.input_cols
        unquoted_input_cols = identifier.get_unescaped_names(self.input_cols)

        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=_SUBPROJECT,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[pandas_udf],
            custom_tags=dict([("autogen", True)]),
        )

        @pandas_udf(  # type: ignore
            is_permanent=False,
            name=batch_inference_udf_name,
            packages=self._get_dependencies(),  # type: ignore
            replace=True,
            session=session,
            statement_params=statement_params,
        )
        def vec_batch_infer(ds: PandasSeries[dict]) -> PandasSeries[dict]:  # type: ignore
            import numpy as np
            import pandas as pd

            input_df = pd.io.json.json_normalize(ds)

            # pd.io.json.json_normalize() doesn't remove quotes around quoted identifiers like snowpakr_df.to_pandas().
            # But trained models have unquoted input column names saved in internal state if trained using snowpark_df
            # or quoted input column names saved in internal state if trained using pandas_df.
            # Model expects exact same columns names in the input df for predict call.

            input_df = input_df[input_cols]  # Select input columns with quoted column names.
            if hasattr(estimator, "feature_names_in_"):
                assert estimator is not None
                missing_features = []
                for i, f in enumerate(estimator.feature_names_in_):
                    if i >= len(input_cols) or (input_cols[i] != f and unquoted_input_cols[i] != f):
                        missing_features.append(f)

                if len(missing_features) > 0:
                    raise ValueError(
                        "The feature names should match with those that were passed during fit.\n"
                        f"Features seen during fit call but not present in the input: {missing_features}\n"
                        f"Features in the input dataframe : {input_cols}\n"
                    )
                input_df.columns = estimator.feature_names_in_
            else:
                # Just rename the column names to unquoted identifiers.
                input_df.columns = (
                    unquoted_input_cols  # Replace the quoted columns identifier with unquoted column ids.
                )
            transformed_numpy_array = getattr(estimator, inference_method)(input_df)
            if (
                isinstance(transformed_numpy_array, list)
                and len(transformed_numpy_array) > 0
                and isinstance(transformed_numpy_array[0], np.ndarray)
            ):
                # In case of multioutput estimators, predict_proba(), decision_function(), etc., functions return
                # a list of ndarrays. We need to concatenate them.
                transformed_numpy_array = np.concatenate(transformed_numpy_array, axis=1)

            if len(transformed_numpy_array.shape) == 3:
                # VotingClassifier will return results of shape (n_classifiers, n_samples, n_classes)
                # when voting = "soft" and flatten_transform = False. We can't handle unflatten transforms,
                # so we ignore flatten_transform flag and flatten the results.
                transformed_numpy_array = np.hstack(transformed_numpy_array)

            if len(transformed_numpy_array.shape) > 1 and transformed_numpy_array.shape[1] != len(
                expected_output_cols_list
            ):
                # HeterogeneousEnsemble's transfrom method produce results with variying shapes
                # from (n_samples, n_estimators) to (n_samples, n_estimators * n_classes).
                # It is hard to predict the response shape without using fragile introspection logic.
                # So, to avoid that we are packing the results into a dataframe of shape (n_samples, 1) with
                # each element being a list.
                if len(expected_output_cols_list) != 1:
                    raise TypeError(
                        "expected_output_cols_list must be same length as transformed array or " "should be of length 1"
                    )
                series = pd.Series(transformed_numpy_array.tolist())
                transformed_pandas_df = pd.DataFrame(series, columns=expected_output_cols_list)
            else:
                transformed_pandas_df = pd.DataFrame(transformed_numpy_array, columns=expected_output_cols_list)
            return transformed_pandas_df.to_dict("records")  # type: ignore

        batch_inference_table_name = f"SNOWML_BATCH_INFERENCE_INPUT_TABLE_{self._get_rand_id()}"

        pass_through_columns = self._get_pass_through_columns(dataset)
        # Run Transform
        query_from_df = str(dataset.queries["queries"][0])

        outer_select_list = pass_through_columns[:]
        inner_select_list = pass_through_columns[:]

        outer_select_list.extend(
            [
                "{object_name}:{column_name}{udf_datatype} as {column_name}".format(
                    object_name=batch_inference_udf_name,
                    column_name=c,
                    udf_datatype=(f"::{expected_output_cols_type}" if expected_output_cols_type else ""),
                )
                for c in expected_output_cols_list
            ]
        )

        inner_select_list.extend(
            [
                "{udf_name}(object_construct_keep_null({input_cols_dict})) AS {udf_name}".format(
                    udf_name=batch_inference_udf_name,
                    input_cols_dict=", ".join([f"'{c}', {c}" for c in self.input_cols]),
                )
            ]
        )

        sql = """WITH {input_table_name} AS ({query})
                    SELECT
                      {outer_select_stmt}
                    FROM (
                      SELECT
                        {inner_select_stmt}
                      FROM {input_table_name}
                    )
               """.format(
            input_table_name=batch_inference_table_name,
            query=query_from_df,
            outer_select_stmt=", ".join(outer_select_list),
            inner_select_stmt=", ".join(inner_select_list),
        )

        return session.sql(sql)

    def _sklearn_inference(
        self, dataset: pd.DataFrame, inference_method: str, expected_output_cols_list: List[str]
    ) -> pd.DataFrame:
        output_cols = expected_output_cols_list.copy()

        # Model expects exact same columns names in the input df for predict call.
        # Given the scenario that user use snowpark DataFrame in fit call, but pandas DataFrame in predict call
        # input cols need to match unquoted / quoted
        input_cols = self.input_cols
        unquoted_input_cols = identifier.get_unescaped_names(self.input_cols)
        quoted_input_cols = identifier.get_escaped_names(unquoted_input_cols)

        estimator = self._sklearn_object

        assert estimator is not None
        features_required_by_estimator = (
            estimator.feature_names_in_ if hasattr(estimator, "feature_names_in_") else unquoted_input_cols
        )
        missing_features = []
        features_in_dataset = set(dataset.columns)
        columns_to_select = []
        for i, f in enumerate(features_required_by_estimator):
            if (
                i >= len(input_cols)
                or (input_cols[i] != f and unquoted_input_cols[i] != f and quoted_input_cols[i] != f)
                or (
                    input_cols[i] not in features_in_dataset
                    and unquoted_input_cols[i] not in features_in_dataset
                    and quoted_input_cols[i] not in features_in_dataset
                )
            ):
                missing_features.append(f)
            elif input_cols[i] in features_in_dataset:
                columns_to_select.append(input_cols[i])
            elif unquoted_input_cols[i] in features_in_dataset:
                columns_to_select.append(unquoted_input_cols[i])
            else:
                columns_to_select.append(quoted_input_cols[i])

        if len(missing_features) > 0:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(
                    "The feature names should match with those that were passed during fit.\n"
                    f"Features seen during fit call but not present in the input: {missing_features}\n"
                    f"Features in the input dataframe : {input_cols}\n"
                ),
            )
        input_df = dataset[columns_to_select]
        input_df.columns = features_required_by_estimator

        transformed_numpy_array = getattr(estimator, inference_method)(input_df)

        if (
            isinstance(transformed_numpy_array, list)
            and len(transformed_numpy_array) > 0
            and isinstance(transformed_numpy_array[0], np.ndarray)
        ):
            # In case of multioutput estimators, predict_proba(), decision_function(), etc., functions return
            # a list of ndarrays. We need to concatenate them.

            # First compute output column names
            if len(output_cols) == len(transformed_numpy_array):
                actual_output_cols = []
                for idx, np_arr in enumerate(transformed_numpy_array):
                    for i in range(1 if len(np_arr.shape) <= 1 else np_arr.shape[1]):
                        actual_output_cols.append(f"{output_cols[idx]}_{i}")
                output_cols = actual_output_cols

            # Concatenate np arrays
            transformed_numpy_array = np.concatenate(transformed_numpy_array, axis=1)

        if len(transformed_numpy_array.shape) == 3:
            # VotingClassifier will return results of shape (n_classifiers, n_samples, n_classes)
            # when voting = "soft" and flatten_transform = False. We can't handle unflatten transforms,
            # so we ignore flatten_transform flag and flatten the results.
            transformed_numpy_array = np.hstack(transformed_numpy_array)

        if len(transformed_numpy_array.shape) == 1:
            transformed_numpy_array = np.reshape(transformed_numpy_array, (-1, 1))

        shape = transformed_numpy_array.shape
        if shape[1] != len(output_cols):
            if len(output_cols) != 1:
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=TypeError(
                        "expected_output_cols_list must be same length as transformed array or " "should be of length 1"
                    ),
                )
            actual_output_cols = []
            for i in range(shape[1]):
                actual_output_cols.append(f"{output_cols[0]}_{i}")
            output_cols = actual_output_cols

        if self._drop_input_cols:
            dataset = pd.DataFrame(data=transformed_numpy_array, columns=output_cols)
        else:
            dataset = dataset.copy()
            dataset[output_cols] = transformed_numpy_array
        return dataset

    @available_if(_original_estimator_has_callable("predict"))  # type: ignore
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    @telemetry.add_stmt_params_to_df(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
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
        """
        super()._check_dataset_type(dataset)
        if isinstance(dataset, DataFrame):
            expected_type_inferred = ""
            # when it is classifier, infer the datatype from label columns
            if expected_type_inferred == "" and "predict" in self.model_signatures:
                expected_type_inferred = convert_sp_to_sf_type(
                    self.model_signatures["predict"].outputs[0].as_snowpark_type()
                )

            output_df = self._batch_inference(
                dataset=dataset,
                inference_method="predict",
                expected_output_cols_list=self.output_cols,
                expected_output_cols_type=expected_type_inferred,
            )
        elif isinstance(dataset, pd.DataFrame):
            output_df = self._sklearn_inference(
                dataset=dataset,
                inference_method="predict",
                expected_output_cols_list=self.output_cols,
            )

        return output_df

    @available_if(_original_estimator_has_callable("transform"))  # type: ignore
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    @telemetry.add_stmt_params_to_df(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
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
        if isinstance(dataset, DataFrame):
            expected_dtype = ""
            if False:  # is child of _BaseHeterogeneousEnsemble
                # transform() method of HeterogeneousEnsemble estimators return responses of varying shapes
                # from (n_samples, n_estimators) to (n_samples, n_estimators * n_classes) (and everything in between)
                # based on init param values. We will convert that to pandas dataframe of shape (n_samples, 1) with
                # each row containing a list of values.
                expected_dtype = "ARRAY"

            output_df = self._batch_inference(
                dataset=dataset,
                inference_method="transform",
                expected_output_cols_list=self.output_cols,
                expected_output_cols_type=expected_dtype,
            )
        elif isinstance(dataset, pd.DataFrame):
            output_df = self._sklearn_inference(
                dataset=dataset,
                inference_method="transform",
                expected_output_cols_list=self.output_cols,
            )

        return output_df

    def _get_output_column_names(self, output_cols_prefix: str) -> List[str]:
        """Returns the list of output columns for predict_proba(), decision_function(), etc.. functions.
        Returns a list with output_cols_prefix as the only element if the estimator is not a classifier.

        Args:
            output_cols_prefix (str): prefix according to the function

        Returns:
            List[str]: output cols with prefix
        """
        if getattr(self._sklearn_object, "classes_", None) is None:
            return [output_cols_prefix]

        assert self._sklearn_object is not None  # keep mypy happy
        classes = self._sklearn_object.classes_
        if isinstance(classes, numpy.ndarray):
            return [f"{output_cols_prefix}{c}" for c in classes.tolist()]
        elif isinstance(classes, list) and len(classes) > 0 and isinstance(classes[0], numpy.ndarray):
            # If the estimator is a multioutput estimator, classes_ will be a list of ndarrays.
            output_cols = []
            for i, cl in enumerate(classes):
                # For binary classification, there is only one output column for each class
                # ndarray as the two classes are complementary.
                if len(cl) == 2:
                    output_cols.append(f"{output_cols_prefix}_{i}_{cl[0]}")
                else:
                    output_cols.extend([f"{output_cols_prefix}_{i}_{c}" for c in cl.tolist()])
            return output_cols
        return []

    @available_if(_original_estimator_has_callable("predict_proba"))  # type: ignore
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    @telemetry.add_stmt_params_to_df(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
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
        if isinstance(dataset, DataFrame):
            output_df = self._batch_inference(
                dataset=dataset,
                inference_method="predict_proba",
                expected_output_cols_list=self._get_output_column_names(output_cols_prefix),
                expected_output_cols_type="float",
            )
        elif isinstance(dataset, pd.DataFrame):
            output_df = self._sklearn_inference(
                dataset=dataset,
                inference_method="predict_proba",
                expected_output_cols_list=self._get_output_column_names(output_cols_prefix),
            )

        return output_df

    @available_if(_original_estimator_has_callable("predict_log_proba"))  # type: ignore
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    @telemetry.add_stmt_params_to_df(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
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
        if isinstance(dataset, DataFrame):
            output_df = self._batch_inference(
                dataset=dataset,
                inference_method="predict_log_proba",
                expected_output_cols_list=self._get_output_column_names(output_cols_prefix),
                expected_output_cols_type="float",
            )
        elif isinstance(dataset, pd.DataFrame):
            output_df = self._sklearn_inference(
                dataset=dataset,
                inference_method="predict_log_proba",
                expected_output_cols_list=self._get_output_column_names(output_cols_prefix),
            )

        return output_df

    @available_if(_original_estimator_has_callable("decision_function"))  # type: ignore
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    @telemetry.add_stmt_params_to_df(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
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
        if isinstance(dataset, DataFrame):
            output_df = self._batch_inference(
                dataset=dataset,
                inference_method="decision_function",
                expected_output_cols_list=self._get_output_column_names(output_cols_prefix),
                expected_output_cols_type="float",
            )
        elif isinstance(dataset, pd.DataFrame):
            output_df = self._sklearn_inference(
                dataset=dataset,
                inference_method="decision_function",
                expected_output_cols_list=self._get_output_column_names(output_cols_prefix),
            )

        return output_df

    @available_if(_original_estimator_has_callable("score"))  # type: ignore
    def score(self, dataset: Union[DataFrame, pd.DataFrame]) -> float:
        """
        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.

        Returns:
            Score.
        """
        self._infer_input_output_cols(dataset)
        super()._check_dataset_type(dataset)
        if isinstance(dataset, pd.DataFrame):
            output_score = self._score_sklearn(dataset)
        elif isinstance(dataset, DataFrame):
            output_score = self._score_snowpark(dataset)
        return output_score

    def _score_sklearn(self, dataset: pd.DataFrame) -> float:
        assert self._sklearn_object is not None and hasattr(self._sklearn_object, "score")  # make type checker happy
        argspec = inspect.getfullargspec(self._sklearn_object.score)
        if "X" in argspec.args:
            args = {"X": dataset[self.input_cols]}
        elif "X_test" in argspec.args:
            args = {"X_test": dataset[self.input_cols]}
        else:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=RuntimeError("Neither 'X' or 'X_test' exist in argument"),
            )

        if self.label_cols:
            label_arg_name = "Y" if "Y" in argspec.args else "y"
            args[label_arg_name] = dataset[self.label_cols].squeeze()

        if self.sample_weight_col is not None and "sample_weight" in argspec.args:
            args["sample_weight"] = dataset[self.sample_weight_col].squeeze()

        score = self._sklearn_object.score(**args)
        return score

    def _score_snowpark(self, dataset: DataFrame) -> float:
        # Specify input columns so column pruing will be enforced
        selected_cols = self._get_active_columns()
        if len(selected_cols) > 0:
            dataset = dataset.select(selected_cols)

        # Extract queries that generated the dataframe. We will need to pass it to score procedure.
        queries = dataset.queries["queries"]

        # Create a temp file and dump the score to that file.
        local_score_file_name = get_temp_file_path()
        with open(local_score_file_name, mode="w+b") as local_score_file:
            cp.dump(self._sklearn_object, local_score_file)

        # Create temp stage to run score.
        score_stage_name = random_name_for_temp_object(TempObjectType.STAGE)
        session = dataset._session
        assert session is not None  # keep mypy happy
        stage_creation_query = f"CREATE OR REPLACE TEMPORARY STAGE {score_stage_name};"
        SqlResultValidator(session=session, query=stage_creation_query).has_dimensions(
            expected_rows=1, expected_cols=1
        ).validate()

        # Use posixpath to construct stage paths
        stage_score_file_name = posixpath.join(score_stage_name, os.path.basename(local_score_file_name))
        score_sproc_name = random_name_for_temp_object(TempObjectType.PROCEDURE)
        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=_SUBPROJECT,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[sproc],
            custom_tags=dict([("autogen", True)]),
        )
        # Put locally serialized score on stage.
        session.file.put(
            local_score_file_name,
            stage_score_file_name,
            auto_compress=False,
            overwrite=True,
            statement_params=statement_params,
        )

        @sproc(
            is_permanent=False,
            name=score_sproc_name,
            packages=["snowflake-snowpark-python"] + self._get_dependencies(),  # type: ignore
            replace=True,
            session=session,
            statement_params=statement_params,
            anonymous=True,
        )
        def score_wrapper_sproc(
            session: Session,
            sql_queries: List[str],
            stage_score_file_name: str,
            input_cols: List[str],
            label_cols: List[str],
            sample_weight_col: Optional[str],
            statement_params: Dict[str, str],
        ) -> float:
            import inspect
            import os
            import tempfile

            import cloudpickle as cp
            import numpy as np  # noqa: F401
            import pandas  # noqa: F401
            import sklearn  # noqa: F401

            for query in sql_queries[:-1]:
                _ = session.sql(query).collect(statement_params=statement_params)
            df = session.sql(sql_queries[-1]).to_pandas(statement_params=statement_params)

            local_score_file = tempfile.NamedTemporaryFile(delete=True)
            local_score_file_name = local_score_file.name
            local_score_file.close()

            session.file.get(stage_score_file_name, local_score_file_name, statement_params=statement_params)

            local_score_file_name_path = os.path.join(local_score_file_name, os.listdir(local_score_file_name)[0])
            with open(local_score_file_name_path, mode="r+b") as local_score_file_obj:
                estimator = cp.load(local_score_file_obj)

            argspec = inspect.getfullargspec(estimator.score)
            if "X" in argspec.args:
                args = {"X": df[input_cols]}
            elif "X_test" in argspec.args:
                args = {"X_test": df[input_cols]}
            else:
                raise RuntimeError("Neither 'X' or 'X_test' exist in argument")

            if label_cols:
                label_arg_name = "Y" if "Y" in argspec.args else "y"
                args[label_arg_name] = df[label_cols].squeeze()

            if sample_weight_col is not None and "sample_weight" in argspec.args:
                args["sample_weight"] = df[sample_weight_col].squeeze()

            result: float = estimator.score(**args)
            return result

        # Call score sproc
        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=_SUBPROJECT,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[Session.call],
            custom_tags=dict([("autogen", True)]),
        )
        score: float = score_wrapper_sproc(
            session,
            queries,
            stage_score_file_name,
            identifier.get_unescaped_names(self.input_cols),
            identifier.get_unescaped_names(self.label_cols),
            identifier.get_unescaped_names(self.sample_weight_col),
            statement_params,
        )

        cleanup_temp_files([local_score_file_name])

        return score

    def _get_model_signatures(self, dataset: Union[DataFrame, pd.DataFrame]) -> None:
        self._model_signature_dict = dict()

        PROB_FUNCTIONS = ["predict_log_proba", "predict_proba", "decision_function"]

        inputs = list(_infer_signature(dataset[self.input_cols], "input"))
        outputs: List[BaseFeatureSpec] = []
        if hasattr(self, "predict"):
            # keep mypy happy
            assert self._sklearn_object is not None and hasattr(self._sklearn_object, "_estimator_type")
            # For classifier, the type of predict is the same as the type of label
            if self._sklearn_object._estimator_type == "classifier":
                # label columns is the desired type for output
                outputs = _infer_signature(dataset[self.label_cols], "output")
                # rename the output columns
                outputs = model_signature_utils.rename_features(outputs, self.output_cols)
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

    @property
    def model_signatures(self) -> Dict[str, ModelSignature]:
        """Returns model signature of current class.

        Raises:
            SnowflakeMLException: If estimator is not fitted, then model signature cannot be inferred

        Returns:
            Dict[str, ModelSignature]: each method and its input output signature
        """
        if self._model_signature_dict is None:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=RuntimeError("Estimator not fitted before accessing property model_signatures!"),
            )
        return self._model_signature_dict

    def to_sklearn(self) -> Any:
        if self._sklearn_object is None:
            self._sklearn_object = self._create_sklearn_object()
        return self._sklearn_object

    def _get_dependencies(self) -> List[str]:
        return self._deps
