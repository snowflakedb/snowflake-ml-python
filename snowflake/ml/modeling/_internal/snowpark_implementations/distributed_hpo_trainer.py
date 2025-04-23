import importlib
import inspect
import io
import os
import posixpath
import sys
import uuid
from typing import Any, Optional, Union

import cloudpickle as cp
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import (
    identifier,
    pkg_version_utils,
    snowpark_dataframe_utils,
    temp_file_utils,
)
from snowflake.ml.modeling._internal.estimator_utils import should_include_sample_weight
from snowflake.ml.modeling._internal.model_specifications import (
    ModelSpecificationsBuilder,
)
from snowflake.ml.modeling._internal.snowpark_implementations.snowpark_trainer import (
    SnowparkModelTrainer,
)
from snowflake.snowpark import DataFrame, Session, functions as F
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    random_name_for_temp_object,
)
from snowflake.snowpark.functions import sproc, udtf
from snowflake.snowpark.row import Row
from snowflake.snowpark.types import IntegerType, StringType, StructField, StructType
from snowflake.snowpark.udtf import UDTFRegistration

cp.register_pickle_by_value(inspect.getmodule(temp_file_utils.get_temp_file_path))
cp.register_pickle_by_value(inspect.getmodule(identifier.get_inferred_name))
cp.register_pickle_by_value(inspect.getmodule(snowpark_dataframe_utils.cast_snowpark_dataframe))
cp.register_pickle_by_value(inspect.getmodule(should_include_sample_weight))

_PROJECT = "ModelDevelopment"
DEFAULT_UDTF_NJOBS = 3
ENABLE_EFFICIENT_MEMORY_USAGE = True
_UDTF_STAGE_NAME = f"MEMORY_EFFICIENT_UDTF_{str(uuid.uuid4()).replace('-', '_')}"


def construct_cv_results(
    estimator: Union[GridSearchCV, RandomizedSearchCV],
    n_split: int,
    param_grid: list[dict[str, Any]],
    cv_results_raw_hex: list[Row],
    cross_validator_indices_length: int,
    parameter_grid_length: int,
) -> tuple[bool, dict[str, Any]]:
    """Construct the cross validation result from the UDF. Because we accelerate the process
    by the number of cross validation number, and the combination of parameter grids.
    Therefore, we need to stick them back together instead of returning the raw result
    to align with original sklearn result.

    Args:
        estimator (Union[GridSearchCV, RandomizedSearchCV]): The sklearn object of estimator
            GridSearchCV or RandomizedSearchCV
        n_split (int): The number of split, which is determined by build_cross_validator.get_n_splits(X, y, groups)
        param_grid (List[Dict[str, Any]]): the list of parameter grid or parameter sampler
        cv_results_raw_hex (List[Row]): the list of cv_results from each cv and parameter grid combination.
            Because UDxF can only return string, and numpy array/masked arrays cannot be encoded in a
            json format. Each cv_result is encoded into hex string.
        cross_validator_indices_length (int): the length of cross validator indices
        parameter_grid_length (int): the length of parameter grid combination

    Raises:
        ValueError: Retrieved empty cross validation results
        ValueError: Cross validator index length is 0
        ValueError: Parameter index length is 0
        ValueError: Retrieved incorrect dataframe dimension from Snowpark's UDTF.
        RuntimeError: Cross validation results are unexpectedly empty for one fold.

    Returns:
        Tuple[bool, Dict[str, Any]]: returns multimetric, cv_results_
    """
    # Filter corner cases: either the snowpark dataframe result is empty; or index length is empty
    if len(cv_results_raw_hex) == 0:
        raise ValueError(
            "Retrieved empty cross validation results from snowpark. Please retry or contact snowflake support."
        )
    if cross_validator_indices_length == 0:
        raise ValueError("Cross validator index length is 0. Was the CV iterator empty? ")
    if parameter_grid_length == 0:
        raise ValueError("Parameter index length is 0. Were there no candidates?")

    # cv_result maintains the original order
    multimetric = False
    # retrieve the cv_results from udtf table; results are encoded by hex and cloudpickle;
    # We are constructing the raw information back to original form
    if len(cv_results_raw_hex) != cross_validator_indices_length * parameter_grid_length:
        raise ValueError(
            "Retrieved incorrect dataframe dimension from Snowpark's UDTF."
            f"Expected {cross_validator_indices_length * parameter_grid_length}, got {len(cv_results_raw_hex)}. "
            "Please retry or contact snowflake support."
        )

    out = []

    for each_cv_result_hex in cv_results_raw_hex:
        # convert the hex string back to cv_results_
        hex_str = bytes.fromhex(each_cv_result_hex[0])
        with io.BytesIO(hex_str) as f_reload:
            each_cv_result = cp.load(f_reload)
            if not each_cv_result:
                raise RuntimeError(
                    "Cross validation response is empty. This issue may be temporary - please try again."
                )
            temp_dict = dict()
            """
            This dictionary has the following keys
            train_scores : dict of scorer name -> float
                Score on training set (for all the scorers),
                returned only if `return_train_score` is `True`.
            test_scores : dict of scorer name -> float
                Score on testing set (for all the scorers).
            fit_time : float
                Time spent for fitting in seconds.
            score_time : float
                Time spent for scoring in seconds.
            """
            if estimator.return_train_score:
                if each_cv_result.get("split0_train_score", None):
                    # for single scorer, the split0_train_score only contains an array with one value
                    temp_dict["train_scores"] = each_cv_result["split0_train_score"][0]
                else:
                    # if multimetric situation, the format would be
                    # {metric_name1: value, metric_name2: value, ...}
                    temp_dict["train_scores"] = {}
                    # For multi-metric evaluation, the scores for all the scorers are available in the
                    # cv_results_ dict at the keys ending with that scorer’s name ('_<scorer_name>')
                    # instead of '_score'.
                    for k, v in each_cv_result.items():
                        if "split0_train_" in k:
                            temp_dict["train_scores"][k[len("split0_train_") :]] = v
            if isinstance(each_cv_result.get("split0_test_score"), np.ndarray):
                temp_dict["test_scores"] = each_cv_result["split0_test_score"][0]
            else:
                temp_dict["test_scores"] = {}
                for k, v in each_cv_result.items():
                    if "split0_test_" in k:
                        temp_dict["test_scores"][k[len("split0_test_") :]] = v
            temp_dict["fit_time"] = each_cv_result["mean_fit_time"][0]
            temp_dict["score_time"] = each_cv_result["mean_score_time"][0]
            out.append(temp_dict)
    first_test_score = out[0]["test_scores"]
    multimetric = isinstance(first_test_score, dict)
    return multimetric, estimator._format_results(param_grid, n_split, out)


def construct_cv_results_memory_efficient_version(
    estimator: Union[GridSearchCV, RandomizedSearchCV],
    n_split: int,
    param_grid: list[dict[str, Any]],
    cv_results_raw_hex: list[Row],
    cross_validator_indices_length: int,
    parameter_grid_length: int,
) -> tuple[Any, dict[str, Any]]:
    """Construct the cross validation result from the UDF.
    The output is a raw dictionary generated by _fit_and_score, encoded into hex binary.
    This function need to decode the string and then call _format_result to stick them back together
    to align with original sklearn result.

    Args:
        estimator (Union[GridSearchCV, RandomizedSearchCV]): The sklearn object of estimator
            GridSearchCV or RandomizedSearchCV
        n_split (int): The number of split, which is determined by build_cross_validator.get_n_splits(X, y, groups)
        param_grid (List[Dict[str, Any]]): the list of parameter grid or parameter sampler
        cv_results_raw_hex (List[Row]): the list of cv_results from each cv and parameter grid combination.
            Because UDxF can only return string, and numpy array/masked arrays cannot be encoded in a
            json format. Each cv_result is encoded into hex string.
        cross_validator_indices_length (int): the length of cross validator indices
        parameter_grid_length (int): the length of parameter grid combination

    Raises:
        ValueError: Retrieved empty cross validation results
        ValueError: Cross validator index length is 0
        ValueError: Parameter index length is 0
        ValueError: Retrieved incorrect dataframe dimension from Snowpark's UDTF.

    Returns:
        Tuple[Any, Dict[str, Any]]: returns first_test_score, cv_results_
    """
    # Filter corner cases: either the snowpark dataframe result is empty; or index length is empty
    if len(cv_results_raw_hex) == 0:
        raise ValueError(
            "Retrieved empty cross validation results from snowpark. Please retry or contact snowflake support."
        )
    if cross_validator_indices_length == 0:
        raise ValueError("Cross validator index length is 0. Was the CV iterator empty? ")
    if parameter_grid_length == 0:
        raise ValueError("Parameter index length is 0. Were there no candidates?")

    all_out = []

    for each_cv_result_hex in cv_results_raw_hex:
        # convert the hex string back to cv_results_
        hex_str = bytes.fromhex(each_cv_result_hex[0])
        with io.BytesIO(hex_str) as f_reload:
            out = cp.load(f_reload)
            all_out.extend(out)

    # because original SearchCV is ranked by parameter first and cv second,
    # to make the memory efficient, we implemented by fitting on cv first and parameter second
    # when retrieving the results back, the ordering should revert back to remain the same result as original SearchCV
    def generate_the_order_by_parameter_index(all_combination_length: int) -> list[int]:
        pattern = []
        for i in range(all_combination_length):
            if i % parameter_grid_length == 0:
                pattern.append(i)
        for i in range(1, parameter_grid_length):
            for j in range(all_combination_length):
                if j % parameter_grid_length == i:
                    pattern.append(j)
        return pattern

    def rerank_array(original_array: list[Any], pattern: list[int]) -> list[Any]:
        reranked_array = []
        for index in pattern:
            reranked_array.append(original_array[index])
        return reranked_array

    pattern = generate_the_order_by_parameter_index(len(all_out))
    reranked_all_out = rerank_array(all_out, pattern)
    first_test_score = all_out[0]["test_scores"]
    return first_test_score, estimator._format_results(param_grid, n_split, reranked_all_out)


cp.register_pickle_by_value(inspect.getmodule(construct_cv_results))
cp.register_pickle_by_value(inspect.getmodule(construct_cv_results_memory_efficient_version))


class DistributedHPOTrainer(SnowparkModelTrainer):
    """
    A class for performing distributed hyperparameter optimization (HPO) using Snowpark.

    This class inherits from SnowparkModelTrainer and extends its functionality
    to support distributed HPO for machine learning models. It enables optimization
    of hyperparameters by distributing the tasks across the warehouse using Snowpark.
    """

    def __init__(
        self,
        estimator: object,
        dataset: DataFrame,
        session: Session,
        input_cols: list[str],
        label_cols: Optional[list[str]],
        sample_weight_col: Optional[str],
        autogenerated: bool = False,
        subproject: str = "",
    ) -> None:
        """
        Initializes the DistributedHPOTrainer with a model, a Snowpark DataFrame, feature, and label column names, etc.

        Args:
            estimator: SKLearn compatible estimator or transformer object.
            dataset: The dataset used for training the model.
            session: Snowflake session object to be used for training.
            input_cols: The name(s) of one or more columns in a DataFrame containing a feature to be used for training.
            label_cols: The name(s) of one or more columns in a DataFrame representing the target variable(s) to learn.
            sample_weight_col: The column name representing the weight of training examples.
            autogenerated: A boolean denoting if the trainer is being used by autogenerated code or not.
            subproject: subproject name to be used in telemetry.
        """
        super().__init__(
            estimator=estimator,
            dataset=dataset,
            session=session,
            input_cols=input_cols,
            label_cols=label_cols,
            sample_weight_col=sample_weight_col,
            autogenerated=autogenerated,
            subproject=subproject,
        )

    # TODO(snandamuri): Copied this code as it is from the snowpark_handler.
    #   Update it to improve the readability.
    def fit_search_snowpark(
        self,
        param_grid: Union[model_selection.ParameterGrid, model_selection.ParameterSampler],
        dataset: DataFrame,
        session: Session,
        estimator: Union[model_selection.GridSearchCV, model_selection.RandomizedSearchCV],
        dependencies: list[str],
        udf_imports: list[str],
        input_cols: list[str],
        label_cols: Optional[list[str]],
        sample_weight_col: Optional[str],
    ) -> Union[model_selection.GridSearchCV, model_selection.RandomizedSearchCV]:
        from itertools import product

        import cachetools
        from sklearn.base import clone, is_classifier
        from sklearn.calibration import check_cv

        # Create one stage for data and for estimators.
        temp_stage_name = random_name_for_temp_object(TempObjectType.STAGE)
        temp_stage_creation_query = f"CREATE OR REPLACE TEMP STAGE {temp_stage_name};"
        session.sql(temp_stage_creation_query).collect()

        # Stage data as parquet file
        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe(dataset)
        remote_file_path = f"{temp_stage_name}/{temp_stage_name}.parquet"
        dataset.write.copy_into_location(  # type:ignore[call-overload]
            remote_file_path, file_format_type="parquet", header=True, overwrite=True
        )
        imports = [f"@{row.name}" for row in session.sql(f"LIST @{temp_stage_name}").collect()]

        # Store GridSearchCV's refit variable. If user set it as False, we don't need to refit it again
        # refit variable can be boolean, string or callable
        original_refit = estimator.refit

        # Create a temp file and dump the estimator to that file.
        estimator_file_name = temp_file_utils.get_temp_file_path()
        params_to_evaluate = []
        for param_to_eval in list(param_grid):
            for k, v in param_to_eval.items():
                param_to_eval[k] = [v]
            params_to_evaluate.append([param_to_eval])

        with open(estimator_file_name, mode="w+b") as local_estimator_file_obj:
            # Set GridSearchCV refit as False and fit it again after retrieving the best param
            estimator.refit = False
            cp.dump(dict(estimator=estimator, param_grid=params_to_evaluate), local_estimator_file_obj)
        stage_estimator_file_name = posixpath.join(temp_stage_name, os.path.basename(estimator_file_name))
        sproc_statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[sproc],
        )
        udtf_statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[udtf],
            custom_tags=dict([("hpo_udtf", True)]),
        )

        # Put locally serialized estimator on stage.
        put_result = session.file.put(
            estimator_file_name,
            temp_stage_name,
            auto_compress=False,
            overwrite=True,
        )
        estimator_location = put_result[0].target
        imports.append(f"@{temp_stage_name}/{estimator_location}")
        temp_file_utils.cleanup_temp_files([estimator_file_name])

        search_sproc_name = random_name_for_temp_object(TempObjectType.PROCEDURE)
        random_udtf_name = random_name_for_temp_object(TempObjectType.TABLE_FUNCTION)

        required_deps = dependencies + [
            "snowflake-snowpark-python<2",
            "fastparquet<2023.11",
            "pyarrow<14",
            "cachetools<6",
        ]

        @sproc(  # type: ignore[misc]
            is_permanent=False,
            name=search_sproc_name,
            packages=required_deps,  # type: ignore[arg-type]
            replace=True,
            session=session,
            anonymous=True,
            imports=imports,  # type: ignore[arg-type]
            statement_params=sproc_statement_params,
        )
        def _distributed_search(
            session: Session,
            imports: list[str],
            stage_estimator_file_name: str,
            input_cols: list[str],
            label_cols: Optional[list[str]],
        ) -> str:
            import os
            import time
            from typing import Iterator

            import cloudpickle as cp
            import pandas as pd
            import pyarrow.parquet as pq
            from sklearn.metrics import check_scoring
            from sklearn.metrics._scorer import (
                _check_multimetric_scoring,
                _MultimetricScorer,
            )

            for import_name in udf_imports:
                importlib.import_module(import_name)

            data_files = [
                filename
                for filename in os.listdir(sys._xoptions["snowflake_import_directory"])
                if filename.startswith(temp_stage_name)
            ]
            partial_df = [
                pq.read_table(os.path.join(sys._xoptions["snowflake_import_directory"], file_name)).to_pandas()
                for file_name in data_files
            ]
            df = pd.concat(partial_df, ignore_index=True)
            df.columns = [identifier.get_inferred_name(col_) for col_ in df.columns]

            X = df[input_cols]
            y = df[label_cols].squeeze() if label_cols else None

            local_estimator_file_name = temp_file_utils.get_temp_file_path()
            session.file.get(stage_estimator_file_name, local_estimator_file_name)

            local_estimator_file_path = os.path.join(
                local_estimator_file_name, os.listdir(local_estimator_file_name)[0]
            )
            with open(local_estimator_file_path, mode="r+b") as local_estimator_file_obj:
                estimator = cp.load(local_estimator_file_obj)["estimator"]

            build_cross_validator = check_cv(estimator.cv, y, classifier=is_classifier(estimator.estimator))
            from sklearn.utils.validation import indexable

            X, y, _ = indexable(X, y, None)
            n_splits = build_cross_validator.get_n_splits(X, y, None)
            # store the cross_validator's test indices only to save space
            cross_validator_indices = [test for _, test in build_cross_validator.split(X, y, None)]
            local_indices_file_name = temp_file_utils.get_temp_file_path()
            with open(local_indices_file_name, mode="w+b") as local_indices_file_obj:
                cp.dump(cross_validator_indices, local_indices_file_obj)

            # Put locally serialized indices on stage.
            put_result = session.file.put(
                local_indices_file_name,
                temp_stage_name,
                auto_compress=False,
                overwrite=True,
            )
            indices_location = put_result[0].target
            imports.append(f"@{temp_stage_name}/{indices_location}")
            cross_validator_indices_length = int(len(cross_validator_indices))
            parameter_grid_length = len(param_grid)

            temp_file_utils.cleanup_temp_files([local_estimator_file_name, local_indices_file_name])

            assert estimator is not None

            @cachetools.cached(cache={})
            def _load_data_into_udf() -> tuple[
                dict[str, pd.DataFrame],
                Union[model_selection.GridSearchCV, model_selection.RandomizedSearchCV],
                pd.DataFrame,
                int,
                list[dict[str, Any]],
            ]:
                import pyarrow.parquet as pq

                data_files = [
                    filename
                    for filename in os.listdir(sys._xoptions["snowflake_import_directory"])
                    if filename.startswith(temp_stage_name)
                ]
                partial_df = [
                    pq.read_table(os.path.join(sys._xoptions["snowflake_import_directory"], file_name)).to_pandas()
                    for file_name in data_files
                ]
                df = pd.concat(partial_df, ignore_index=True)
                df.columns = [identifier.get_inferred_name(col_) for col_ in df.columns]

                # load estimator
                local_estimator_file_path = os.path.join(
                    sys._xoptions["snowflake_import_directory"], f"{estimator_location}"
                )
                with open(local_estimator_file_path, mode="rb") as local_estimator_file_obj:
                    estimator_objects = cp.load(local_estimator_file_obj)
                    estimator = estimator_objects["estimator"]
                    params_to_evaluate = estimator_objects["param_grid"]

                # load indices
                local_indices_file_path = os.path.join(
                    sys._xoptions["snowflake_import_directory"], f"{indices_location}"
                )
                with open(local_indices_file_path, mode="rb") as local_indices_file_obj:
                    indices = cp.load(local_indices_file_obj)

                argspec = inspect.getfullargspec(estimator.fit)
                args = {"X": df[input_cols]}

                if label_cols:
                    label_arg_name = "Y" if "Y" in argspec.args else "y"
                    args[label_arg_name] = df[label_cols].squeeze()

                if sample_weight_col is not None:
                    args["sample_weight"] = df[sample_weight_col].squeeze()
                return args, estimator, indices, len(df), params_to_evaluate

            class SearchCV:
                def __init__(self) -> None:
                    args, estimator, indices, data_length, params_to_evaluate = _load_data_into_udf()
                    self.args = args
                    self.estimator = estimator
                    self.indices = indices
                    self.data_length = data_length
                    self.params_to_evaluate = params_to_evaluate

                def process(self, params_idx: int, cv_idx: int) -> Iterator[tuple[str]]:
                    # Assign parameter to GridSearchCV
                    if hasattr(estimator, "param_grid"):
                        self.estimator.param_grid = self.params_to_evaluate[params_idx]
                    # Assign parameter to RandomizedSearchCV
                    else:
                        self.estimator.param_distributions = self.params_to_evaluate[params_idx]
                    # cross validator's indices: we stored test indices only (to save space);
                    # use the full indices to re-construct the train indices back.
                    full_indices = np.array([i for i in range(self.data_length)])
                    test_indice = self.indices[cv_idx]
                    train_indice = np.setdiff1d(full_indices, test_indice)
                    # assign the tuple of train and test indices to estimator's original cross validator
                    self.estimator.cv = [(train_indice, test_indice)]
                    self.estimator.fit(**self.args)
                    # If the cv_results_ is empty, then the udtf table will have different number of output rows
                    # from the input rows. Raise ValueError.
                    if not self.estimator.cv_results_:
                        raise RuntimeError(
                            """Cross validation results are unexpectedly empty for one fold.
                            This issue may be temporary - please try again."""
                        )
                    # Encode the dictionary of cv_results_ as binary (in hex format) to send it back
                    # because udtf doesn't allow numpy within json file
                    binary_cv_results = None
                    with io.BytesIO() as f:
                        cp.dump(self.estimator.cv_results_, f)
                        f.seek(0)
                        binary_cv_results = f.getvalue().hex()
                    yield (binary_cv_results,)

                def end_partition(self) -> None:
                    ...

            session.udtf.register(
                SearchCV,
                output_schema=StructType([StructField("CV_RESULTS", StringType())]),
                input_types=[IntegerType(), IntegerType()],
                name=random_udtf_name,
                packages=required_deps,  # type: ignore[arg-type]
                replace=True,
                is_permanent=False,
                imports=imports,  # type: ignore[arg-type]
                statement_params=udtf_statement_params,
            )

            HP_TUNING = F.table_function(random_udtf_name)

            # param_indices is for the index for each parameter grid;
            # cv_indices is for the index for each cross_validator's fold;
            # param_cv_indices is for the index for the product of (len(param_indices) * len(cv_indices))
            param_indices, cv_indices = [], []
            for param_idx, cv_idx in product(
                [param_index for param_index in range(parameter_grid_length)],
                [cv_index for cv_index in range(cross_validator_indices_length)],
            ):
                param_indices.append(param_idx)
                cv_indices.append(cv_idx)

            indices_info_pandas = pd.DataFrame(
                {
                    "PARAM_IND": param_indices,
                    "CV_IND": cv_indices,
                    "PARAM_CV_IND": [i for i in range(cross_validator_indices_length * parameter_grid_length)],
                }
            )
            indices_info_sp = session.create_dataframe(indices_info_pandas)
            # execute udtf by querying HP_TUNING table
            HP_raw_results = indices_info_sp.select(
                F.cast(indices_info_sp["PARAM_CV_IND"], IntegerType()).as_("PARAM_CV_IND"),
                (
                    HP_TUNING(indices_info_sp["PARAM_IND"], indices_info_sp["CV_IND"]).over(
                        partition_by=indices_info_sp["PARAM_CV_IND"]
                    )
                ),
            )
            # multimetric, cv_results_, best_param_index, scorers
            multimetric, cv_results_ = construct_cv_results(
                estimator,
                n_splits,
                list(param_grid),
                HP_raw_results.select("CV_RESULTS").sort(F.col("PARAM_CV_IND")).collect(),
                cross_validator_indices_length,
                parameter_grid_length,
            )

            estimator.cv_results_ = cv_results_
            estimator.multimetric_ = multimetric

            # Reconstruct the sklearn estimator.
            refit_metric = "score"
            if callable(estimator.scoring):
                scorers = estimator.scoring
            elif estimator.scoring is None or isinstance(estimator.scoring, str):
                scorers = check_scoring(estimator.estimator, estimator.scoring)
            else:
                scorers = _check_multimetric_scoring(estimator.estimator, estimator.scoring)
                estimator._check_refit_for_multimetric(scorers)
                refit_metric = original_refit
                scorers = _MultimetricScorer(scorers=scorers)

            estimator.scorer_ = scorers

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(estimator.scoring) and estimator.multimetric_:
                refit_metric = original_refit

            # For multi-metric evaluation, store the best_index_, best_params_ and
            # best_score_ iff refit is one of the scorer names
            # In single metric evaluation, refit_metric is "score"
            if original_refit or not estimator.multimetric_:
                estimator.best_index_ = estimator._select_best_index(original_refit, refit_metric, cv_results_)
                if not callable(original_refit):
                    # With a non-custom callable, we can select the best score
                    # based on the best index
                    estimator.best_score_ = cv_results_[f"mean_test_{refit_metric}"][estimator.best_index_]
                estimator.best_params_ = cv_results_["params"][estimator.best_index_]

            if original_refit:
                estimator.best_estimator_ = clone(estimator.estimator).set_params(
                    **clone(estimator.best_params_, safe=False)
                )

                # Let the sproc use all cores to refit.
                estimator.n_jobs = -1 if not estimator.n_jobs else estimator.n_jobs

                # process the input as args
                argspec = inspect.getfullargspec(estimator.fit)
                args = {"X": X}
                if label_cols:
                    label_arg_name = "Y" if "Y" in argspec.args else "y"
                    args[label_arg_name] = y
                if sample_weight_col is not None and should_include_sample_weight(estimator, "fit"):
                    args["sample_weight"] = df[sample_weight_col].squeeze()
                estimator.refit = original_refit
                refit_start_time = time.time()
                estimator.best_estimator_.fit(**args)
                refit_end_time = time.time()
                estimator.refit_time_ = refit_end_time - refit_start_time

                if hasattr(estimator.best_estimator_, "feature_names_in_"):
                    estimator.feature_names_in_ = estimator.best_estimator_.feature_names_in_

            local_result_file_name = temp_file_utils.get_temp_file_path()

            with open(local_result_file_name, mode="w+b") as local_result_file_obj:
                cp.dump(estimator, local_result_file_obj)

            session.file.put(
                local_result_file_name,
                temp_stage_name,
                auto_compress=False,
                overwrite=True,
            )
            temp_file_utils.cleanup_temp_files([local_result_file_name])

            # Note: you can add something like  + "|" + str(df) to the return string
            # to pass debug information to the caller.
            return str(os.path.basename(local_result_file_name))

        sproc_export_file_name = _distributed_search(
            session,
            imports,
            stage_estimator_file_name,
            input_cols,
            label_cols,
        )

        local_estimator_path = temp_file_utils.get_temp_file_path()
        session.file.get(
            posixpath.join(temp_stage_name, sproc_export_file_name),
            local_estimator_path,
        )

        with open(os.path.join(local_estimator_path, sproc_export_file_name), mode="r+b") as result_file_obj:
            fit_estimator = cp.load(result_file_obj)

        temp_file_utils.cleanup_temp_files([local_estimator_path])

        return fit_estimator

    def fit_search_snowpark_enable_efficient_memory_usage(
        self,
        param_grid: Union[model_selection.ParameterGrid, model_selection.ParameterSampler],
        dataset: DataFrame,
        session: Session,
        estimator: Union[model_selection.GridSearchCV, model_selection.RandomizedSearchCV],
        dependencies: list[str],
        udf_imports: list[str],
        input_cols: list[str],
        label_cols: Optional[list[str]],
        sample_weight_col: Optional[str],
    ) -> Union[model_selection.GridSearchCV, model_selection.RandomizedSearchCV]:
        from itertools import product

        from sklearn.base import clone, is_classifier
        from sklearn.calibration import check_cv

        # Create one stage for data and for estimators.
        temp_stage_name = random_name_for_temp_object(TempObjectType.STAGE)
        temp_stage_creation_query = f"CREATE OR REPLACE TEMP STAGE {temp_stage_name};"
        session.sql(temp_stage_creation_query).collect()

        # Stage data as parquet file
        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe(dataset)
        dataset_file_name = "dataset"
        remote_file_path = f"{temp_stage_name}/{dataset_file_name}.parquet"
        dataset.write.copy_into_location(  # type:ignore[call-overload]
            remote_file_path, file_format_type="parquet", header=True, overwrite=True
        )
        imports = [f"@{row.name}" for row in session.sql(f"LIST @{temp_stage_name}/{dataset_file_name}").collect()]

        # Create a temp file and dump the estimator to that file.
        estimator_file_name = temp_file_utils.get_temp_file_path()
        params_to_evaluate = list(param_grid)
        CONSTANTS: dict[str, Any] = dict()
        CONSTANTS["dataset_snowpark_cols"] = dataset.columns
        CONSTANTS["n_candidates"] = len(params_to_evaluate)
        CONSTANTS["_N_JOBS"] = estimator.n_jobs
        CONSTANTS["_PRE_DISPATCH"] = estimator.pre_dispatch

        with open(estimator_file_name, mode="w+b") as local_estimator_file_obj:
            cp.dump(dict(estimator=estimator, param_grid=params_to_evaluate), local_estimator_file_obj)
        stage_estimator_file_name = posixpath.join(temp_stage_name, os.path.basename(estimator_file_name))
        sproc_statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[sproc],
        )
        udtf_statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[udtf],
            custom_tags=dict([("hpo_memory_efficient", True)]),
        )
        from snowflake.ml.modeling._internal.snowpark_implementations.distributed_search_udf_file import (
            execute_template,
        )

        # Put locally serialized estimator on stage.
        session.file.put(
            estimator_file_name,
            temp_stage_name,
            auto_compress=False,
            overwrite=True,
        )
        estimator_location = os.path.basename(estimator_file_name)
        imports.append(f"@{temp_stage_name}/{estimator_location}")
        temp_file_utils.cleanup_temp_files([estimator_file_name])
        CONSTANTS["estimator_location"] = estimator_location

        search_sproc_name = random_name_for_temp_object(TempObjectType.PROCEDURE)
        random_udtf_name = random_name_for_temp_object(TempObjectType.FUNCTION)

        required_deps = dependencies + [
            "snowflake-snowpark-python<2",
            "fastparquet<2023.11",
            "pyarrow<14",
            "cachetools<6",
        ]

        @sproc(  # type: ignore[misc]
            is_permanent=False,
            name=search_sproc_name,
            packages=required_deps,  # type: ignore[arg-type]
            replace=True,
            session=session,
            anonymous=True,
            imports=imports,  # type: ignore[arg-type]
            statement_params=sproc_statement_params,
        )
        def _distributed_search(
            session: Session,
            imports: list[str],
            stage_estimator_file_name: str,
            input_cols: list[str],
            label_cols: Optional[list[str]],
        ) -> str:
            import os
            import time

            import cloudpickle as cp
            import pandas as pd
            import pyarrow.parquet as pq
            from sklearn.metrics import check_scoring
            from sklearn.metrics._scorer import (
                _check_multimetric_scoring,
                _MultimetricScorer,
            )
            from sklearn.utils.validation import _check_method_params, indexable

            # import packages in sproc
            for import_name in udf_imports:
                importlib.import_module(import_name)

            # os.cpu_count() returns the number of logical CPUs in the system. Returns None if undetermined.
            _NUM_CPUs = os.cpu_count() or 1

            # load dataset
            data_files = [
                filename
                for filename in os.listdir(sys._xoptions["snowflake_import_directory"])
                if filename.startswith(dataset_file_name)
            ]
            partial_df = [
                pq.read_table(os.path.join(sys._xoptions["snowflake_import_directory"], file_name)).to_pandas()
                for file_name in data_files
            ]
            df = pd.concat(partial_df, ignore_index=True)
            df.columns = [identifier.get_inferred_name(col_) for col_ in df.columns]

            X = df[input_cols]
            y = df[label_cols].squeeze() if label_cols else None
            DATA_LENGTH = len(df)
            fit_params = {}
            if sample_weight_col:
                fit_params["sample_weight"] = df[sample_weight_col].squeeze()

            local_estimator_file_folder_name = temp_file_utils.get_temp_file_path()
            session.file.get(stage_estimator_file_name, local_estimator_file_folder_name)

            local_estimator_file_path = os.path.join(
                local_estimator_file_folder_name, os.listdir(local_estimator_file_folder_name)[0]
            )
            with open(local_estimator_file_path, mode="r+b") as local_estimator_file_obj:
                estimator = cp.load(local_estimator_file_obj)["estimator"]

            # preprocess the attributes - (1) scorer
            refit_metric = "score"
            if callable(estimator.scoring):
                scorers = estimator.scoring
            elif estimator.scoring is None or isinstance(estimator.scoring, str):
                scorers = check_scoring(estimator.estimator, estimator.scoring)
            else:
                scorers = _check_multimetric_scoring(estimator.estimator, estimator.scoring)
                estimator._check_refit_for_multimetric(scorers)
                refit_metric = estimator.refit
                scorers = _MultimetricScorer(scorers=scorers)

            # preprocess the attributes - (2) check fit_params
            groups = None
            X, y, _ = indexable(X, y, groups)
            fit_params = _check_method_params(X, fit_params)

            # preprocess the attributes - (3) safe clone base estimator
            base_estimator = clone(estimator.estimator)

            # preprocess the attributes - (4) check cv
            build_cross_validator = check_cv(estimator.cv, y, classifier=is_classifier(estimator.estimator))
            n_splits = build_cross_validator.get_n_splits(X, y, groups)

            # preprocess the attributes - (5) generate fit_and_score_kwargs
            fit_and_score_kwargs = dict(
                scorer=scorers,
                fit_params=fit_params,
                score_params=None,
                return_train_score=estimator.return_train_score,
                return_n_test_samples=True,
                return_times=True,
                return_parameters=False,
                error_score=estimator.error_score,
                verbose=estimator.verbose,
            )

            # (1) store the cross_validator's test indices only to save space
            cross_validator_indices = [test for _, test in build_cross_validator.split(X, y, None)]
            local_indices_file_name = temp_file_utils.get_temp_file_path()
            with open(local_indices_file_name, mode="w+b") as local_indices_file_obj:
                cp.dump(cross_validator_indices, local_indices_file_obj)

            # Put locally serialized indices on stage.
            session.file.put(
                local_indices_file_name,
                temp_stage_name,
                auto_compress=False,
                overwrite=True,
            )
            indices_location = os.path.basename(local_indices_file_name)
            imports.append(f"@{temp_stage_name}/{indices_location}")

            # (2) store the base estimator
            local_base_estimator_file_name = temp_file_utils.get_temp_file_path()
            with open(local_base_estimator_file_name, mode="w+b") as local_base_estimator_file_obj:
                cp.dump(base_estimator, local_base_estimator_file_obj)
            session.file.put(
                local_base_estimator_file_name,
                temp_stage_name,
                auto_compress=False,
                overwrite=True,
            )
            base_estimator_location = os.path.basename(local_base_estimator_file_name)
            imports.append(f"@{temp_stage_name}/{base_estimator_location}")

            # (3) store the fit_and_score_kwargs
            local_fit_and_score_kwargs_file_name = temp_file_utils.get_temp_file_path()
            with open(local_fit_and_score_kwargs_file_name, mode="w+b") as local_fit_and_score_kwargs_file_obj:
                cp.dump(fit_and_score_kwargs, local_fit_and_score_kwargs_file_obj)
            session.file.put(
                local_fit_and_score_kwargs_file_name,
                temp_stage_name,
                auto_compress=False,
                overwrite=True,
            )
            fit_and_score_kwargs_location = os.path.basename(local_fit_and_score_kwargs_file_name)
            imports.append(f"@{temp_stage_name}/{fit_and_score_kwargs_location}")

            CONSTANTS["input_cols"] = input_cols
            CONSTANTS["label_cols"] = label_cols
            CONSTANTS["DATA_LENGTH"] = DATA_LENGTH
            CONSTANTS["n_splits"] = n_splits
            CONSTANTS["indices_location"] = indices_location
            CONSTANTS["base_estimator_location"] = base_estimator_location
            CONSTANTS["fit_and_score_kwargs_location"] = fit_and_score_kwargs_location

            # (6) store the constants
            local_constant_file_name = temp_file_utils.get_temp_file_path(prefix="constant")
            with open(local_constant_file_name, mode="w+b") as local_indices_file_obj:
                cp.dump(CONSTANTS, local_indices_file_obj)

            # Put locally serialized indices on stage.
            session.file.put(
                local_constant_file_name,
                temp_stage_name,
                auto_compress=False,
                overwrite=True,
            )
            constant_location = os.path.basename(local_constant_file_name)
            imports.append(f"@{temp_stage_name}/{constant_location}")

            temp_file_utils.cleanup_temp_files(
                [
                    local_estimator_file_folder_name,
                    local_indices_file_name,
                    local_base_estimator_file_name,
                    local_base_estimator_file_name,
                    local_fit_and_score_kwargs_file_name,
                    local_constant_file_name,
                ]
            )

            cross_validator_indices_length = int(len(cross_validator_indices))
            parameter_grid_length = len(param_grid)

            assert estimator is not None

            # Instantiate UDTFRegistration with the session object
            udtf_registration = UDTFRegistration(session)

            import tempfile

            # delete is set to False to support Windows environment
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                udf_code = execute_template
                f.file.write(udf_code)
                f.file.flush()

                # Use catchall exception handling and a finally block to clean up the _UDTF_STAGE_NAME
                try:
                    # Create one stage for data and for estimators.
                    # Because only permanent functions support _sf_node_singleton for now, therefore,
                    # UDTF creation would change to is_permanent=True, and manually drop the stage after UDTF is done
                    _stage_creation_query_udtf = f"CREATE OR REPLACE STAGE {_UDTF_STAGE_NAME};"
                    session.sql(_stage_creation_query_udtf).collect()

                    # Register the UDTF function from the file
                    udtf_registration.register_from_file(
                        file_path=f.name,
                        handler_name="SearchCV",
                        name=random_udtf_name,
                        output_schema=StructType(
                            [StructField("FIRST_IDX", IntegerType()), StructField("EACH_CV_RESULTS", StringType())]
                        ),
                        input_types=[IntegerType(), IntegerType(), IntegerType()],
                        replace=True,
                        imports=imports,  # type: ignore[arg-type]
                        stage_location=_UDTF_STAGE_NAME,
                        is_permanent=True,
                        packages=required_deps,  # type: ignore[arg-type]
                        statement_params=udtf_statement_params,
                    )

                    HP_TUNING = F.table_function(random_udtf_name)

                    # param_indices is for the index for each parameter grid;
                    # cv_indices is for the index for each cross_validator's fold;
                    # param_cv_indices is for the index for the product of (len(param_indices) * len(cv_indices))
                    cv_indices, param_indices = zip(
                        *product(range(cross_validator_indices_length), range(parameter_grid_length))
                    )

                    indices_info_pandas = pd.DataFrame(
                        {
                            "IDX": [
                                i // _NUM_CPUs for i in range(parameter_grid_length * cross_validator_indices_length)
                            ],
                            "PARAM_IND": param_indices,
                            "CV_IND": cv_indices,
                        }
                    )

                    indices_info_sp = session.create_dataframe(indices_info_pandas)
                    # execute udtf by querying HP_TUNING table
                    HP_raw_results = indices_info_sp.select(
                        (
                            HP_TUNING(
                                indices_info_sp["IDX"], indices_info_sp["PARAM_IND"], indices_info_sp["CV_IND"]
                            ).over(partition_by="IDX")
                        ),
                    )

                    first_test_score, cv_results_ = construct_cv_results_memory_efficient_version(
                        estimator,
                        n_splits,
                        list(param_grid),
                        HP_raw_results.select("EACH_CV_RESULTS").sort(F.col("FIRST_IDX")).collect(),
                        cross_validator_indices_length,
                        parameter_grid_length,
                    )

                    estimator.cv_results_ = cv_results_
                    estimator.multimetric_ = isinstance(first_test_score, dict)

                    # check refit_metric now for a callable scorer that is multimetric
                    if callable(estimator.scoring) and estimator.multimetric_:
                        estimator._check_refit_for_multimetric(first_test_score)
                        refit_metric = estimator.refit

                    # For multi-metric evaluation, store the best_index_, best_params_ and
                    # best_score_ iff refit is one of the scorer names
                    # In single metric evaluation, refit_metric is "score"
                    if estimator.refit or not estimator.multimetric_:
                        estimator.best_index_ = estimator._select_best_index(estimator.refit, refit_metric, cv_results_)
                        if not callable(estimator.refit):
                            # With a non-custom callable, we can select the best score
                            # based on the best index
                            estimator.best_score_ = cv_results_[f"mean_test_{refit_metric}"][estimator.best_index_]
                        estimator.best_params_ = cv_results_["params"][estimator.best_index_]

                    if estimator.refit:
                        estimator.best_estimator_ = clone(base_estimator).set_params(
                            **clone(estimator.best_params_, safe=False)
                        )

                        # Let the sproc use all cores to refit.
                        estimator.n_jobs = estimator.n_jobs or -1

                        # process the input as args
                        argspec = inspect.getfullargspec(estimator.fit)
                        args = {"X": X}
                        if label_cols:
                            label_arg_name = "Y" if "Y" in argspec.args else "y"
                            args[label_arg_name] = y
                        if sample_weight_col is not None:
                            args["sample_weight"] = df[sample_weight_col].squeeze()
                        # estimator.refit = original_refit
                        refit_start_time = time.time()
                        estimator.best_estimator_.fit(**args)
                        refit_end_time = time.time()
                        estimator.refit_time_ = refit_end_time - refit_start_time

                        if hasattr(estimator.best_estimator_, "feature_names_in_"):
                            estimator.feature_names_in_ = estimator.best_estimator_.feature_names_in_

                    # Store the only scorer not as a dict for single metric evaluation
                    estimator.scorer_ = scorers
                    estimator.n_splits_ = n_splits

                    local_result_file_name = temp_file_utils.get_temp_file_path()

                    with open(local_result_file_name, mode="w+b") as local_result_file_obj:
                        cp.dump(estimator, local_result_file_obj)

                    session.file.put(
                        local_result_file_name,
                        temp_stage_name,
                        auto_compress=False,
                        overwrite=True,
                    )

                    # Clean up the stages and files
                    session.sql(f"DROP STAGE IF EXISTS {_UDTF_STAGE_NAME}")

                    temp_file_utils.cleanup_temp_files([local_result_file_name])

                    return str(os.path.basename(local_result_file_name))
                finally:
                    # Clean up the stages
                    session.sql(f"DROP STAGE IF EXISTS {_UDTF_STAGE_NAME}")

        sproc_export_file_name = _distributed_search(
            session,
            imports,
            stage_estimator_file_name,
            input_cols,
            label_cols,
        )

        local_estimator_path = temp_file_utils.get_temp_file_path()
        session.file.get(
            posixpath.join(temp_stage_name, sproc_export_file_name),
            local_estimator_path,
        )

        with open(os.path.join(local_estimator_path, sproc_export_file_name), mode="r+b") as result_file_obj:
            fit_estimator = cp.load(result_file_obj)

        temp_file_utils.cleanup_temp_files(local_estimator_path)

        return fit_estimator

    def train(self) -> object:
        """
        Runs hyper parameter optimization by distributing the tasks across warehouse.

        Returns:
            Trained model
        """
        model_spec = ModelSpecificationsBuilder.build(model=self.estimator)
        assert isinstance(self.estimator, model_selection.GridSearchCV) or isinstance(
            self.estimator, model_selection.RandomizedSearchCV
        )
        if hasattr(self.estimator.estimator, "n_jobs") and self.estimator.estimator.n_jobs in [
            None,
            -1,
        ]:
            self.estimator.estimator.n_jobs = DEFAULT_UDTF_NJOBS

        if isinstance(self.estimator, model_selection.GridSearchCV):
            param_grid = model_selection.ParameterGrid(self.estimator.param_grid)
        elif isinstance(self.estimator, model_selection.RandomizedSearchCV):
            param_grid = model_selection.ParameterSampler(
                self.estimator.param_distributions,
                n_iter=self.estimator.n_iter,
                random_state=self.estimator.random_state,
            )
        relaxed_dependencies = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=model_spec.pkgDependencies, session=self.session
        )
        if ENABLE_EFFICIENT_MEMORY_USAGE:
            return self.fit_search_snowpark_enable_efficient_memory_usage(
                param_grid=param_grid,
                dataset=self.dataset,
                session=self.session,
                estimator=self.estimator,
                dependencies=relaxed_dependencies,
                udf_imports=["sklearn"],
                input_cols=self.input_cols,
                label_cols=self.label_cols,
                sample_weight_col=self.sample_weight_col,
            )

        return self.fit_search_snowpark(
            param_grid=param_grid,
            dataset=self.dataset,
            session=self.session,
            estimator=self.estimator,
            dependencies=relaxed_dependencies,
            udf_imports=["sklearn"],
            input_cols=self.input_cols,
            label_cols=self.label_cols,
            sample_weight_col=self.sample_weight_col,
        )
