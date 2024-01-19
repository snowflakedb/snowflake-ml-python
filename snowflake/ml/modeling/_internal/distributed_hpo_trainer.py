import importlib
import inspect
import io
import os
import posixpath
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cloudpickle as cp
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import (
    identifier,
    pkg_version_utils,
    snowpark_dataframe_utils,
)
from snowflake.ml._internal.utils.temp_file_utils import (
    cleanup_temp_files,
    get_temp_file_path,
)
from snowflake.ml.modeling._internal.model_specifications import (
    ModelSpecificationsBuilder,
)
from snowflake.ml.modeling._internal.snowpark_trainer import SnowparkModelTrainer
from snowflake.snowpark import DataFrame, Session, functions as F
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    random_name_for_temp_object,
)
from snowflake.snowpark.functions import sproc, udtf
from snowflake.snowpark.row import Row
from snowflake.snowpark.types import IntegerType, StringType, StructField, StructType

cp.register_pickle_by_value(inspect.getmodule(get_temp_file_path))
cp.register_pickle_by_value(inspect.getmodule(identifier.get_inferred_name))

_PROJECT = "ModelDevelopment"
DEFAULT_UDTF_NJOBS = 3


def construct_cv_results(
    estimator: Union[GridSearchCV, RandomizedSearchCV],
    n_split: int,
    param_grid: List[Dict[str, Any]],
    cv_results_raw_hex: List[Row],
    cross_validator_indices_length: int,
    parameter_grid_length: int,
) -> Tuple[bool, Dict[str, Any]]:
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


cp.register_pickle_by_value(inspect.getmodule(construct_cv_results))


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
        input_cols: List[str],
        label_cols: Optional[List[str]],
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
        dependencies: List[str],
        udf_imports: List[str],
        input_cols: List[str],
        label_cols: Optional[List[str]],
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
        estimator_file_name = get_temp_file_path()
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

        search_sproc_name = random_name_for_temp_object(TempObjectType.PROCEDURE)
        random_udtf_name = random_name_for_temp_object(TempObjectType.FUNCTION)

        required_deps = dependencies + [
            "snowflake-snowpark-python<2",
            "fastparquet<2023.11",
            "pyarrow<14",
            "cachetools<5",
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
            imports: List[str],
            stage_estimator_file_name: str,
            input_cols: List[str],
            label_cols: Optional[List[str]],
        ) -> str:
            import os
            import time
            from typing import Iterator

            import cloudpickle as cp
            import pandas as pd
            import pyarrow.parquet as pq
            from sklearn.metrics import check_scoring
            from sklearn.metrics._scorer import _check_multimetric_scoring

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

            local_estimator_file_name = get_temp_file_path()
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
            local_indices_file_name = get_temp_file_path()
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

            assert estimator is not None

            @cachetools.cached(cache={})
            def _load_data_into_udf() -> Tuple[
                Dict[str, pd.DataFrame],
                Union[model_selection.GridSearchCV, model_selection.RandomizedSearchCV],
                pd.DataFrame,
                int,
                List[Dict[str, Any]],
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

                if sample_weight_col is not None and "sample_weight" in argspec.args:
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

                def process(self, params_idx: int, cv_idx: int) -> Iterator[Tuple[str]]:
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
                if sample_weight_col is not None and "sample_weight" in argspec.args:
                    args["sample_weight"] = df[sample_weight_col].squeeze()
                estimator.refit = original_refit
                refit_start_time = time.time()
                estimator.best_estimator_.fit(**args)
                refit_end_time = time.time()
                estimator.refit_time_ = refit_end_time - refit_start_time

                if hasattr(estimator.best_estimator_, "feature_names_in_"):
                    estimator.feature_names_in_ = estimator.best_estimator_.feature_names_in_

            local_result_file_name = get_temp_file_path()

            with open(local_result_file_name, mode="w+b") as local_result_file_obj:
                cp.dump(estimator, local_result_file_obj)

            session.file.put(
                local_result_file_name,
                temp_stage_name,
                auto_compress=False,
                overwrite=True,
            )

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

        local_estimator_path = get_temp_file_path()
        session.file.get(
            posixpath.join(temp_stage_name, sproc_export_file_name),
            local_estimator_path,
        )

        with open(os.path.join(local_estimator_path, sproc_export_file_name), mode="r+b") as result_file_obj:
            fit_estimator = cp.load(result_file_obj)

        cleanup_temp_files([local_estimator_path])

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
