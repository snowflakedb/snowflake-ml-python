import importlib
import inspect
import io
import json
import os
import posixpath
import sys
from collections import defaultdict
from math import ceil
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import cloudpickle as cp
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn import model_selection

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils.query_result_checker import SqlResultValidator
from snowflake.ml._internal.utils.temp_file_utils import (
    cleanup_temp_files,
    get_temp_file_path,
)
from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    random_name_for_temp_object,
)
from snowflake.snowpark.functions import col, pandas_udf, sproc, udf
from snowflake.snowpark.types import PandasSeries

_PROJECT = "ModelDevelopment"


def _get_rand_id() -> str:
    """
    Generate random id to be used in sproc and stage names.

    Returns:
        Random id string usable in sproc, table, and stage names.
    """
    return str(uuid4()).replace("-", "_").upper()


class SnowparkHandlers:
    def __init__(self, class_name: str, subproject: str) -> None:
        self._class_name = class_name
        self._subproject = subproject

    def _fit_wrapper_sproc(
        self, dependencies: List[str], session: Session, statement_params: Dict[str, str], fit_sproc_imports: List[str]
    ) -> Any:
        fit_sproc_name = random_name_for_temp_object(TempObjectType.PROCEDURE)

        @sproc(  # type: ignore
            is_permanent=False,
            name=fit_sproc_name,
            packages=dependencies,  # type: ignore
            replace=True,
            session=session,
            statement_params=statement_params,
            anonymous=True,
        )
        def fit_wrapper_sproc(
            session: Session,
            sql_queries: List[str],
            stage_transform_file_name: str,
            stage_result_file_name: str,
            input_cols: List[str],
            label_cols: List[str],
            sample_weight_col: Optional[str],
            statement_params: Dict[str, str],
        ) -> str:
            import inspect
            import os
            import tempfile

            import cloudpickle as cp
            import pandas as pd

            for import_name in fit_sproc_imports:
                importlib.import_module(import_name)

            # Execute snowpark queries and obtain the results as pandas dataframe
            # NB: this implies that the result data must fit into memory.
            for query in sql_queries[:-1]:
                _ = session.sql(query).collect(statement_params=statement_params)
            sp_df = session.sql(sql_queries[-1])
            df: pd.DataFrame = sp_df.to_pandas(statement_params=statement_params)
            df.columns = sp_df.columns

            local_transform_file = tempfile.NamedTemporaryFile(delete=True)
            local_transform_file_name = local_transform_file.name
            local_transform_file.close()

            session.file.get(stage_transform_file_name, local_transform_file_name, statement_params=statement_params)

            local_transform_file_path = os.path.join(
                local_transform_file_name, os.listdir(local_transform_file_name)[0]
            )
            with open(local_transform_file_path, mode="r+b") as local_transform_file_obj:
                estimator = cp.load(local_transform_file_obj)

            argspec = inspect.getfullargspec(estimator.fit)
            args = {"X": df[input_cols]}
            if label_cols:
                label_arg_name = "Y" if "Y" in argspec.args else "y"
                args[label_arg_name] = df[label_cols].squeeze()

            if sample_weight_col is not None and "sample_weight" in argspec.args:
                args["sample_weight"] = df[sample_weight_col].squeeze()

            estimator.fit(**args)

            local_result_file = tempfile.NamedTemporaryFile(delete=True)
            local_result_file_name = local_result_file.name
            local_result_file.close()

            with open(local_result_file_name, mode="w+b") as local_result_file_obj:
                cp.dump(estimator, local_result_file_obj)

            session.file.put(
                local_result_file_name,
                stage_result_file_name,
                auto_compress=False,
                overwrite=True,
                statement_params=statement_params,
            )

            # Note: you can add something like  + "|" + str(df) to the return string
            # to pass debug information to the caller.
            return str(os.path.basename(local_result_file_name))

        return fit_wrapper_sproc

    def fit_snowpark(
        self,
        dataset: DataFrame,
        session: Session,
        estimator: object,
        dependencies: List[str],
        fit_sproc_imports: List[str],
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> object:
        # Extract query that generated the datafrome. We will need to pass it to the fit procedure.
        queries = dataset.queries["queries"]

        # Create a temp file and dump the transform to that file.
        local_transform_file_name = get_temp_file_path()
        with open(local_transform_file_name, mode="w+b") as local_transform_file:
            cp.dump(estimator, local_transform_file)

        # Create temp stage to run fit.
        transform_stage_name = random_name_for_temp_object(TempObjectType.STAGE)
        stage_creation_query = f"CREATE OR REPLACE TEMPORARY STAGE {transform_stage_name};"
        SqlResultValidator(session=session, query=stage_creation_query).has_dimensions(
            expected_rows=1, expected_cols=1
        ).validate()

        # Use posixpath to construct stage paths
        stage_transform_file_name = posixpath.join(transform_stage_name, os.path.basename(local_transform_file_name))
        stage_result_file_name = posixpath.join(transform_stage_name, os.path.basename(local_transform_file_name))
        local_result_file_name = get_temp_file_path()

        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
            api_calls=[sproc],
            custom_tags=dict([("autogen", True)]),
        )
        # Put locally serialized transform on stage.
        session.file.put(
            local_transform_file_name,
            stage_transform_file_name,
            auto_compress=False,
            overwrite=True,
            statement_params=statement_params,
        )

        # Call fit sproc
        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
            api_calls=[Session.call],
            custom_tags=dict([("autogen", True)]),
        )

        fit_wrapper = self._fit_wrapper_sproc(dependencies, session, statement_params, fit_sproc_imports)

        sproc_export_file_name = fit_wrapper(
            session,
            queries,
            stage_transform_file_name,
            stage_result_file_name,
            input_cols,
            label_cols,
            sample_weight_col,
            statement_params,
        )

        if "|" in sproc_export_file_name:
            fields = sproc_export_file_name.strip().split("|")
            sproc_export_file_name = fields[0]
            if len(fields) > 1:
                print("\n".join(fields[1:]))

        session.file.get(
            posixpath.join(stage_result_file_name, sproc_export_file_name),
            local_result_file_name,
            statement_params=statement_params,
        )

        with open(os.path.join(local_result_file_name, sproc_export_file_name), mode="r+b") as result_file_obj:
            fit_estimator = cp.load(result_file_obj)

        cleanup_temp_files([local_transform_file_name, local_result_file_name])

        return fit_estimator

    def batch_inference(
        self,
        dataset: DataFrame,
        session: Session,
        estimator: object,
        dependencies: List[str],
        inference_method: str,
        input_cols: List[str],
        pass_through_columns: List[str],
        expected_output_cols_list: List[str],
        expected_output_cols_type: str = "",
    ) -> DataFrame:
        # Register vectorized UDF for batch inference
        batch_inference_udf_name = random_name_for_temp_object(TempObjectType.FUNCTION)
        snowpark_cols = dataset.select(input_cols).columns

        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
            api_calls=[pandas_udf],
            custom_tags=dict([("autogen", True)]),
        )

        @pandas_udf(  # type: ignore
            is_permanent=False,
            name=batch_inference_udf_name,
            packages=dependencies,  # type: ignore
            replace=True,
            session=session,
            statement_params=statement_params,
        )
        def vec_batch_infer(ds: PandasSeries[dict]) -> PandasSeries[dict]:  # type: ignore
            import numpy as np
            import pandas as pd

            input_df = pd.json_normalize(ds)

            # pd.json_normalize() doesn't remove quotes around quoted identifiers like snowpakr_df.to_pandas().
            # But trained models have unquoted input column names saved in internal state if trained using snowpark_df
            # or quoted input column names saved in internal state if trained using pandas_df.
            # Model expects exact same columns names in the input df for predict call.

            input_df = input_df[input_cols]  # Select input columns with quoted column names.
            if hasattr(estimator, "feature_names_in_"):
                missing_features = []
                for i, f in enumerate(getattr(estimator, "feature_names_in_", {})):
                    if i >= len(input_cols) or (input_cols[i] != f and snowpark_cols[i] != f):
                        missing_features.append(f)

                if len(missing_features) > 0:
                    raise ValueError(
                        "The feature names should match with those that were passed during fit.\n"
                        f"Features seen during fit call but not present in the input: {missing_features}\n"
                        f"Features in the input dataframe : {input_cols}\n"
                    )
                input_df.columns = getattr(estimator, "feature_names_in_", {})
            else:
                # Just rename the column names to unquoted identifiers.
                input_df.columns = snowpark_cols  # Replace the quoted columns identifier with unquoted column ids.
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
                # HeterogeneousEnsemble's transform method produce results with variying shapes
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

        batch_inference_table_name = f"SNOWML_BATCH_INFERENCE_INPUT_TABLE_{_get_rand_id()}"

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
                    input_cols_dict=", ".join([f"'{c}', {c}" for c in input_cols]),
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

    def score_snowpark(
        self,
        dataset: DataFrame,
        session: Session,
        estimator: object,
        dependencies: List[str],
        score_sproc_imports: List[str],
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> float:
        # Extract queries that generated the dataframe. We will need to pass it to score procedure.
        queries = dataset.queries["queries"]

        # Create a temp file and dump the score to that file.
        local_score_file_name = get_temp_file_path()
        with open(local_score_file_name, mode="w+b") as local_score_file:
            cp.dump(estimator, local_score_file)

        # Create temp stage to run score.
        score_stage_name = random_name_for_temp_object(TempObjectType.STAGE)
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
            subproject=self._subproject,
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
            packages=dependencies,  # type: ignore
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

            for import_name in score_sproc_imports:
                importlib.import_module(import_name)

            for query in sql_queries[:-1]:
                _ = session.sql(query).collect(statement_params=statement_params)
            sp_df = session.sql(sql_queries[-1])
            df = sp_df.to_pandas(statement_params=statement_params)
            df.columns = sp_df.columns

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
            subproject=self._subproject,
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
            input_cols,
            label_cols,
            sample_weight_col,
            statement_params,
        )

        cleanup_temp_files([local_score_file_name])

        return score

    def _fit_search_snowpark(
        self,
        param_list: Union[Dict[str, Any], List[Dict[str, Any]]],
        dataset: DataFrame,
        session: Session,
        estimator: Union[model_selection.GridSearchCV, model_selection.RandomizedSearchCV],
        dependencies: List[str],
        udf_imports: List[str],
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> Dict[str, Union[float, Dict[str, Any]]]:
        import cachetools

        from snowflake.ml.fileset.fileset import _cast_snowpark_dataframe

        # Create one stage for data and for estimators.
        temp_stage_name = random_name_for_temp_object(TempObjectType.STAGE)
        temp_stage_creation_query = f"CREATE OR REPLACE TEMP STAGE {temp_stage_name};"
        session.sql(temp_stage_creation_query).collect()

        # Stage data.
        # TODO: add index column to the staged data
        dataset = _cast_snowpark_dataframe(dataset)
        remote_file_path = f"{temp_stage_name}/{temp_stage_name}.parquet"
        dataset.write.copy_into_location(  # type:ignore[call-overload]
            remote_file_path, file_format_type="parquet", header=True, overwrite=True, single=True
        )

        imports = [f"@{row.name}" for row in session.sql(f"LIST @{temp_stage_name}").collect()]

        # Create a temp file and dump the transform to that file.
        local_transform_file_name = get_temp_file_path()
        with open(local_transform_file_name, mode="w+b") as local_transform_file:
            cp.dump(estimator, local_transform_file)
        put_result = session.file.put(
            local_transform_file_name,
            temp_stage_name,
            auto_compress=False,
            overwrite=True,
        )
        estimator_location = put_result[0].target
        imports.append(f"@{temp_stage_name}/{estimator_location}")

        # Create estimators with subset of param grid.
        # TODO: Decide how to choose parallelization factor.
        parallel_factor = 8

        assert estimator is not None
        params_to_evaluate = list(param_list)
        max_params_per_estimator = ceil(len(params_to_evaluate) / parallel_factor)
        param_chunks = [
            params_to_evaluate[x : x + max_params_per_estimator]
            for x in range(0, len(params_to_evaluate), max_params_per_estimator)
        ]
        target_params = []
        for param_chunk in param_chunks:
            param_chunk_dist: Any = defaultdict(set)
            for d in param_chunk:
                assert isinstance(d, dict)
                for k, v in d.items():
                    param_chunk_dist[k].add(v)
            for k, v in param_chunk_dist.items():
                param_chunk_dist[k] = list(v)

            target_params.append(param_chunk_dist)

        @cachetools.cached(cache={})
        def _load_data_into_udf() -> Dict[str, pd.DataFrame]:
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

            argspec = inspect.getfullargspec(estimator.fit)
            args = {"X": df[input_cols]}

            if label_cols:
                label_arg_name = "Y" if "Y" in argspec.args else "y"
                args[label_arg_name] = df[label_cols].squeeze()

            if sample_weight_col is not None and "sample_weight" in argspec.args:
                args["sample_weight"] = df[sample_weight_col].squeeze()

            return args

        # Register UDF for batch search
        batch_search_udf_name = random_name_for_temp_object(TempObjectType.FUNCTION)

        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[udf],
        )

        @udf(  # type: ignore
            is_permanent=False,
            name=batch_search_udf_name,
            packages=dependencies + ["pyarrow", "fastparquet"],  # type: ignore
            replace=True,
            session=session,
            statement_params=statement_params,
            imports=imports,  # type: ignore
        )
        def batch_search(param: dict) -> Dict[str, str]:  # type: ignore
            import io
            import json

            import numpy as np  # noqa: F401
            import pandas as pd  # noqa: F401

            for import_name in udf_imports:
                importlib.import_module(import_name)

            args = _load_data_into_udf()

            local_transform_file_path = os.path.join(
                sys._xoptions["snowflake_import_directory"], f"{estimator_location}"
            )
            results = {}
            with open(local_transform_file_path, mode="rb") as local_transform_file_obj:
                estimator = cp.load(local_transform_file_obj)
                estimator.param_grid = param
                estimator.fit(**args)
                results["BEST_SCORE"] = estimator.best_score_
                results["BEST_PARAMS"] = json.dumps(estimator.best_params_)
                # write cv_results into a cloudpickle file to serialize dict of numpy array and masked array
                with io.BytesIO() as f:
                    cp.dump(estimator.cv_results_, f)
                    f.seek(0)
                    results["CV_RESULTS"] = f.getvalue().hex()
            return results

        # Run Transform
        param_df = session.create_dataframe(
            pd.DataFrame({"PARAMS": target_params, "INDEX": [i for i in range(len(target_params))]})
        )
        results: DataFrame = param_df.with_columns(
            ["BEST_PARAMS", "BEST_SCORE", "CV_RESULTS"],
            [
                batch_search("PARAMS")["BEST_PARAMS"],  # type: ignore
                batch_search("PARAMS")["BEST_SCORE"],  # type: ignore
                batch_search("PARAMS")["CV_RESULTS"],  # type: ignore
            ],
        )
        # best param and best score needs to be sorted
        sorted_result = results.sort(col("BEST_SCORE").desc(), "INDEX")
        best_param = json.loads(json.loads(sorted_result.select("BEST_PARAMS").first()[0]))
        best_score = float(sorted_result.select("BEST_SCORE").first()[0])

        # cv_result maintains the original order
        cv_results_ = dict()
        for val in results.select("CV_RESULTS").collect():
            # retrieved string had one more double quote in the front and end of the string.
            # use [1:-1] to remove the extra double quotes
            hex_str = bytes.fromhex(val[0][1:-1])
            with io.BytesIO(hex_str) as f_reload:
                each_cv_result = cp.load(f_reload)
                for k, v in each_cv_result.items():
                    if k not in cv_results_:
                        cv_results_[k] = v
                    else:
                        cv_results_[k] = np.concatenate([cv_results_[k], v])

        cv_results_["rank_test_score"] = rankdata(-cv_results_["mean_test_score"], method="min")
        return {"best_param": best_param, "best_score": best_score, "cv_results": cv_results_}
