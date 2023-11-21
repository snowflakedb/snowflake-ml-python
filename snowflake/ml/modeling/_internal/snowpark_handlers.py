import importlib
import inspect
import io
import os
import posixpath
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import cloudpickle as cp
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn import model_selection

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.env_utils import SNOWML_SPROC_ENV
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions,
    modeling_error_messages,
)
from snowflake.ml._internal.utils import identifier, snowpark_dataframe_utils
from snowflake.ml._internal.utils.query_result_checker import SqlResultValidator
from snowflake.ml._internal.utils.temp_file_utils import (
    cleanup_temp_files,
    get_temp_file_path,
)
from snowflake.snowpark import (
    DataFrame,
    Session,
    exceptions as snowpark_exceptions,
    functions as F,
)
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    random_name_for_temp_object,
)
from snowflake.snowpark.functions import col, pandas_udf, sproc, udtf
from snowflake.snowpark.stored_procedure import StoredProcedure
from snowflake.snowpark.types import (
    IntegerType,
    PandasSeries,
    StringType,
    StructField,
    StructType,
)

cp.register_pickle_by_value(inspect.getmodule(get_temp_file_path))
cp.register_pickle_by_value(inspect.getmodule(identifier.get_inferred_name))

_PROJECT = "ModelDevelopment"


class WrapperProvider:
    def __init__(self) -> None:
        self.imports: List[str] = []
        self.dependencies: List[str] = []

    def get_fit_wrapper_function(
        self,
    ) -> Callable[[Any, List[str], str, str, List[str], List[str], Optional[str], Dict[str, str]], str]:
        imports = self.imports  # In order for the sproc to not resolve this reference in snowflake.ml

        def fit_wrapper_function(
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

            import cloudpickle as cp
            import pandas as pd

            for import_name in imports:
                importlib.import_module(import_name)

            # Execute snowpark queries and obtain the results as pandas dataframe
            # NB: this implies that the result data must fit into memory.
            for query in sql_queries[:-1]:
                _ = session.sql(query).collect(statement_params=statement_params)
            sp_df = session.sql(sql_queries[-1])
            df: pd.DataFrame = sp_df.to_pandas(statement_params=statement_params)
            df.columns = sp_df.columns

            local_transform_file_name = get_temp_file_path()

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

            local_result_file_name = get_temp_file_path()

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

        return fit_wrapper_function


class SklearnWrapperProvider(WrapperProvider):
    def __init__(self) -> None:
        import sklearn

        self.imports: List[str] = ["sklearn"]

        # TODO(snandamuri): Replace cloudpickle with joblib after latest version of joblib is added to snowflake conda.
        self.dependencies: List[str] = [
            f"numpy=={np.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"cloudpickle=={cp.__version__}",
        ]


class XGBoostWrapperProvider(WrapperProvider):
    def __init__(self) -> None:
        import xgboost

        self.imports: List[str] = ["xgboost"]
        self.dependencies = [
            f"numpy=={np.__version__}",
            f"xgboost=={xgboost.__version__}",
            f"cloudpickle=={cp.__version__}",
        ]


class LightGBMWrapperProvider(WrapperProvider):
    def __init__(self) -> None:
        import lightgbm

        self.imports: List[str] = ["lightgbm"]
        self.dependencies = [
            f"numpy=={np.__version__}",
            f"lightgbm=={lightgbm.__version__}",
            f"cloudpickle=={cp.__version__}",
        ]


class SklearnModelSelectionWrapperProvider(WrapperProvider):
    def __init__(self) -> None:
        import xgboost

        self.imports: List[str] = ["sklearn", "xgboost"]
        self.dependencies = [
            f"numpy=={np.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"cloudpickle=={cp.__version__}",
            f"xgboost=={xgboost.__version__}",
        ]

        # Only include lightgbm in the dependencies if it is installed.
        try:
            import lightgbm
        except ModuleNotFoundError:
            pass
        else:
            self.imports.append("lightgbm")
            self.dependencies.append(f"lightgbm=={lightgbm.__version__}")


def _get_rand_id() -> str:
    """
    Generate random id to be used in sproc and stage names.

    Returns:
        Random id string usable in sproc, table, and stage names.
    """
    return str(uuid4()).replace("-", "_").upper()


class SnowparkHandlers:
    def __init__(
        self, class_name: str, subproject: str, wrapper_provider: WrapperProvider, autogenerated: Optional[bool] = False
    ) -> None:
        self._class_name = class_name
        self._subproject = subproject
        self._wrapper_provider = wrapper_provider
        self._autogenerated = autogenerated

    def _get_fit_wrapper_sproc(
        self, dependencies: List[str], session: Session, statement_params: Dict[str, str]
    ) -> StoredProcedure:
        # If the sproc already exists, don't register.
        if not hasattr(session, "_FIT_WRAPPER_SPROCS"):
            session._FIT_WRAPPER_SPROCS: Dict[str, StoredProcedure] = {}  # type: ignore[attr-defined, misc]

        fit_sproc_key = self._wrapper_provider.__class__.__name__
        if fit_sproc_key in session._FIT_WRAPPER_SPROCS:  # type: ignore[attr-defined]
            fit_sproc: StoredProcedure = session._FIT_WRAPPER_SPROCS[fit_sproc_key]  # type: ignore[attr-defined]
            return fit_sproc

        fit_sproc_name = random_name_for_temp_object(TempObjectType.PROCEDURE)

        fit_wrapper_sproc = session.sproc.register(
            func=self._wrapper_provider.get_fit_wrapper_function(),
            is_permanent=False,
            name=fit_sproc_name,
            packages=dependencies,  # type: ignore[arg-type]
            replace=True,
            session=session,
            statement_params=statement_params,
        )

        session._FIT_WRAPPER_SPROCS[fit_sproc_key] = fit_wrapper_sproc  # type: ignore[attr-defined]

        return fit_wrapper_sproc

    def fit_pandas(
        self,
        dataset: pd.DataFrame,
        estimator: object,
        input_cols: List[str],
        label_cols: Optional[List[str]],
        sample_weight_col: Optional[str],
    ) -> object:
        assert hasattr(estimator, "fit")  # Keep mypy happy
        argspec = inspect.getfullargspec(estimator.fit)
        args = {"X": dataset[input_cols]}

        if label_cols:
            label_arg_name = "Y" if "Y" in argspec.args else "y"
            args[label_arg_name] = dataset[label_cols].squeeze()

        if sample_weight_col is not None and "sample_weight" in argspec.args:
            args["sample_weight"] = dataset[sample_weight_col].squeeze()

        return estimator.fit(**args)

    def fit_snowpark(
        self,
        dataset: DataFrame,
        session: Session,
        estimator: object,
        dependencies: List[str],
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> Any:
        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(dataset)

        # If we are already in a stored procedure, no need to kick off another one.
        if SNOWML_SPROC_ENV in os.environ:
            statement_params = telemetry.get_function_usage_statement_params(
                project=_PROJECT,
                subproject=self._subproject,
                function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
                api_calls=[Session.call],
                custom_tags=dict([("autogen", True)]) if self._autogenerated else None,
            )
            pd_df: pd.DataFrame = dataset.to_pandas(statement_params=statement_params)
            pd_df.columns = dataset.columns
            return self.fit_pandas(pd_df, estimator, input_cols, label_cols, sample_weight_col)

        # Extract query that generated the dataframe. We will need to pass it to the fit procedure.
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
            custom_tags=dict([("autogen", True)]) if self._autogenerated else None,
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
            custom_tags=dict([("autogen", True)]) if self._autogenerated else None,
        )

        fit_wrapper_sproc = self._get_fit_wrapper_sproc(dependencies, session, statement_params)

        try:
            sproc_export_file_name: str = fit_wrapper_sproc(
                session,
                queries,
                stage_transform_file_name,
                stage_result_file_name,
                input_cols,
                label_cols,
                sample_weight_col,
                statement_params,
            )
        except snowpark_exceptions.SnowparkClientException as e:
            if "fit() missing 1 required positional argument: 'y'" in str(e):
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=RuntimeError(modeling_error_messages.ATTRIBUTE_NOT_SET.format("label_cols")),
                ) from e
            raise e

        if "|" in sproc_export_file_name:
            fields = sproc_export_file_name.strip().split("|")
            sproc_export_file_name = fields[0]

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
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame:
        # Register vectorized UDF for batch inference
        batch_inference_udf_name = random_name_for_temp_object(TempObjectType.FUNCTION)
        snowpark_cols = dataset.select(input_cols).columns
        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(dataset)

        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
            api_calls=[pandas_udf],
            custom_tags=dict([("autogen", True)]) if self._autogenerated else None,
        )

        @pandas_udf(  # type: ignore[arg-type, misc]
            is_permanent=False,
            name=batch_inference_udf_name,
            packages=dependencies,  # type: ignore[arg-type]
            replace=True,
            session=session,
            statement_params=statement_params,
        )
        def vec_batch_infer(ds: PandasSeries[dict]) -> PandasSeries[dict]:  # type: ignore[type-arg]
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
            inference_res = getattr(estimator, inference_method)(input_df, *args, **kwargs)
            if isinstance(inference_res, list) and len(inference_res) > 0 and isinstance(inference_res[0], np.ndarray):
                # In case of multioutput estimators, predict_proba, decision_function etc., functions return a list of
                # ndarrays. We need to concatenate them.
                transformed_numpy_array = np.concatenate(inference_res, axis=1)
            elif (
                isinstance(inference_res, tuple) and len(inference_res) > 0 and isinstance(inference_res[0], np.ndarray)
            ):
                # In case of kneighbors, functions return a tuple of ndarrays.
                transformed_numpy_array = np.stack(inference_res, axis=1)
            else:
                transformed_numpy_array = inference_res

            if (len(transformed_numpy_array.shape) == 3) and inference_method != "kneighbors":
                # VotingClassifier will return results of shape (n_classifiers, n_samples, n_classes)
                # when voting = "soft" and flatten_transform = False. We can't handle unflatten transforms,
                # so we ignore flatten_transform flag and flatten the results.
                transformed_numpy_array = np.hstack(transformed_numpy_array)  # type: ignore[call-overload]

            if len(transformed_numpy_array.shape) > 1:
                if transformed_numpy_array.shape[1] != len(expected_output_cols_list):
                    # HeterogeneousEnsemble's transform method produce results with variying shapes
                    # from (n_samples, n_estimators) to (n_samples, n_estimators * n_classes).
                    # It is hard to predict the response shape without using fragile introspection logic.
                    # So, to avoid that we are packing the results into a dataframe of shape (n_samples, 1) with
                    # each element being a list.
                    if len(expected_output_cols_list) != 1:
                        raise TypeError(
                            "expected_output_cols_list must be same length as transformed array or "
                            "should be of length 1"
                        )
                    series = pd.Series(transformed_numpy_array.tolist())
                    transformed_pandas_df = pd.DataFrame(series, columns=expected_output_cols_list)
                else:
                    transformed_pandas_df = pd.DataFrame(
                        transformed_numpy_array.tolist(), columns=expected_output_cols_list
                    )
            else:
                transformed_pandas_df = pd.DataFrame(transformed_numpy_array, columns=expected_output_cols_list)

            return transformed_pandas_df.to_dict("records")  # type: ignore[no-any-return]

        batch_inference_table_name = f"SNOWML_BATCH_INFERENCE_INPUT_TABLE_{_get_rand_id()}"

        # Run Transform
        query_from_df = str(dataset.queries["queries"][0])

        outer_select_list = pass_through_columns[:]
        inner_select_list = pass_through_columns[:]

        outer_select_list.extend(
            [
                "{object_name}:{column_name}{udf_datatype} as {column_name}".format(
                    object_name=batch_inference_udf_name,
                    column_name=identifier.get_inferred_name(c),
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

    def score_pandas(
        self,
        dataset: pd.DataFrame,
        estimator: object,
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> float:
        assert hasattr(estimator, "score")  # make type checker happy
        argspec = inspect.getfullargspec(estimator.score)
        if "X" in argspec.args:
            args = {"X": dataset[input_cols]}
        elif "X_test" in argspec.args:
            args = {"X_test": dataset[input_cols]}
        else:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=RuntimeError("Neither 'X' or 'X_test' exist in argument"),
            )

        if len(label_cols) > 0:
            label_arg_name = "Y" if "Y" in argspec.args else "y"
            args[label_arg_name] = dataset[label_cols].squeeze()

        if sample_weight_col is not None and "sample_weight" in argspec.args:
            args["sample_weight"] = dataset[sample_weight_col].squeeze()

        score = estimator.score(**args)
        assert isinstance(score, float)  # make type checker happy

        return score

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
        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(dataset)
        if SNOWML_SPROC_ENV in os.environ:
            statement_params = telemetry.get_function_usage_statement_params(
                project=_PROJECT,
                subproject=self._subproject,
                function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
                api_calls=[Session.call],
                custom_tags=dict([("autogen", True)]) if self._autogenerated else None,
            )
            pd_df: pd.DataFrame = dataset.to_pandas(statement_params=statement_params)
            pd_df.columns = dataset.columns
            return self.score_pandas(pd_df, estimator, input_cols, label_cols, sample_weight_col)

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
            custom_tags=dict([("autogen", True)]) if self._autogenerated else None,
        )
        # Put locally serialized score on stage.
        session.file.put(
            local_score_file_name,
            stage_score_file_name,
            auto_compress=False,
            overwrite=True,
            statement_params=statement_params,
        )

        @sproc(  # type: ignore[misc]
            is_permanent=False,
            name=score_sproc_name,
            packages=dependencies,  # type: ignore[arg-type]
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

            import cloudpickle as cp

            for import_name in score_sproc_imports:
                importlib.import_module(import_name)

            for query in sql_queries[:-1]:
                _ = session.sql(query).collect(statement_params=statement_params)
            sp_df = session.sql(sql_queries[-1])
            df: pd.DataFrame = sp_df.to_pandas(statement_params=statement_params)
            df.columns = sp_df.columns

            local_score_file_name = get_temp_file_path()
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
            custom_tags=dict([("autogen", True)]) if self._autogenerated else None,
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

    def fit_search_snowpark(
        self,
        param_grid: Union[model_selection.ParameterGrid, model_selection.ParameterSampler],
        dataset: DataFrame,
        session: Session,
        estimator: Union[model_selection.GridSearchCV, model_selection.RandomizedSearchCV],
        dependencies: List[str],
        udf_imports: List[str],
        input_cols: List[str],
        label_cols: List[str],
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

        # Stage data.
        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe(dataset)
        remote_file_path = f"{temp_stage_name}/{temp_stage_name}.parquet"
        dataset.write.copy_into_location(  # type:ignore[call-overload]
            remote_file_path, file_format_type="parquet", header=True, overwrite=True
        )
        imports = [f"@{row.name}" for row in session.sql(f"LIST @{temp_stage_name}").collect()]

        # Store GridSearchCV's refit variable. If user set it as False, we don't need to refit it again
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
            custom_tags=dict([("autogen", True)]) if self._autogenerated else None,
        )
        udtf_statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[udtf],
            custom_tags=dict([("autogen", True)]) if self._autogenerated else None,
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
            label_cols: List[str],
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
            df.columns = [identifier.get_inferred_name(col) for col in df.columns]

            X = df[input_cols]
            y = df[label_cols].squeeze()

            local_estimator_file_name = get_temp_file_path()
            session.file.get(stage_estimator_file_name, local_estimator_file_name)

            local_estimator_file_path = os.path.join(
                local_estimator_file_name, os.listdir(local_estimator_file_name)[0]
            )
            with open(local_estimator_file_path, mode="r+b") as local_estimator_file_obj:
                estimator = cp.load(local_estimator_file_obj)["estimator"]

            cv_orig = check_cv(estimator.cv, y, classifier=is_classifier(estimator.estimator))
            indices = [test for _, test in cv_orig.split(X, y)]
            local_indices_file_name = get_temp_file_path()
            with open(local_indices_file_name, mode="w+b") as local_indices_file_obj:
                cp.dump(indices, local_indices_file_obj)

            # Put locally serialized indices on stage.
            put_result = session.file.put(
                local_indices_file_name,
                temp_stage_name,
                auto_compress=False,
                overwrite=True,
            )
            indices_location = put_result[0].target
            imports.append(f"@{temp_stage_name}/{indices_location}")
            indices_len = len(indices)

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
                df.columns = [identifier.get_inferred_name(col) for col in df.columns]

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

                def process(self, params_idx: int, idx: int) -> Iterator[Tuple[str]]:
                    if hasattr(estimator, "param_grid"):
                        self.estimator.param_grid = self.params_to_evaluate[params_idx]
                    else:
                        self.estimator.param_distributions = self.params_to_evaluate[params_idx]
                    full_indices = np.array([i for i in range(self.data_length)])
                    test_indice = self.indices[idx]
                    train_indice = np.setdiff1d(full_indices, test_indice)
                    self.estimator.cv = [(train_indice, test_indice)]
                    self.estimator.fit(**self.args)
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

            idx_length = int(indices_len)
            params_length = len(param_grid)
            idxs = [i for i in range(idx_length)]
            param_indices, training_indices = [], []
            for param_idx, cv_idx in product([param_index for param_index in range(params_length)], idxs):
                param_indices.append(param_idx)
                training_indices.append(cv_idx)

            pd_df = pd.DataFrame(
                {
                    "PARAMS": param_indices,
                    "TRAIN_IND": training_indices,
                    "PARAM_INDEX": [i for i in range(idx_length * params_length)],
                }
            )
            df = session.create_dataframe(pd_df)
            results = df.select(
                F.cast(df["PARAM_INDEX"], IntegerType()).as_("PARAM_INDEX"),
                (HP_TUNING(df["PARAMS"], df["TRAIN_IND"]).over(partition_by=df["PARAM_INDEX"])),
            )

            # cv_result maintains the original order
            multimetric = False
            cv_results_ = dict()
            scorers = set()
            for i, val in enumerate(results.select("CV_RESULTS").sort(col("PARAM_INDEX")).collect()):
                # retrieved string had one more double quote in the front and end of the string.
                # use [1:-1] to remove the extra double quotes
                hex_str = bytes.fromhex(val[0])
                with io.BytesIO(hex_str) as f_reload:
                    each_cv_result = cp.load(f_reload)
                    for k, v in each_cv_result.items():
                        cur_cv = i % idx_length
                        key = k
                        if "split0_test_" in k:
                            # For multi-metric evaluation, the scores for all the scorers are available in the
                            # cv_results_ dict at the keys ending with that scorer’s name ('_<scorer_name>')
                            # instead of '_score'.
                            scorers.add(k[len("split0_test_") :])
                            key = k.replace("split0_test", f"split{cur_cv}_test")
                        elif k.startswith("param"):
                            if cur_cv != 0:
                                key = False
                        if key:
                            if key not in cv_results_:
                                cv_results_[key] = v
                            else:
                                cv_results_[key] = np.concatenate([cv_results_[key], v])

            multimetric = len(scorers) > 1
            # Use numpy to re-calculate all the information in cv_results_ again
            # Generally speaking, reshape all the results into the (scorers+2, idx_length, params_length) shape,
            # and average them by the idx_length;
            # idx_length is the number of cv folds; params_length is the number of parameter combinations
            scores = [
                np.reshape(
                    np.concatenate([cv_results_[f"split{cur_cv}_test_{score}"] for cur_cv in range(idx_length)]),
                    (idx_length, -1),
                )
                for score in scorers
            ]

            fit_score_test_matrix = np.stack(
                [
                    np.reshape(cv_results_["mean_fit_time"], (idx_length, -1)),
                    np.reshape(cv_results_["mean_score_time"], (idx_length, -1)),
                ]
                + scores
            )

            mean_fit_score_test_matrix = np.mean(fit_score_test_matrix, axis=1)
            std_fit_score_test_matrix = np.std(fit_score_test_matrix, axis=1)
            cv_results_["std_fit_time"] = std_fit_score_test_matrix[0]
            cv_results_["mean_fit_time"] = mean_fit_score_test_matrix[0]
            cv_results_["std_score_time"] = std_fit_score_test_matrix[1]
            cv_results_["mean_score_time"] = mean_fit_score_test_matrix[1]
            for idx, score in enumerate(scorers):
                cv_results_[f"std_test_{score}"] = std_fit_score_test_matrix[idx + 2]
                cv_results_[f"mean_test_{score}"] = mean_fit_score_test_matrix[idx + 2]
                # re-compute the ranking again with mean_test_<score>.
                cv_results_[f"rank_test_{score}"] = rankdata(-cv_results_[f"mean_test_{score}"], method="min")
                # The best param is the highest ranking (which is 1) and we choose the first time ranking 1 appeared.
                # If all scores are `nan`, `rankdata` will also produce an array of `nan` values.
                # In that case, default to first index.
                best_param_index = (
                    np.where(cv_results_[f"rank_test_{score}"] == 1)[0][0]
                    if not np.isnan(cv_results_[f"rank_test_{score}"]).all()
                    else 0
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
                estimator.best_params_ = cv_results_["params"][best_param_index]

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
