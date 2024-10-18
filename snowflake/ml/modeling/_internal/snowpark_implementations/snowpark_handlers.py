import importlib
import inspect
import os
import posixpath
import sys
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cloudpickle as cp
import pandas as pd

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import (
    identifier,
    pkg_version_utils,
    snowpark_dataframe_utils,
    temp_file_utils,
)
from snowflake.ml._internal.utils.query_result_checker import SqlResultValidator
from snowflake.ml.modeling._internal import estimator_utils
from snowflake.ml.modeling._internal.estimator_utils import (
    handle_inference_result,
    should_include_sample_weight,
)
from snowflake.snowpark import DataFrame, Session, functions as F, types as T
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    random_name_for_temp_object,
)

cp.register_pickle_by_value(inspect.getmodule(temp_file_utils.get_temp_file_path))
cp.register_pickle_by_value(inspect.getmodule(identifier.get_inferred_name))
cp.register_pickle_by_value(inspect.getmodule(handle_inference_result))
cp.register_pickle_by_value(inspect.getmodule(should_include_sample_weight))


_PROJECT = "ModelDevelopment"


def _get_rand_id() -> str:
    """
    Generate random id to be used in sproc and stage names.

    Returns:
        Random id string usable in sproc, table, and stage names.
    """
    return str(uuid4()).replace("-", "_").upper()


class SnowparkTransformHandlers:
    def __init__(
        self,
        dataset: DataFrame,
        estimator: object,
        class_name: str,
        subproject: str,
        autogenerated: Optional[bool] = False,
    ) -> None:
        """
        Args:
            dataset: The dataset to run transform functions on.
            estimator: The estimator used to run transforms.
            class_name: class name to be used in telemetry.
            subproject: subproject to be used in telemetry.
            autogenerated: Whether the class was autogenerated from a template.
        """
        self.dataset = dataset
        self.estimator = estimator
        self._class_name = class_name
        self._subproject = subproject
        self._autogenerated = autogenerated

    def batch_inference(
        self,
        inference_method: str,
        input_cols: List[str],
        expected_output_cols: List[str],
        session: Session,
        dependencies: List[str],
        drop_input_cols: Optional[bool] = False,
        expected_output_cols_type: Optional[str] = "",
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame:
        """Run batch inference on the given dataset.

        Args:
            session: An active Snowpark Session.
            dependencies: List of dependencies for the transformer.
            inference_method: the name of the method used by `estimator` to run inference.
            input_cols: List of feature columns for inference.
            expected_output_cols: column names (in order) of the output dataset.
            drop_input_cols: Boolean to determine whether to drop the input columns from the output dataset.
            expected_output_cols_type: Expected type of the output columns.
            args: additional positional arguments.
            kwargs: additional keyword args.

        Returns:
            A new dataset of the same type as the input dataset.
        """

        dependencies = self._get_validated_snowpark_dependencies(session, dependencies)
        dataset = self.dataset

        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
            api_calls=[F.pandas_udf],
            custom_tags={"autogen": True} if self._autogenerated else None,
        )

        temp_stage_name = estimator_utils.create_temp_stage(session)

        estimator_file_name = estimator_utils.upload_model_to_stage(
            stage_name=temp_stage_name,
            estimator=self.estimator,
            session=session,
            statement_params=statement_params,
        )
        imports = [f"@{temp_stage_name}/{estimator_file_name}"]

        # Register vectorized UDF for batch inference
        batch_inference_udf_name = random_name_for_temp_object(TempObjectType.FUNCTION)

        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(dataset)
        # Align the input_cols with snowpark dataframe's column name
        # This step also makes sure that the every col in input_cols exists in the current dataset
        snowpark_cols = dataset.select(input_cols).columns

        # Infer the datatype from input dataset's schema for batch inference
        # This is required before registering the UDTF
        fields = dataset.select(input_cols).schema.fields
        input_datatypes = []
        for field in fields:
            input_datatypes.append(field.datatype)

        # TODO(xjiang): for optimization, use register_from_file to reduce duplicate loading estimator object
        # or use cachetools here
        def load_estimator() -> object:
            estimator_file_path = os.path.join(sys._xoptions["snowflake_import_directory"], f"{estimator_file_name}")
            with open(estimator_file_path, mode="rb") as local_estimator_file_obj:
                estimator_object = cp.load(local_estimator_file_obj)
            return estimator_object

        @F.pandas_udf(  # type: ignore[arg-type, misc]
            is_permanent=False,
            name=batch_inference_udf_name,
            packages=dependencies,  # type: ignore[arg-type]
            replace=True,
            session=session,
            statement_params=statement_params,
            input_types=[T.PandasDataFrameType(input_datatypes)],
            imports=imports,  # type: ignore[arg-type]
        )
        def vec_batch_infer(input_df: pd.DataFrame) -> T.PandasSeries[dict]:  # type: ignore[type-arg]
            import numpy as np  # noqa: F401
            import pandas as pd

            input_df.columns = snowpark_cols

            estimator = load_estimator()

            if hasattr(estimator, "n_jobs"):
                # Vectorized UDF cannot handle joblib multiprocessing right now, deactivate the n_jobs
                estimator.n_jobs = 1
            inference_res = getattr(estimator, inference_method)(input_df, *args, **kwargs)

            transformed_numpy_array, _ = handle_inference_result(
                inference_res=inference_res,
                output_cols=expected_output_cols,
                inference_method=inference_method,
                within_udf=True,
            )

            if len(transformed_numpy_array.shape) > 1:
                if transformed_numpy_array.shape[1] != len(expected_output_cols):
                    series = pd.Series(transformed_numpy_array.tolist())
                    transformed_pandas_df = pd.DataFrame(series, columns=expected_output_cols)
                else:
                    transformed_pandas_df = pd.DataFrame(transformed_numpy_array.tolist(), columns=expected_output_cols)
            else:
                transformed_pandas_df = pd.DataFrame(transformed_numpy_array, columns=expected_output_cols)

            return transformed_pandas_df.to_dict("records")  # type: ignore[no-any-return]

        # Run Transform and get intermediate result
        INTERMEDIATE_OBJ_NAME = "tmp_result"
        # Use snowpark_cols can make sure the name ordering of the input dataframe
        # and only select those columns to put into vectorized udf
        output_obj = F.call_udf(batch_inference_udf_name, [F.col(col_name) for col_name in snowpark_cols])
        df_res: DataFrame = dataset.with_column(INTERMEDIATE_OBJ_NAME, output_obj)

        # Prepare the output
        output_cols = []
        output_col_names = []
        # When there is no expected_output_cols_type, default set it as StringType
        # snowpark cannot handle empty string, so this step give "string" value
        if expected_output_cols_type == "":
            expected_output_cols_type = "string"
        assert expected_output_cols_type is not None
        for output_feature in expected_output_cols:
            output_cols.append(F.col(INTERMEDIATE_OBJ_NAME)[output_feature].astype(expected_output_cols_type))
            output_col_names.append(identifier.get_inferred_name(output_feature))

        # Extract output from INTERMEDIATE_OBJ_NAME and drop that column
        df_res = df_res.with_columns(
            output_col_names,
            output_cols,
        ).drop(INTERMEDIATE_OBJ_NAME)

        if drop_input_cols:
            df_res = df_res.drop(*input_cols)

        return df_res

    def score(
        self,
        input_cols: List[str],
        label_cols: List[str],
        session: Session,
        dependencies: List[str],
        score_sproc_imports: List[str],
        sample_weight_col: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """Score the given test dataset.

        Args:
            session: An active Snowpark Session.
            dependencies: score function dependencies.
            score_sproc_imports: imports for score stored procedure.
            input_cols: List of feature columns for inference.
            label_cols: List of label columns for scoring.
            sample_weight_col: A column assigning relative weights to each row for scoring.
            args: additional positional arguments.
            kwargs: additional keyword args.

        Returns:
            An accuracy score for the model on the given test data.
        """
        dependencies = self._get_validated_snowpark_dependencies(session, dependencies)
        dependencies.append("snowflake-snowpark-python")
        dataset = self.dataset
        estimator = self.estimator
        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(dataset)

        # Extract queries that generated the dataframe. We will need to pass it to score procedure.
        queries = dataset.queries["queries"]

        # Create a temp file and dump the score to that file.
        local_score_file_name = temp_file_utils.get_temp_file_path()
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
            api_calls=[F.sproc],
            custom_tags={"autogen": True} if self._autogenerated else None,
        )
        # Put locally serialized score on stage.
        session.file.put(
            local_score_file_name,
            stage_score_file_name,
            auto_compress=False,
            overwrite=True,
            statement_params=statement_params,
        )

        @F.sproc(  # type: ignore[misc]
            is_permanent=False,
            name=score_sproc_name,
            packages=dependencies,  # type: ignore[arg-type]
            replace=True,
            session=session,
            statement_params=statement_params,
            anonymous=True,
            execute_as="caller",
        )
        def score_wrapper_sproc(
            session: Session,
            sql_queries: List[str],
            stage_score_file_name: str,
            input_cols: List[str],
            label_cols: List[str],
            sample_weight_col: Optional[str],
            score_statement_params: Dict[str, str],
        ) -> float:
            import inspect
            import os

            import cloudpickle as cp

            for import_name in score_sproc_imports:
                importlib.import_module(import_name)

            for query in sql_queries[:-1]:
                _ = session.sql(query).collect(statement_params=score_statement_params)
            sp_df = session.sql(sql_queries[-1])
            df: pd.DataFrame = sp_df.to_pandas(statement_params=score_statement_params)
            df.columns = sp_df.columns

            local_score_file_name = temp_file_utils.get_temp_file_path()
            session.file.get(stage_score_file_name, local_score_file_name, statement_params=score_statement_params)

            local_score_file_name_path = os.path.join(local_score_file_name, os.listdir(local_score_file_name)[0])
            with open(local_score_file_name_path, mode="r+b") as local_score_file_obj:
                estimator = cp.load(local_score_file_obj)

            params = inspect.signature(estimator.score).parameters
            if "X" in params:
                args = {"X": df[input_cols]}
            elif "X_test" in params:
                args = {"X_test": df[input_cols]}
            else:
                raise RuntimeError("Neither 'X' or 'X_test' exist in argument")

            if label_cols:
                label_arg_name = "Y" if "Y" in params else "y"
                args[label_arg_name] = df[label_cols].squeeze()

            # Sample weight is not included in search estimators parameters, check the underlying estimator.
            if sample_weight_col is not None and should_include_sample_weight(estimator, "score"):
                args["sample_weight"] = df[sample_weight_col].squeeze()

            result: float = estimator.score(**args)
            return result

        # Call score sproc
        score_statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[Session.call],
            custom_tags={"autogen": True} if self._autogenerated else None,
        )

        kwargs = telemetry.get_sproc_statement_params_kwargs(score_wrapper_sproc, score_statement_params)
        score: float = score_wrapper_sproc(
            session,
            queries,
            stage_score_file_name,
            input_cols,
            label_cols,
            sample_weight_col,
            score_statement_params,
            **kwargs,
        )

        temp_file_utils.cleanup_temp_files([local_score_file_name])

        return score

    def _get_validated_snowpark_dependencies(self, session: Session, dependencies: List[str]) -> List[str]:
        """A helper function to validate dependencies and return the available packages that exists
        in the snowflake anaconda channel

        Args:
            session: the active snowpark Session
            dependencies: unvalidated dependencies

        Returns:
            A list of packages present in the snoflake conda channel.
        """

        return pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=dependencies, session=session, subproject=self._subproject
        )
