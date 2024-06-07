import importlib
import inspect
import os
import posixpath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle as cp
import pandas as pd

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions,
    modeling_error_messages,
)
from snowflake.ml._internal.utils import (
    identifier,
    pkg_version_utils,
    snowpark_dataframe_utils,
    temp_file_utils,
)
from snowflake.ml.modeling._internal import estimator_utils
from snowflake.ml.modeling._internal.estimator_utils import handle_inference_result
from snowflake.ml.modeling._internal.model_specifications import (
    ModelSpecifications,
    ModelSpecificationsBuilder,
)
from snowflake.snowpark import DataFrame, Session, exceptions as snowpark_exceptions
from snowflake.snowpark._internal import utils as snowpark_utils
from snowflake.snowpark.stored_procedure import StoredProcedure

cp.register_pickle_by_value(inspect.getmodule(temp_file_utils.get_temp_file_path))
cp.register_pickle_by_value(inspect.getmodule(identifier.get_inferred_name))
cp.register_pickle_by_value(inspect.getmodule(handle_inference_result))

_PROJECT = "ModelDevelopment"
_ENABLE_ANONYMOUS_SPROC = False


class SnowparkModelTrainer:
    """
    A class for training models on Snowflake data using the Sproc.

    TODO (snandamuri): Introduce the concept of executor that would take the training function
    and execute it on the target environments like, local, Snowflake warehouse, or SPCS, etc.
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
        Initializes the SnowparkModelTrainer with a model, a Snowpark DataFrame, feature, and label column names.

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
        self.estimator = estimator
        self.dataset = dataset
        self.session = session
        self.input_cols = input_cols
        self.label_cols = label_cols
        self.sample_weight_col = sample_weight_col
        self._autogenerated = autogenerated
        self._subproject = subproject
        self._class_name = estimator.__class__.__name__

    def _fetch_model_from_stage(self, dir_path: str, file_name: str, statement_params: Dict[str, str]) -> object:
        """
        Downloads the serialized model from a stage location and unpickles it.

        Args:
            dir_path: Stage directory path where results are stored.
            file_name: File name with in the directory where results are stored.
            statement_params: Statement params to be attached to the SQL queries issue form this method.

        Returns:
            Deserialized model object.
        """
        local_result_file_name = temp_file_utils.get_temp_file_path()
        self.session.file.get(
            posixpath.join(dir_path, file_name),
            local_result_file_name,
            statement_params=statement_params,
        )

        with open(os.path.join(local_result_file_name, file_name), mode="r+b") as result_file_obj:
            fit_estimator = cp.load(result_file_obj)

        temp_file_utils.cleanup_temp_files([local_result_file_name])
        return fit_estimator

    def _build_fit_wrapper_sproc(
        self,
        model_spec: ModelSpecifications,
    ) -> Callable[[Any, List[str], str, List[str], List[str], Optional[str], Dict[str, str]], str]:
        """
        Constructs and returns a python stored procedure function to be used for training model.

        Args:
            model_spec: ModelSpecifications object that contains model specific information
                like required imports, package dependencies, etc.

        Returns:
            A callable that can be registered as a stored procedure.
        """
        imports = model_spec.imports  # In order for the sproc to not resolve this reference in snowflake.ml

        def fit_wrapper_function(
            session: Session,
            sql_queries: List[str],
            temp_stage_name: str,
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

            local_transform_file_name = temp_file_utils.get_temp_file_path()

            session.file.get(
                stage_location=temp_stage_name,
                target_directory=local_transform_file_name,
                statement_params=statement_params,
            )

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

            local_result_file_name = temp_file_utils.get_temp_file_path()

            with open(local_result_file_name, mode="w+b") as local_result_file_obj:
                cp.dump(estimator, local_result_file_obj)

            session.file.put(
                local_file_name=local_result_file_name,
                stage_location=temp_stage_name,
                auto_compress=False,
                overwrite=True,
                statement_params=statement_params,
            )

            # Note: you can add something like  + "|" + str(df) to the return string
            # to pass debug information to the caller.
            return str(os.path.basename(local_result_file_name))

        return fit_wrapper_function

    def _get_fit_wrapper_sproc_anonymous(self, statement_params: Dict[str, str]) -> StoredProcedure:
        model_spec = ModelSpecificationsBuilder.build(model=self.estimator)
        fit_sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)

        relaxed_dependencies = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=model_spec.pkgDependencies, session=self.session
        )

        fit_wrapper_sproc = self.session.sproc.register(
            func=self._build_fit_wrapper_sproc(model_spec=model_spec),
            is_permanent=False,
            name=fit_sproc_name,
            packages=["snowflake-snowpark-python"] + relaxed_dependencies,  # type: ignore[arg-type]
            replace=True,
            session=self.session,
            statement_params=statement_params,
            anonymous=True,
        )

        return fit_wrapper_sproc

    def _get_fit_wrapper_sproc(self, statement_params: Dict[str, str]) -> StoredProcedure:
        # If the sproc already exists, don't register.
        if not hasattr(self.session, "_FIT_WRAPPER_SPROCS"):
            self.session._FIT_WRAPPER_SPROCS: Dict[str, StoredProcedure] = {}  # type: ignore[attr-defined, misc]

        model_spec = ModelSpecificationsBuilder.build(model=self.estimator)
        fit_sproc_key = model_spec.__class__.__name__
        if fit_sproc_key in self.session._FIT_WRAPPER_SPROCS:  # type: ignore[attr-defined]
            fit_sproc: StoredProcedure = self.session._FIT_WRAPPER_SPROCS[fit_sproc_key]  # type: ignore[attr-defined]
            return fit_sproc

        fit_sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)

        relaxed_dependencies = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=model_spec.pkgDependencies, session=self.session
        )

        fit_wrapper_sproc = self.session.sproc.register(
            func=self._build_fit_wrapper_sproc(model_spec=model_spec),
            is_permanent=False,
            name=fit_sproc_name,
            packages=["snowflake-snowpark-python"] + relaxed_dependencies,  # type: ignore[arg-type]
            replace=True,
            session=self.session,
            statement_params=statement_params,
        )

        self.session._FIT_WRAPPER_SPROCS[fit_sproc_key] = fit_wrapper_sproc  # type: ignore[attr-defined]

        return fit_wrapper_sproc

    def _build_fit_predict_wrapper_sproc(
        self,
        model_spec: ModelSpecifications,
    ) -> Callable[[Session, List[str], str, List[str], Dict[str, str], bool, List[str], str], str]:
        """
        Constructs and returns a python stored procedure function to be used for training model.

        Args:
            model_spec: ModelSpecifications object that contains model specific information
                like required imports, package dependencies, etc.

        Returns:
            A callable that can be registered as a stored procedure.
        """
        imports = model_spec.imports  # In order for the sproc to not resolve this reference in snowflake.ml

        def fit_predict_wrapper_function(
            session: Session,
            sql_queries: List[str],
            temp_stage_name: str,
            input_cols: List[str],
            statement_params: Dict[str, str],
            drop_input_cols: bool,
            expected_output_cols_list: List[str],
            fit_predict_result_name: str,
        ) -> str:
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

            local_transform_file_name = temp_file_utils.get_temp_file_path()

            session.file.get(
                stage_location=temp_stage_name,
                target_directory=local_transform_file_name,
                statement_params=statement_params,
            )

            local_transform_file_path = os.path.join(
                local_transform_file_name, os.listdir(local_transform_file_name)[0]
            )
            with open(local_transform_file_path, mode="r+b") as local_transform_file_obj:
                estimator = cp.load(local_transform_file_obj)

            fit_predict_result = estimator.fit_predict(X=df[input_cols])

            local_result_file_name = temp_file_utils.get_temp_file_path()

            with open(local_result_file_name, mode="w+b") as local_result_file_obj:
                cp.dump(estimator, local_result_file_obj)

            session.file.put(
                local_file_name=local_result_file_name,
                stage_location=temp_stage_name,
                auto_compress=False,
                overwrite=True,
                statement_params=statement_params,
            )

            # store the predict output
            if drop_input_cols:
                fit_predict_result_pd = pd.DataFrame(data=fit_predict_result, columns=expected_output_cols_list)
            else:
                df = df.copy()
                # in case the output column name overlap with the input column names,
                # remove the ones in input column names
                remove_dataset_col_name_exist_in_output_col = list(set(df.columns) - set(expected_output_cols_list))
                fit_predict_result_pd = pd.concat(
                    [
                        df[remove_dataset_col_name_exist_in_output_col],
                        pd.DataFrame(data=fit_predict_result, columns=expected_output_cols_list),
                    ],
                    axis=1,
                )

            # write into a temp table in sproc and load the table from outside
            session.write_pandas(
                fit_predict_result_pd, fit_predict_result_name, auto_create_table=True, table_type="temp"
            )

            # Note: you can add something like  + "|" + str(df) to the return string
            # to pass debug information to the caller.
            return str(os.path.basename(local_result_file_name))

        return fit_predict_wrapper_function

    def _build_fit_transform_wrapper_sproc(
        self,
        model_spec: ModelSpecifications,
    ) -> Callable[
        [
            Session,
            List[str],
            str,
            List[str],
            Optional[List[str]],
            Optional[str],
            Dict[str, str],
            bool,
            List[str],
            str,
        ],
        str,
    ]:
        """
        Constructs and returns a python stored procedure function to be used for training model.

        Args:
            model_spec: ModelSpecifications object that contains model specific information
                like required imports, package dependencies, etc.

        Returns:
            A callable that can be registered as a stored procedure.
        """
        imports = model_spec.imports  # In order for the sproc to not resolve this reference in snowflake.ml

        def fit_transform_wrapper_function(
            session: Session,
            sql_queries: List[str],
            temp_stage_name: str,
            input_cols: List[str],
            label_cols: Optional[List[str]],
            sample_weight_col: Optional[str],
            statement_params: Dict[str, str],
            drop_input_cols: bool,
            expected_output_cols_list: List[str],
            fit_transform_result_name: str,
        ) -> str:
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

            local_transform_file_name = temp_file_utils.get_temp_file_path()

            session.file.get(
                stage_location=temp_stage_name,
                target_directory=local_transform_file_name,
                statement_params=statement_params,
            )

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

            fit_transform_result = estimator.fit_transform(**args)

            local_result_file_name = temp_file_utils.get_temp_file_path()

            with open(local_result_file_name, mode="w+b") as local_result_file_obj:
                cp.dump(estimator, local_result_file_obj)

            session.file.put(
                local_file_name=local_result_file_name,
                stage_location=temp_stage_name,
                auto_compress=False,
                overwrite=True,
                statement_params=statement_params,
            )

            transformed_numpy_array, output_cols = handle_inference_result(
                inference_res=fit_transform_result,
                output_cols=expected_output_cols_list,
                inference_method="fit_transform",
                within_udf=True,
            )

            if len(transformed_numpy_array.shape) > 1:
                if transformed_numpy_array.shape[1] != len(output_cols):
                    series = pd.Series(transformed_numpy_array.tolist())
                    transformed_pandas_df = pd.DataFrame(series, columns=output_cols)
                else:
                    transformed_pandas_df = pd.DataFrame(transformed_numpy_array.tolist(), columns=output_cols)
            else:
                transformed_pandas_df = pd.DataFrame(transformed_numpy_array, columns=output_cols)

            # store the transform output
            if not drop_input_cols:
                df = df.copy()
                # in case the output column name overlap with the input column names,
                # remove the ones in input column names
                remove_dataset_col_name_exist_in_output_col = list(set(df.columns) - set(output_cols))
                transformed_pandas_df = pd.concat(
                    [df[remove_dataset_col_name_exist_in_output_col], transformed_pandas_df], axis=1
                )

            # write into a temp table in sproc and load the table from outside
            session.write_pandas(
                transformed_pandas_df,
                fit_transform_result_name,
                auto_create_table=True,
                table_type="temp",
                quote_identifiers=False,
            )

            return str(os.path.basename(local_result_file_name))

        return fit_transform_wrapper_function

    def _get_fit_predict_wrapper_sproc_anonymous(self, statement_params: Dict[str, str]) -> StoredProcedure:
        model_spec = ModelSpecificationsBuilder.build(model=self.estimator)

        fit_predict_sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)

        relaxed_dependencies = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=model_spec.pkgDependencies, session=self.session
        )

        fit_predict_wrapper_sproc = self.session.sproc.register(
            func=self._build_fit_predict_wrapper_sproc(model_spec=model_spec),
            is_permanent=False,
            name=fit_predict_sproc_name,
            packages=["snowflake-snowpark-python"] + relaxed_dependencies,  # type: ignore[arg-type]
            replace=True,
            session=self.session,
            statement_params=statement_params,
            anonymous=True,
        )

        return fit_predict_wrapper_sproc

    def _get_fit_predict_wrapper_sproc(self, statement_params: Dict[str, str]) -> StoredProcedure:
        # If the sproc already exists, don't register.
        if not hasattr(self.session, "_FIT_WRAPPER_SPROCS"):
            self.session._FIT_WRAPPER_SPROCS: Dict[str, StoredProcedure] = {}  # type: ignore[attr-defined, misc]

        model_spec = ModelSpecificationsBuilder.build(model=self.estimator)
        fit_predict_sproc_key = model_spec.__class__.__name__ + "_fit_predict"
        if fit_predict_sproc_key in self.session._FIT_WRAPPER_SPROCS:  # type: ignore[attr-defined]
            fit_sproc: StoredProcedure = self.session._FIT_WRAPPER_SPROCS[  # type: ignore[attr-defined]
                fit_predict_sproc_key
            ]
            return fit_sproc

        fit_predict_sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)

        relaxed_dependencies = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=model_spec.pkgDependencies, session=self.session
        )

        fit_predict_wrapper_sproc = self.session.sproc.register(
            func=self._build_fit_predict_wrapper_sproc(model_spec=model_spec),
            is_permanent=False,
            name=fit_predict_sproc_name,
            packages=["snowflake-snowpark-python"] + relaxed_dependencies,  # type: ignore[arg-type]
            replace=True,
            session=self.session,
            statement_params=statement_params,
        )

        self.session._FIT_WRAPPER_SPROCS[  # type: ignore[attr-defined]
            fit_predict_sproc_key
        ] = fit_predict_wrapper_sproc

        return fit_predict_wrapper_sproc

    def _get_fit_transform_wrapper_sproc_anonymous(self, statement_params: Dict[str, str]) -> StoredProcedure:
        model_spec = ModelSpecificationsBuilder.build(model=self.estimator)

        fit_transform_sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)

        relaxed_dependencies = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=model_spec.pkgDependencies, session=self.session
        )

        fit_transform_wrapper_sproc = self.session.sproc.register(
            func=self._build_fit_transform_wrapper_sproc(model_spec=model_spec),
            is_permanent=False,
            name=fit_transform_sproc_name,
            packages=["snowflake-snowpark-python"] + relaxed_dependencies,  # type: ignore[arg-type]
            replace=True,
            session=self.session,
            statement_params=statement_params,
            anonymous=True,
        )
        return fit_transform_wrapper_sproc

    def _get_fit_transform_wrapper_sproc(self, statement_params: Dict[str, str]) -> StoredProcedure:
        # If the sproc already exists, don't register.
        if not hasattr(self.session, "_FIT_WRAPPER_SPROCS"):
            self.session._FIT_WRAPPER_SPROCS: Dict[str, StoredProcedure] = {}  # type: ignore[attr-defined, misc]

        model_spec = ModelSpecificationsBuilder.build(model=self.estimator)
        fit_transform_sproc_key = model_spec.__class__.__name__ + "_fit_transform"
        if fit_transform_sproc_key in self.session._FIT_WRAPPER_SPROCS:  # type: ignore[attr-defined]
            fit_sproc: StoredProcedure = self.session._FIT_WRAPPER_SPROCS[  # type: ignore[attr-defined]
                fit_transform_sproc_key
            ]
            return fit_sproc

        fit_transform_sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)

        relaxed_dependencies = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=model_spec.pkgDependencies, session=self.session
        )

        fit_transform_wrapper_sproc = self.session.sproc.register(
            func=self._build_fit_transform_wrapper_sproc(model_spec=model_spec),
            is_permanent=False,
            name=fit_transform_sproc_name,
            packages=["snowflake-snowpark-python"] + relaxed_dependencies,  # type: ignore[arg-type]
            replace=True,
            session=self.session,
            statement_params=statement_params,
        )

        self.session._FIT_WRAPPER_SPROCS[  # type: ignore[attr-defined]
            fit_transform_sproc_key
        ] = fit_transform_wrapper_sproc

        return fit_transform_wrapper_sproc

    def train(self) -> object:
        """
        Trains the model by pushing down the compute into Snowflake using stored procedures.

        Returns:
            Trained model

        Raises:
            e: Raises an exception if any of Snowflake operations fail because of any reason.
            SnowflakeMLException: Know exception are caught and rethrow with more detailed error message.
        """
        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(self.dataset)

        # TODO(snandamuri) : Handle the already in a stored procedure case in the in builder.

        # Extract query that generated the dataframe. We will need to pass it to the fit procedure.
        queries = dataset.queries["queries"]

        temp_stage_name = estimator_utils.create_temp_stage(self.session)
        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
            api_calls=[Session.call],
            custom_tags={"autogen": True} if self._autogenerated else None,
        )
        estimator_utils.upload_model_to_stage(
            stage_name=temp_stage_name,
            estimator=self.estimator,
            session=self.session,
            statement_params=statement_params,
        )
        # Call fit sproc

        if _ENABLE_ANONYMOUS_SPROC:
            fit_wrapper_sproc = self._get_fit_wrapper_sproc_anonymous(statement_params=statement_params)
        else:
            fit_wrapper_sproc = self._get_fit_wrapper_sproc(statement_params=statement_params)

        try:
            sproc_export_file_name: str = fit_wrapper_sproc(
                self.session,
                queries,
                temp_stage_name,
                self.input_cols,
                self.label_cols,
                self.sample_weight_col,
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

        return self._fetch_model_from_stage(
            dir_path=temp_stage_name,
            file_name=sproc_export_file_name,
            statement_params=statement_params,
        )

    def train_fit_predict(
        self,
        expected_output_cols_list: List[str],
        drop_input_cols: Optional[bool] = False,
    ) -> Tuple[Union[DataFrame, pd.DataFrame], object]:
        """Trains the model by pushing down the compute into Snowflake using stored procedures.
        This API is different from fit itself because it would also provide the predict
        output.

        Args:
            expected_output_cols_list (List[str]): The output columns
                name as a list. Defaults to None.
            drop_input_cols (Optional[bool]): Boolean to determine drop
                the input columns from the output dataset or not

        Returns:
            Tuple[Union[DataFrame, pd.DataFrame], object]: [predicted dataset, estimator]
        """
        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(self.dataset)

        # Extract query that generated the dataframe. We will need to pass it to the fit procedure.
        queries = dataset.queries["queries"]

        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
            api_calls=[Session.call],
            custom_tags={"autogen": True} if self._autogenerated else None,
        )

        temp_stage_name = estimator_utils.create_temp_stage(self.session)
        estimator_utils.upload_model_to_stage(
            stage_name=temp_stage_name,
            estimator=self.estimator,
            session=self.session,
            statement_params=statement_params,
        )

        # Call fit sproc
        if _ENABLE_ANONYMOUS_SPROC:
            fit_predict_wrapper_sproc = self._get_fit_predict_wrapper_sproc_anonymous(statement_params=statement_params)
        else:
            fit_predict_wrapper_sproc = self._get_fit_predict_wrapper_sproc(statement_params=statement_params)

        fit_predict_result_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE)

        sproc_export_file_name: str = fit_predict_wrapper_sproc(
            self.session,
            queries,
            temp_stage_name,
            self.input_cols,
            statement_params,
            drop_input_cols,
            expected_output_cols_list,
            fit_predict_result_name,
        )

        output_result_sp = self.session.table(fit_predict_result_name)
        fitted_estimator = self._fetch_model_from_stage(
            dir_path=temp_stage_name,
            file_name=sproc_export_file_name,
            statement_params=statement_params,
        )

        return output_result_sp, fitted_estimator

    def train_fit_transform(
        self,
        expected_output_cols_list: List[str],
        drop_input_cols: Optional[bool] = False,
    ) -> Tuple[Union[DataFrame, pd.DataFrame], object]:
        """Trains the model by pushing down the compute into Snowflake using stored procedures.
        This API is different from fit itself because it would also provide the transform
        output.

        Args:
            expected_output_cols_list (List[str]): The output columns
                name as a list. Defaults to None.
            drop_input_cols (Optional[bool]): Boolean to determine whether to
                drop the input columns from the output dataset.

        Returns:
            Tuple[Union[DataFrame, pd.DataFrame], object]: [transformed dataset, estimator]
        """
        dataset = snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(self.dataset)

        # Extract query that generated the dataframe. We will need to pass it to the fit procedure.
        queries = dataset.queries["queries"]

        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
            api_calls=[Session.call],
            custom_tags={"autogen": True} if self._autogenerated else None,
        )

        temp_stage_name = estimator_utils.create_temp_stage(self.session)
        estimator_utils.upload_model_to_stage(
            stage_name=temp_stage_name,
            estimator=self.estimator,
            session=self.session,
            statement_params=statement_params,
        )

        # Call fit sproc
        if _ENABLE_ANONYMOUS_SPROC:
            fit_transform_wrapper_sproc = self._get_fit_transform_wrapper_sproc_anonymous(
                statement_params=statement_params
            )
        else:
            fit_transform_wrapper_sproc = self._get_fit_transform_wrapper_sproc(statement_params=statement_params)

        fit_transform_result_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE)

        sproc_export_file_name: str = fit_transform_wrapper_sproc(
            self.session,
            queries,
            temp_stage_name,
            self.input_cols,
            self.label_cols,
            self.sample_weight_col,
            statement_params,
            drop_input_cols,
            expected_output_cols_list,
            fit_transform_result_name,
        )

        output_result_sp = self.session.table(fit_transform_result_name)
        fitted_estimator = self._fetch_model_from_stage(
            dir_path=temp_stage_name,
            file_name=sproc_export_file_name,
            statement_params=statement_params,
        )

        return output_result_sp, fitted_estimator
