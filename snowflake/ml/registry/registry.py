import warnings
from types import ModuleType
from typing import Any, Dict, List, Optional, Union, overload

import pandas as pd

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import (
    Model,
    ModelVersion,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.monitoring import model_monitor, model_monitor_version
from snowflake.ml.monitoring._manager import model_monitor_manager
from snowflake.ml.monitoring.entities import model_monitor_config
from snowflake.ml.registry._manager import model_manager
from snowflake.snowpark import session

_TELEMETRY_PROJECT = "MLOps"
_MODEL_TELEMETRY_SUBPROJECT = "ModelManagement"

_MODEL_MONITORING_UNIMPLEMENTED_ERROR = "Model Monitoring is not implemented in python yet."
_MODEL_MONITORING_DISABLED_ERROR = (
    """Must enable monitoring to use this method. Please set `options={"enable_monitoring": True}` in the Registry"""
)


class Registry:
    def __init__(
        self,
        session: session.Session,
        *,
        database_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Opens a registry within a pre-created Snowflake schema.

        Args:
            session: The Snowpark Session to connect with Snowflake.
            database_name: The name of the database. If None, the current database of the session
                will be used. Defaults to None.
            schema_name: The name of the schema. If None, the current schema of the session
                will be used. If there is no active schema, the PUBLIC schema will be used. Defaults to None.
            options: Optional set of configurations to modify registry.
                Registry Options include:
                - enable_monitoring: Feature flag to indicate whether registry can be used for monitoring.

        Raises:
            ValueError: When there is no specified or active database in the session.
        """
        if database_name:
            self._database_name = sql_identifier.SqlIdentifier(database_name)
        else:
            session_db = session.get_current_database()
            if session_db:
                self._database_name = sql_identifier.SqlIdentifier(session_db)
            else:
                raise ValueError("You need to provide a database to use registry.")

        if schema_name:
            self._schema_name = sql_identifier.SqlIdentifier(schema_name)
        elif database_name:
            self._schema_name = sql_identifier.SqlIdentifier("PUBLIC")
        else:
            session_schema = session.get_current_schema()
            self._schema_name = (
                sql_identifier.SqlIdentifier(session_schema)
                if session_schema
                else sql_identifier.SqlIdentifier("PUBLIC")
            )

        self._model_manager = model_manager.ModelManager(
            session,
            database_name=self._database_name,
            schema_name=self._schema_name,
        )

        self.enable_monitoring = options.get("enable_monitoring", True) if options else True
        if self.enable_monitoring:
            monitor_statement_params = telemetry.get_statement_params(
                project=telemetry.TelemetryProject.MLOPS.value,
                subproject=telemetry.TelemetrySubProject.MONITORING.value,
            )

            self._model_monitor_manager = model_monitor_manager.ModelMonitorManager(
                session=session,
                database_name=self._database_name,
                schema_name=self._schema_name,
                statement_params=monitor_statement_params,
            )

    @property
    def location(self) -> str:
        """Get the location (database.schema) of the registry."""
        return ".".join([self._database_name.identifier(), self._schema_name.identifier()])

    @overload
    def log_model(
        self,
        model: model_types.SupportedModelType,
        *,
        model_name: str,
        version_name: Optional[str] = None,
        comment: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        artifact_repository_map: Optional[Dict[str, str]] = None,
        target_platforms: Optional[List[model_types.SupportedTargetPlatformType]] = None,
        python_version: Optional[str] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        user_files: Optional[Dict[str, List[str]]] = None,
        code_paths: Optional[List[str]] = None,
        ext_modules: Optional[List[ModuleType]] = None,
        task: model_types.Task = model_types.Task.UNKNOWN,
        options: Optional[model_types.ModelSaveOption] = None,
    ) -> ModelVersion:
        """
        Log a model with various parameters and metadata, or a ModelVersion object.

        Args:
            model: Supported model or ModelVersion object.
                - Supported model: Model object of supported types such as Scikit-learn, XGBoost, LightGBM, Snowpark ML,
                PyTorch, TorchScript, Tensorflow, Tensorflow Keras, MLFlow, HuggingFace Pipeline, Sentence Transformers,
                or Custom Model.
                - ModelVersion: Source ModelVersion object used to create the new ModelVersion object.
            model_name: Name to identify the model. This must be a valid Snowflake SQL Identifier. Alphanumeric
                characters and underscores are permitted.
                See https://docs.snowflake.com/en/sql-reference/identifiers-syntax for more.
            version_name: Version identifier for the model. Combination of model_name and version_name must be unique.
                If not specified, a random name will be generated.
            comment: Comment associated with the model version. Defaults to None.
            metrics: A JSON serializable dictionary containing metrics linked to the model version. Defaults to None.
            conda_dependencies: List of Conda package specifications. Use "[channel::]package [operator version]" syntax
                to specify a dependency. It is a recommended way to specify your dependencies using conda. When channel
                is not specified, Snowflake Anaconda Channel will be used. Defaults to None.
            pip_requirements: List of Pip package specifications. Defaults to None.
                Models with pip requirements are currently only runnable in Snowpark Container Services.
                See https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/container for more.
                Models with pip requirements specified will not be executable in Snowflake Warehouse where all
                dependencies must be retrieved from Snowflake Anaconda Channel.
            artifact_repository_map: Specifies a mapping of package channels or platforms to custom artifact
                repositories. Defaults to None. Currently, the mapping applies only to warehouse execution.
                Note : This feature is currently in Private Preview; please contact your Snowflake account team
                to enable it.
                Format: {channel_name: artifact_repository_name}, where:
                   - channel_name: The name of the Conda package channel (e.g., 'condaforge') or 'pip' for pip packages.
                   - artifact_repository_name: The name or URL of the repository to fetch packages from.
            target_platforms: List of target platforms to run the model. The only acceptable inputs are a combination of
                {"WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"}. Defaults to None.
            python_version: Python version in which the model is run. Defaults to None.
            signatures: Model data signatures for inputs and outputs for various target methods. If it is None,
                sample_input_data would be used to infer the signatures for those models that cannot automatically
                infer the signature. If not None, sample_input_data should not be specified. Defaults to None.
            sample_input_data: Sample input data to infer model signatures from.
                It would also be used as background data in explanation and to capture data lineage. Defaults to None.
            user_files: Dictionary where the keys are subdirectories, and values are lists of local file name
                strings. The local file name strings can include wildcards (? or *) for matching multiple files.
            code_paths: List of directories containing code to import. Defaults to None.
            ext_modules: List of external modules to pickle with the model object.
                Only supported when logging the following types of model:
                Scikit-learn, Snowpark ML, PyTorch, TorchScript and Custom Model. Defaults to None.
            task: The task of the Model Version. It is an enum class Task with values TABULAR_REGRESSION,
                TABULAR_BINARY_CLASSIFICATION, TABULAR_MULTI_CLASSIFICATION, TABULAR_RANKING, or UNKNOWN. By default,
                it is set to Task.UNKNOWN and may be overridden by inferring from the Model Object.
            options (Dict[str, Any], optional): Additional model saving options.

                Model Saving Options include:

                - embed_local_ml_library: Embed local Snowpark ML into the code directory or folder.
                    Override to True if the local Snowpark ML version is not available in the Snowflake Anaconda
                    Channel. Otherwise, defaults to False
                - relax_version: Whether to relax the version constraints of the dependencies when running in the
                    Warehouse. It detects any ==x.y.z in specifiers and replaced with >=x.y, <(x+1). Defaults to True.
                - function_type: Set the method function type globally. To set method function types individually see
                    function_type in model_options.
                - target_methods: List of target methods to register when logging the model.
                  This option is not used in MLFlow models. Defaults to None, in which case the model handler's
                  default target methods will be used.
                - save_location: Location to save the model and metadata.
                - method_options: Per-method saving options. This dictionary has method names as keys and dictionary
                    values with the desired options.

                    The following are the available method options:

                    - case_sensitive: Indicates whether the method and its signature should be case sensitive.
                        This means when you refer the method in the SQL, you need to double quote it.
                        This will be helpful if you need case to tell apart your methods or features, or you have
                        non-alphabetic characters in your method or feature name. Defaults to False.
                    - max_batch_size: Maximum batch size that the method could accept in the Snowflake Warehouse.
                        Defaults to None, determined automatically by Snowflake.
                    - function_type: One of supported model method function types (FUNCTION or TABLE_FUNCTION).
        Returns:
            ModelVersion: ModelVersion object corresponding to the model just logged.
        """

        ...

    @overload
    def log_model(
        self,
        model: ModelVersion,
        *,
        model_name: str,
        version_name: Optional[str] = None,
    ) -> ModelVersion:
        """
        Log a model with a ModelVersion object.

        Args:
            model: Source ModelVersion object used to create the new ModelVersion object.
            model_name: Name to identify the model.
            version_name: Version identifier for the model. Combination of model_name and version_name must be unique.
                If not specified, a random name will be generated.
        """
        ...

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_MODEL_TELEMETRY_SUBPROJECT,
        func_params_to_log=[
            "model_name",
            "version_name",
            "comment",
            "metrics",
            "conda_dependencies",
            "pip_requirements",
            "artifact_repository_map",
            "target_platforms",
            "python_version",
            "signatures",
        ],
    )
    def log_model(
        self,
        model: Union[model_types.SupportedModelType, ModelVersion],
        *,
        model_name: str,
        version_name: Optional[str] = None,
        comment: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        artifact_repository_map: Optional[Dict[str, str]] = None,
        target_platforms: Optional[List[model_types.SupportedTargetPlatformType]] = None,
        python_version: Optional[str] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        user_files: Optional[Dict[str, List[str]]] = None,
        code_paths: Optional[List[str]] = None,
        ext_modules: Optional[List[ModuleType]] = None,
        task: model_types.Task = model_types.Task.UNKNOWN,
        options: Optional[model_types.ModelSaveOption] = None,
    ) -> ModelVersion:
        """
        Log a model with various parameters and metadata, or a ModelVersion object.

        Args:
            model: Supported model or ModelVersion object.
                - Supported model: Model object of supported types such as Scikit-learn, XGBoost, LightGBM, Snowpark ML,
                PyTorch, TorchScript, Tensorflow, Tensorflow Keras, MLFlow, HuggingFace Pipeline, Sentence Transformers,
                or Custom Model.
                - ModelVersion: Source ModelVersion object used to create the new ModelVersion object.
            model_name: Name to identify the model. This must be a valid Snowflake SQL Identifier. Alphanumeric
                characters and underscores are permitted.
                See https://docs.snowflake.com/en/sql-reference/identifiers-syntax for more.
            version_name: Version identifier for the model. Combination of model_name and version_name must be unique.
                If not specified, a random name will be generated.
            comment: Comment associated with the model version. Defaults to None.
            metrics: A JSON serializable dictionary containing metrics linked to the model version. Defaults to None.
            conda_dependencies: List of Conda package specifications. Use "[channel::]package [operator version]" syntax
                to specify a dependency. It is a recommended way to specify your dependencies using conda. When channel
                is not specified, Snowflake Anaconda Channel will be used. Defaults to None.
            pip_requirements: List of Pip package specifications. Defaults to None.
                Models with pip requirements are currently only runnable in Snowpark Container Services.
                See https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/container for more.
                Models with pip requirements specified will not be executable in Snowflake Warehouse where all
                dependencies must be retrieved from Snowflake Anaconda Channel.
            artifact_repository_map: Specifies a mapping of package channels or platforms to custom artifact
                repositories. Defaults to None. Currently, the mapping applies only to warehouse execution.
                Note : This feature is currently in Private Preview; please contact your Snowflake account team to
                enable it.
                Format: {channel_name: artifact_repository_name}, where:
                   - channel_name: The name of the Conda package channel (e.g., 'condaforge') or 'pip' for pip packages.
                   - artifact_repository_name: The name or URL of the repository to fetch packages from.
            target_platforms: List of target platforms to run the model. The only acceptable inputs are a combination of
                {"WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"}. Defaults to None.
            python_version: Python version in which the model is run. Defaults to None.
            signatures: Model data signatures for inputs and outputs for various target methods. If it is None,
                sample_input_data would be used to infer the signatures for those models that cannot automatically
                infer the signature. If not None, sample_input_data should not be specified. Defaults to None.
            sample_input_data: Sample input data to infer model signatures from.
                It would also be used as background data in explanation and to capture data lineage. Defaults to None.
            user_files: Dictionary where the keys are subdirectories, and values are lists of local file name
                strings. The local file name strings can include wildcards (? or *) for matching multiple files.
            code_paths: List of directories containing code to import. Defaults to None.
            ext_modules: List of external modules to pickle with the model object.
                Only supported when logging the following types of model:
                Scikit-learn, Snowpark ML, PyTorch, TorchScript and Custom Model. Defaults to None.
            task: The task of the Model Version. It is an enum class Task with values TABULAR_REGRESSION,
                TABULAR_BINARY_CLASSIFICATION, TABULAR_MULTI_CLASSIFICATION, TABULAR_RANKING, or UNKNOWN. By default,
                it is set to Task.UNKNOWN and may be overridden by inferring from the Model Object.
            options (Dict[str, Any], optional): Additional model saving options.

                Model Saving Options include:

                - embed_local_ml_library: Embed local Snowpark ML into the code directory or folder.
                    Override to True if the local Snowpark ML version is not available in the Snowflake Anaconda
                    Channel. Otherwise, defaults to False
                - relax_version: Whether to relax the version constraints of the dependencies when running in the
                    Warehouse. It detects any ==x.y.z in specifiers and replaced with >=x.y, <(x+1). Defaults to True.
                - function_type: Set the method function type globally. To set method function types individually see
                  function_type in model_options.
                - target_methods: List of target methods to register when logging the model.
                  This option is not used in MLFlow models. Defaults to None, in which case the model handler's
                  default target methods will be used.
                - save_location: Location to save the model and metadata.
                - method_options: Per-method saving options. This dictionary has method names as keys and dictionary
                    values with the desired options. See the example below.

                    The following are the available method options:

                    - case_sensitive: Indicates whether the method and its signature should be case sensitive.
                        This means when you refer the method in the SQL, you need to double quote it.
                        This will be helpful if you need case to tell apart your methods or features, or you have
                        non-alphabetic characters in your method or feature name. Defaults to False.
                    - max_batch_size: Maximum batch size that the method could accept in the Snowflake Warehouse.
                        Defaults to None, determined automatically by Snowflake.
                    - function_type: One of supported model method function types (FUNCTION or TABLE_FUNCTION).

        Raises:
            ValueError: If extra arguments are specified ModelVersion is provided.

        Returns:
            ModelVersion: ModelVersion object corresponding to the model just logged.

        Example::

            from snowflake.ml.registry import Registry

            # create a session
            session = ...

            registry = Registry(session=session)

            # Define `method_options` for each inference method if needed.
            method_options={
              "predict": {
                "case_sensitive": True
              }
            }

            registry.log_model(
              model=model,
              model_name="my_model",
              options={"method_options": method_options},
            )
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )
        if isinstance(model, ModelVersion):
            # check that no arguments are provided other than the ones for copy model.
            invalid_args = [
                comment,
                conda_dependencies,
                pip_requirements,
                artifact_repository_map,
                target_platforms,
                python_version,
                signatures,
                sample_input_data,
                user_files,
                code_paths,
                ext_modules,
                options,
            ]
            for arg in invalid_args:
                if arg is not None:
                    raise ValueError(
                        "When calling log_model with a ModelVersion, only model_name and version_name may be specified."
                    )
            if task is not model_types.Task.UNKNOWN:
                raise ValueError("`task` cannot be specified when calling log_model with a ModelVersion.")

        if pip_requirements:
            warnings.warn(
                "Models logged specifying `pip_requirements` can not be executed "
                "in Snowflake Warehouse where all dependencies are required to be retrieved "
                "from Snowflake Anaconda Channel.",
                category=UserWarning,
                stacklevel=1,
            )
        return self._model_manager.log_model(
            model=model,
            model_name=model_name,
            version_name=version_name,
            comment=comment,
            metrics=metrics,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            artifact_repository_map=artifact_repository_map,
            target_platforms=target_platforms,
            python_version=python_version,
            signatures=signatures,
            sample_input_data=sample_input_data,
            user_files=user_files,
            code_paths=code_paths,
            ext_modules=ext_modules,
            task=task,
            options=options,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_MODEL_TELEMETRY_SUBPROJECT,
    )
    def get_model(self, model_name: str) -> Model:
        """Get the model object by its name.

        Args:
            model_name: The name of the model.

        Returns:
            The corresponding model object.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )
        return self._model_manager.get_model(model_name=model_name, statement_params=statement_params)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_MODEL_TELEMETRY_SUBPROJECT,
    )
    def models(self) -> List[Model]:
        """Get all models in the schema where the registry is opened.

        Returns:
            A list of Model objects representing all models in the opened registry.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )
        return self._model_manager.models(statement_params=statement_params)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_MODEL_TELEMETRY_SUBPROJECT,
    )
    def show_models(self) -> pd.DataFrame:
        """Show information of all models in the schema where the registry is opened.

        Returns:
            A Pandas DataFrame containing information of all models in the schema.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )
        return self._model_manager.show_models(statement_params=statement_params)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_MODEL_TELEMETRY_SUBPROJECT,
    )
    def delete_model(self, model_name: str) -> None:
        """
        Delete the model by its name.

        Args:
            model_name: The name of the model to be deleted.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )

        self._model_manager.delete_model(model_name=model_name, statement_params=statement_params)

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    @snowpark._internal.utils.private_preview(version=model_monitor_version.SNOWFLAKE_ML_MONITORING_MIN_VERSION)
    def add_monitor(
        self,
        name: str,
        source_config: model_monitor_config.ModelMonitorSourceConfig,
        model_monitor_config: model_monitor_config.ModelMonitorConfig,
    ) -> model_monitor.ModelMonitor:
        """Add a Model Monitor to the Registry.

        Args:
            name: Name of Model Monitor to create.
            source_config: Configuration options of table for Model Monitor.
            model_monitor_config: Configuration options of Model Monitor.

        Returns:
            The newly added Model Monitor object.

        Raises:
            ValueError: If monitoring is not enabled in the Registry.
        """
        if not self.enable_monitoring:
            raise ValueError(_MODEL_MONITORING_DISABLED_ERROR)
        return self._model_monitor_manager.add_monitor(name, source_config, model_monitor_config)

    @overload
    def get_monitor(self, model_version: model_version_impl.ModelVersion) -> model_monitor.ModelMonitor:
        """Get a Model Monitor on a Model Version from the Registry.

        Args:
            model_version: Model Version for which to retrieve the Model Monitor.
        """
        ...

    @overload
    def get_monitor(self, name: str) -> model_monitor.ModelMonitor:
        """Get a Model Monitor by name from the Registry.

        Args:
            name: Name of Model Monitor to retrieve.
        """
        ...

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    @snowpark._internal.utils.private_preview(version=model_monitor_version.SNOWFLAKE_ML_MONITORING_MIN_VERSION)
    def get_monitor(
        self, *, name: Optional[str] = None, model_version: Optional[model_version_impl.ModelVersion] = None
    ) -> model_monitor.ModelMonitor:
        """Get a Model Monitor from the Registry.

        Args:
            name: Name of Model Monitor to retrieve.
            model_version: Model Version for which to retrieve the Model Monitor.

        Returns:
            The fetched Model Monitor.

        Raises:
            ValueError: If monitoring is not enabled in the Registry.
        """
        if not self.enable_monitoring:
            raise ValueError(_MODEL_MONITORING_DISABLED_ERROR)
        if name is not None:
            return self._model_monitor_manager.get_monitor(name=name)
        elif model_version is not None:
            return self._model_monitor_manager.get_monitor_by_model_version(model_version)
        else:
            raise ValueError("Must provide either `name` or `model_version` to get ModelMonitor")

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    @snowpark._internal.utils.private_preview(version=model_monitor_version.SNOWFLAKE_ML_MONITORING_MIN_VERSION)
    def show_model_monitors(self) -> List[snowpark.Row]:
        """Show all model monitors in the registry.

        Returns:
            List of snowpark.Row containing metadata for each model monitor.

        Raises:
            ValueError: If monitoring is not enabled in the Registry.
        """
        if not self.enable_monitoring:
            raise ValueError(_MODEL_MONITORING_DISABLED_ERROR)
        return self._model_monitor_manager.show_model_monitors()

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    @snowpark._internal.utils.private_preview(version=model_monitor_version.SNOWFLAKE_ML_MONITORING_MIN_VERSION)
    def delete_monitor(self, name: str) -> None:
        """Delete a Model Monitor by name from the Registry.

        Args:
            name: Name of the Model Monitor to delete.

        Raises:
            ValueError: If monitoring is not enabled in the registry.
        """
        if not self.enable_monitoring:
            raise ValueError(_MODEL_MONITORING_DISABLED_ERROR)
        self._model_monitor_manager.delete_monitor(name)
