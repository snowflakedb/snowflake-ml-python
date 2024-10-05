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
from snowflake.ml.monitoring._client import (
    model_monitor,
    model_monitor_manager,
    model_monitor_version,
)
from snowflake.ml.monitoring.entities import model_monitor_config
from snowflake.ml.registry._manager import model_manager
from snowflake.snowpark import session

_TELEMETRY_PROJECT = "MLOps"
_MODEL_TELEMETRY_SUBPROJECT = "ModelManagement"


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
            session, database_name=self._database_name, schema_name=self._schema_name
        )

        self.enable_monitoring = options.get("enable_monitoring", False) if options else False
        if self.enable_monitoring:
            monitor_statement_params = telemetry.get_statement_params(
                project=telemetry.TelemetryProject.MLOPS.value,
                subproject=telemetry.TelemetrySubProject.MONITORING.value,
            )

            self._model_monitor_manager = model_monitor_manager.ModelMonitorManager(
                session=session,
                database_name=self._database_name,
                schema_name=self._schema_name,
                create_if_not_exists=True,  # TODO: Support static setup method to configure schema for monitoring.
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
        python_version: Optional[str] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        code_paths: Optional[List[str]] = None,
        ext_modules: Optional[List[ModuleType]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
    ) -> ModelVersion:
        """
        Log a model with various parameters and metadata.

        Args:
            model: Model object of supported types such as Scikit-learn, XGBoost, LightGBM, Snowpark ML,
                PyTorch, TorchScript, Tensorflow, Tensorflow Keras, MLFlow, HuggingFace Pipeline,
                Sentence Transformers, or Custom Model.
            model_name: Name to identify the model.
            version_name: Version identifier for the model. Combination of model_name and version_name must be unique.
                If not specified, a random name will be generated.
            comment: Comment associated with the model version. Defaults to None.
            metrics: A JSON serializable dictionary containing metrics linked to the model version. Defaults to None.
            signatures: Model data signatures for inputs and outputs for various target methods. If it is None,
                sample_input_data would be used to infer the signatures for those models that cannot automatically
                infer the signature. If not None, sample_input_data should not be specified. Defaults to None.
            sample_input_data: Sample input data to infer model signatures from. Defaults to None.
            conda_dependencies: List of Conda package specifications. Use "[channel::]package [operator version]" syntax
                to specify a dependency. It is a recommended way to specify your dependencies using conda. When channel
                is not specified, Snowflake Anaconda Channel will be used. Defaults to None.
            pip_requirements: List of Pip package specifications. Defaults to None.
                Currently it is not supported since Model can only executed in Snowflake Warehouse where all
                dependencies are required to be retrieved from Snowflake Anaconda Channel.
            python_version: Python version in which the model is run. Defaults to None.
            code_paths: List of directories containing code to import. Defaults to None.
            ext_modules: List of external modules to pickle with the model object.
                Only supported when logging the following types of model:
                Scikit-learn, Snowpark ML, PyTorch, TorchScript and Custom Model. Defaults to None.
            options (Dict[str, Any], optional): Additional model saving options.
                Model Saving Options include:
                - embed_local_ml_library: Embed local Snowpark ML into the code directory or folder.
                    Override to True if the local Snowpark ML version is not available in the Snowflake Anaconda
                    Channel. Otherwise, defaults to False
                - relax_version: Whether or not relax the version constraints of the dependencies when running in the
                    Warehouse. It detects any ==x.y.z in specifiers and replaced with >=x.y, <(x+1). Defaults to True.
                - function_type: Set the method function type globally. To set method function types individually see
                  function_type in model_options.
                - method_options: Per-method saving options including:
                    - case_sensitive: Indicates whether the method and its signature should be case sensitive.
                        This means when you refer the method in the SQL, you need to double quote it.
                        This will be helpful if you need case to tell apart your methods or features, or you have
                        non-alphabetic characters in your method or feature name. Defaults to False.
                    - max_batch_size: Maximum batch size that the method could accept in the Snowflake Warehouse.
                        Defaults to None, determined automatically by Snowflake.
                    - function_type: One of supported model method function types (FUNCTION or TABLE_FUNCTION).
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
        python_version: Optional[str] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
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
            model_name: Name to identify the model.
            version_name: Version identifier for the model. Combination of model_name and version_name must be unique.
                If not specified, a random name will be generated.
            comment: Comment associated with the model version. Defaults to None.
            metrics: A JSON serializable dictionary containing metrics linked to the model version. Defaults to None.
            signatures: Model data signatures for inputs and outputs for various target methods. If it is None,
                sample_input_data would be used to infer the signatures for those models that cannot automatically
                infer the signature. If not None, sample_input_data should not be specified. Defaults to None.
            sample_input_data: Sample input data to infer model signatures from. Defaults to None.
            conda_dependencies: List of Conda package specifications. Use "[channel::]package [operator version]" syntax
                to specify a dependency. It is a recommended way to specify your dependencies using conda. When channel
                is not specified, Snowflake Anaconda Channel will be used. Defaults to None.
            pip_requirements: List of Pip package specifications. Defaults to None.
                Currently it is not supported since Model can only executed in Snowflake Warehouse where all
                dependencies are required to be retrieved from Snowflake Anaconda Channel.
            python_version: Python version in which the model is run. Defaults to None.
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
                - relax_version: Whether or not relax the version constraints of the dependencies when running in the
                    Warehouse. It detects any ==x.y.z in specifiers and replaced with >=x.y, <(x+1). Defaults to True.
                - function_type: Set the method function type globally. To set method function types individually see
                  function_type in model_options.
                - method_options: Per-method saving options including:
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
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )
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
            python_version=python_version,
            signatures=signatures,
            sample_input_data=sample_input_data,
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
        table_config: model_monitor_config.ModelMonitorTableConfig,
        model_monitor_config: model_monitor_config.ModelMonitorConfig,
        *,
        add_dashboard_udtfs: bool = False,
    ) -> model_monitor.ModelMonitor:
        """Add a Model Monitor to the Registry

        Args:
            name: Name of Model Monitor to create
            table_config: Configuration options of table for ModelMonitor.
            model_monitor_config: Configuration options of ModelMonitor.
            add_dashboard_udtfs: Add UDTFs useful for creating a dashboard.

        Returns:
            The newly added ModelMonitor object.

        Raises:
            ValueError: If monitoring feature flag is not enabled.
        """
        if not self.enable_monitoring:
            raise ValueError(
                "Must enable monitoring in Registry to use this method. Please set the `enable_monitoring=True` option"
            )

        # TODO: Change to fully qualified source table reference to allow table to live in different DB.
        return self._model_monitor_manager.add_monitor(
            name, table_config, model_monitor_config, add_dashboard_udtfs=add_dashboard_udtfs
        )

    @overload
    def get_monitor(self, model_version: model_version_impl.ModelVersion) -> model_monitor.ModelMonitor:
        """Get a Model Monitor on a ModelVersion from the Registry

        Args:
            model_version: ModelVersion for which to retrieve the ModelMonitor.
        """
        ...

    @overload
    def get_monitor(self, name: str) -> model_monitor.ModelMonitor:
        """Get a Model Monitor from the Registry

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
        """Get a Model Monitor from the Registry

        Args:
            name: Name of Model Monitor to retrieve.
            model_version: ModelVersion for which to retrieve the ModelMonitor.

        Returns:
            The fetched ModelMonitor.

        Raises:
            ValueError: If monitoring feature flag is not enabled.
            ValueError: If neither name nor model_version specified.
        """
        if not self.enable_monitoring:
            raise ValueError(
                "Must enable monitoring in Registry to use this method. Please set the `enable_monitoring=True` option"
            )
        if name is not None:
            return self._model_monitor_manager.get_monitor(name=name)
        elif model_version is not None:
            return self._model_monitor_manager.get_monitor_by_model_version(model_version=model_version)
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
            ValueError: If monitoring feature flag is not enabled.
        """
        if not self.enable_monitoring:
            raise ValueError(
                "Must enable monitoring in Registry to use this method. Please set the `enable_monitoring=True` option"
            )
        return self._model_monitor_manager.show_model_monitors()

    @telemetry.send_api_usage_telemetry(
        project=telemetry.TelemetryProject.MLOPS.value,
        subproject=telemetry.TelemetrySubProject.MONITORING.value,
    )
    @snowpark._internal.utils.private_preview(version=model_monitor_version.SNOWFLAKE_ML_MONITORING_MIN_VERSION)
    def delete_monitor(self, name: str) -> None:
        """Delete a Model Monitor from the Registry

        Args:
            name: Name of the Model Monitor to delete.

        Raises:
            ValueError: If monitoring feature flag is not enabled.
        """
        if not self.enable_monitoring:
            raise ValueError(
                "Must enable monitoring in Registry to use this method. Please set the `enable_monitoring=True` option"
            )
        self._model_monitor_manager.delete_monitor(name)
