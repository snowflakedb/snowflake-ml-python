from types import ModuleType
from typing import Any, Dict, List, Optional

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import (
    Model,
    ModelVersion,
    model_signature,
    type_hints as model_types,
)
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
    ) -> None:
        """Open a registry in a **pre-created** Snowflake schema.

        Args:
            session: The Snowpark Session to connect with Snowflake.
            database_name: The name of the database. If None, will use the current database of the session.
                Defaults to None.
            schema_name: The name of the database. If None, will use the current schema of the session. If there is no
                active schema, PUBLIC schema will be used. Defaults to None.

        Raises:
            ValueError: Raised when there is either no database specified or active in the session.
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

    @property
    def location(self) -> str:
        """Get the location (database.schema) of the registry."""
        return ".".join([self._database_name.identifier(), self._schema_name.identifier()])

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_MODEL_TELEMETRY_SUBPROJECT,
    )
    def log_model(
        self,
        model: model_types.SupportedModelType,
        *,
        model_name: str,
        version_name: str,
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
        """Log a model.

        Args:
            model: Model Python object. Can be the following types:
                Scikit-learn, XGBoost, Snowpark ML, PyTorch, TorchScript, Tensorflow, Tensorflow Keras, MLFlow,
                HuggingFace Pipeline, Peft-finetuned LLM, and Custom Model.
            model_name: A string as name.
            version_name: A string as version. Model_name and version_name combination must be unique.
            comment: Comment to the about to log with the model version. Defaults to None
            metrics: A JSON serializable dictionary containing the metrics linked to the model version.
                Default to None.
            signatures: Model data signatures for inputs and output for every target methods. If it is None,
                sample_input_data would be used to infer the signatures for those models that cannot automatically
                infer the signature. If not None, sample_input should not be specified. Defaults to None.
            sample_input_data: Sample input data to infer the model signatures from. If it is None, signatures must be
                specified if the model cannot automatically infer the signature. If not None, signatures should not be
                specified. Defaults to None.
            conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to
                specify a dependency. It is a recommended way to specify your dependencies using conda. When channel is
                not specified, Snowflake Anaconda Channel will be used. Defaults to None.
            pip_requirements: List of Pip package specs.  Defaults to None.
            python_version: A string of python version where model is run. Used for user override. If specified as None,
                current version would be captured. Defaults to None.
            code_paths: A List of directory of code to import.  Defaults to None.
            ext_modules: A list of external modules that user might want to get pickled with model object.
                Only supported when logging the following types of model:
                Scikit-learn, Snowpark ML, PyTorch, TorchScript and Custom Model.
                Defaults to None.
            options: Additional Model Saving Options.

        Model Saving Options includes:

        - embed_local_ml_library:
            Embedding local Snowpark ML into the code directory of the folder. Default to False
            if the local Snowpark ML version is available in Snowflake Anaconda Channel. Default to True
            if the local Snowpark ML version is not available in Snowflake Anaconda Channel.
        - method_options:
            A Dictionary indicating per-method saving options including:

            - case_sensitive:
                A boolean indicating that the method, include its signature should be considered
                as case sensitive. This means when you refer the method in the SQL, you need to double quote it.
                This will be helpful if you need case to tell apart your methods or features, or you have
                non-alphabetic characters in your method or feature name.
            - max_batch_size:
                An integer indicating that the max batch size that the method could accept when
                using in the Snowflake Warehouse. If None, it will be determined by Snowflake automatically.
                Default to None.

        Returns:
            A ModelVersion object corresponding to the model just get logged.
        """

        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
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
            options=options,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_MODEL_TELEMETRY_SUBPROJECT,
    )
    def get_model(self, model_name: str) -> Model:
        """Get the model object.

        Args:
            model_name: The model name.

        Returns:
            The model object.
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
    def show_models(self) -> List[Model]:
        """Show all models in the schema where the registry is opened.

        Returns:
            A List of Model object representing all models in the schema where the registry is opened.
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
        """Delete the model.

        Args:
            model_name: The model name to be deleted.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )

        self._model_manager.delete_model(model_name=model_name, statement_params=statement_params)
