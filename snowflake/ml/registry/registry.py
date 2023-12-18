from types import ModuleType
from typing import Dict, List, Optional

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._client.model import model_impl, model_version_impl
from snowflake.ml.model._client.ops import model_ops
from snowflake.ml.model._model_composer import model_composer
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

        self._model_ops = model_ops.ModelOperator(
            session, database_name=self._database_name, schema_name=self._schema_name
        )

    @property
    def location(self) -> str:
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
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        python_version: Optional[str] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        code_paths: Optional[List[str]] = None,
        ext_modules: Optional[List[ModuleType]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
    ) -> model_version_impl.ModelVersion:
        """Log a model.

        Args:
            model: Model Python object
            model_name: A string as name.
            version_name: A string as version. model_name and version_name combination must be unique.
            signatures: Model data signatures for inputs and output for every target methods. If it is None,
                sample_input_data would be used to infer the signatures for those models that cannot automatically
                infer the signature. If not None, sample_input should not be specified. Defaults to None.
            sample_input_data: Sample input data to infer the model signatures from. If it is None, signatures must be
                specified if the model cannot automatically infer the signature. If not None, signatures should not be
                specified. Defaults to None.
            conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to
                specify a dependency. It is a recommended way to specify your dependencies using conda. When channel is
                not specified, Snowflake Anaconda Channel will be used.
            pip_requirements: List of Pip package specs.
            python_version: A string of python version where model is run. Used for user override. If specified as None,
                current version would be captured. Defaults to None.
            code_paths: Directory of code to import.
            ext_modules: External modules that user might want to get pickled with model object. Defaults to None.
            options: Model specific kwargs.

        Returns:
            A ModelVersion object corresponding to the model just get logged.
        """

        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )
        model_name_id = sql_identifier.SqlIdentifier(model_name)

        version_name_id = sql_identifier.SqlIdentifier(version_name)

        stage_path = self._model_ops.prepare_model_stage_path(
            statement_params=statement_params,
        )

        mc = model_composer.ModelComposer(self._model_ops._session, stage_path=stage_path)
        mc.save(
            name=model_name_id.resolved(),
            model=model,
            signatures=signatures,
            sample_input=sample_input_data,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            python_version=python_version,
            code_paths=code_paths,
            ext_modules=ext_modules,
            options=options,
        )
        self._model_ops.create_from_stage(
            composed_model=mc,
            model_name=model_name_id,
            version_name=version_name_id,
            statement_params=statement_params,
        )

        return model_version_impl.ModelVersion._ref(
            self._model_ops,
            model_name=model_name_id,
            version_name=version_name_id,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_MODEL_TELEMETRY_SUBPROJECT,
    )
    def get_model(self, model_name: str) -> model_impl.Model:
        """Get the model object.

        Args:
            model_name: The model name.

        Raises:
            ValueError: Raised when the model requested does not exist.

        Returns:
            The model object.
        """
        model_name_id = sql_identifier.SqlIdentifier(model_name)

        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )
        if self._model_ops.validate_existence(
            model_name=model_name_id,
            statement_params=statement_params,
        ):
            return model_impl.Model._ref(
                self._model_ops,
                model_name=model_name_id,
            )
        else:
            raise ValueError(f"Unable to find model {model_name}")

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_MODEL_TELEMETRY_SUBPROJECT,
    )
    def list_models(self) -> List[model_impl.Model]:
        """List all models in the schema where the registry is opened.

        Returns:
            A List of Model= object representing all models in the schema where the registry is opened.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )
        model_names = self._model_ops.list_models_or_versions(
            statement_params=statement_params,
        )
        return [
            model_impl.Model._ref(
                self._model_ops,
                model_name=model_name,
            )
            for model_name in model_names
        ]

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_MODEL_TELEMETRY_SUBPROJECT,
    )
    def delete_model(self, model_name: str) -> None:
        """Delete the model.

        Args:
            model_name: The model name, can be fully qualified one.
                If not, use database name and schema name of the registry.
        """
        model_name_id = sql_identifier.SqlIdentifier(model_name)

        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_MODEL_TELEMETRY_SUBPROJECT,
        )

        self._model_ops.delete_model_or_version(
            model_name=model_name_id,
            statement_params=statement_params,
        )
