import functools
import json
import sys
import warnings
from typing import Any, Optional, Union
from urllib.parse import quote

from snowflake import snowpark
from snowflake.ml import model as ml_model, registry
from snowflake.ml._internal.human_readable_id import hrid_generator
from snowflake.ml._internal.utils import connection_params, sql_identifier
from snowflake.ml.experiment import (
    _entities as entities,
    _experiment_info as experiment_info,
)
from snowflake.ml.experiment._client import (
    artifact,
    experiment_tracking_sql_client as sql_client,
)
from snowflake.ml.model import type_hints
from snowflake.ml.utils import sql_client as sql_client_utils

DEFAULT_EXPERIMENT_NAME = sql_identifier.SqlIdentifier("DEFAULT")


class ExperimentTracking:
    """
    Class to manage experiments in Snowflake.
    """

    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "ExperimentTracking":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        session: snowpark.Session,
        *,
        database_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> None:
        """
        Initializes experiment tracking within a pre-created schema.
        This is a singleton class, so if an instance already exists, it will not reinitialize.

        Args:
            session: The Snowpark Session to connect with Snowflake.
            database_name: The name of the database. If None, the current database of the session
                will be used. Defaults to None.
            schema_name: The name of the schema. If None, the current schema of the session
                will be used. If there is no active schema, the PUBLIC schema will be used. Defaults to None.

        Raises:
            ValueError: If no database is provided and no active database exists in the session.
        """
        if hasattr(self, "_initialized"):
            warnings.warn(
                "ExperimentTracking is a singleton class. Reusing the existing instance, which has the setting:\n"
                f"    Database: {self._database_name}, Schema: {self._schema_name}\n"
                "To change the database or schema, use the database_name and schema_name arguments to set_experiment.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Declare types for mypy
        self._database_name: sql_identifier.SqlIdentifier
        self._schema_name: sql_identifier.SqlIdentifier
        self._sql_client: sql_client.ExperimentTrackingSQLClient

        if database_name:
            self._database_name = sql_identifier.SqlIdentifier(database_name)
        elif session_db := session.get_current_database():
            self._database_name = sql_identifier.SqlIdentifier(session_db)
        else:
            raise ValueError("You need to provide a database to use experiment tracking.")

        if schema_name:
            self._schema_name = sql_identifier.SqlIdentifier(schema_name)
        elif session_schema := session.get_current_schema():
            self._schema_name = sql_identifier.SqlIdentifier(session_schema)
        else:
            self._schema_name = sql_identifier.SqlIdentifier("PUBLIC")

        self._sql_client = sql_client.ExperimentTrackingSQLClient(
            session,
            database_name=self._database_name,
            schema_name=self._schema_name,
        )
        self._registry = registry.Registry(
            session=session,
            database_name=self._database_name,
            schema_name=self._schema_name,
        )
        self._session = session

        # The experiment in context
        self._experiment: Optional[entities.Experiment] = None
        # The run in context
        self._run: Optional[entities.Run] = None

        self._initialized = True

    def __getstate__(self) -> dict[str, Any]:
        parent_state = (
            super().__getstate__()  # type: ignore[misc] # object.__getstate__ appears in 3.11
            if hasattr(super(), "__getstate__")
            else self.__dict__
        )
        state = dict(parent_state)  # Create a copy so we can safely modify the state

        # Remove unpicklable attributes
        state["_session"] = None
        state["_sql_client"] = None
        state["_registry"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)  # type: ignore[misc]
        else:
            self.__dict__.update(state)

        # Restore unpicklable attributes
        options: dict[str, Any] = connection_params.SnowflakeLoginOptions()
        options["client_session_keep_alive"] = True  # Needed for long-running training jobs
        self._session = snowpark.Session.builder.configs(options).getOrCreate()
        self._sql_client = sql_client.ExperimentTrackingSQLClient(
            session=self._session,
            database_name=self._database_name,
            schema_name=self._schema_name,
        )
        self._registry = registry.Registry(
            session=self._session,
            database_name=self._database_name,
            schema_name=self._schema_name,
        )

    def set_experiment(
        self,
        experiment_name: str,
        database_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> entities.Experiment:
        """
        Set the experiment in context. Creates a new experiment if it doesn't exist.

        Args:
            experiment_name: The name of the experiment.
            database_name: The name of the database. If None, reuse the current database. Defaults to None.
            schema_name: The name of the schema. If None, the behavior depends on whether `database_name` is specified.
                If `database_name` is specified, the schema is set to "PUBLIC".
                If `database_name` is not specified, reuse the current schema. Defaults to None.

        Returns:
            Experiment: The experiment that was set.
        """
        if database_name is not None:
            if schema_name is None:
                schema_name = "PUBLIC"
        database_name = (
            sql_identifier.SqlIdentifier(database_name) if database_name is not None else self._database_name
        )
        schema_name = sql_identifier.SqlIdentifier(schema_name) if schema_name is not None else self._schema_name

        experiment_name = sql_identifier.SqlIdentifier(experiment_name)
        if (
            self._experiment
            and self._experiment.name == experiment_name
            and self._database_name == database_name
            and self._schema_name == schema_name
        ):
            return self._experiment

        self._update_database_and_schema(database_name, schema_name)
        self._sql_client.create_experiment(
            experiment_name=experiment_name,
            creation_mode=sql_client_utils.CreationMode(if_not_exists=True),
        )
        self._experiment = entities.Experiment(experiment_name=experiment_name)
        self._run = None
        return self._experiment

    def delete_experiment(
        self,
        experiment_name: str,
        database_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> None:
        """
        Delete an experiment.

        Args:
            experiment_name: The name of the experiment.
            database_name: The name of the database. If None, reuse the current database.
                Must be specified if `schema_name` is specified. Defaults to None.
            schema_name: The name of the schema. If None, reuse the current schema.
                Must be specified if `database_name` is specified. Defaults to None.

        Raises:
            ValueError: If database_name is specified but schema_name is not.
        """
        if (database_name is None) ^ (schema_name is None):  # if only one of database_name and schema_name is set
            raise ValueError(
                "If one of database_name and schema_name is specified, the other one must also be specified."
            )
        database_name = (
            sql_identifier.SqlIdentifier(database_name) if database_name is not None else self._database_name
        )
        schema_name = sql_identifier.SqlIdentifier(schema_name) if schema_name is not None else self._schema_name

        self._sql_client.drop_experiment(
            database_name=database_name,
            schema_name=schema_name,
            experiment_name=sql_identifier.SqlIdentifier(experiment_name),
        )
        if (
            self._experiment
            and self._experiment.name == experiment_name
            and self._database_name == database_name
            and self._schema_name == schema_name
        ):
            self._experiment = None
            self._run = None

    @functools.wraps(registry.Registry.log_model)
    def log_model(
        self,
        model: Union[type_hints.SupportedModelType, ml_model.ModelVersion],
        *,
        model_name: str,
        **kwargs: Any,
    ) -> ml_model.ModelVersion:
        run = self._get_or_start_run()
        with experiment_info.ExperimentInfoPatcher(experiment_info=run._get_experiment_info()):
            return self._registry.log_model(model, model_name=model_name, **kwargs)

    def start_run(
        self,
        run_name: Optional[str] = None,
    ) -> entities.Run:
        """
        Start a new run. If a run name of an existing run is provided, resumes the run if it is running.

        Args:
            run_name: The name of the run. If None, a default name will be generated.

        Returns:
            Run: The run that was started or resumed.

        Raises:
            RuntimeError: If a run is already active. If a run with the same name exists but is not running.
        """
        if self._run:
            raise RuntimeError("A run is already active. Please end the current run before starting a new one.")
        experiment = self._get_or_set_experiment()

        if run_name is None:
            run_name = self._generate_run_name(experiment)
        elif runs := self._sql_client.show_runs_in_experiment(experiment_name=experiment.name, like=run_name):
            if "RUNNING" != json.loads(runs[0][sql_client.RUN_METADATA_COL_NAME])["status"]:
                raise RuntimeError(f"Run {run_name} exists but cannot be resumed as it is no longer running.")
            else:
                self._run = entities.Run(
                    experiment_tracking=self,
                    experiment_name=experiment.name,
                    run_name=sql_identifier.SqlIdentifier(run_name),
                )
                return self._run

        run_name = sql_identifier.SqlIdentifier(run_name)
        self._sql_client.add_run(
            experiment_name=experiment.name,
            run_name=run_name,
        )
        self._run = entities.Run(experiment_tracking=self, experiment_name=experiment.name, run_name=run_name)
        return self._run

    def end_run(self, run_name: Optional[str] = None) -> None:
        """
        End the current run if no run name is provided. Otherwise, the specified run is ended.

        Args:
            run_name: The name of the run to be ended. If None, the current run is ended.

        Raises:
            RuntimeError: If no run is active.
        """
        if not self._experiment:
            raise RuntimeError("No experiment set. Please set an experiment before ending a run.")
        experiment_name = self._experiment.name

        if run_name:
            run_name = sql_identifier.SqlIdentifier(run_name)
        elif self._run:
            run_name = self._run.name
        else:
            raise RuntimeError("No run is active. Please start a run before ending it.")

        self._sql_client.commit_run(
            experiment_name=experiment_name,
            run_name=run_name,
        )
        if self._run and run_name == self._run.name:
            self._run = None
        self._print_urls(experiment_name=experiment_name, run_name=run_name)

    def delete_run(
        self,
        run_name: str,
    ) -> None:
        """
        Delete a run.

        Args:
            run_name: The name of the run to be deleted.

        Raises:
            RuntimeError: If no experiment is set.
        """
        if not self._experiment:
            raise RuntimeError("No experiment set. Please set an experiment before deleting a run.")
        self._sql_client.drop_run(
            experiment_name=self._experiment.name,
            run_name=sql_identifier.SqlIdentifier(run_name),
        )
        if self._run and self._run.name == run_name:
            self._run = None

    def log_metric(
        self,
        key: str,
        value: float,
        step: int = 0,
    ) -> None:
        """
        Log a metric under the current run. If no run is active, this method will create a new run.

        Args:
            key: The name of the metric.
            value: The value of the metric.
            step: The step of the metric. Defaults to 0.
        """
        self.log_metrics(metrics={key: value}, step=step)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int = 0,
    ) -> None:
        """
        Log metrics under the current run. If no run is active, this method will create a new run.

        Args:
            metrics: Dictionary containing metric keys and float values.
            step: The step of the metrics. Defaults to 0.

        Raises:
            snowpark.exceptions.SnowparkSQLException: If logging metrics fails due to Snowflake SQL errors,
                except for run metadata size limit errors which will issue a warning instead of raising.
        """
        run = self._get_or_start_run()
        metrics_list = []
        for key, value in metrics.items():
            metrics_list.append(entities.Metric(key, value, step))
        try:
            self._sql_client.modify_run_add_metrics(
                experiment_name=run.experiment_name,
                run_name=run.name,
                metrics=json.dumps([metric.to_dict() for metric in metrics_list]),
            )
        except snowpark.exceptions.SnowparkSQLException as e:
            if e.sql_error_code == 400003:  # EXPERIMENT_RUN_PROPERTY_SIZE_LIMIT_EXCEEDED
                run._warn_about_run_metadata_size(e.message)
            else:
                raise

    def log_param(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Log a parameter under the current run. If no run is active, this method will create a new run.

        Args:
            key: The name of the parameter.
            value: The value of the parameter. Values can be of any type, but will be converted to string.
        """
        self.log_params({key: value})

    def log_params(
        self,
        params: dict[str, Any],
    ) -> None:
        """
        Log parameters under the current run. If no run is active, this method will create a new run.

        Args:
            params: Dictionary containing parameter keys and values. Values can be of any type, but will be converted
                to string.

        Raises:
            snowpark.exceptions.SnowparkSQLException: If logging parameters fails due to Snowflake SQL errors,
                except for run metadata size limit errors which will issue a warning instead of raising.
        """
        run = self._get_or_start_run()
        params_list = []
        for key, value in params.items():
            params_list.append(entities.Param(key, str(value)))
        try:
            self._sql_client.modify_run_add_params(
                experiment_name=run.experiment_name,
                run_name=run.name,
                params=json.dumps([param.to_dict() for param in params_list]),
            )
        except snowpark.exceptions.SnowparkSQLException as e:
            if e.sql_error_code == 400003:  # EXPERIMENT_RUN_PROPERTY_SIZE_LIMIT_EXCEEDED
                run._warn_about_run_metadata_size(e.message)
            else:
                raise

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log an artifact or a directory of artifacts under the current run. If no run is active, this method will create
        a new run.

        Args:
            local_path: The path to the local file or directory to write.
            artifact_path: The directory within the run directory to write the artifacts to. If None, the artifacts will
                be logged in the root directory of the run.
        """
        run = self._get_or_start_run()
        for file_path, dest_artifact_path in artifact.get_put_path_pairs(local_path, artifact_path or ""):
            self._sql_client.put_artifact(
                experiment_name=run.experiment_name,
                run_name=run.name,
                artifact_path=dest_artifact_path,
                file_path=file_path,
            )

    def list_artifacts(
        self,
        run_name: str,
        artifact_path: Optional[str] = None,
    ) -> list[artifact.ArtifactInfo]:
        """
        List artifacts for a given run within the current experiment.

        Args:
            run_name: Name of the run to list artifacts from.
            artifact_path: Optional subdirectory within the run's artifact directory to scope the listing.
                If None, lists from the root of the run's artifact directory.

        Returns:
            A list of artifact entries under the specified path.

        Raises:
            RuntimeError: If no experiment is currently set.
        """
        if not self._experiment:
            raise RuntimeError("No experiment set. Please set an experiment before listing artifacts.")

        return self._sql_client.list_artifacts(
            experiment_name=self._experiment.name,
            run_name=sql_identifier.SqlIdentifier(run_name),
            artifact_path=artifact_path or "",
        )

    def download_artifacts(
        self,
        run_name: str,
        artifact_path: Optional[str] = None,
        target_path: Optional[str] = None,
    ) -> None:
        """
        Download artifacts from a run to a local directory.

        Args:
            run_name: Name of the run to download artifacts from.
            artifact_path: Optional path to file or subdirectory within the run's artifact directory to download.
                If None, downloads all artifacts from the root of the run's artifact directory.
            target_path: Optional local directory to download files into. If None, downloads into the
                current working directory.

        Raises:
            RuntimeError: If no experiment is currently set.
        """
        if not self._experiment:
            raise RuntimeError("No experiment set. Please set an experiment before downloading artifacts.")

        artifacts = self.list_artifacts(run_name=run_name, artifact_path=artifact_path or "")
        for relative_path, local_dir in artifact.get_download_path_pairs(artifacts, target_path or ""):
            self._sql_client.get_artifact(
                experiment_name=self._experiment.name,
                run_name=sql_identifier.SqlIdentifier(run_name),
                artifact_path=relative_path,
                target_path=local_dir,
            )

    def _get_or_set_experiment(self) -> entities.Experiment:
        if self._experiment:
            return self._experiment
        return self.set_experiment(experiment_name=DEFAULT_EXPERIMENT_NAME)

    def _get_or_start_run(self) -> entities.Run:
        if self._run:
            return self._run
        return self.start_run()

    def _generate_run_name(self, experiment: entities.Experiment) -> sql_identifier.SqlIdentifier:
        generator = hrid_generator.HRID16()
        existing_runs = self._sql_client.show_runs_in_experiment(experiment_name=experiment.name)
        existing_run_names = [row[sql_client.RUN_NAME_COL_NAME] for row in existing_runs]
        for _ in range(1000):
            run_name = generator.generate()[1]
            if run_name not in existing_run_names:
                return sql_identifier.SqlIdentifier(run_name)
        raise RuntimeError("Random run name generation failed.")

    def _update_database_and_schema(
        self, database_name: sql_identifier.SqlIdentifier, schema_name: sql_identifier.SqlIdentifier
    ) -> None:
        self._database_name = database_name
        self._schema_name = schema_name
        self._sql_client = sql_client.ExperimentTrackingSQLClient(
            session=self._session,
            database_name=database_name,
            schema_name=schema_name,
        )
        self._registry = registry.Registry(
            session=self._session,
            database_name=database_name,
            schema_name=schema_name,
        )

    def _print_urls(
        self,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
        scheme: str = "https",
        host: str = "app.snowflake.com",
    ) -> None:

        experiment_url = (
            f"{scheme}://{host}/_deeplink/#/experiments"
            f"/databases/{quote(str(self._database_name))}"
            f"/schemas/{quote(str(self._schema_name))}"
            f"/experiments/{quote(str(experiment_name))}"
        )
        run_url = experiment_url + f"/runs/{quote(str(run_name))}"
        sys.stdout.write(f"🏃 View run {run_name} at: {run_url}\n")
        sys.stdout.write(f"🧪 View experiment at: {experiment_url}\n")
