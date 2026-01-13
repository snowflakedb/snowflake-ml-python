import inspect
import json
import uuid
from typing import Any, Callable, Optional

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.registry import registry
from snowflake.snowpark import exceptions as sp_exceptions
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    test_env_utils,
)


class RegistryModelTestBase(common_test_base.CommonTestBase):
    REGISTRY_TEST_FN_LIST = ["_test_registry_model", "_test_registry_model_from_model_version"]

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()

        self._run_id = uuid.uuid4().hex
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()

        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self.session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        super().tearDown()

    def _test_registry_model(
        self,
        model: model_types.SupportedModelType,
        prediction_assert_fns: dict[str, tuple[Any, Callable[[Any], Any]]],
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        additional_dependencies: Optional[list[str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
        signatures: Optional[dict[str, model_signature.ModelSignature]] = None,
        additional_version_suffix: Optional[str] = None,
        function_type_assert: Optional[dict[str, model_manifest_schema.ModelMethodFunctionTypes]] = None,
        pip_requirements: Optional[list[str]] = None,
        artifact_repository_map: Optional[dict[str, str]] = None,
        resource_constraint: Optional[dict[str, str]] = None,
        code_paths: Optional[list[model_types.CodePathLike]] = None,
    ) -> None:
        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        if additional_dependencies:
            conda_dependencies.extend(additional_dependencies)

        version_suffix = self._run_id
        if additional_version_suffix:
            version_suffix = version_suffix + "_" + additional_version_suffix

        # Get the name of the caller as the model name
        name = f"model_{inspect.stack()[1].function}"
        version = f"ver_{version_suffix}"

        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=sample_input_data,
            conda_dependencies=conda_dependencies,
            options=options,
            signatures=signatures,
            artifact_repository_map=artifact_repository_map,
            resource_constraint=resource_constraint,
            pip_requirements=pip_requirements,
            code_paths=code_paths,
        )

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            res = mv.run(test_input, function_name=target_method)
            check_func(res)

        if function_type_assert:
            res = mv.show_functions()
            for f in res:
                if f["target_method"] in function_type_assert.keys():
                    self.assertEqual(f["target_method_function_type"], function_type_assert[f["target_method"]].value)

        self.registry.show_models()

        self.registry.delete_model(model_name=name)

        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    def _test_registry_model_target_platforms(
        self,
        model: model_types.SupportedModelType,
        prediction_assert_fns: dict[str, tuple[Any, Callable[[Any], Any]]],
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        additional_dependencies: Optional[list[str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
        pip_requirements: Optional[list[str]] = None,
        artifact_repository_map: Optional[dict[str, str]] = None,
        target_platforms: Optional[list[str]] = None,
        expect_error: Optional[bool] = False,
    ) -> None:
        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        if additional_dependencies:
            conda_dependencies.extend(additional_dependencies)

        version_suffix = self._run_id

        # Get the name of the caller as the model name
        name = f"model_{inspect.stack()[1].function}"
        version = f"ver_{version_suffix}"

        if expect_error:
            with self.assertRaises(sp_exceptions.SnowparkSQLException):
                _ = self.registry.log_model(
                    model=model,
                    model_name=name,
                    version_name=version,
                    sample_input_data=sample_input_data,
                    conda_dependencies=conda_dependencies,
                    options=options,
                    signatures=None,
                    target_platforms=target_platforms,
                    artifact_repository_map=artifact_repository_map,
                    pip_requirements=pip_requirements,
                )
            return
        else:
            mv = self.registry.log_model(
                model=model,
                model_name=name,
                version_name=version,
                sample_input_data=sample_input_data,
                conda_dependencies=conda_dependencies,
                options=options,
                signatures=None,
                target_platforms=target_platforms,
                artifact_repository_map=artifact_repository_map,
                pip_requirements=pip_requirements,
            )

        show_versions_res = self.session.sql(f"SHOW VERSIONS IN MODEL {mv.model_name}").collect()
        runnable_in = json.loads(show_versions_res[0]["runnable_in"])

        self.assertEqual(runnable_in, target_platforms)

        if "WAREHOUSE" in runnable_in:
            for target_method, (test_input, check_func) in prediction_assert_fns.items():
                res = mv.run(test_input, function_name=target_method)
                check_func(res)

        self.registry.delete_model(model_name=name)

        self.assertNotIn(mv.model_name, [m.name for m in self.registry.models()])

    def _test_registry_model_from_model_version(
        self,
        model: model_types.SupportedModelType,
        prediction_assert_fns: dict[str, tuple[Any, Callable[[Any], Any]]],
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        additional_dependencies: Optional[list[str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
        signatures: Optional[dict[str, model_signature.ModelSignature]] = None,
        additional_version_suffix: Optional[str] = None,
        function_type_assert: Optional[dict[str, model_manifest_schema.ModelMethodFunctionTypes]] = None,
        pip_requirements: Optional[list[str]] = None,
        artifact_repository_map: Optional[dict[str, str]] = None,
        resource_constraint: Optional[dict[str, str]] = None,
    ) -> None:
        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        if additional_dependencies:
            conda_dependencies.extend(additional_dependencies)

        version_suffix = self._run_id
        if additional_version_suffix:
            version_suffix = version_suffix + "_" + additional_version_suffix

        # Get the name of the caller as the model name
        source_name = f"source_model_{inspect.stack()[1].function}"
        name = f"model_{inspect.stack()[1].function}"
        source_version = f"source_ver_{version_suffix}"
        version = f"ver_{version_suffix}"
        source_mv = self.registry.log_model(
            model=model,
            model_name=source_name,
            version_name=source_version,
            sample_input_data=sample_input_data,
            conda_dependencies=conda_dependencies,
            options=options,
            signatures=signatures,
            artifact_repository_map=artifact_repository_map,
            resource_constraint=resource_constraint,
            pip_requirements=pip_requirements,
        )

        # Create a new model when the model doesn't exist
        mv = self.registry.log_model(
            model=source_mv,
            model_name=name,
            version_name=version,
        )

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            res = mv.run(test_input, function_name=target_method)
            check_func(res)

        if function_type_assert:
            res = mv.show_functions()
            for f in res:
                if f["target_method"] in function_type_assert.keys():
                    self.assertEqual(f["target_method_function_type"], function_type_assert[f["target_method"]].value)

        self.registry.show_models()

        # Add a version when the model exists
        version2 = f"ver_{version_suffix}_2"
        mv2 = self.registry.log_model(
            model=source_mv,
            model_name=name,
            version_name=version2,
        )

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            res = mv2.run(test_input, function_name=target_method)
            check_func(res)

        if function_type_assert:
            res = mv2.show_functions()
            for f in res:
                if f["target_method"] in function_type_assert.keys():
                    self.assertEqual(f["target_method_function_type"], function_type_assert[f["target_method"]].value)

        self.registry.show_models()
        self.registry.delete_model(model_name=name)
        self.assertNotIn(mv2.model_name, [m.name for m in self.registry.models()])
