import warnings
from typing import Any, cast
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml._internal import env_utils
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.registry._manager import model_parameter_reconciler
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import Session


class ModelParameterReconcilerTest(parameterized.TestCase):
    """Test cases for ModelParameterReconciler functionality."""

    def setUp(self) -> None:
        """Set up common test fixtures."""
        self.database_name = sql_identifier.SqlIdentifier("TEST_DB")
        self.schema_name = sql_identifier.SqlIdentifier("TEST_SCHEMA")

    def _create_reconciler(self, **kwargs: Any) -> model_parameter_reconciler.ModelParameterReconciler:
        """Helper to create reconciler with default context."""
        mock_session = kwargs.get("session", mock.MagicMock())
        return model_parameter_reconciler.ModelParameterReconciler(
            session=mock_session,
            database_name=kwargs.get("database_name", self.database_name),
            schema_name=kwargs.get("schema_name", self.schema_name),
            conda_dependencies=kwargs.get("conda_dependencies"),
            pip_requirements=kwargs.get("pip_requirements"),
            target_platforms=kwargs.get("target_platforms"),
            artifact_repository_map=kwargs.get("artifact_repository_map"),
            options=kwargs.get("options"),
            python_version=kwargs.get("python_version"),
            statement_params=kwargs.get("statement_params"),
        )

    def test_artifact_repository_map_none(self) -> None:
        """Test that None artifact_repository_map returns None."""
        reconciler = self._create_reconciler(artifact_repository_map=None)
        result = reconciler.reconcile()
        self.assertIsNone(result.artifact_repository_map)

    def test_artifact_repository_map_transformation(self) -> None:
        """Test artifact_repository_map transformation to fully qualified names."""
        reconciler = self._create_reconciler(
            artifact_repository_map={"pip": "my_repo", "conda": "OTHER_DB.PUBLIC.conda_repo"}
        )
        result = reconciler.reconcile()

        expected = {"pip": "TEST_DB.TEST_SCHEMA.MY_REPO", "conda": "OTHER_DB.PUBLIC.CONDA_REPO"}
        self.assertEqual(result.artifact_repository_map, expected)

    def test_save_location_none_options(self) -> None:
        """Test that None options returns None save_location."""
        reconciler = self._create_reconciler(options=None)
        result = reconciler.reconcile()
        self.assertIsNone(result.save_location)

    def test_save_location_extraction(self) -> None:
        """Test save_location extraction from options."""
        reconciler = self._create_reconciler(options={"save_location": "/tmp/my_model", "enable_explainability": True})
        result = reconciler.reconcile()
        self.assertEqual(result.save_location, "/tmp/my_model")

    def test_save_location_missing_from_options(self) -> None:
        """Test that options without save_location returns None."""
        reconciler = self._create_reconciler(options={"enable_explainability": True, "relax_version": False})
        result = reconciler.reconcile()
        self.assertIsNone(result.save_location)

    def test_targets_warehouse(self) -> None:
        """Test _targets_warehouse method with various target platform configurations."""
        self.assertTrue(model_parameter_reconciler.ModelParameterReconciler._targets_warehouse(None))
        self.assertFalse(model_parameter_reconciler.ModelParameterReconciler._targets_warehouse([]))
        self.assertFalse(
            model_parameter_reconciler.ModelParameterReconciler._targets_warehouse(
                [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]
            )
        )
        self.assertTrue(
            model_parameter_reconciler.ModelParameterReconciler._targets_warehouse(
                [model_types.TargetPlatform.WAREHOUSE]
            )
        )
        self.assertTrue(
            model_parameter_reconciler.ModelParameterReconciler._targets_warehouse(
                [
                    model_types.TargetPlatform.WAREHOUSE,
                    model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
                ]
            )
        )
        self.assertTrue(model_parameter_reconciler.ModelParameterReconciler._targets_warehouse(["WAREHOUSE"]))
        self.assertFalse(
            model_parameter_reconciler.ModelParameterReconciler._targets_warehouse(["SNOWPARK_CONTAINER_SERVICES"])
        )
        self.assertTrue(
            model_parameter_reconciler.ModelParameterReconciler._targets_warehouse(
                ["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"]
            )
        )

    def test_pip_requirements_warehouse_warnings(self) -> None:
        """Test pip_requirements warnings for various warehouse targeting scenarios."""
        reconciler = self._create_reconciler(
            pip_requirements=["pandas>=1.3.0"],
            artifact_repository_map=None,
            target_platforms=None,
        )
        with self.assertWarnsRegex(
            UserWarning, "Models logged specifying `pip_requirements` cannot be executed in a Snowflake Warehouse"
        ):
            reconciler.reconcile()

        reconciler = self._create_reconciler(
            pip_requirements=["pandas>=1.3.0"],
            artifact_repository_map=None,
            target_platforms=[model_types.TargetPlatform.WAREHOUSE],
        )
        with self.assertWarnsRegex(
            UserWarning, "Models logged specifying `pip_requirements` cannot be executed in a Snowflake Warehouse"
        ):
            reconciler.reconcile()

        reconciler = self._create_reconciler(
            pip_requirements=["pandas>=1.3.0"],
            artifact_repository_map=None,
            target_platforms=["WAREHOUSE"],
        )
        with self.assertWarnsRegex(
            UserWarning, "Models logged specifying `pip_requirements` cannot be executed in a Snowflake Warehouse"
        ):
            reconciler.reconcile()

        reconciler = self._create_reconciler(
            pip_requirements=["pandas>=1.3.0"],
            artifact_repository_map=None,
            target_platforms=[
                model_types.TargetPlatform.WAREHOUSE,
                model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
            ],
        )
        with self.assertWarnsRegex(
            UserWarning, "Models logged specifying `pip_requirements` cannot be executed in a Snowflake Warehouse"
        ):
            reconciler.reconcile()

    def test_pip_requirements_no_warnings(self) -> None:
        """Test scenarios where pip_requirements warnings are not raised."""

        reconciler = self._create_reconciler(
            pip_requirements=["pandas>=1.3.0"],
            artifact_repository_map={"pip": "my_repo"},
            target_platforms=None,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reconciler.reconcile()
            pip_warnings = [
                warning for warning in w if "Models logged specifying `pip_requirements`" in str(warning.message)
            ]
            self.assertEqual(len(pip_warnings), 0)

        reconciler = self._create_reconciler(
            pip_requirements=None,
            artifact_repository_map=None,
            target_platforms=None,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reconciler.reconcile()
            pip_warnings = [
                warning for warning in w if "Models logged specifying `pip_requirements`" in str(warning.message)
            ]
            self.assertEqual(len(pip_warnings), 0)

        reconciler = self._create_reconciler(
            pip_requirements=["pandas>=1.3.0"],
            artifact_repository_map=None,
            target_platforms=[model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reconciler.reconcile()
            pip_warnings = [
                warning for warning in w if "Models logged specifying `pip_requirements`" in str(warning.message)
            ]
            self.assertEqual(len(pip_warnings), 0)

        reconciler = self._create_reconciler(
            pip_requirements=["pandas>=1.3.0"],
            artifact_repository_map={"pip": "my_repo"},
            target_platforms=[model_types.TargetPlatform.WAREHOUSE],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reconciler.reconcile()
            pip_warnings = [
                warning for warning in w if "Models logged specifying `pip_requirements`" in str(warning.message)
            ]
            self.assertEqual(len(pip_warnings), 0)

    def test_has_table_function(self) -> None:
        """Test _has_table_function method with various options configurations."""
        reconciler = self._create_reconciler(options=None)
        self.assertFalse(reconciler._has_table_function())

        reconciler = self._create_reconciler(options={})
        self.assertFalse(reconciler._has_table_function())

        reconciler = self._create_reconciler(options={"function_type": "TABLE_FUNCTION"})
        self.assertTrue(reconciler._has_table_function())

        reconciler = self._create_reconciler(
            options={
                "method_options": {
                    "predict": {"function_type": "FUNCTION"},
                    "predict_proba": {"function_type": "TABLE_FUNCTION"},
                }
            }
        )
        self.assertTrue(reconciler._has_table_function())

        reconciler = self._create_reconciler(
            options={
                "method_options": {
                    "predict": {"function_type": "FUNCTION"},
                    "predict_proba": {"function_type": "FUNCTION"},
                }
            }
        )
        self.assertFalse(reconciler._has_table_function())

    def test_reconcile_target_platforms_user_specified(self) -> None:
        """Test _reconcile_target_platforms with user-specified platforms."""
        reconciler = self._create_reconciler(target_platforms=["WAREHOUSE"])
        result = reconciler.reconcile()
        self.assertEqual(result.target_platforms, [model_types.TargetPlatform.WAREHOUSE])

        reconciler = self._create_reconciler(target_platforms=[model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES])
        result = reconciler.reconcile()
        self.assertEqual(result.target_platforms, [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES])

        reconciler = self._create_reconciler(
            target_platforms=["WAREHOUSE", model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]
        )
        result = reconciler.reconcile()
        self.assertEqual(
            result.target_platforms,
            [model_types.TargetPlatform.WAREHOUSE, model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
        )

    def test_reconcile_target_platforms_defaults(self) -> None:
        """Test _reconcile_target_platforms default behavior in various scenarios."""
        with mock.patch("snowflake.ml._internal.env.IN_ML_RUNTIME", True):
            reconciler = self._create_reconciler(target_platforms=None, options=None)
            result = reconciler.reconcile()
            self.assertEqual(result.target_platforms, [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES])

        with mock.patch("snowflake.ml._internal.env.IN_ML_RUNTIME", False):
            reconciler = self._create_reconciler(target_platforms=None, options={"function_type": "TABLE_FUNCTION"})
            result = reconciler.reconcile()
            self.assertEqual(result.target_platforms, [model_types.TargetPlatform.WAREHOUSE])

        with mock.patch("snowflake.ml._internal.env.IN_ML_RUNTIME", False):
            reconciler = self._create_reconciler(target_platforms=None, options=None)
            result = reconciler.reconcile()
            self.assertIsNone(result.target_platforms)

    def test_is_warehouse_runnable(self) -> None:
        """Test _is_warehouse_runnable logic using conda channels and pip requirements."""
        reconciler = self._create_reconciler()
        self.assertTrue(reconciler._is_warehouse_runnable({}))
        self.assertTrue(reconciler._is_warehouse_runnable({"": ["pkg1"]}))
        self.assertTrue(reconciler._is_warehouse_runnable({"https://repo.anaconda.com/pkgs/snowflake": ["pkg1"]}))

        self.assertFalse(reconciler._is_warehouse_runnable({"conda-forge": ["pkg1"]}))

        reconciler = self._create_reconciler(pip_requirements=["numpy"])
        self.assertFalse(reconciler._is_warehouse_runnable({}))

    def test_explainability_validation(self) -> None:
        """Test explainability validation logic for different platform configurations."""

        reconciler = self._create_reconciler(target_platforms=[model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES])
        result = reconciler.reconcile()
        assert result.options is not None
        self.assertEqual(result.options["enable_explainability"], False)

        reconciler = self._create_reconciler(
            target_platforms=[
                model_types.TargetPlatform.WAREHOUSE,
                model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
            ],
            options={"enable_explainability": True},
        )
        with self.assertWarnsRegex(
            UserWarning, "Explain function will only be available for model deployed to warehouse"
        ):
            result = reconciler.reconcile()
        assert result.options is not None
        self.assertEqual(result.options["enable_explainability"], True)

        reconciler = self._create_reconciler(
            conda_dependencies=["conda-forge::python-package==1.0.0"],
            target_platforms=[model_types.TargetPlatform.WAREHOUSE],
        )
        result = reconciler.reconcile()
        assert result.options is not None
        self.assertEqual(result.options["enable_explainability"], False)

        reconciler = self._create_reconciler(
            conda_dependencies=["conda-forge::python-package==1.0.0"],
            target_platforms=[model_types.TargetPlatform.WAREHOUSE],
            options={"enable_explainability": True},
        )
        with self.assertRaisesRegex(ValueError, "`enable_explainability` cannot be set to True.*not runnable in WH"):
            reconciler.reconcile()

    def test_embed_local_ml_library_logic(self) -> None:
        """Test embed_local_ml_library auto-setting logic."""
        with mock.patch.object(env_utils, "get_matched_package_versions_in_information_schema") as mock_get_versions:

            mock_get_versions.return_value = {}

            reconciler = self._create_reconciler(
                target_platforms=[model_types.TargetPlatform.WAREHOUSE], options={"embed_local_ml_library": False}
            )
            result = reconciler.reconcile()
            assert result.options is not None
            self.assertTrue(result.options["embed_local_ml_library"])

            mock_get_versions.return_value = {"snowflake-ml-python": ["1.0.0"]}
            reconciler = self._create_reconciler(
                target_platforms=[model_types.TargetPlatform.WAREHOUSE], options={"embed_local_ml_library": False}
            )
            result = reconciler.reconcile()
            assert result.options is not None
            self.assertFalse(result.options["embed_local_ml_library"])

            reconciler = self._create_reconciler(
                target_platforms=[model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
                options={"embed_local_ml_library": False},
            )
            result = reconciler.reconcile()
            assert result.options is not None
            self.assertFalse(result.options["embed_local_ml_library"])

    def test_relax_version_logic(self) -> None:
        """Test relax_version auto-setting and validation logic."""

        reconciler = self._create_reconciler(pip_requirements=["xgboost==1.2.3"])
        with mock.patch.object(model_parameter_reconciler.logger, "info") as mock_info:
            result = reconciler.reconcile()
            assert result.options is not None
            self.assertFalse(result.options["relax_version"])
            mock_info.assert_called_with(
                "Setting `relax_version=False` as this model will run in Snowpark Container Services "
                "or in Warehouse with a specified artifact_repository_map where exact version "
                " specifications will be honored."
            )

        reconciler = self._create_reconciler(target_platforms=[model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES])
        with mock.patch.object(model_parameter_reconciler.logger, "info") as mock_info:
            result = reconciler.reconcile()
            assert result.options is not None
            self.assertFalse(result.options["relax_version"])
            mock_info.assert_called_with(
                "Setting `relax_version=False` as this model will run in Snowpark Container Services "
                "or in Warehouse with a specified artifact_repository_map where exact version "
                " specifications will be honored."
            )

        reconciler = self._create_reconciler()
        with self.assertWarnsRegex(UserWarning, "`relax_version` is not set and therefore defaulted to True"):
            result = reconciler.reconcile()
            assert result.options is not None
            self.assertTrue(result.options["relax_version"])

        reconciler = self._create_reconciler(pip_requirements=["xgboost==1.2.3"], options=None)
        with mock.patch.object(model_parameter_reconciler.logger, "info") as mock_info:
            result = reconciler.reconcile()
            assert result.options is not None
            self.assertFalse(result.options["relax_version"])
            mock_info.assert_called_with(
                "Setting `relax_version=False` as this model will run in Snowpark Container Services "
                "or in Warehouse with a specified artifact_repository_map where exact version "
                " specifications will be honored."
            )

        reconciler = self._create_reconciler(options={"relax_version": False})
        result = reconciler.reconcile()
        assert result.options is not None
        self.assertFalse(result.options["relax_version"])

        reconciler = self._create_reconciler(options={"relax_version": True})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = reconciler.reconcile()
            assert result.options is not None
            self.assertTrue(result.options["relax_version"])

        reconciler = self._create_reconciler(pip_requirements=["xgboost==1.2.3"], options={"relax_version": True})
        with self.assertRaises(exceptions.SnowflakeMLException) as cm:
            reconciler.reconcile()
        self.assertEqual(cm.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn(
            "Setting `relax_version=True` is only allowed for models to be run in Warehouse with "
            "Snowflake Conda Channel dependencies",
            str(cm.exception),
        )

        reconciler = self._create_reconciler(
            target_platforms=[model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES], options={"relax_version": True}
        )
        with self.assertRaises(exceptions.SnowflakeMLException) as cm:
            reconciler.reconcile()
        self.assertEqual(cm.exception.error_code, error_codes.INVALID_ARGUMENT)
        self.assertIn(
            "Setting `relax_version=True` is only allowed for models to be run in Warehouse with "
            "Snowflake Conda Channel dependencies",
            str(cm.exception),
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"disable_explainability": True, "target_platforms": [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]},
        {
            "disable_explainability": False,
            "target_platforms": [
                model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
                model_types.TargetPlatform.WAREHOUSE,
            ],
        },
        {"disable_explainability": False, "target_platforms": []},
        {
            "disable_explainability": True,
            "conda_dependencies": ["python-package1==1.0.0", "conda-forge::python-package2==1.1.0"],
        },
        {
            "disable_explainability": False,
            "conda_dependencies": [
                "python-package1==1.0.0",
                "https://repo.anaconda.com/pkgs/snowflake::python-package2",
            ],
        },
        {"disable_explainability": True, "pip_requirements": ["python-package==1.0.0"]},
        {"disable_explainability": False, "pip_requirements": None},
    )
    def test_explainability_parameter_reconciliation(self, disable_explainability: bool, **kwargs: Any) -> None:
        """Test explainability parameter reconciliation matching original model_composer test structure."""
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(Session, m_session)

        reconciler = model_parameter_reconciler.ModelParameterReconciler(
            session=c_session,
            database_name=self.database_name,
            schema_name=self.schema_name,
            conda_dependencies=kwargs.get("conda_dependencies"),
            pip_requirements=kwargs.get("pip_requirements"),
            target_platforms=kwargs.get("target_platforms"),
            options=None,
        )

        if kwargs.get("conda_dependencies") == ["python-package1==1.0.0", "conda-forge::python-package2==1.1.0"]:
            mock_conda_result = {"conda-forge": ["python-package2"]}
        else:
            mock_conda_result = {}

        with mock.patch.object(
            env_utils, "validate_conda_dependency_string_list", return_value=mock_conda_result
        ) as mock_validate_conda:
            with mock.patch.object(
                env_utils,
                "get_matched_package_versions_in_information_schema",
                return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
            ) as mock_get_versions:

                result = reconciler.reconcile()

                mock_validate_conda.assert_called_once()
                if kwargs.get("target_platforms") != [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]:
                    mock_get_versions.assert_called_once()
                else:
                    mock_get_versions.assert_not_called()

                if disable_explainability:
                    assert result.options is not None
                    self.assertEqual(result.options["enable_explainability"], False)
                else:
                    assert result.options is not None
                    self.assertNotIn("enable_explainability", result.options)

        if disable_explainability:
            reconciler = model_parameter_reconciler.ModelParameterReconciler(
                session=c_session,
                database_name=self.database_name,
                schema_name=self.schema_name,
                conda_dependencies=kwargs.get("conda_dependencies"),
                pip_requirements=kwargs.get("pip_requirements"),
                target_platforms=kwargs.get("target_platforms"),
                options={"enable_explainability": True},
            )

            with mock.patch.object(env_utils, "validate_conda_dependency_string_list", return_value=mock_conda_result):
                with mock.patch.object(
                    env_utils,
                    "get_matched_package_versions_in_information_schema",
                    return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
                ):

                    with self.assertRaisesRegex(
                        ValueError,
                        "`enable_explainability` cannot be set to True when the model is not runnable in WH "
                        "or the target platforms include SPCS.",
                    ):
                        reconciler.reconcile()

    @parameterized.parameters(  # type: ignore[misc]
        {"target_platforms": [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]},
        {
            "target_platforms": [
                model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
                model_types.TargetPlatform.WAREHOUSE,
            ]
        },
    )
    def test_embed_ml_library_information_schema_check(
        self, target_platforms: list[model_types.TargetPlatform]
    ) -> None:
        """Test embed_local_ml_library information schema check matching original model_composer test structure."""
        m_session = mock_session.MockSession(conn=None, test_case=self)
        c_session = cast(Session, m_session)

        reconciler = model_parameter_reconciler.ModelParameterReconciler(
            session=c_session,
            database_name=self.database_name,
            schema_name=self.schema_name,
            conda_dependencies=None,
            pip_requirements=None,
            target_platforms=cast(list[model_types.SupportedTargetPlatformType], target_platforms),
            options=None,
        )

        with mock.patch.object(env_utils, "validate_conda_dependency_string_list", return_value={}):
            with mock.patch.object(
                env_utils,
                "get_matched_package_versions_in_information_schema",
                return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
            ) as mock_get_versions:

                reconciler.reconcile()

                if target_platforms == [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]:
                    mock_get_versions.assert_not_called()
                else:
                    mock_get_versions.assert_called_once()


if __name__ == "__main__":
    absltest.main()
