import warnings
from typing import Any

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.registry._manager import model_parameter_reconciler


class ModelParameterReconcilerTest(absltest.TestCase):
    """Test cases for ModelParameterReconciler functionality."""

    def setUp(self) -> None:
        """Set up common test fixtures."""
        self.database_name = sql_identifier.SqlIdentifier("TEST_DB")
        self.schema_name = sql_identifier.SqlIdentifier("TEST_SCHEMA")

    def _create_reconciler(self, **kwargs: Any) -> model_parameter_reconciler.ModelParameterReconciler:
        """Helper to create reconciler with default context."""
        return model_parameter_reconciler.ModelParameterReconciler(
            database_name=kwargs.get("database_name", self.database_name),
            schema_name=kwargs.get("schema_name", self.schema_name),
            conda_dependencies=kwargs.get("conda_dependencies"),
            pip_requirements=kwargs.get("pip_requirements"),
            target_platforms=kwargs.get("target_platforms"),
            artifact_repository_map=kwargs.get("artifact_repository_map"),
            options=kwargs.get("options"),
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

    def test_parameter_passthrough(self) -> None:
        """Test that other parameters are passed through unchanged."""
        conda_deps = ["numpy==1.21.0"]
        pip_reqs = ["pandas>=1.3.0"]
        target_platforms = [model_types.TargetPlatform.WAREHOUSE]

        reconciler = self._create_reconciler(
            conda_dependencies=conda_deps, pip_requirements=pip_reqs, target_platforms=target_platforms
        )
        result = reconciler.reconcile()

        self.assertEqual(result.conda_dependencies, conda_deps)
        self.assertEqual(result.pip_requirements, pip_reqs)
        self.assertEqual(result.target_platforms, target_platforms)

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
        """Test scenarios where warnings are not raised from pip_requirements."""

        reconciler = self._create_reconciler(
            pip_requirements=["pandas>=1.3.0"],
            artifact_repository_map={"pip": "my_repo"},
            target_platforms=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reconciler.reconcile()

        reconciler = self._create_reconciler(
            pip_requirements=None,
            artifact_repository_map=None,
            target_platforms=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reconciler.reconcile()

        reconciler = self._create_reconciler(
            pip_requirements=["pandas>=1.3.0"],
            artifact_repository_map=None,
            target_platforms=[model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reconciler.reconcile()

        reconciler = self._create_reconciler(
            pip_requirements=["pandas>=1.3.0"],
            artifact_repository_map={"pip": "my_repo"},
            target_platforms=[model_types.TargetPlatform.WAREHOUSE],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reconciler.reconcile()


if __name__ == "__main__":
    absltest.main()
