import uuid

import inflection
import pandas as pd
import registry_model_test_base
import xgboost
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection

from snowflake.ml.model import type_hints


class TestRegistryTargetPlatformsInteg(registry_model_test_base.RegistryModelTestBase):
    def _create_artifact_repository(self) -> tuple[str, str, str]:
        """Create temporary API integration and artifact repository for testing.

        Returns:
            Tuple of (api_integration_name, artifact_repo_name, artifact_repository_map_value)
        """
        session = self.registry._model_manager._model_ops._session
        schema = session.get_current_schema()
        db = session.get_current_database()

        # Create unique names for API integration and artifact repository
        unique_id = str(uuid.uuid4()).replace("-", "_")[:8]
        api_integration_name = f"test_pypi_api_integration_{unique_id}"
        artifact_repo_name = f"test_artifact_repo_{unique_id}"

        try:
            # Create API integration
            session.sql(
                f"CREATE API INTEGRATION {api_integration_name} " "API_PROVIDER = PYPI " "ENABLED = TRUE"
            ).collect()

            # Create artifact repository
            session.sql(
                f"CREATE ARTIFACT REPOSITORY {db}.{schema}.{artifact_repo_name} "
                f"API_INTEGRATION = {api_integration_name} "
                "TYPE = 'pip'"
            ).collect()

            return api_integration_name, artifact_repo_name, f"{db}.{schema}.{artifact_repo_name}"

        except Exception as e:
            # Clean up on failure
            self._cleanup_artifact_repository(api_integration_name, artifact_repo_name)
            raise e

    def _cleanup_artifact_repository(self, api_integration_name: str, artifact_repo_name: str) -> None:
        """Clean up temporary API integration and artifact repository.

        Args:
            api_integration_name: Name of the API integration to drop
            artifact_repo_name: Name of the artifact repository to drop
        """
        session = self.registry._model_manager._model_ops._session
        schema = session.get_current_schema()
        db = session.get_current_database()

        if artifact_repo_name:
            try:
                session.sql(f"DROP ARTIFACT REPOSITORY IF EXISTS {db}.{schema}.{artifact_repo_name}").collect()
            except Exception as e:
                print(f"Warning: Failed to drop artifact repository {artifact_repo_name}: {e}")

        if api_integration_name:
            try:
                session.sql(f"DROP API INTEGRATION IF EXISTS {api_integration_name}").collect()
            except Exception as e:
                print(f"Warning: Failed to drop API integration {api_integration_name}: {e}")

    @parameterized.product(  # type: ignore[misc]
        target_platforms_and_dependency_combinations=[
            {
                "target_platforms": [type_hints.TargetPlatform.WAREHOUSE.value],
                "pip_requirements": None,
                "conda_dependencies": None,
                "artifact_repository_map": None,
                "expect_error": False,
            },
            {
                "target_platforms": [type_hints.TargetPlatform.SNOWPARK_CONTAINER_SERVICES.value],
                "pip_requirements": None,
                "conda_dependencies": None,
                "artifact_repository_map": None,
                "expect_error": False,
            },
            {
                "target_platforms": [type_hints.TargetPlatform.WAREHOUSE.value],
                "pip_requirements": ["prophet"],
                "conda_dependencies": None,
                "artifact_repository_map": None,
                "expect_error": True,
            },
            {
                "target_platforms": [type_hints.TargetPlatform.SNOWPARK_CONTAINER_SERVICES.value],
                "pip_requirements": ["prophet"],
                "conda_dependencies": None,
                "artifact_repository_map": None,
                "expect_error": False,
            },
            {
                "target_platforms": [type_hints.TargetPlatform.WAREHOUSE.value],
                "pip_requirements": None,
                "conda_dependencies": ["conda-forge::prophet"],
                "artifact_repository_map": False,
                "expect_error": True,
            },
            {
                "target_platforms": [type_hints.TargetPlatform.WAREHOUSE.value],
                "pip_requirements": None,
                "conda_dependencies": ["prophet"],
                "artifact_repository_map": False,
                "expect_error": False,
            },
            {
                "target_platforms": [type_hints.TargetPlatform.WAREHOUSE.value],
                "pip_requirements": ["prophet", "pandas==2.1.4"],  # Pin pandas version to override snowpark
                "conda_dependencies": None,
                "artifact_repository_map": True,
                "expect_error": False,
            },
        ]
    )
    def test_log_model_target_platforms_argument(
        self,
        target_platforms_and_dependency_combinations: dict[any, any],
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        regressor.fit(cal_X_train, cal_y_train)

        def _check_predict_fn(res: pd.DataFrame) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(regressor.predict(cal_X_test), columns=res.columns),
                check_dtype=False,
            )

        artifact_repository_map = None
        api_integration_name = None
        artifact_repo_name = None

        # Create artifact repository if needed for this test case
        if target_platforms_and_dependency_combinations["artifact_repository_map"]:
            api_integration_name, artifact_repo_name, repo_path = self._create_artifact_repository()
            artifact_repository_map = {"pip": repo_path}

        try:
            self._test_registry_model_target_platforms(
                model=regressor,
                sample_input_data=cal_X_test,
                target_platforms=target_platforms_and_dependency_combinations["target_platforms"],
                pip_requirements=target_platforms_and_dependency_combinations["pip_requirements"],
                additional_dependencies=target_platforms_and_dependency_combinations["conda_dependencies"],
                artifact_repository_map=artifact_repository_map,
                prediction_assert_fns={
                    "predict": (
                        cal_X_test,
                        _check_predict_fn,
                    ),
                },
                options={"enable_explainability": False},
                expect_error=target_platforms_and_dependency_combinations["expect_error"],
            )
        finally:
            # Clean up resources if they were created
            if target_platforms_and_dependency_combinations["artifact_repository_map"]:
                self._cleanup_artifact_repository(api_integration_name, artifact_repo_name)


if __name__ == "__main__":
    absltest.main()
