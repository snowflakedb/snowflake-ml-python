import inflection
import pandas as pd
import registry_model_test_base
import xgboost
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection

from snowflake.ml.model import type_hints


class TestRegistryTargetPlatformsInteg(registry_model_test_base.RegistryModelTestBase):
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
        regressor = xgboost.XGBRegressor(n_estimators=10, reg_lambda=1, gamma=0, max_depth=3, n_jobs=1)
        regressor.fit(cal_X_train, cal_y_train)

        def _check_predict_fn(res: pd.DataFrame) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(regressor.predict(cal_X_test), columns=res.columns),
                check_dtype=False,
            )

        artifact_repository_map = None

        # Use public PyPI shared repository if needed for this test case
        if target_platforms_and_dependency_combinations["artifact_repository_map"]:
            artifact_repository_map = {"pip": "snowflake.snowpark.pypi_shared_repository"}

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


if __name__ == "__main__":
    absltest.main()
