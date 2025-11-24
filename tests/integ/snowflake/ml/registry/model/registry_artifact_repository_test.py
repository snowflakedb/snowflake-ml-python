import inflection
import pandas as pd
import registry_model_test_base
import xgboost
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection


class TestRegistryArtifactRepositoryInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
        test_artifact_repository=[False, True],
    )
    def test_log_model_artifact_repository_argument(
        self,
        registry_test_fn: str,
        test_artifact_repository: bool,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)
        regressor = xgboost.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, n_jobs=1)
        regressor.fit(cal_X_train, cal_y_train)

        def _check_predict_fn(res: pd.DataFrame) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame(regressor.predict(cal_X_test), columns=res.columns),
                check_dtype=False,
            )

        artifact_repository_map = None

        if test_artifact_repository:
            artifact_repository_map = {"pip": "snowflake.snowpark.pypi_shared_repository"}

        getattr(self, registry_test_fn)(
            model=regressor,
            sample_input_data=cal_X_test,
            pip_requirements=["rich"] if artifact_repository_map else None,
            artifact_repository_map=artifact_repository_map,
            prediction_assert_fns={
                "predict": (
                    cal_X_test,
                    _check_predict_fn,
                ),
            },
            options={"enable_explainability": False},
        )


if __name__ == "__main__":
    absltest.main()
