"""Integ: pip-only + Warehouse ``mv.run`` with reconciler-injected pip artifact repo (no user map)."""

import pandas as pd
import registry_model_test_base
from absl.testing import absltest

from snowflake.ml.model import custom_model, type_hints


class _PipOnlyRequestsModel(custom_model.CustomModel):
    """Minimal custom model that depends on a pip-only package (``requests``)."""

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        import requests  # noqa: F401

        return pd.DataFrame({"output": input["value"] * 2})


class TestRegistryPipOnlyWarehouseArtifactInjectionInteg(registry_model_test_base.RegistryModelTestBase):
    def test_pip_only_warehouse_inference_injected_artifact_repository_map(self) -> None:
        """Registry reconciler injects shared PyPI repo; user does not pass ``artifact_repository_map``."""
        import snowflake.ml.model.parameters.enable_pip_only_packaging  # noqa: F401

        test_input = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        model = _PipOnlyRequestsModel(custom_model.ModelContext())

        def _check_predict(res: pd.DataFrame) -> None:
            pd.testing.assert_frame_equal(
                res,
                model.predict(test_input),
                check_dtype=False,
            )

        self._test_registry_model_target_platforms(
            model=model,
            sample_input_data=test_input,
            target_platforms=[type_hints.TargetPlatform.WAREHOUSE.value],
            pip_requirements=["requests>=2.28.0"],
            additional_dependencies=None,
            conda_dependencies=[],
            artifact_repository_map=None,
            prediction_assert_fns={
                "predict": (
                    test_input,
                    _check_predict,
                ),
            },
            options={"enable_explainability": False},
            expect_error=False,
        )


if __name__ == "__main__":
    absltest.main()
