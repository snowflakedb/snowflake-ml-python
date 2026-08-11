"""Integ: log model + warehouse inference using an external test PyPI artifact repository."""

import logging

import pandas as pd
import registry_model_test_base
from absl.testing import absltest

from snowflake.ml.model import custom_model, model_signature
from tests.integ.snowflake.ml.registry import artifact_repository_integ_util as util

logger = logging.getLogger(__name__)

# Package published only on the external test PyPI index.
_TEST_WHL_PACKAGE = "test-whl-package"


class _TestWhlPackageModel(custom_model.CustomModel):
    """Custom model that imports ``test-whl-package`` at inference time.

    The package is not installed in the local test env; provide an explicit
    ``signatures=`` at log time so packaging does not invoke ``predict`` locally.
    """

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        import test_whl_package  # noqa: F401

        return pd.DataFrame({"output": input["value"] * 2})


class TestRegistryExternalTestPypiArtifactRepositoryInteg(registry_model_test_base.RegistryModelTestBase):
    def test_log_model_and_infer_with_external_test_pypi(self) -> None:
        """Create external test PyPI repo, log a model that needs ``test-whl-package``, run inference."""
        if util.get_private_pypi_credentials() is None:
            skip_reason = f"{util.PRIVATE_PYPI_USERNAME_ENV} / {util.PRIVATE_PYPI_PASSWORD_ENV} not set"
            logger.warning(
                "Skipping external test PyPI artifact repository integ test: %s",
                skip_reason,
            )
            self.skipTest(skip_reason)

        test_input = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        model = _TestWhlPackageModel(custom_model.ModelContext())
        # Explicit signature avoids local predict (and thus local import of the private wheel).
        sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="value", dtype=model_signature.DataType.DOUBLE)],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.DOUBLE)],
        )

        def _check_predict(res: pd.DataFrame) -> None:
            pd.testing.assert_frame_equal(
                res,
                pd.DataFrame({"output": test_input["value"] * 2}),
                check_dtype=False,
            )

        with util.external_test_pypi_artifact_repository(
            self.session,
            database=self._test_db,
            schema=self._test_schema,
            run_id=self._run_id,
        ) as repo:
            self._test_registry_model(
                model=model,
                sample_input_data=test_input,
                signatures={"predict": sig},
                pip_requirements=[_TEST_WHL_PACKAGE],
                artifact_repository_map=repo.artifact_repository_map,
                prediction_assert_fns={
                    "predict": (
                        test_input,
                        _check_predict,
                    ),
                },
                options={"enable_explainability": False},
            )


if __name__ == "__main__":
    absltest.main()
