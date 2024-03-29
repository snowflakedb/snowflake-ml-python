from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml
from absl.testing import absltest

from snowflake.ml.model import model_signature
from snowflake.ml.registry import model_registry
from snowflake.snowpark import DataFrame as SnowparkDataFrame
from tests.integ.snowflake.ml.test_utils import (
    model_factory,
    spcs_integ_test_base,
    test_env_utils,
)


def is_valid_yaml(yaml_string) -> bool:
    try:
        yaml.safe_load(yaml_string)
        return True
    except yaml.YAMLError:
        return False


class TestModelRegistryIntegSnowServiceBase(spcs_integ_test_base.SpcsIntegTestBase):
    def setUp(self) -> None:
        super().setUp()
        model_registry.create_model_registry(
            session=self._session, database_name=self._test_db, schema_name=self._test_schema
        )
        self.registry = model_registry.ModelRegistry(
            session=self._session, database_name=self._test_db, schema_name=self._test_schema
        )

    def tearDown(self) -> None:
        super().tearDown()

    def _test_snowservice_deployment(
        self,
        model_name: str,
        model_version: str,
        prepare_model_and_feature_fn: Callable[[], Tuple[Any, Any, Any]],
        deployment_options: Dict[str, Any],
        prediction_assert_fn: Callable[[Any, Union[pd.DataFrame, SnowparkDataFrame]], Any],
        pip_requirements: Optional[List[str]] = None,
        conda_dependencies: Optional[List[str]] = None,
        embed_local_ml_library: Optional[bool] = True,
        omit_target_method_when_deploy: bool = False,
    ) -> None:
        model, test_features, *_ = prepare_model_and_feature_fn()
        if omit_target_method_when_deploy:
            target_method = deployment_options.pop("target_method")
        else:
            target_method = deployment_options["target_method"]

        if hasattr(model, "predict_with_device"):
            local_prediction = model.predict_with_device(test_features, model_factory.DEVICE.CPU)
        else:
            local_prediction = getattr(model, target_method)(test_features)

        # In test, latest snowpark version might not be in conda channel yet, which can cause image build to fail.
        # Instead we rely on snowpark version on information.schema table. Note that this will not affect end user
        # as by the time they use it, the latest snowpark should be available in conda already.
        conda_dependencies = conda_dependencies or []
        conda_dependencies.append(test_env_utils.get_latest_package_version_spec_in_conda("snowflake-snowpark-python"))

        self.registry.log_model(
            model_name=model_name,
            model_version=model_version,
            model=model,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            signatures={target_method: model_signature.infer_signature(test_features, local_prediction)},
            options={"embed_local_ml_library": embed_local_ml_library},
        )

        model_ref = model_registry.ModelReference(
            registry=self.registry, model_name=model_name, model_version=model_version
        )

        deployment_name = f"{model_name}_{model_version}_deployment"
        deployment_options["deployment_name"] = deployment_name
        deploy_info = model_ref.deploy(**deployment_options)  # type: ignore[attr-defined]
        deploy_details = deploy_info["details"]
        self.assertNotEmpty(deploy_details)
        self.assertTrue(deploy_details["service_info"])
        self.assertTrue(deploy_details["service_function_sql"])

        remote_prediction = model_ref.predict(deployment_name, test_features)
        prediction_assert_fn(local_prediction, remote_prediction)

        model_deployment_list = model_ref.list_deployments().to_pandas()  # type: ignore[attr-defined]
        self.assertEqual(model_deployment_list.shape[0], 1)
        self.assertEqual(model_deployment_list["MODEL_NAME"][0], model_name)
        self.assertEqual(model_deployment_list["MODEL_VERSION"][0], model_version)
        self.assertEqual(model_deployment_list["DEPLOYMENT_NAME"][0], deployment_name)

        deployment = self.registry._get_deployment(
            model_name=model_name, model_version=model_version, deployment_name=deployment_name
        )
        service_name = f"service_{deployment['MODEL_ID']}"
        model_ref.delete_deployment(deployment_name=deployment_name)  # type: ignore[attr-defined]
        self.assertEqual(model_ref.list_deployments().to_pandas().shape[0], 0)  # type: ignore[attr-defined]

        service_lst = self._session.sql(f"SHOW SERVICES LIKE '{service_name}' in account;").collect()
        self.assertEqual(len(service_lst), 0, "Service was not deleted successfully")
        self.assertEqual(self.registry.list_models().to_pandas().shape[0], 1)
        self.registry.delete_model(model_name=model_name, model_version=model_version, delete_artifact=True)
        self.assertEqual(self.registry.list_models().to_pandas().shape[0], 0)


if __name__ == "__main__":
    absltest.main()
