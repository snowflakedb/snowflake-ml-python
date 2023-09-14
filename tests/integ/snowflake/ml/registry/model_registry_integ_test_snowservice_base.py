import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest import SkipTest

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import model_signature
from snowflake.ml.registry import model_registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    model_factory,
    test_env_utils,
)


class TestModelRegistryIntegSnowServiceBase(parameterized.TestCase):
    _SNOWSERVICE_CONNECTION_NAME = "regtest"
    _TEST_CPU_COMPUTE_POOL = "REGTEST_INFERENCE_CPU_POOL"
    _TEST_GPU_COMPUTE_POOL = "REGTEST_INFERENCE_GPU_POOL"
    _RUN_ID = uuid.uuid4().hex[:2]
    _TEST_DB = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "db").upper()
    _TEST_SCHEMA = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(_RUN_ID, "schema").upper()

    @classmethod
    def setUpClass(cls) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        try:
            login_options = connection_params.SnowflakeLoginOptions(connection_name=cls._SNOWSERVICE_CONNECTION_NAME)
        except KeyError:
            raise SkipTest(
                "SnowService connection parameters not present: skipping "
                "TestModelRegistryIntegWithSnowServiceDeployment."
            )
        cls._session = Session.builder.configs(
            {
                **login_options,
                **{"database": cls._TEST_DB, "schema": cls._TEST_SCHEMA},
            }
        ).create()
        cls._db_manager = db_manager.DBManager(cls._session)
        cls._db_manager.cleanup_databases(expire_hours=6)
        model_registry.create_model_registry(
            session=cls._session, database_name=cls._TEST_DB, schema_name=cls._TEST_SCHEMA
        )
        cls.registry = model_registry.ModelRegistry(
            session=cls._session, database_name=cls._TEST_DB, schema_name=cls._TEST_SCHEMA
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._db_manager.drop_database(cls._TEST_DB)
        cls._session.close()

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
        conda_dependencies.append(test_env_utils.get_latest_package_versions_in_conda("snowflake-snowpark-python"))

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
        model_ref.deploy(**deployment_options)  # type: ignore[attr-defined]

        remote_prediction = model_ref.predict(deployment_name, test_features)
        prediction_assert_fn(local_prediction, remote_prediction)

        model_deployment_list = model_ref.list_deployments().to_pandas()  # type: ignore[attr-defined]
        self.assertEqual(model_deployment_list.shape[0], 1)
        self.assertEqual(model_deployment_list["MODEL_NAME"][0], model_name)
        self.assertEqual(model_deployment_list["MODEL_VERSION"][0], model_version)
        self.assertEqual(model_deployment_list["DEPLOYMENT_NAME"][0], deployment_name)

        model_ref.delete_deployment(deployment_name=deployment_name)  # type: ignore[attr-defined]
        self.assertEqual(model_ref.list_deployments().to_pandas().shape[0], 0)  # type: ignore[attr-defined]

        self.assertEqual(self.registry.list_models().to_pandas().shape[0], 1)
        self.registry.delete_model(model_name=model_name, model_version=model_version, delete_artifact=True)
        self.assertEqual(self.registry.list_models().to_pandas().shape[0], 0)


if __name__ == "__main__":
    absltest.main()
