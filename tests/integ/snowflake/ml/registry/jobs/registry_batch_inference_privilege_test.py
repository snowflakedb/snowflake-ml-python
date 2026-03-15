import logging
import uuid
from typing import Callable, TypeVar

import pandas as pd
from absl.testing import absltest
from sklearn import datasets, linear_model

from snowflake.ml.model.batch import JobSpec, OutputSpec
from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base
from tests.integ.snowflake.ml.test_utils import db_manager

logger = logging.getLogger(__name__)

T = TypeVar("T")


@absltest.skip("Temporarily skipped due to privilege test instability")
class RegistryBatchInferencePrivilegeTest(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Integration tests verifying privilege requirements for batch inference via SPCS Jobs.

    Validates that a user with READ privilege on a model can invoke
    batch inference via mv.run_batch().

    Required privileges for batch inference:
        - USAGE on DATABASE and SCHEMA containing the model
        - USAGE on WAREHOUSE
        - READ on MODEL (required for accessing model artifacts/stage files during job creation)
        - USAGE on COMPUTE POOL
        - READ/WRITE on output stage
        - READ/WRITE on IMAGE REPOSITORY (for user operations)
        - SERVICE READ/SERVICE WRITE on IMAGE REPOSITORY (for MODEL_BUILD service to push images)
        - CREATE SERVICE on SCHEMA
        - BIND SERVICE ENDPOINT on ACCOUNT (for service to bind to endpoints)

    Note: USAGE privilege alone is NOT sufficient for batch inference because
    run_batch() needs to access model artifacts (stage files) which requires READ.

    Note: The MODEL_BUILD service (which uses Kaniko to build container images)
    requires SERVICE WRITE privilege on the image repository, not just WRITE.
    SERVICE READ/WRITE are special privileges for container services.

    Test Isolation:
        Each test method gets a fresh session (see CommonTestBase.setUp/tearDown),
        so role changes do not leak between tests or affect parallel execution.
    """

    def setUp(self) -> None:
        super().setUp()

        self._admin_role = self.session.get_current_role().strip('"')
        self._test_warehouse = self.session.get_current_warehouse()

        self._read_role = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "BATCH_READ_ROLE"
        ).upper()

        self._db_manager.create_role(self._read_role)

        current_user = self.session.get_current_user().strip('"')
        self.session.sql(f"GRANT ROLE {self._read_role} TO USER {current_user}").collect()

        self.session.sql(f"GRANT USAGE ON DATABASE {self._test_db} TO ROLE {self._read_role}").collect()
        self.session.sql(
            f"GRANT USAGE ON SCHEMA {self._test_db}.{self._test_schema} TO ROLE {self._read_role}"
        ).collect()

        try:
            self.session.sql(f"GRANT USAGE ON WAREHOUSE {self._test_warehouse} TO ROLE {self._read_role}").collect()
        except Exception:
            pass

    def tearDown(self) -> None:
        self.session.use_role(self._admin_role)
        self._db_manager.drop_role(self._read_role, if_exists=True)

        super().tearDown()

    def _run_as_role(self, role: str, fn: Callable[[], T]) -> T:
        """Execute a callable under a specific role with secondary roles disabled."""
        prev_role = self.session.get_current_role()
        try:
            self.session.sql("USE SECONDARY ROLES NONE").collect()
            self.session.use_role(role)
            return fn()
        finally:
            self.session.use_role(prev_role)
            self.session.sql("USE SECONDARY ROLES ALL").collect()

    def test_read_can_run_batch_inference(self) -> None:
        """A user with READ privilege on model can run batch inference."""
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        classifier = linear_model.LogisticRegression()
        classifier.fit(iris_X, iris_y)

        iris_pandas_df = pd.DataFrame(iris_X, columns=[f"input_feature_{i}" for i in range(iris_X.shape[1])])
        iris_pandas_df[self._INDEX_COL] = range(len(iris_pandas_df))

        job_name = f"BATCH_PRIV_TEST_{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{job_name}/output/"

        model_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "BATCH_PRIV_MODEL"
        ).upper()
        version_name = "V1"

        mv = self.registry.log_model(
            model=classifier,
            model_name=model_name,
            version_name=version_name,
            sample_input_data=iris_X,
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"enable_explainability": False, "embed_local_ml_library": True},
        )
        fully_qualified_model_name = mv.fully_qualified_model_name

        self.session.use_role(self._admin_role)

        self.session.sql(f"GRANT READ ON MODEL {fully_qualified_model_name} TO ROLE {self._read_role}").collect()

        try:
            self.session.sql(
                f"GRANT USAGE ON COMPUTE POOL {self._TEST_CPU_COMPUTE_POOL} TO ROLE {self._read_role}"
            ).collect()
        except Exception as e:
            logger.warning(f"Could not grant USAGE on compute pool: {e}")

        try:
            self.session.sql(
                f"GRANT READ, WRITE ON STAGE {self._test_db}.{self._test_schema}.{self._test_stage} "
                f"TO ROLE {self._read_role}"
            ).collect()
        except Exception as e:
            logger.warning(f"Could not grant READ, WRITE on stage: {e}")

        try:
            self.session.sql(
                f"GRANT READ, WRITE ON IMAGE REPOSITORY {self._test_db}.{self._test_schema}.{self._test_image_repo} "
                f"TO ROLE {self._read_role}"
            ).collect()
        except Exception as e:
            logger.warning(f"Could not grant READ, WRITE on image repository: {e}")

        try:
            self.session.sql(
                f"GRANT SERVICE READ, SERVICE WRITE ON IMAGE REPOSITORY "
                f"{self._test_db}.{self._test_schema}.{self._test_image_repo} TO ROLE {self._read_role}"
            ).collect()
        except Exception as e:
            logger.warning(f"Could not grant SERVICE READ, SERVICE WRITE on image repository: {e}")

        try:
            self.session.sql(
                f"GRANT CREATE SERVICE ON SCHEMA {self._test_db}.{self._test_schema} TO ROLE {self._read_role}"
            ).collect()
        except Exception as e:
            logger.warning(f"Could not grant CREATE SERVICE on schema: {e}")

        try:
            self.session.sql(f"GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE {self._read_role}").collect()
        except Exception as e:
            logger.warning(f"Could not grant BIND SERVICE ENDPOINT on account: {e}")

        def _test_batch_inference() -> None:
            """Test batch inference via mv.run_batch()."""
            input_df = self.session.create_dataframe(iris_pandas_df)

            reg = registry.Registry(self.session)
            model = reg.get_model(model_name)
            mv_as_read = model.version(version_name)

            batch_job = mv_as_read.run_batch(
                input_df,
                compute_pool=self._TEST_CPU_COMPUTE_POOL,
                output_spec=OutputSpec(stage_location=output_stage_location),
                job_spec=JobSpec(
                    job_name=job_name,
                    num_workers=1,
                    replicas=1,
                    function_name="predict",
                    image_repo=f"{self._test_db}.{self._test_schema}.{self._test_image_repo}",
                ),
            )

            batch_job.wait()
            self.assertEqual(batch_job.status, "DONE", f"Job status is {batch_job.status}, expected DONE")

            output_df = self.session.read.option("on_error", "CONTINUE").parquet(output_stage_location)
            self.assertEqual(output_df.count(), input_df.count())

        self._run_as_role(self._read_role, _test_batch_inference)


if __name__ == "__main__":
    absltest.main()
