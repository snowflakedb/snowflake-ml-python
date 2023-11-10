from absl.testing import absltest

from snowflake.ml._internal.utils import identifier, query_result_checker
from snowflake.ml.model._deploy_client.utils import (
    image_registry_client,
    snowservice_client,
)
from tests.integ.snowflake.ml.test_utils import spcs_integ_test_base


class ImageRegistryClientIntegTest(spcs_integ_test_base.SpcsIntegTestBase):
    def setUp(self) -> None:
        super().setUp()
        self._TEST_REPO = "TEST_REPO"
        client = snowservice_client.SnowServiceClient(self._session)
        client.create_image_repo(
            identifier.get_schema_level_object_identifier(self._test_db, self._test_schema, self._TEST_REPO)
        )

    def tearDown(self) -> None:
        super().tearDown()

    def _get_repo_url(self) -> str:
        """Retrieve repo url.

        Returns: repo url, sample repo url format: org-account.registry.snowflakecomputing.com/db/schema/repo.
        """
        sql = (
            f"SHOW IMAGE REPOSITORIES LIKE '{self._TEST_REPO}' "
            f"IN SCHEMA {'.'.join([self._test_db, self._test_schema])}"
        )
        result = (
            query_result_checker.SqlResultValidator(
                session=self._session,
                query=sql,
            )
            .has_column("repository_url")
            .has_dimensions(expected_rows=1)
            .validate()
        )
        return result[0]["repository_url"]

    def test_copy_from_docker_hub_to_spcs_registry(self) -> None:
        repo_url = self._get_repo_url()
        dest_image = "/".join([repo_url, "kaniko-project/executor:v1.16.0-debug"])
        client = image_registry_client.ImageRegistryClient(self._session)
        self.assertFalse(client.image_exists(dest_image))
        client.copy_image(
            "gcr.io/kaniko-project/executor@sha256:b8c0977f88f24dbd7cbc2ffe5c5f824c410ccd0952a72cc066efc4b6dfbb52b6",
            dest_image,
        )
        self.assertTrue(client.image_exists(dest_image))


if __name__ == "__main__":
    absltest.main()
