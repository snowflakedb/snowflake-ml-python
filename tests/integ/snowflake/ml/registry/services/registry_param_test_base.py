"""Base class for param integration tests across model types.

Provides shared REST infrastructure for testing ParamSpec parameters across
invocation paths (mv.run, REST flat, REST split, REST records, REST wide).

Subclasses define their own models, param constants, and assertions.
"""

from typing import Any

import requests
import retrying

from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)


class ParamTestBase(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Shared REST param testing infrastructure for all model types.

    TODO: Migrate TestTransformerParamsInteg to also inherit from this class.
    """

    def _rest_post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        target_method: str = "predict",
    ) -> requests.Response:
        """Retried POST to REST inference endpoint."""
        return retrying.retry(
            wait_exponential_multiplier=1000,
            wait_exponential_max=10000,
            stop_max_attempt_number=3,
            retry_on_result=(
                registry_model_deployment_test_base.RegistryModelDeploymentTestBase.retry_if_result_status_retriable
            ),
        )(requests.post)(
            f"https://{endpoint}/{target_method.replace('_', '-')}",
            json=payload,
            auth=self._get_auth_for_inference(endpoint),
            timeout=60,
        )

    def _assert_rest_ok(
        self,
        endpoint: str,
        payload: dict[str, Any],
        target_method: str = "predict",
        label: str = "",
    ) -> requests.Response:
        """Assert REST POST returns 200 and return the response."""
        tag = f"[{label}] " if label else ""
        response = self._rest_post(endpoint, payload, target_method)
        self.assertEqual(
            response.status_code,
            200,
            f"{tag}Expected 200, got {response.status_code}: {response.text[:300]}",
        )
        return response

    def _assert_rest_400(
        self,
        endpoint: str,
        payload: dict[str, Any],
        target_method: str = "predict",
        label: str = "",
    ) -> requests.Response:
        """Assert REST POST returns 400 and return the response."""
        tag = f"[{label}] " if label else ""
        response = self._rest_post(endpoint, payload, target_method)
        self.assertEqual(
            response.status_code,
            400,
            f"{tag}Expected 400, got {response.status_code}: {response.text[:300]}",
        )
        return response

    def _parse_rest_rows(self, response: requests.Response) -> list[dict[str, Any]]:
        """Parse REST response {"data": [[row_id, {row_dict}], ...]} into list of row dicts."""
        return [row[1] for row in response.json()["data"]]

    def _get_service_name(self, mv: Any) -> str:
        """Get the first service name from a model version."""
        return mv.list_services().loc[0, "name"]
