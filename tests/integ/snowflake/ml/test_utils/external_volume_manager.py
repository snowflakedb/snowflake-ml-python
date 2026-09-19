import logging
import os
import time
import uuid

from snowflake import snowpark

logger = logging.getLogger(__name__)

# Iceberg integ backends are account-specific. Set these env vars in the test
# environment rather than copying ARNs and buckets into every test module.
ICEBERG_S3_BASE_URL_ENV = "SNOWML_ICEBERG_S3_BASE_URL"
ICEBERG_AWS_ROLE_ARN_ENV = "SNOWML_ICEBERG_AWS_ROLE_ARN"
ICEBERG_AWS_EXTERNAL_ID_ENV = "SNOWML_ICEBERG_AWS_EXTERNAL_ID"
ICEBERG_AZURE_BASE_URL_ENV = "SNOWML_ICEBERG_AZURE_BASE_URL"
ICEBERG_AZURE_TENANT_ID_ENV = "SNOWML_ICEBERG_AZURE_TENANT_ID"

ICEBERG_VOLUME_NAME_PREFIX = "MLPLATFORMTEST_ICEBERG_"
_AWS_VOLUME_PREFIX = "MLPLATFORMTEST_ICEBERG_AWS_S3"
_AZURE_VOLUME_PREFIX = "MLPLATFORMTEST_ICEBERG_AZURE_BLOB"

# Fallbacks used only when the matching env var is unset, so existing CI keeps
# working. Prefer setting the env vars so these values are not the source of truth.
_DEFAULT_ICEBERG_S3_BASE_URL = "s3://mlplatform-iceberg-test/ml-platform/"
_DEFAULT_ICEBERG_AWS_ROLE_ARN = "arn:aws:iam::736112632310:role/MLPlatformTestIcebergRole"
_DEFAULT_ICEBERG_AWS_EXTERNAL_ID = "MLPLATFORMTEST_SFCRole=MLPlatformExternalVolume="
_DEFAULT_ICEBERG_AZURE_BASE_URL = "azure://mlplatformtesticeberg.blob.core.windows.net/iceberg-data/"
_DEFAULT_ICEBERG_AZURE_TENANT_ID = "075f576f-6f9a-4955-8d99-4086736225c9"


def _env(name: str, default: str) -> str:
    """Return an Iceberg integ-test setting from the environment, or ``default``.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or empty.

    Returns:
        The stripped environment value, or ``default``.
    """
    value = os.environ.get(name, "").strip()
    return value or default


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")


def iceberg_volume_name_prefix(provider: str) -> str:
    """Return the CREATE EXTERNAL VOLUME name prefix for ``provider``.

    Args:
        provider: Cloud provider, ``AWS`` or ``AZURE``.

    Returns:
        Volume name prefix.

    Raises:
        ValueError: If ``provider`` is not a supported Iceberg backend.
    """
    if provider == "AWS":
        return _AWS_VOLUME_PREFIX
    if provider == "AZURE":
        return _AZURE_VOLUME_PREFIX
    raise ValueError(f"Unsupported Iceberg storage provider: {provider}")


def iceberg_storage_locations_sql(*, provider: str) -> str:
    """Build the STORAGE_LOCATIONS SQL fragment for an Iceberg external volume.

    Backend identity comes from environment variables so test modules do not
    embed account ARNs or bucket URLs.

    Args:
        provider: Cloud provider, ``AWS`` or ``AZURE``.

    Returns:
        SQL fragment for the ``STORAGE_LOCATIONS`` clause.

    Raises:
        ValueError: If ``provider`` is not a supported Iceberg backend.
    """
    if provider == "AWS":
        base_url = _sql_literal(_env(ICEBERG_S3_BASE_URL_ENV, _DEFAULT_ICEBERG_S3_BASE_URL))
        role_arn = _sql_literal(_env(ICEBERG_AWS_ROLE_ARN_ENV, _DEFAULT_ICEBERG_AWS_ROLE_ARN))
        external_id = _sql_literal(_env(ICEBERG_AWS_EXTERNAL_ID_ENV, _DEFAULT_ICEBERG_AWS_EXTERNAL_ID))
        return f"""
                (
                    NAME                 = 'prod-iceberg-s3'
                    STORAGE_PROVIDER     = 'S3'
                    STORAGE_BASE_URL     = '{base_url}'
                    STORAGE_AWS_ROLE_ARN = '{role_arn}'
                    STORAGE_AWS_EXTERNAL_ID = '{external_id}'
                )
            """
    if provider == "AZURE":
        base_url = _sql_literal(_env(ICEBERG_AZURE_BASE_URL_ENV, _DEFAULT_ICEBERG_AZURE_BASE_URL))
        tenant_id = _sql_literal(_env(ICEBERG_AZURE_TENANT_ID_ENV, _DEFAULT_ICEBERG_AZURE_TENANT_ID))
        return f"""
                (
                    NAME = 'prod-iceberg-azure'
                    STORAGE_PROVIDER = 'AZURE'
                    STORAGE_BASE_URL = '{base_url}'
                    AZURE_TENANT_ID = '{tenant_id}'
                )
            """
    raise ValueError(f"Unsupported Iceberg storage provider: {provider}")


class ExternalVolumeManager:
    """Manager for external volumes (account-level objects) used with Iceberg tables."""

    # One shared volume per provider per process. Tests isolate themselves with
    # unique ``base_location`` prefixes instead of provisioning a new account-level
    # volume on every call.
    _shared_volumes: dict[str, str] = {}

    def __init__(self, session: snowpark.Session) -> None:
        self._session = session

    def show_external_volumes(self, volume_name_pattern: str) -> snowpark.DataFrame:
        """Show external volumes matching the given pattern.

        Args:
            volume_name_pattern: Pattern to match volume names (e.g., 'MLPLATFORMTEST_%').

        Returns:
            DataFrame with external volume information.
        """
        return self._session.sql(f"SHOW EXTERNAL VOLUMES LIKE '{volume_name_pattern}'")

    def create_external_volume(
        self,
        volume_name: str,
        storage_locations_sql: str,
    ) -> str:
        """Create an external volume with the given storage locations.

        Args:
            volume_name: Name for the external volume.
            storage_locations_sql: SQL fragment for STORAGE_LOCATIONS clause.

        Returns:
            The created volume name.
        """
        self._session.sql(
            f"""
            CREATE EXTERNAL VOLUME {volume_name}
            STORAGE_LOCATIONS = ({storage_locations_sql})
            """
        ).collect()
        return volume_name

    def create_iceberg_external_volume(
        self,
        provider: str = "AWS",
        *,
        volume_name: str | None = None,
    ) -> str:
        """Create an Iceberg external volume for ``provider``.

        Args:
            provider: Cloud provider, ``AWS`` or ``AZURE``.
            volume_name: Explicit volume name. Generated when omitted.

        Returns:
            The created volume name.
        """
        name = volume_name or f"{iceberg_volume_name_prefix(provider)}_{uuid.uuid4().hex[:8].upper()}"
        return self.create_external_volume(name, iceberg_storage_locations_sql(provider=provider))

    def get_or_create_shared_iceberg_volume(self, *, provider: str = "AWS") -> str:
        """Return a process-wide Iceberg volume for ``provider``, creating it once.

        Args:
            provider: Cloud provider, ``AWS`` or ``AZURE``.

        Returns:
            The shared external volume name.
        """
        volume_name = type(self)._shared_volumes.get(provider)
        if volume_name is None:
            volume_name = self.create_iceberg_external_volume(provider)
            type(self)._shared_volumes[provider] = volume_name
        return volume_name

    def new_iceberg_base_location(self) -> str:
        """Return a unique Iceberg ``base_location`` prefix for one table or feature view.

        Returns:
            A unique path under the shared external volume.
        """
        return f"test_{uuid.uuid4().hex}/"

    def try_drop_shared_iceberg_volumes(self) -> None:
        """Best-effort drop of process-wide shared Iceberg volumes."""
        for provider, volume_name in list(type(self)._shared_volumes.items()):
            if self.try_drop_external_volume(volume_name):
                type(self)._shared_volumes.pop(provider, None)

    def drop_external_volume(self, volume_name: str, if_exists: bool = True) -> None:
        """Drop an external volume.

        Args:
            volume_name: Name of the external volume to drop.
            if_exists: If True, don't error if volume doesn't exist.
        """
        if_exists_sql = " IF EXISTS" if if_exists else ""
        self._session.sql(f"DROP EXTERNAL VOLUME{if_exists_sql} {volume_name}").collect()

    def try_drop_external_volume(
        self,
        volume_name: str,
        *,
        attempts: int = 3,
        retry_delay_s: float = 10.0,
    ) -> bool:
        """Drop an external volume without failing the caller if it is still in use.

        A volume cannot be dropped while tables reference it, and tables that were just
        dropped can take a moment to release it, so the drop is retried. A volume that
        still cannot be dropped is left for ``cleanup_external_volumes`` on a later run.

        Args:
            volume_name: Name of the external volume to drop.
            attempts: How many times to try the drop before giving up.
            retry_delay_s: Seconds to wait between attempts.

        Returns:
            True if the volume was dropped, False if it was left behind.
        """
        for attempt in range(1, attempts + 1):
            try:
                self.drop_external_volume(volume_name, if_exists=True)
                return True
            except Exception:
                if attempt == attempts:
                    logger.warning(
                        "Could not drop external volume %s after %d attempts; "
                        "leaving it for the stale-volume sweep.",
                        volume_name,
                        attempts,
                        exc_info=True,
                    )
                    return False
                logger.debug(
                    "Could not drop external volume %s on attempt %d/%d; retrying in %.1fs.",
                    volume_name,
                    attempt,
                    attempts,
                    retry_delay_s,
                    exc_info=True,
                )
                time.sleep(retry_delay_s)
        return False

    def cleanup_external_volumes(
        self,
        prefix: str,
        expire_days: int = 1,
    ) -> None:
        """Clean up stale external volumes matching the prefix.

        Each volume is dropped independently: a volume that is still referenced by a table
        cannot be dropped, and aborting the whole sweep on the first such volume would let
        the rest accumulate indefinitely. This is the safety net ``try_drop_external_volume``
        defers to, so it has to keep going.

        Args:
            prefix: Prefix pattern to match (e.g., 'MLPLATFORMTEST_ICEBERG_').
            expire_days: Only delete volumes older than this many days.
        """
        try:
            volumes_df = self.show_external_volumes(f"{prefix}%")
            stale_volumes = volumes_df.filter(
                f"\"created_on\" < dateadd('day', {-expire_days}, current_timestamp())"
            ).collect()
        except Exception:
            logger.warning("Could not list stale external volumes for prefix %s.", prefix, exc_info=True)
            return

        for row in stale_volumes:
            volume_name = row["name"]
            try:
                self.drop_external_volume(volume_name, if_exists=True)
            except Exception:
                logger.warning(
                    "Could not drop stale external volume %s; continuing with the rest of the sweep.",
                    volume_name,
                    exc_info=True,
                )
