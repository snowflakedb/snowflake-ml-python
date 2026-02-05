from snowflake import snowpark


class ExternalVolumeManager:
    """Manager for external volumes (account-level objects) used with Iceberg tables."""

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

    def drop_external_volume(self, volume_name: str, if_exists: bool = True) -> None:
        """Drop an external volume.

        Args:
            volume_name: Name of the external volume to drop.
            if_exists: If True, don't error if volume doesn't exist.
        """
        if_exists_sql = " IF EXISTS" if if_exists else ""
        self._session.sql(f"DROP EXTERNAL VOLUME{if_exists_sql} {volume_name}").collect()

    def cleanup_external_volumes(
        self,
        prefix: str,
        expire_days: int = 1,
    ) -> None:
        """Clean up stale external volumes matching the prefix.

        Args:
            prefix: Prefix pattern to match (e.g., 'MLPLATFORMTEST_ICEBERG_').
            expire_days: Only delete volumes older than this many days.
        """
        try:
            volumes_df = self.show_external_volumes(f"{prefix}%")
            stale_volumes = volumes_df.filter(
                f"\"created_on\" < dateadd('day', {-expire_days}, current_timestamp())"
            ).collect()
            for row in stale_volumes:
                self.drop_external_volume(row["name"], if_exists=True)
        except Exception:
            pass  # Best effort cleanup
