"""Unit tests for OFT SHOW row ``store_type`` → ``OnlineConfig`` reconstruction."""

import json
import logging
from typing import Optional
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized

from snowflake.ml.feature_store import feature_store as fs_mod
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import OnlineConfig, OnlineStoreType
from snowflake.snowpark import Row


class StoreTypeFromOftShowRowTest(parameterized.TestCase):
    """Tests for ``_store_type_from_oft_show_row``."""

    @parameterized.named_parameters(  # type: ignore[misc]
        ("postgres_upper", "POSTGRES", OnlineStoreType.POSTGRES),
        ("postgres_lower", "postgres", OnlineStoreType.POSTGRES),
        ("hybrid_table", "hybrid_table", OnlineStoreType.HYBRID_TABLE),
        ("hybrid_table_upper", "HYBRID_TABLE", OnlineStoreType.HYBRID_TABLE),
        ("missing_column", None, OnlineStoreType.HYBRID_TABLE),
    )
    def test_store_type_mapping(self, store_type_value: Optional[str], expected: OnlineStoreType) -> None:
        if store_type_value is None:
            row = Row(TARGET_LAG="10s")
        else:
            row = Row(TARGET_LAG="10s", STORE_TYPE=store_type_value)
        self.assertEqual(fs_mod._store_type_from_oft_show_row(row), expected)

    def test_lowercase_store_type_key(self) -> None:
        row = Row(target_lag="10s", store_type="postgres")
        self.assertEqual(fs_mod._store_type_from_oft_show_row(row), OnlineStoreType.POSTGRES)

    def test_empty_or_none_store_type_defaults_hybrid(self) -> None:
        self.assertEqual(
            fs_mod._store_type_from_oft_show_row(Row(TARGET_LAG="10s", STORE_TYPE=None)),
            OnlineStoreType.HYBRID_TABLE,
        )
        self.assertEqual(
            fs_mod._store_type_from_oft_show_row(Row(TARGET_LAG="10s", STORE_TYPE="")),
            OnlineStoreType.HYBRID_TABLE,
        )

    def test_unknown_store_type_warns_and_defaults_hybrid(self) -> None:
        row = Row(TARGET_LAG="10s", STORE_TYPE="unknown_backend")
        with self.assertLogs(fs_mod.logger.name, level=logging.WARNING) as log_ctx:
            got = fs_mod._store_type_from_oft_show_row(row)
        self.assertEqual(got, OnlineStoreType.HYBRID_TABLE)
        self.assertTrue(any("Unknown SHOW ONLINE FEATURE TABLES store_type" in m for m in log_ctx.output))


class DetermineOnlineConfigFromOftTest(absltest.TestCase):
    """Tests for ``FeatureStore._determine_online_config_from_oft`` with mocked SHOW rows."""

    def _make_fs_with_oft_row(self, oft_row: Row) -> FeatureStore:
        fs = object.__new__(FeatureStore)
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[oft_row]))
        return fs

    def test_postgres_store_type_in_online_config_json(self) -> None:
        oft_row = Row(
            TARGET_LAG="15 seconds",
            REFRESH_MODE="AUTO",
            SCHEDULING_STATE="STARTED",
            STORE_TYPE="POSTGRES",
        )
        fs = self._make_fs_with_oft_row(oft_row)
        json_str = fs._determine_online_config_from_oft("my_fv", "v1")
        cfg = OnlineConfig.from_json(json_str)
        self.assertTrue(cfg.enable)
        self.assertEqual(cfg.store_type, OnlineStoreType.POSTGRES)

    def test_missing_store_type_defaults_hybrid_in_json(self) -> None:
        oft_row = Row(
            TARGET_LAG="1 minute",
            REFRESH_MODE="AUTO",
            SCHEDULING_STATE="STARTED",
        )
        fs = self._make_fs_with_oft_row(oft_row)
        json_str = fs._determine_online_config_from_oft("my_fv", "v1")
        cfg = OnlineConfig.from_json(json_str)
        self.assertEqual(cfg.store_type, OnlineStoreType.HYBRID_TABLE)

    def test_include_online_service_metadata_preserves_store_type(self) -> None:
        oft_row = Row(
            TARGET_LAG="20 seconds",
            REFRESH_MODE="FULL",
            SCHEDULING_STATE="SUSPENDED",
            STORE_TYPE="postgres",
        )
        fs = self._make_fs_with_oft_row(oft_row)
        json_str = fs._determine_online_config_from_oft("my_fv", "v1", include_online_service_metadata=True)
        data = json.loads(json_str)
        self.assertEqual(data["store_type"], "postgres")
        self.assertEqual(data["refresh_mode"], "FULL")
        self.assertEqual(data["scheduling_state"], "SUSPENDED")


if __name__ == "__main__":
    absltest.main()
