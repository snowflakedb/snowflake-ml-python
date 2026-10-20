"""Unit tests for the ``initialization_warehouse`` plumbing in feature_store."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml.feature_store import feature_store as fs_mod
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    StorageConfig,
    StorageFormat,
)


def _make_fv(**kwargs: Any) -> FeatureView:
    mock_df = MagicMock()
    mock_df.columns = ["user_id", "amount"]
    mock_df.queries = {"queries": ["SELECT * FROM source"]}
    return FeatureView(
        name="test_fv",
        entities=[Entity(name="user", join_keys=["user_id"])],
        feature_df=mock_df,
        refresh_freq="1d",
        warehouse="small_wh",
        **kwargs,
    )


class InitializationWarehouseClauseTest(absltest.TestCase):
    """The clause builder emits CREATE-time DDL; read-back uses the SHOW column."""

    def test_clause_empty_when_unset(self) -> None:
        fv = _make_fv()
        self.assertEqual(fs_mod._initialization_warehouse_clause(fv), "")

    def test_clause_present_when_set(self) -> None:
        fv = _make_fv(initialization_warehouse="large_wh")
        clause = fs_mod._initialization_warehouse_clause(fv)
        self.assertIn("INITIALIZATION_WAREHOUSE = LARGE_WH", clause)


class CreateDynamicTableQueryTest(absltest.TestCase):
    """``_create_dynamic_table_query`` does not touch session state, so it can be
    exercised on a bare FeatureStore instance."""

    def setUp(self) -> None:
        self._fs = object.__new__(fs_mod.FeatureStore)

    def _build(self, fv: FeatureView) -> str:
        return self._fs._create_dynamic_table_query(
            override_clause="",
            table_name="DB.SCH.TEST_FV$V1",
            column_descs="",
            schedule_task=False,
            feature_view=fv,
            tagging_clause="TAG_A = 'x'",
            warehouse="SMALL_WH",
        )

    def test_dynamic_table_emits_initialization_warehouse_when_set(self) -> None:
        query = self._build(_make_fv(initialization_warehouse="large_wh"))
        self.assertIn("WAREHOUSE = SMALL_WH", query)
        self.assertIn("INITIALIZATION_WAREHOUSE = LARGE_WH", query)

    def test_dynamic_table_omits_clause_when_unset(self) -> None:
        query = self._build(_make_fv())
        self.assertIn("WAREHOUSE = SMALL_WH", query)
        self.assertNotIn("INITIALIZATION_WAREHOUSE", query)

    def test_iceberg_table_emits_initialization_warehouse_when_set(self) -> None:
        fv = _make_fv(
            initialization_warehouse="large_wh",
            storage_config=StorageConfig(
                format=StorageFormat.ICEBERG,
                external_volume="MY_VOLUME",
                base_location="feature_store/test_fv",
            ),
        )
        query = self._build(fv)
        self.assertIn("DYNAMIC ICEBERG TABLE", query)
        self.assertIn("INITIALIZATION_WAREHOUSE = LARGE_WH", query)


class DynamicTableInitializeParseTest(absltest.TestCase):
    """``_DT_INITIALIZE_PATTERN`` recovers ``INITIALIZE`` from the Dynamic Table
    DDL in ``SHOW DYNAMIC TABLES.text`` during reconstruction.

    The capture must be bounded to the parameter value so a quoted literal in
    the query body (e.g. ``'click'``) is not swallowed, which previously made
    reconstruction fail ``FeatureView`` validation for imperatively-registered
    SQL feature views.
    """

    def _parse(self, text: str) -> str | None:
        m = re.match(fs_mod._DT_INITIALIZE_PATTERN, text)
        return m.group("initialize") if m is not None else None

    def test_imperative_case_when_literal_parses_value_only(self) -> None:
        text = (
            "CREATE DYNAMIC TABLE DB.SCH.F_APPEND_ONLY_CASE$V1(UUID, EVENT_TS, CLICKS) "
            "initialize = 'ON_CREATE' warehouse = USER_WH AS "
            "SELECT UUID, MAX(EVENT_TS) AS EVENT_TS,\n"
            "SUM(CASE WHEN EVENT_TYPE = 'click' THEN 1 ELSE 0 END) AS CLICKS "
            "FROM EVENTS GROUP BY UUID"
        )
        self.assertEqual(self._parse(text), "ON_CREATE")

    def test_imperative_count_if_and_in_list_literals(self) -> None:
        text = (
            "CREATE DYNAMIC TABLE DB.SCH.E_APPEND_ONLY$V1(UUID, CLICKS, INTERACTIONS) "
            "initialize = 'ON_CREATE' warehouse = USER_WH AS "
            "SELECT UUID, COUNT_IF(EVENT_TYPE = 'click') AS CLICKS, "
            "COUNT_IF(EVENT_TYPE IN ('click', 'view')) AS INTERACTIONS "
            "FROM EVENTS GROUP BY UUID"
        )
        self.assertEqual(self._parse(text), "ON_CREATE")

    def test_imperative_no_string_literals(self) -> None:
        text = (
            "CREATE DYNAMIC TABLE DB.SCH.G$V1(UUID, TOTAL) "
            "initialize = 'ON_CREATE' warehouse = USER_WH AS "
            "SELECT UUID, SUM(AMOUNT) AS TOTAL FROM EVENTS GROUP BY UUID"
        )
        self.assertEqual(self._parse(text), "ON_CREATE")

    def test_on_schedule_value_round_trips(self) -> None:
        text = (
            "CREATE DYNAMIC TABLE DB.SCH.H$V1(UUID, CLICKS) "
            "initialize = 'ON_SCHEDULE' warehouse = USER_WH AS "
            "SELECT UUID, SUM(CASE WHEN EVENT_TYPE = 'click' THEN 1 ELSE 0 END) AS CLICKS "
            "FROM EVENTS GROUP BY UUID"
        )
        self.assertEqual(self._parse(text), "ON_SCHEDULE")

    def test_declarative_create_or_replace_does_not_match(self) -> None:
        text = (
            "CREATE OR REPLACE DYNAMIC TABLE DB.SCH.D$V1(UUID, CLICKS) "
            "initialize = 'ON_CREATE' warehouse = USER_WH AS "
            "SELECT UUID, SUM(CASE WHEN EVENT_TYPE = 'click' THEN 1 ELSE 0 END) AS CLICKS "
            "FROM EVENTS GROUP BY UUID"
        )
        self.assertIsNone(self._parse(text))


if __name__ == "__main__":
    absltest.main()
