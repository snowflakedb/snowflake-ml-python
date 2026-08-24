"""Tests for ``imperative_executor.fetch_feature_view_rows``.

Pins hypothesis H3 from
``plans/offline_bfv_state_fix_b9da0006.plan.md``.

The new helper mirrors :func:`imperative_executor.fetch_entity_rows`:

- Lazy-imports ``FeatureStore`` and constructs it via
  :func:`assert_feature_store_initialized` (init-first invariant).
- Calls ``FeatureStore.list_feature_views().collect()``.
- Translates each row into the narrow Phase-1 contract dict (see
  Section 7 of the plan).

The fix surfaces offline-only BFVs that ``SHOW ONLINE FEATURE TABLES``
cannot enumerate — :func:`fetch_feature_view_rows` is the
authoritative imperative-API path and is the only addition in
Phase 1.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestFetchFeatureViewRowsImportable:
    """RED before fix — the helper does not exist yet, so the import
    raises ``ImportError`` / ``AttributeError``.  Once Phase 1 lands,
    this test goes GREEN and becomes the load-bearing import contract.
    """

    def test_helper_is_exported_from_imperative_executor(self) -> None:
        from snowflake.ml.feature_store.decl import imperative_executor  # noqa: F401

        assert hasattr(
            imperative_executor, "fetch_feature_view_rows"
        ), "imperative_executor.fetch_feature_view_rows must exist (Phase 1 deliverable)"

    def test_helper_is_exported_from_decl_api(self) -> None:
        """The thin facade in ``decl_api`` is the only entry point
        the CLI may call (per ``docs/DEVELOPMENT_STANDARDS.md`` §
        "snowml decl/ Package — Library Rules" rule 14).
        """
        from snowflake.ml.feature_store.decl import api as decl_api

        assert hasattr(
            decl_api, "fetch_feature_view_rows"
        ), "decl_api.fetch_feature_view_rows must exist (Phase 1 deliverable)"


class TestFetchFeatureViewRowsTranslation:
    """Verify ``fetch_feature_view_rows`` translates
    ``FeatureStore.list_feature_views()`` output into the Phase 1
    contract (Section 7 of the plan).
    """

    def _make_row(
        self,
        *,
        name: str,
        version: str,
        database_name: str,
        schema_name: str,
        kind: str,
        entities: Any,  # list[str] | str (JSON-string form)
        online_config_json: Any,  # str | None
        target_lag: Any,  # str | None
        refresh_freq: Any,  # str | None
        warehouse: str,
        desc: str,
    ) -> dict[str, Any]:
        """Mirror the ``_LIST_FEATURE_VIEW_SCHEMA`` row shape that
        ``FeatureStore.list_feature_views().collect()`` produces.

        Field names lower-cased to match the Snowpark dataframe
        ``Row.as_dict()`` convention.

        Args:
            name: Base FV name.
            version: FV version string.
            database_name: Snowflake database column.
            schema_name: Snowflake schema column.
            kind: ``BATCH`` / ``STREAMING`` / ``REALTIME`` kind discriminator.
            entities: Entity names (list or JSON-string form).
            online_config_json: ``online_config`` JSON column or ``None``.
            target_lag: Snowflake target-lag string or ``None``.
            refresh_freq: Refresh frequency string or ``None``.
            warehouse: Warehouse column.
            desc: Description column.

        Returns:
            Row dict in the ``_LIST_FEATURE_VIEW_SCHEMA`` shape.
        """
        return {
            "name": name,
            "version": version,
            "database_name": database_name,
            "schema_name": schema_name,
            "created_on": "2024-01-01 00:00:00",
            "owner": "ROLE_X",
            "desc": desc,
            "entities": entities,
            "refresh_freq": refresh_freq,
            "refresh_mode": "INCREMENTAL",
            "scheduling_state": "ACTIVE",
            "warehouse": warehouse,
            "cluster_by": None,
            "online_config": online_config_json,
            "storage_config": '{"format": "snowflake"}',
            "stream_config": None,
            "kind": kind,
            "target_lag": target_lag,
        }

    def test_offline_only_batch_fv_translated_with_online_enabled_false(self) -> None:
        """An offline-only BFV row carries ``online_config`` whose
        ``enable=False``; the helper must surface this as
        ``online_enabled: False`` in the translated dict.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_view_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = [
            self._make_row(
                name="MY_BATCH_FV",
                version="V1",
                database_name="DB",
                schema_name="SCH",
                kind="BATCH",
                entities=["USER_ID"],
                online_config_json=json.dumps({"enable": False, "target_lag": "0 seconds", "store_type": "postgres"}),
                target_lag="1 minute",
                refresh_freq="1 minute",
                warehouse="WH",
                desc="",
            )
        ]
        fs = MagicMock(name="FeatureStore")
        fs.list_feature_views.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_view_rows(session, "DB", "SCH", "WH")

        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "MY_BATCH_FV"
        assert row["version"] == "V1"
        assert row["database_name"] == "DB"
        assert row["schema_name"] == "SCH"
        assert row["kind"] == "BATCH"
        assert row["entities"] == ["USER_ID"]
        assert row["online_enabled"] is False
        assert row["target_lag"] == "1 minute"
        assert row["physical_dt_name"] == "MY_BATCH_FV$V1"

    def test_online_batch_fv_translated_with_online_enabled_true(self) -> None:
        """An online BFV row (``online_config.enable=True``) must
        translate to ``online_enabled: True`` so the merge path in
        :func:`state.fetch_applied_state` knows the OFT path is
        authoritative.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_view_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = [
            self._make_row(
                name="MY_ONLINE_FV",
                version="V1",
                database_name="DB",
                schema_name="SCH",
                kind="BATCH",
                entities=["USER_ID"],
                online_config_json=json.dumps({"enable": True, "target_lag": "1 minute", "store_type": "postgres"}),
                target_lag="1 minute",
                refresh_freq="1 minute",
                warehouse="WH",
                desc="",
            )
        ]
        fs = MagicMock(name="FeatureStore")
        fs.list_feature_views.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_view_rows(session, "DB", "SCH", "WH")

        assert rows[0]["online_enabled"] is True

    def test_entities_parsed_when_returned_as_json_string(self) -> None:
        """``list_feature_views`` may surface ``entities`` either
        as a Python list (Snowpark ArrayType) or as a JSON string
        (cursor-side serialization).  The helper must parse the
        JSON-string form so downstream code has a real ``list[str]``.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_view_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = [
            self._make_row(
                name="FV",
                version="V1",
                database_name="DB",
                schema_name="SCH",
                kind="BATCH",
                entities='["USER_ID","SESSION_ID"]',  # JSON string form
                online_config_json=None,
                target_lag="1 minute",
                refresh_freq="1 minute",
                warehouse="WH",
                desc="",
            )
        ]
        fs = MagicMock(name="FeatureStore")
        fs.list_feature_views.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_view_rows(session, "DB", "SCH", "WH")

        assert rows[0]["entities"] == ["USER_ID", "SESSION_ID"]

    def test_empty_dataframe_yields_empty_list(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_view_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = []
        fs = MagicMock(name="FeatureStore")
        fs.list_feature_views.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_view_rows(session, "DB", "SCH", "WH")

        assert rows == []


class TestFetchFeatureViewRowsInitFirst:
    """Init-first guard — uninitialised schemas raise
    :class:`FeatureStoreNotInitializedError` (matching
    ``fetch_entity_rows`` behaviour).
    """

    def test_raises_feature_store_not_initialized_when_tags_missing(self) -> None:
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )
        from snowflake.ml.feature_store.decl.errors import (
            FeatureStoreNotInitializedError,
        )
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_view_rows,
        )

        session = MagicMock(name="session")
        missing_tag = snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=ValueError("Feature store internal tag SNOWML_FEATURE_STORE_OBJECT does not exist."),
        )
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            side_effect=missing_tag,
        ), pytest.raises(FeatureStoreNotInitializedError):
            fetch_feature_view_rows(session, "DB", "SCH", "WH")

        # Negative pin: no raw SQL fallback against the session.
        session.sql.assert_not_called()


class TestFetchFeatureViewRowsApiFacade:
    """The ``decl_api.fetch_feature_view_rows`` thin facade delegates
    to the executor — this is what the CLI manager will call.
    """

    def test_decl_api_facade_delegates_to_executor(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.decl.imperative_executor.fetch_feature_view_rows",
            return_value=[{"name": "FV", "version": "V1"}],
        ) as mock_fetch:
            result = decl_api.fetch_feature_view_rows(session, "DB", "SCH", "WH")

        mock_fetch.assert_called_once_with(session, "DB", "SCH", "WH")
        assert result == [{"name": "FV", "version": "V1"}]


class TestCanonicalizeEnumsInPlace:
    """Pin the enum-canonicalisation contract on the offline-only
    BatchFV serialisation path.

    ``FeatureViewSpec.to_dict()`` returns Python ``Enum`` instances for
    ``kind`` (``FeatureViewKind``) and ``feature_aggregation_method``
    (``FeatureAggregationMethod``).  The live ``DESCRIBE … TYPE =
    SPECIFICATION`` path returns strings (Snowflake serialises the
    spec server-side), so the divergence is invisible to the
    online-OFT recovery path and only surfaces on offline-only BFV
    export via :func:`_serialize_batch_fv_spec`.  Without
    canonicalisation, the exporter writes the enum-bearing payload
    through PyYAML and emits
    ``!!python/object/apply:snowflake.ml.feature_store.spec.enums.<Enum>``
    tags that ``yaml.safe_load`` rejects on the next
    ``snow feature plan``.
    """

    def test_top_level_enum_value_replaced_with_value(self) -> None:
        from enum import Enum

        from snowflake.ml.feature_store.decl.imperative_executor import (
            _canonicalize_enums_in_place,
        )

        class _Kind(Enum):
            BATCH = "BatchFeatureView"

        payload: dict[str, Any] = {"kind": _Kind.BATCH}
        _canonicalize_enums_in_place(payload)

        assert payload == {"kind": "BatchFeatureView"}

    def test_nested_enum_in_inner_spec_replaced_with_value(self) -> None:
        from enum import Enum

        from snowflake.ml.feature_store.decl.imperative_executor import (
            _canonicalize_enums_in_place,
        )

        class _Method(Enum):
            TILES = "tiles"

        payload: dict[str, Any] = {"spec": {"feature_aggregation_method": _Method.TILES}}
        _canonicalize_enums_in_place(payload)

        assert payload == {"spec": {"feature_aggregation_method": "tiles"}}

    def test_enum_inside_list_replaced_with_value(self) -> None:
        from enum import Enum

        from snowflake.ml.feature_store.decl.imperative_executor import (
            _canonicalize_enums_in_place,
        )

        class _Kind(Enum):
            A = "A"
            B = "B"

        payload = {"things": [_Kind.A, _Kind.B, "C"]}
        _canonicalize_enums_in_place(payload)

        assert payload == {"things": ["A", "B", "C"]}

    def test_non_enum_payload_is_untouched(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _canonicalize_enums_in_place,
        )

        payload = {
            "kind": "BatchFeatureView",
            "spec": {
                "feature_aggregation_method": "tiles",
                "entities": ["USER_ID"],
            },
        }
        _canonicalize_enums_in_place(payload)

        assert payload == {
            "kind": "BatchFeatureView",
            "spec": {
                "feature_aggregation_method": "tiles",
                "entities": ["USER_ID"],
            },
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
