"""Phase 3 RED tests — ``imperative_executor.fetch_feature_group_rows``.

Pins the read-path bridge from the declarative library to the imperative
``FeatureStore.list_feature_groups()``.  Mirrors the contract of
``fetch_entity_rows`` / ``fetch_feature_view_rows``:

* Lazy-imports ``FeatureStore`` and constructs it via
  :func:`assert_feature_store_initialized` (init-first invariant).
* Returns plain ``list[dict]`` rows in a narrow shape consumable by
  :func:`state.fetch_applied_state`.
* Re-exported as a thin facade on :mod:`decl.api`.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from snowflake.ml.test_utils import pytest_driver


def _make_row(
    *,
    name: str = "USER_FRAUD_FG",
    version: str = "V1",
    desc: str = "",
    owner: str = "ROLE_X",
    auto_prefix: bool = True,
    sources: list[dict[str, Any]] | None = None,
    output_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Mirror the shape of one row from ``FeatureStore.list_feature_groups()``.

    Snowpark serialises the ``SOURCES`` / ``OUTPUT_COLUMNS`` columns as
    JSON strings (the metadata store persists them as JSON variants); the
    helper must accept that shape.

    Args:
        name: FG name (Snowflake-uppercased identifier).
        version: FG version string.
        desc: Optional description carried in the ``DESC`` column.
        owner: Owning role string.
        auto_prefix: Whether the imperative side will auto-prefix the
            FG's output column names with the contributing FV name.
        sources: Optional list of source-FV dicts. Each dict is the
            imperative-row shape (``fv_name`` / ``fv_version``); when
            ``None``, a single default ref is materialised so the
            row is well-formed.
        output_columns: Optional list of derived output-column names
            in the imperative-row shape.

    Returns:
        Dict in the exact shape one row of
        ``FeatureStore.list_feature_groups()`` would return — keyed by
        the upper-case Snowpark column names with ``SOURCES`` and
        ``OUTPUT_COLUMNS`` JSON-encoded as strings.
    """
    return {
        "NAME": name,
        "VERSION": version,
        "DESC": desc,
        "OWNER": owner,
        "AUTO_PREFIX": auto_prefix,
        "SOURCES": json.dumps(
            sources
            if sources is not None
            else [
                {"fv_name": "USER_CLICK_STATS", "fv_version": "V1"},
            ]
        ),
        "OUTPUT_COLUMNS": json.dumps(output_columns) if output_columns is not None else None,
    }


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


class TestFetchFeatureGroupRowsImportable:
    def test_helper_exported_from_imperative_executor(self) -> None:
        from snowflake.ml.feature_store.decl import imperative_executor

        assert hasattr(imperative_executor, "fetch_feature_group_rows"), (
            "imperative_executor.fetch_feature_group_rows must exist " "(Phase 3 deliverable)"
        )

    def test_helper_exported_from_decl_api(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        assert hasattr(decl_api, "fetch_feature_group_rows"), (
            "decl_api.fetch_feature_group_rows must exist " "(Phase 3 deliverable, CLI entry point)"
        )


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


class TestFetchFeatureGroupRowsTranslation:
    def test_single_row_translated(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_group_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = [
            _make_row(
                name="USER_FRAUD_FG",
                version="V1",
                desc="Combined user signals.",
                auto_prefix=True,
                sources=[
                    {"fv_name": "USER_CLICK_STATS", "fv_version": "V1"},
                    {
                        "fv_name": "USER_TXN_STATS",
                        "fv_version": "V1",
                        "slice_columns": ["TOTAL_SPEND_30D"],
                        "alias": "txn",
                    },
                ],
                output_columns=["A", "B"],
            )
        ]
        fs = MagicMock(name="FeatureStore")
        fs.list_feature_groups.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_group_rows(session, "DB", "SCH", "WH")

        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "USER_FRAUD_FG"
        assert row["version"] == "V1"
        assert row["desc"] == "Combined user signals."
        assert row["owner"] == "ROLE_X"
        assert row["auto_prefix"] is True
        # Sources surfaces as a real Python list[dict] (JSON-decoded).
        assert isinstance(row["sources"], list)
        assert row["sources"][0] == {"fv_name": "USER_CLICK_STATS", "fv_version": "V1"}
        assert row["sources"][1]["slice_columns"] == ["TOTAL_SPEND_30D"]
        assert row["sources"][1]["alias"] == "txn"
        # Output columns also JSON-decoded.
        assert row["output_columns"] == ["A", "B"]
        assert row["database_name"] == "DB"
        assert row["schema_name"] == "SCH"

    def test_sources_pre_decoded_list_passes_through(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_group_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = [
            {
                "NAME": "FG",
                "VERSION": "V1",
                "DESC": "",
                "OWNER": "OWNER",
                "AUTO_PREFIX": False,
                "SOURCES": [{"fv_name": "FV_A", "fv_version": "V1"}],
                "OUTPUT_COLUMNS": ["X"],
            }
        ]
        fs = MagicMock(name="FeatureStore")
        fs.list_feature_groups.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_group_rows(session, "DB", "SCH", "WH")

        assert rows[0]["sources"] == [{"fv_name": "FV_A", "fv_version": "V1"}]
        assert rows[0]["auto_prefix"] is False

    def test_empty_dataframe_yields_empty_list(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_group_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = []
        fs = MagicMock(name="FeatureStore")
        fs.list_feature_groups.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_group_rows(session, "DB", "SCH", "WH")

        assert rows == []


# ---------------------------------------------------------------------------
# Strict-fail on missing name/version
# ---------------------------------------------------------------------------


class TestFetchFeatureGroupRowsStrictFail:
    """Pin the contract that a Snowpark row mapping to an empty ``name``
    or ``version`` raises ``ValueError`` instead of silently skipping
    the row.

    The previous silent-skip behaviour masked upstream regressions
    (e.g. a Snowpark ``_LIST_FEATURE_GROUP_SCHEMA`` column-rename or a
    legacy ``FeatureGroupMetadata`` row written without a version
    field): the FG would simply disappear from the export, the next
    ``snow feature plan`` would diff against an empty applied state,
    and the operator would see a spurious CREATE_FG instead of the
    real metadata problem.  Strict-fail surfaces the root cause at
    fetch time with a row-shape diagnostic the operator can act on.
    """

    def test_empty_version_raises_value_error(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_group_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = [_make_row(name="USER_FRAUD_FG", version="")]
        fs = MagicMock(name="FeatureStore")
        fs.list_feature_groups.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ), pytest.raises(ValueError, match="USER_FRAUD_FG"):
            fetch_feature_group_rows(session, "DB", "SCH", "WH")

    def test_empty_name_raises_value_error(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_group_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        df.collect.return_value = [_make_row(name="", version="V1")]
        fs = MagicMock(name="FeatureStore")
        fs.list_feature_groups.return_value = df

        # The exception message references the offending row shape so
        # the operator can correlate to the broken metadata entry.
        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ), pytest.raises(ValueError, match="(?i)name"):
            fetch_feature_group_rows(session, "DB", "SCH", "WH")

    def test_uppercase_version_key_is_honoured(self) -> None:
        """Regression guard: ``_row_get`` must find ``VERSION`` (the
        canonical Snowpark casing for the lowercase-defined
        ``_LIST_FEATURE_GROUP_SCHEMA`` field) so a well-formed row
        translates without tripping the strict-fail.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_group_rows,
        )

        session = MagicMock(name="session")
        df = MagicMock(name="DataFrame")
        # Force-feed the upper-case Snowpark shape (what cursor paths
        # surface).  _row_get must find VERSION via its case-fallback.
        df.collect.return_value = [
            {
                "NAME": "USER_FRAUD_FG",
                "VERSION": "V1",
                "DESC": "",
                "OWNER": "ROLE",
                "AUTO_PREFIX": True,
                "SOURCES": json.dumps([{"fv_name": "FV_A", "fv_version": "V1"}]),
                "OUTPUT_COLUMNS": None,
            }
        ]
        fs = MagicMock(name="FeatureStore")
        fs.list_feature_groups.return_value = df

        with patch(
            "snowflake.ml.feature_store.feature_store.FeatureStore",
            return_value=fs,
        ):
            rows = fetch_feature_group_rows(session, "DB", "SCH", "WH")

        assert len(rows) == 1
        assert rows[0]["name"] == "USER_FRAUD_FG"
        assert rows[0]["version"] == "V1"


# ---------------------------------------------------------------------------
# Init-first
# ---------------------------------------------------------------------------


class TestFetchFeatureGroupRowsInitFirst:
    def test_raises_feature_store_not_initialized_when_tags_missing(self) -> None:
        from snowflake.ml._internal.exceptions import (
            error_codes,
            exceptions as snowml_exceptions,
        )
        from snowflake.ml.feature_store.decl.errors import (
            FeatureStoreNotInitializedError,
        )
        from snowflake.ml.feature_store.decl.imperative_executor import (
            fetch_feature_group_rows,
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
            fetch_feature_group_rows(session, "DB", "SCH", "WH")

        # Negative pin: no raw-SQL fallback.
        session.sql.assert_not_called()


# ---------------------------------------------------------------------------
# decl_api facade
# ---------------------------------------------------------------------------


class TestFetchFeatureGroupRowsApiFacade:
    def test_decl_api_facade_delegates_to_executor(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        session = MagicMock(name="session")
        with patch(
            "snowflake.ml.feature_store.decl.imperative_executor.fetch_feature_group_rows",
            return_value=[{"name": "FG", "version": "V1"}],
        ) as mock_fetch:
            result = decl_api.fetch_feature_group_rows(session, "DB", "SCH", "WH")

        mock_fetch.assert_called_once_with(session, "DB", "SCH", "WH")
        assert result == [{"name": "FG", "version": "V1"}]


if __name__ == "__main__":
    pytest_driver.main()
