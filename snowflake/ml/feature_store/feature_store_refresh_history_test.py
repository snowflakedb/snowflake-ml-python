"""Unit tests for ``FeatureStore._get_offline_refresh_history`` SQL composition."""

import warnings
from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import FeatureView, FeatureViewVersion
from snowflake.ml.feature_store.metadata_manager import StreamingMetadata


class _FakeFSConfig:
    """Minimal stand-in for FeatureStore._config used by _get_offline_refresh_history."""

    def __init__(self, *, database: str = "MY_DB", schema: str = "MY_SCH") -> None:
        self.database = SqlIdentifier(database)
        self.schema = SqlIdentifier(schema)


class _CapturingSession:
    """Captures every ``session.sql(...)`` call and returns a sentinel DataFrame."""

    def __init__(self) -> None:
        self.sql_calls: list[str] = []

    def sql(self, query: str) -> object:
        self.sql_calls.append(query)
        return MagicMock(name="DataFrame")


def _make_streaming_fv(*, name: str = "fv", version: str = "v1") -> FeatureView:
    """Streaming FV stub bypassing stream_config validation.

    ``_get_offline_refresh_history`` only inspects ``is_streaming``, ``name``,
    and ``version``, so a real stream_config is not needed.
    """
    fv = object.__new__(FeatureView)
    object.__setattr__(fv, "_name", SqlIdentifier(name))
    object.__setattr__(fv, "_version", FeatureViewVersion(version))
    object.__setattr__(fv, "_stream_config", None)
    object.__setattr__(fv, "_is_streaming_marker", True)
    return fv


def _make_offline_fv(*, name: str = "fv", version: str = "v1") -> FeatureView:
    fv = object.__new__(FeatureView)
    object.__setattr__(fv, "_name", SqlIdentifier(name))
    object.__setattr__(fv, "_version", FeatureViewVersion(version))
    object.__setattr__(fv, "_stream_config", None)
    object.__setattr__(fv, "_is_streaming_marker", False)
    return fv


def _make_fs(
    session: _CapturingSession,
    *,
    streaming_meta_for: tuple[FeatureView, StreamingMetadata] | None = None,
) -> FeatureStore:
    fs = object.__new__(FeatureStore)
    object.__setattr__(fs, "_session", session)
    object.__setattr__(fs, "_config", _FakeFSConfig())
    metadata_manager = MagicMock()
    if streaming_meta_for is not None:
        fv, meta = streaming_meta_for

        def _get(name: str, version: str) -> StreamingMetadata | None:
            if name == str(fv.name) and version == str(fv.version):
                return meta
            return None

        metadata_manager.get_streaming_metadata.side_effect = _get
    else:
        metadata_manager.get_streaming_metadata.return_value = None
    object.__setattr__(fs, "_metadata_manager", metadata_manager)
    return fs


class GetOfflineRefreshHistoryTest(absltest.TestCase):
    """Tests for SQL composition in ``_get_offline_refresh_history``."""

    def test_non_streaming_fv_dt_only(self) -> None:
        """Non-streaming FVs produce DT-only SQL with no UNION ALL."""
        session = _CapturingSession()
        fs = _make_fs(session)
        fv = _make_offline_fv(name="MY_FV", version="V1")

        fs._get_offline_refresh_history(fv, verbose=False)

        self.assertEqual(len(session.sql_calls), 1)
        sql = session.sql_calls[0]
        self.assertIn("DYNAMIC_TABLE_REFRESH_HISTORY", sql)
        self.assertNotIn("UNION ALL", sql)
        self.assertNotIn("TASK_HISTORY", sql)
        self.assertNotIn("BACKFILL", sql)

    def test_streaming_without_backfill_metadata_dt_only(self) -> None:
        """Legacy streaming FVs (no backfill_root_task_name) produce DT-only SQL."""
        session = _CapturingSession()
        fv = _make_streaming_fv(name="MY_FV", version="V1")
        # streaming metadata exists but has no backfill task names.
        meta = StreamingMetadata(stream_source_name="SRC", transformation_fn_name="fn")
        fs = _make_fs(session, streaming_meta_for=(fv, meta))

        fs._get_offline_refresh_history(fv, verbose=False)

        self.assertEqual(len(session.sql_calls), 1)
        sql = session.sql_calls[0]
        self.assertIn("DYNAMIC_TABLE_REFRESH_HISTORY", sql)
        self.assertNotIn("UNION ALL", sql)
        self.assertNotIn("TASK_HISTORY", sql)

    def test_streaming_with_backfill_metadata_unions_task_history(self) -> None:
        """Streaming FVs with a backfill root task UNION ALL with TASK_HISTORY rows.

        User-facing projection: backfill rows are labeled with the FV's
        physical name (matching the dynamic-table rows above) and only
        ``REFRESH_ACTION='BACKFILL'`` distinguishes them. The internal
        finalizer task is filtered out — it has no user-meaningful
        refresh semantics.
        """
        session = _CapturingSession()
        fv = _make_streaming_fv(name="MY_FV", version="V1")
        meta = StreamingMetadata(
            stream_source_name="SRC",
            transformation_fn_name="fn",
            backfill_root_task_name="MY_DB.MY_SCH.MY_FV$V1$BACKFILL_ROOT",
            backfill_finalize_task_name="MY_DB.MY_SCH.MY_FV$V1$BACKFILL_FINALIZE",
        )
        fs = _make_fs(session, streaming_meta_for=(fv, meta))

        fs._get_offline_refresh_history(fv, verbose=False)

        self.assertEqual(len(session.sql_calls), 1)
        sql = session.sql_calls[0]
        self.assertIn("DYNAMIC_TABLE_REFRESH_HISTORY", sql)
        self.assertIn("TASK_HISTORY", sql)
        self.assertIn("UNION ALL", sql)
        self.assertIn("'BACKFILL'", sql)
        # NAME column projects the FV's physical name as a literal — not
        # the underlying task name, so users see "MY_FV$V1" in both
        # incremental and backfill rows.
        self.assertIn("'MY_FV$V1'", sql)
        # User-visible match set: exact root + LIKE on window children.
        self.assertIn("NAME = 'MY_FV$V1$BACKFILL_ROOT'", sql)
        self.assertIn("NAME LIKE 'MY_FV$V1$BACKFILL_W%'", sql)
        # Finalizer task is intentionally excluded from the user-facing UNION.
        self.assertNotIn("BACKFILL_FINALIZE", sql)
        self.assertIn("ORDER BY refresh_start_time DESC", sql)

    def test_streaming_verbose_emits_warning_and_returns_dt_only(self) -> None:
        """Verbose mode on a streaming FV emits a UserWarning and keeps the DT-only schema."""
        session = _CapturingSession()
        fv = _make_streaming_fv(name="MY_FV", version="V1")
        meta = StreamingMetadata(
            stream_source_name="SRC",
            transformation_fn_name="fn",
            backfill_root_task_name="MY_DB.MY_SCH.MY_FV$V1$BACKFILL_ROOT",
            backfill_finalize_task_name="MY_DB.MY_SCH.MY_FV$V1$BACKFILL_FINALIZE",
        )
        fs = _make_fs(session, streaming_meta_for=(fv, meta))

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fs._get_offline_refresh_history(fv, verbose=True)

        warning_messages = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
        self.assertTrue(
            any("backfill task history is not included in verbose mode" in m for m in warning_messages),
            f"expected warning, got: {warning_messages!r}",
        )

        self.assertEqual(len(session.sql_calls), 1)
        sql = session.sql_calls[0]
        self.assertIn("SELECT *", sql)
        self.assertIn("DYNAMIC_TABLE_REFRESH_HISTORY", sql)
        self.assertNotIn("UNION ALL", sql)
        self.assertNotIn("TASK_HISTORY", sql)

    def test_non_streaming_verbose_no_warning(self) -> None:
        """Verbose mode on a non-streaming FV does not emit the streaming warning."""
        session = _CapturingSession()
        fs = _make_fs(session)
        fv = _make_offline_fv(name="MY_FV", version="V1")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fs._get_offline_refresh_history(fv, verbose=True)

        for w in captured:
            self.assertNotIn("backfill task history", str(w.message))


if __name__ == "__main__":
    absltest.main()
