"""Tests for the implicit ``online=True`` contract on streaming / realtime FVs.

``StreamingFeatureView`` and ``RealtimeFeatureView`` are always online by
design — the Snowflake runtime materialises an Online Feature Table for
every deployed instance.  Authors should therefore not have to type
``online=True`` / ``online: true`` on these kinds, and an explicit
``online=False`` is a contradiction that must be rejected at load time
with a clear, named error.

These tests cover the four authoring entry points the loader and the
Python form expose:

1. ``StreamingFeatureView(name=...)`` direct subclass construction.
2. ``RealtimeFeatureView(name=...)`` direct subclass construction.
3. ``FeatureView.model_validate({"kind": "StreamingFeatureView", ...})``
   YAML-form parity — the path the loader's ``_dict_to_spec`` falls
   through when ``kind_map`` does not match (defensive coverage).
4. ``compile_to_spec(...)`` against a streaming dict with ``online``
   omitted — the on-wire compiled spec must still carry
   ``online_store_type == "postgres"``.

``BatchFeatureView`` is unchanged — its ``online`` field stays
first-class because batch FVs can legitimately be offline-only.
"""

import pytest
from pydantic import ValidationError

from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import (
    BatchFeatureView,
    FeatureView,
    RealtimeFeatureView,
    StreamingFeatureView,
)
from snowflake.ml.test_utils import pytest_driver


class TestStreamingFeatureViewOnlineDefault:
    """``StreamingFeatureView()`` is always online by design."""

    def test_default_online_is_true(self) -> None:
        fv = StreamingFeatureView(name="MY_STREAM_FV")
        assert fv.online is True

    def test_explicit_online_true_still_works(self) -> None:
        """Back-compat: existing YAML / Python that sets ``online=True`` continues to load."""
        fv = StreamingFeatureView(name="MY_STREAM_FV", online=True)
        assert fv.online is True

    def test_explicit_online_false_is_rejected(self) -> None:
        """An explicit ``online=False`` is a contradiction — fail loud."""
        with pytest.raises(ValidationError) as excinfo:
            StreamingFeatureView(name="MY_STREAM_FV", online=False)
        msg = str(excinfo.value)
        assert "MY_STREAM_FV" in msg
        assert "always online by design" in msg


class TestRealtimeFeatureViewOnlineDefault:
    """``RealtimeFeatureView()`` mirrors the streaming contract."""

    def test_default_online_is_true(self) -> None:
        fv = RealtimeFeatureView(name="MY_REALTIME_FV")
        assert fv.online is True

    def test_explicit_online_true_still_works(self) -> None:
        fv = RealtimeFeatureView(name="MY_REALTIME_FV", online=True)
        assert fv.online is True

    def test_explicit_online_false_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RealtimeFeatureView(name="MY_REALTIME_FV", online=False)
        msg = str(excinfo.value)
        assert "MY_REALTIME_FV" in msg
        assert "always online by design" in msg


class TestBatchFeatureViewOnlineUnchanged:
    """``BatchFeatureView`` keeps the legacy ``online: bool = False`` default."""

    def test_default_online_is_false(self) -> None:
        fv = BatchFeatureView(name="MY_BATCH_FV")
        assert fv.online is False

    def test_explicit_online_true_accepted(self) -> None:
        fv = BatchFeatureView(name="MY_BATCH_FV", online=True)
        assert fv.online is True

    def test_explicit_online_false_accepted(self) -> None:
        """Batch FVs are legitimately offline-only — no rejection."""
        fv = BatchFeatureView(name="MY_BATCH_FV", online=False)
        assert fv.online is False


class TestKindDispatchYamlParity:
    """``FeatureView.model_validate({"kind": ..., ...})`` honours the contract.

    The loader's ``_dict_to_spec`` normally dispatches to the matching
    subclass via the ``kind`` discriminator, but the base-class
    validation path is also exercised by direct callers (and is the
    cleanest single source of truth for the always-online rule).
    """

    def test_streaming_kind_defaults_online_true(self) -> None:
        fv = FeatureView.model_validate({"kind": "StreamingFeatureView", "name": "MY_STREAM_FV"})
        assert fv.online is True

    def test_realtime_kind_defaults_online_true(self) -> None:
        fv = FeatureView.model_validate({"kind": "RealtimeFeatureView", "name": "MY_REALTIME_FV"})
        assert fv.online is True

    def test_streaming_kind_rejects_explicit_online_false(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            FeatureView.model_validate(
                {
                    "kind": "StreamingFeatureView",
                    "name": "MY_STREAM_FV",
                    "online": False,
                }
            )
        msg = str(excinfo.value)
        assert "MY_STREAM_FV" in msg
        assert "always online by design" in msg

    def test_realtime_kind_rejects_explicit_online_false(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            FeatureView.model_validate(
                {
                    "kind": "RealtimeFeatureView",
                    "name": "MY_REALTIME_FV",
                    "online": False,
                }
            )
        msg = str(excinfo.value)
        assert "MY_REALTIME_FV" in msg
        assert "always online by design" in msg

    def test_batch_kind_does_not_alter_online(self) -> None:
        fv = FeatureView.model_validate({"kind": "BatchFeatureView", "name": "MY_BATCH_FV", "online": False})
        assert fv.online is False


class TestSpecCompilerIntegration:
    """``compile_to_spec`` honours the implicit-online default for streaming."""

    def test_streaming_without_online_compiles_to_postgres(self) -> None:
        """A streaming spec dict that omits ``online`` must still produce
        ``online_store_type: postgres`` in the compiled root."""
        spec_dict = {
            "kind": "StreamingFeatureView",
            "name": "MY_STREAM_FV",
            "version": "V1",
            "entities": ["USER_ID"],
            "timestamp_col": "TIMESTAMP",
            "feature_granularity_sec": 300,
            "sources": [
                {
                    "name": "EVENTS",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "TIMESTAMP", "type": "TimestampType"},
                    ],
                }
            ],
            "features": [],
        }
        result = compile_to_spec(spec_dict, "DEMO_DB", "PUBLIC")
        assert result.get("online_store_type") == "postgres"

    def test_realtime_without_online_compiles_to_postgres(self) -> None:
        spec_dict = {
            "kind": "RealtimeFeatureView",
            "name": "MY_REALTIME_FV",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [],
            "features": [],
        }
        result = compile_to_spec(spec_dict, "DEMO_DB", "PUBLIC")
        assert result.get("online_store_type") == "postgres"

    def test_batch_without_online_compiles_offline(self) -> None:
        """Batch default stays offline so the compiler must not emit
        ``online_store_type`` when the authoring dict omits ``online``."""
        spec_dict = {
            "kind": "BatchFeatureView",
            "name": "MY_BATCH_FV",
            "version": "V1",
            "entities": ["USER_ID"],
            "sources": [
                {
                    "name": "ORDERS",
                    "source_type": "Batch",
                    "table": "RAW_ORDERS",
                }
            ],
            "features": [],
        }
        result = compile_to_spec(spec_dict, "DEMO_DB", "PUBLIC")
        assert "online_store_type" not in result


if __name__ == "__main__":
    pytest_driver.main()
