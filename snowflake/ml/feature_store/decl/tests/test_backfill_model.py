"""Tests for the FV-level ``Backfill`` Pydantic model and ``FeatureView.backfill``.

This is the authoring surface for the declarative backfill block introduced
to align with the imperative API:

* Streaming FVs map ``Backfill.{table, start_time}`` onto
  ``StreamConfig.backfill_df`` + ``StreamConfig.backfill_start_time``.
* Batch FVs map ``Backfill.{overwrite, initialize}`` onto
  ``FeatureStore.register_feature_view(overwrite=...)`` and
  ``FeatureView(initialize=...)``.

Cross-field validation lives on ``FeatureView`` itself (it is the layer that
knows the FV kind).  An empty ``backfill: {}`` block is valid and is treated
as a no-op so authors can opt in incrementally without restating defaults.
"""

from __future__ import annotations

import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from snowflake.ml.feature_store.decl.loader import _dict_to_spec
from snowflake.ml.feature_store.decl.spec_models import Backfill, FeatureView
from snowflake.ml.test_utils import pytest_driver


def _streaming_fv_payload(extra: dict[str, Any]) -> dict[str, Any]:
    """Minimal StreamingFeatureView dict for ``_dict_to_spec``.

    Args:
        extra: Extra keys to merge into the payload (typically the
            ``backfill`` block under test).

    Returns:
        Streaming FV payload ready for ``_dict_to_spec``.
    """
    base: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": "USER_CLICK_BACKFILL_DECL",
        "version": "V1",
        "entities": ["USER_ID"],
        "sources": [{"name": "src", "source_type": "Stream"}],
    }
    base.update(extra)
    return base


def _batch_fv_payload(extra: dict[str, Any]) -> dict[str, Any]:
    """Minimal BatchFeatureView dict for ``_dict_to_spec``.

    Args:
        extra: Extra keys to merge into the payload (typically the
            ``backfill`` block under test).

    Returns:
        Batch FV payload ready for ``_dict_to_spec``.
    """
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "ORDERS_TOTAL_DECL",
        "version": "V1",
        "entities": ["ORDER_ID"],
        "sources": [{"name": "src", "source_type": "Table"}],
    }
    base.update(extra)
    return base


class TestBackfillModelStandalone:
    """Direct construction of the ``Backfill`` model.

    The model carries four orthogonal optional fields (``table`` /
    ``start_time`` for streaming, ``overwrite`` / ``initialize`` for batch).
    Cross-field-kind validation lives on ``FeatureView`` — this class only
    verifies the model itself accepts each field shape.
    """

    def test_default_construction_is_empty_block(self) -> None:
        """An empty ``Backfill()`` is the documented no-op shape."""
        b = Backfill()
        assert b.table is None
        assert b.start_time is None
        assert b.overwrite is None or b.overwrite is False
        assert b.initialize is None

    def test_streaming_table_only(self) -> None:
        b = Backfill(table="JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL")
        assert b.table == "JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL"
        assert b.start_time is None

    def test_streaming_start_time_iso_string(self) -> None:
        """Authoring YAML supplies ``start_time`` as an ISO 8601 string."""
        b = Backfill(table="HIST", start_time="2026-05-18T22:00:00")
        assert b.table == "HIST"
        # The model must preserve the value losslessly in some form the
        # executor can later convert to ``datetime`` — either a string or a
        # ``datetime`` instance is acceptable.
        assert b.start_time is not None

    def test_streaming_start_time_datetime_object(self) -> None:
        """Authoring code (or exporter round-trip) may pass a ``datetime``."""
        ts = datetime.datetime(2026, 5, 18, 22, 0, 0)
        b = Backfill(table="HIST", start_time=ts)
        assert b.start_time is not None

    def test_batch_overwrite_only(self) -> None:
        b = Backfill(overwrite=True)
        assert b.overwrite is True
        assert b.table is None
        assert b.start_time is None
        assert b.initialize is None

    def test_batch_initialize_on_create(self) -> None:
        b = Backfill(initialize="ON_CREATE")
        assert b.initialize == "ON_CREATE"

    def test_batch_initialize_on_schedule(self) -> None:
        b = Backfill(initialize="ON_SCHEDULE")
        assert b.initialize == "ON_SCHEDULE"

    def test_batch_initialize_rejects_unknown_token(self) -> None:
        """Only the two imperative tokens are accepted."""
        with pytest.raises(ValidationError) as exc:
            Backfill(initialize="WHENEVER")  # type: ignore[arg-type]
        msg = str(exc.value)
        assert "ON_CREATE" in msg or "ON_SCHEDULE" in msg, (
            "Backfill.initialize validator must surface the legal tokens "
            f"(ON_CREATE / ON_SCHEDULE) in the error; got: {msg}"
        )

    def test_round_trip_through_model_dump(self) -> None:
        """``model_dump`` then ``model_validate`` must preserve every field."""
        b = Backfill(
            table="DB.SCHEMA.HIST",
            start_time="2026-05-18T22:00:00",
            overwrite=True,
            initialize="ON_SCHEDULE",
        )
        dumped = b.model_dump()
        assert dumped["table"] == "DB.SCHEMA.HIST"
        assert dumped["overwrite"] is True
        assert dumped["initialize"] == "ON_SCHEDULE"
        rebuilt = Backfill.model_validate(dumped)
        assert rebuilt.table == "DB.SCHEMA.HIST"
        assert rebuilt.overwrite is True
        assert rebuilt.initialize == "ON_SCHEDULE"


class TestFeatureViewBackfillField:
    """``FeatureView.backfill`` is the canonical authoring surface.

    These tests cover construction only.  Cross-field validation against the
    FV ``kind`` lives on the FeatureView model itself and is exercised in
    :class:`TestFeatureViewBackfillCrossKindValidation`.
    """

    def test_default_backfill_is_none(self) -> None:
        """FVs without a ``backfill:`` block should keep ``None`` for it."""
        fv = FeatureView(name="USER_CLICK_BACKFILL_DECL", version="V1")
        assert hasattr(fv, "backfill"), (
            "FeatureView must expose a `backfill` field so authors can map "
            "Backfill onto StreamConfig (streaming) or "
            "register_feature_view(overwrite=...) (batch)."
        )
        assert fv.backfill is None

    def test_streaming_fv_with_backfill_table(self) -> None:
        fv = FeatureView(
            name="USER_CLICK_BACKFILL_DECL",
            kind="StreamingFeatureView",
            version="V1",
            backfill=Backfill(
                table="JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL",
                start_time="2026-05-18T22:00:00",
            ),
        )
        assert fv.backfill is not None
        assert fv.backfill.table == "JKEW_DB.JKEW_SCHEMA.RAW_CLICK_HISTORY_DECL"

    def test_streaming_fv_with_empty_backfill_block_valid(self) -> None:
        """``backfill: {}`` is a valid no-op (matches the synthesized
        sentinel-row backfill behaviour the executor falls back to when
        ``Backfill.table`` is missing)."""
        fv = FeatureView(
            name="USER_CLICK_BACKFILL_DECL",
            kind="StreamingFeatureView",
            version="V1",
            backfill=Backfill(),
        )
        assert fv.backfill is not None

    def test_batch_fv_with_backfill_overwrite(self) -> None:
        fv = FeatureView(
            name="ORDERS_TOTAL_DECL",
            kind="BatchFeatureView",
            version="V1",
            backfill=Backfill(overwrite=True, initialize="ON_CREATE"),
        )
        assert fv.backfill is not None
        assert fv.backfill.overwrite is True
        assert fv.backfill.initialize == "ON_CREATE"


class TestFeatureViewBackfillCrossKindValidation:
    """Cross-field validation between ``FeatureView.kind`` and ``backfill``.

    The imperative API treats backfill very differently between streaming
    (``StreamConfig``) and batch (``register_feature_view(overwrite=...)``).
    Mixing fields silently is more dangerous than rejecting them, so the
    declarative model surfaces the misuse as a ``ValidationError`` with a
    message that names the offending field.  Both the Python constructor
    and the YAML / JSON loader (``_dict_to_spec``) must raise — the
    messages contain ``"is not valid on"`` so the loader re-raises instead
    of degrading to a bare SpecBase.
    """

    def test_streaming_fv_rejects_overwrite(self) -> None:
        with pytest.raises(ValidationError) as exc:
            FeatureView(
                name="USER_CLICK_BACKFILL_DECL",
                kind="StreamingFeatureView",
                version="V1",
                backfill=Backfill(overwrite=True),
            )
        msg = str(exc.value)
        assert "overwrite" in msg.lower()
        assert "is not valid on" in msg
        assert "streaming" in msg.lower(), (
            "Error must point at the streaming/batch kind mismatch, not just " f"a generic Pydantic message; got: {msg}"
        )

    def test_streaming_fv_rejects_initialize(self) -> None:
        with pytest.raises(ValidationError) as exc:
            FeatureView(
                name="USER_CLICK_BACKFILL_DECL",
                kind="StreamingFeatureView",
                version="V1",
                backfill=Backfill(initialize="ON_CREATE"),
            )
        msg = str(exc.value)
        assert "initialize" in msg.lower()
        assert "is not valid on" in msg

    def test_batch_fv_rejects_table(self) -> None:
        with pytest.raises(ValidationError) as exc:
            FeatureView(
                name="ORDERS_TOTAL_DECL",
                kind="BatchFeatureView",
                version="V1",
                backfill=Backfill(table="DB.SCHEMA.HIST"),
            )
        msg = str(exc.value)
        assert "table" in msg.lower()
        assert "is not valid on" in msg
        assert "batch" in msg.lower()

    def test_batch_fv_rejects_start_time(self) -> None:
        with pytest.raises(ValidationError) as exc:
            FeatureView(
                name="ORDERS_TOTAL_DECL",
                kind="BatchFeatureView",
                version="V1",
                backfill=Backfill(start_time="2026-05-18T22:00:00"),
            )
        msg = str(exc.value)
        assert "start_time" in msg.lower()
        assert "is not valid on" in msg

    def test_yaml_loader_streaming_fv_rejects_overwrite(self) -> None:
        """YAML / JSON path via ``_dict_to_spec`` must re-raise, not SpecBase."""
        with pytest.raises(ValidationError) as exc:
            _dict_to_spec(_streaming_fv_payload({"backfill": {"overwrite": True}}))
        msg = str(exc.value)
        assert "overwrite" in msg.lower()
        assert "is not valid on" in msg

    def test_yaml_loader_streaming_fv_rejects_initialize(self) -> None:
        with pytest.raises(ValidationError) as exc:
            _dict_to_spec(_streaming_fv_payload({"backfill": {"initialize": "ON_CREATE"}}))
        msg = str(exc.value)
        assert "initialize" in msg.lower()
        assert "is not valid on" in msg

    def test_yaml_loader_batch_fv_rejects_table(self) -> None:
        with pytest.raises(ValidationError) as exc:
            _dict_to_spec(_batch_fv_payload({"backfill": {"table": "DB.SCHEMA.HIST"}}))
        msg = str(exc.value)
        assert "table" in msg.lower()
        assert "is not valid on" in msg

    def test_yaml_loader_batch_fv_rejects_start_time(self) -> None:
        with pytest.raises(ValidationError) as exc:
            _dict_to_spec(_batch_fv_payload({"backfill": {"start_time": "2026-05-18T22:00:00"}}))
        msg = str(exc.value)
        assert "start_time" in msg.lower()
        assert "is not valid on" in msg


if __name__ == "__main__":
    pytest_driver.main()
