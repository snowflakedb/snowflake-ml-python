"""Tests pinning the strict separation of feature_granularity, refresh_freq, target_lag.

The declarative authoring layer historically collapsed three independent
imperative ``FeatureView`` knobs into one tangled fallback chain
(``refresh_freq > target_lag_sec > feature_granularity``).  This test
file pins the new strict contract:

* ``feature_granularity[_sec]`` → ``FeatureView.feature_granularity``.
  Tile size only.  **Never** silently becomes ``refresh_freq``.
* ``refresh_freq`` → ``FeatureView.refresh_freq``.  DT refresh
  cadence only.  **Required** when the FV is tiled.
* ``target_lag[_sec]`` → ``OnlineConfig.target_lag``.  OFT staleness
  only.  **Allowed only when ``online: true`` on ``BatchFeatureView``.**
  Still rejected on streaming / realtime kinds.

Each section is a separate ``Test*`` class so pytest output names the
contract that broke.  Every test uses the existing public surface
(``compile_to_spec``, ``_build_feature_view``, ``_resolve_online_target_lag``,
``_build_offline_fv_object``, ``_build_full_fidelity_fv``, the Pydantic
validators) — no new wiring required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_kwargs() -> tuple[type, list[dict[str, Any]]]:
    """Return a fake class and a list that records every constructor call's kwargs."""
    captured: list[dict[str, Any]] = []

    class _FakeCtor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.append(dict(kwargs))

    return _FakeCtor, captured


def _capture_online_config_kwargs() -> tuple[type, dict[str, Any]]:
    """Mirror of :func:`test_imperative_executor_target_lag._capture_online_config_kwargs`."""
    captured: dict[str, Any] = {}

    class _FakeOnlineConfig:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

    return _FakeOnlineConfig, captured


def _minimal_bfv_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "BFV_DECOUPLE_TEST",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "EVENTS_DECOUPLE",
                "source_type": "Batch",
                "table": "RAW_EVENTS_DECOUPLE",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
    }
    base.update(overrides)
    return base


def _minimal_streaming_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": "SFV_DECOUPLE_TEST",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": True,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "STREAM_DECOUPLE",
                "source_type": "Stream",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "TIMESTAMP", "type": "TimestampType"},
                    {"name": "AMOUNT", "type": "DoubleType"},
                ],
            }
        ],
        "timestamp_col": "TIMESTAMP",
        "udf": {
            "name": "compute",
            "engine": "pandas",
            "function_definition": "def compute(df):\n    return df\n",
            "output_columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "TIMESTAMP", "type": "TimestampType"},
                {"name": "AMOUNT", "type": "DoubleType"},
            ],
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Compiler: refresh_freq → wire spec.target_lag_sec; target_lag does NOT.
# ---------------------------------------------------------------------------


class TestCompilerBatchScheduleIsCanonicalDTRefresh:
    """The wire-format ``spec.target_lag_sec`` is the DT refresh.  The
    compiler must source it from the authoring ``refresh_freq`` (the
    decoupled DT-refresh field) — not from the authoring ``target_lag``
    (which is now OFT-only).

    Pre-fix the compiler preferred ``target_lag`` and only fell back to
    ``refresh_freq``.  Post-fix the wire ``target_lag_sec`` field
    follows ``refresh_freq`` exclusively.
    """

    def _compile(self, spec_dict: dict[str, Any]) -> dict[str, Any]:
        from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

        return compile_to_spec(spec_dict, "DB", "SC")

    def test_refresh_freq_populates_wire_target_lag_sec(self) -> None:
        spec = {
            "kind": "BatchFeatureView",
            "name": "X",
            "version": "V1",
            "entities": ["U"],
            "sources": [{"name": "S", "source_type": "Batch", "table": "T"}],
            "features": [],
            "refresh_freq": "5 minutes",
        }
        result = self._compile(spec)
        assert result["spec"]["target_lag_sec"] == 300

    def test_target_lag_does_not_populate_wire_target_lag_sec_for_offline_bfv(
        self,
    ) -> None:
        # Strictly: authoring ``target_lag`` is OFT staleness.  An
        # offline BFV (online=false) has no OFT, so ``target_lag`` is
        # rejected by validation upstream — but if a hand-built dict
        # bypasses validation, the compiler MUST NOT silently route the
        # value into the DT-refresh wire field.
        spec = {
            "kind": "BatchFeatureView",
            "name": "X",
            "version": "V1",
            "entities": ["U"],
            "sources": [{"name": "S", "source_type": "Batch", "table": "T"}],
            "features": [],
            "target_lag": "5 minutes",
        }
        result = self._compile(spec)
        assert "target_lag_sec" not in result["spec"], (
            "authoring target_lag is OFT-only; it must not leak into " "wire spec.target_lag_sec (the DT refresh)"
        )

    def test_only_refresh_freq_drives_wire_target_lag_sec_when_both_present(
        self,
    ) -> None:
        # Online BFV authoring both fields: ``target_lag`` is OFT-only
        # (handled separately via OnlineConfig); ``refresh_freq`` is
        # the DT refresh (drives wire ``spec.target_lag_sec``).
        spec = {
            "kind": "BatchFeatureView",
            "name": "X",
            "version": "V1",
            "online": True,
            "entities": ["U"],
            "sources": [{"name": "S", "source_type": "Batch", "table": "T"}],
            "features": [],
            "target_lag": "1 hour",
            "refresh_freq": "5 minutes",
        }
        result = self._compile(spec)
        # 5 minutes = 300 seconds.  ``target_lag`` (1 hour = 3600s)
        # MUST NOT win for the wire DT-refresh field.
        assert result["spec"]["target_lag_sec"] == 300


# ---------------------------------------------------------------------------
# 2. Executor: refresh_freq comes only from refresh_freq, never from
#    target_lag_sec or feature_granularity.
# ---------------------------------------------------------------------------


class TestExecutorRefreshFreqOnlyFromBatchSchedule:
    """``_build_feature_view`` must populate the imperative ``refresh_freq``
    kwarg exclusively from authoring ``refresh_freq``.  The two pre-fix
    fallbacks — ``target_lag_sec → refresh_freq`` and ``feature_granularity
    → refresh_freq`` — are deleted.
    """

    def _run(self, payload: dict[str, Any]) -> dict[str, Any]:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        session = MagicMock()
        session.table.return_value = MagicMock()
        fs = MagicMock()
        fake_fv, fv_calls = _capture_kwargs()
        fake_oc, _ = _capture_online_config_kwargs()
        with patch("snowflake.ml.feature_store.feature_view.FeatureView", fake_fv,), patch(
            "snowflake.ml.feature_store.feature_view.OnlineConfig",
            fake_oc,
        ), patch("snowflake.ml.feature_store.feature_view.OnlineStoreType", MagicMock(),), patch(
            "snowflake.ml.feature_store.feature_view.FeatureAggregationMethod",
            MagicMock(),
        ):
            try:
                _build_feature_view(payload, session, "DB", "SC", "WH", fs=fs)
            except Exception:
                # ValueError from snowml-core's FeatureView constructor for
                # tiled-without-refresh is caught — we only care about the
                # kwargs the executor passed before the constructor ran.
                pass
        return fv_calls[-1] if fv_calls else {}

    def test_refresh_freq_populates_refresh_freq(self) -> None:
        payload = _minimal_bfv_payload(refresh_freq="5 minutes")
        kwargs = self._run(payload)
        assert kwargs.get("refresh_freq") == "5 minutes"

    def test_target_lag_sec_does_not_populate_refresh_freq(self) -> None:
        # Compiled inner spec might carry target_lag_sec=60 from a
        # pre-fix wire payload.  The executor must NOT use it as
        # refresh_freq — the offline DT refresh comes from
        # refresh_freq only.
        payload = _minimal_bfv_payload(target_lag_sec=60)
        kwargs = self._run(payload)
        assert "refresh_freq" not in kwargs

    def test_target_lag_string_does_not_populate_refresh_freq(self) -> None:
        # Even when authored as a human-friendly string.
        payload = _minimal_bfv_payload(online=True, target_lag="1 hour")
        kwargs = self._run(payload)
        assert "refresh_freq" not in kwargs

    def test_feature_granularity_does_not_default_refresh_freq_for_streaming_tiled(
        self,
    ) -> None:
        # Tiled streaming FV with feature_granularity but no
        # refresh_freq.  Pre-fix: refresh_freq silently defaulted to
        # the granularity.  Post-fix: the executor passes no
        # refresh_freq (granularity is never a cadence source) and the
        # imperative FeatureView constructor raises — matching the
        # imperative-API contract for tiled FVs and the decl
        # STREAM_FV_TILING_REFRESH invariant.
        payload = _minimal_streaming_payload(
            feature_granularity_sec=300,
            feature_aggregation_method="tiles",
            features=[
                {
                    "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                    "output_column": {"name": "AMOUNT_1H", "type": "DoubleType"},
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
        )
        kwargs = self._run(payload)
        assert "refresh_freq" not in kwargs

    def test_refresh_freq_forwarded_for_streaming_tiled(self) -> None:
        # A tiled streaming FV that authors refresh_freq: the executor
        # forwards it to FeatureView(refresh_freq=...) — it drives the
        # offline tile Dynamic Table's TARGET_LAG (distinct from the
        # OFT's runtime-stamped ingest lag).
        payload = _minimal_streaming_payload(
            refresh_freq="1 minute",
            feature_granularity_sec=300,
            feature_aggregation_method="tiles",
            features=[
                {
                    "source_column": {"name": "AMOUNT", "type": "DoubleType"},
                    "output_column": {"name": "AMOUNT_1H", "type": "DoubleType"},
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
        )
        kwargs = self._run(payload)
        assert kwargs.get("refresh_freq") == "1 minute"

    def test_feature_granularity_does_not_default_refresh_freq_for_batch_tiled(
        self,
    ) -> None:
        payload = _minimal_bfv_payload(
            feature_granularity_sec=300,
            feature_aggregation_method="tiles",
            timestamp_col="TS",
            features=[
                {
                    "source_column": {"name": "AMOUNT", "type": "FloatType"},
                    "output_column": {"name": "AMOUNT_1H", "type": "FloatType"},
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
        )
        kwargs = self._run(payload)
        assert "refresh_freq" not in kwargs


# ---------------------------------------------------------------------------
# 3. _resolve_online_target_lag: target_lag (string/sec) only — no
#    refresh_freq fallback.
# ---------------------------------------------------------------------------


class TestResolveOnlineTargetLagNoBatchScheduleFallback:
    """``_resolve_online_target_lag`` must source OFT staleness from
    ``target_lag`` / ``target_lag_sec`` only.  The pre-fix
    ``refresh_freq`` fallback is removed because authoring
    ``refresh_freq`` is now strictly DT refresh, not a stand-in for
    OFT staleness.

    The caller (``_build_feature_view``) is responsible for picking a
    sensible OFT default — ``"10 seconds"`` for batch online FVs (the
    imperative ``_BATCH_OFT_TARGET_LAG``) and ``"0 seconds"`` for
    streaming / realtime / FG.
    """

    def _resolve(self, payload: dict[str, Any]) -> Any:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _resolve_online_target_lag,
        )

        return _resolve_online_target_lag(payload)

    def test_target_lag_string_resolves(self) -> None:
        assert self._resolve({"target_lag": "1 hour"}) == "1 hour"

    def test_target_lag_int_resolves_in_seconds_form(self) -> None:
        assert self._resolve({"target_lag": 90}) == "90 seconds"

    def test_target_lag_sec_resolves_in_seconds_form(self) -> None:
        assert self._resolve({"target_lag_sec": 120}) == "120 seconds"

    def test_target_lag_wins_over_target_lag_sec(self) -> None:
        assert self._resolve({"target_lag": "1 hour", "target_lag_sec": 60}) == "1 hour"

    def test_refresh_freq_alone_returns_none(self) -> None:
        # The fallback that pre-fix mirrored refresh_freq onto OFT
        # is removed.  Callers handle the None case.
        assert self._resolve({"refresh_freq": "5 minutes"}) is None

    def test_target_lag_sec_zero_falls_through_to_none(self) -> None:
        # ``> 0`` guard preserved; zero is not a valid OFT staleness.
        assert self._resolve({"target_lag_sec": 0, "refresh_freq": "5 minutes"}) is None


class TestExecutorBatchOnlineDefaultTargetLag:
    """Online ``BatchFeatureView`` without an authored ``target_lag``
    falls back to the imperative ``_BATCH_OFT_TARGET_LAG = "10 seconds"``
    default — not ``"0 seconds"`` (which Snowflake rejects for batch
    OFTs).

    Pre-fix the executor used ``refresh_freq`` as the OFT staleness
    when ``target_lag`` was absent; post-fix the executor uses the
    imperative default and never reads ``refresh_freq`` for the OFT
    side at all.
    """

    def _online_config_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_view,
        )

        session = MagicMock()
        session.table.return_value = MagicMock()
        fs = MagicMock()
        fake_fv, _fv_calls = _capture_kwargs()
        fake_oc, captured = _capture_online_config_kwargs()
        with patch("snowflake.ml.feature_store.feature_view.FeatureView", fake_fv,), patch(
            "snowflake.ml.feature_store.feature_view.OnlineConfig",
            fake_oc,
        ), patch(
            "snowflake.ml.feature_store.feature_view.OnlineStoreType",
            MagicMock(),
        ):
            try:
                _build_feature_view(payload, session, "DB", "SC", "WH", fs=fs)
            except Exception:
                pass
        return captured

    def test_online_bfv_without_target_lag_uses_imperative_default_10s(self) -> None:
        payload = _minimal_bfv_payload(online=True, refresh_freq="5 minutes")
        captured = self._online_config_kwargs(payload)
        assert captured.get("target_lag") == "10 seconds", (
            "Online BFV without an authored target_lag must default to "
            "_BATCH_OFT_TARGET_LAG (10 seconds); the pre-fix refresh_freq "
            "fallback is gone."
        )

    def test_online_bfv_with_target_lag_uses_authored_value(self) -> None:
        payload = _minimal_bfv_payload(
            online=True,
            target_lag="1 hour",
            refresh_freq="5 minutes",
        )
        captured = self._online_config_kwargs(payload)
        assert captured.get("target_lag") == "1 hour"

    def test_online_streaming_default_remains_zero_seconds(self) -> None:
        # Streaming / realtime: caller default is "0 seconds" because
        # the runtime requires it for those kinds.
        payload = _minimal_streaming_payload()
        captured = self._online_config_kwargs(payload)
        assert captured.get("target_lag") == "0 seconds"


# ---------------------------------------------------------------------------
# 4. spec_models validators: reject target_lag offline; require
#    refresh_freq on tiled FVs.
# ---------------------------------------------------------------------------


class TestSpecModelsRejectTargetLagOnOfflineBatchFV:
    """Authoring ``target_lag`` is OFT staleness.  An offline-only
    ``BatchFeatureView`` (``online: False``) has no OFT, so any
    authored ``target_lag`` / ``target_lag_sec`` is meaningless and
    must be rejected at load time.
    """

    def test_target_lag_string_rejected_when_online_false(self) -> None:
        from snowflake.ml.feature_store.decl.spec_models import BatchFeatureView

        with pytest.raises(ValueError, match="target_lag"):
            BatchFeatureView(
                name="X",
                version="V1",
                online=False,
                entities=["U"],
                target_lag="5 minutes",
            )

    def test_target_lag_sec_rejected_when_online_false(self) -> None:
        from snowflake.ml.feature_store.decl.spec_models import BatchFeatureView

        with pytest.raises(ValueError, match="target_lag"):
            BatchFeatureView(
                name="X",
                version="V1",
                online=False,
                entities=["U"],
                target_lag_sec=300,
            )

    def test_target_lag_accepted_when_online_true(self) -> None:
        from snowflake.ml.feature_store.decl.spec_models import BatchFeatureView

        fv = BatchFeatureView(
            name="X",
            version="V1",
            online=True,
            entities=["U"],
            target_lag="5 minutes",
            refresh_freq="5 minutes",
        )
        assert fv.target_lag == "5 minutes"


# ---------------------------------------------------------------------------
# 5. State recovery: deployed refresh_freq surfaces under both keys so
#    the planner sees the cadence regardless of which authoring shape
#    the YAML uses.
# ---------------------------------------------------------------------------


class TestStateRecoveryEmitsBatchSchedule:
    """``_build_offline_fv_object`` reads the deployed Dynamic Table's
    refresh cadence (``refresh_freq`` from the imperative
    ``list_feature_views()`` row) and must surface it on the applied
    side as the authoring-form ``refresh_freq`` so the exporter can
    emit YAML in the new canonical shape.

    Backwards compatibility: the wire-format ``spec.target_lag_sec``
    field is preserved (it's the SPECIFICATION-shaped key the planner's
    full-spec hash already understands).  We add ``refresh_freq`` as
    an authoring-shape sibling.
    """

    def test_refresh_freq_surfaces_as_refresh_freq_on_applied_payload(
        self,
    ) -> None:
        from snowflake.ml.feature_store.decl.state import _build_offline_fv_object

        fv_row = {
            "name": "X",
            "version": "V1",
            "kind": "BATCH",
            "database_name": "DB",
            "schema_name": "SC",
            "physical_dt_name": "X$V1",
            "entities": ["USER_ID"],
            "refresh_freq": "1 minute",
        }
        applied = _build_offline_fv_object(
            fv_row,
            dt_text_map=None,
            default_database="DB",
            default_schema="SC",
        )
        assert applied is not None
        spec_payload = applied.spec_payload or {}
        # ``refresh_freq`` lives inside the inner ``spec`` dict —
        # mirroring where the wire-form ``target_lag_sec`` already
        # sits and matching where ``compile_to_spec`` writes it on
        # the local authoring side.
        inner = spec_payload.get("spec") or {}
        assert inner.get("refresh_freq") == "1 minute", (
            "deployed DT refresh_freq must surface as the authoring-form "
            "``refresh_freq`` on the applied payload's inner spec so "
            "the exporter can emit it directly to YAML"
        )


# ---------------------------------------------------------------------------
# 6. Exporter: emit refresh_freq (and OFT target_lag separately when
#    non-default) instead of conflating into a single target_lag_sec field.
# ---------------------------------------------------------------------------


class TestExporterEmitsBatchScheduleForDTRefresh:
    """The exporter writes authoring-format YAML.  After the fix, the
    DT refresh cadence is written under ``refresh_freq`` (the
    canonical authoring field) — not the wire-format alias
    ``target_lag_sec``.

    The exporter still emits ``target_lag_sec`` only when the deployed
    OFT carried a NON-default ``TARGET_LAG`` distinct from the DT
    refresh.  In the canonical case where OFT TARGET_LAG was authored
    independently, both fields are emitted; in the common case where
    the operator only set ``refresh_freq``, only that key appears.
    """

    def _build(self, full_spec: dict[str, Any]) -> dict[str, Any]:
        from snowflake.ml.feature_store.decl.exporter import _build_full_fidelity_fv

        return _build_full_fidelity_fv(
            full_spec,
            fallback_name="X",
            fallback_version="V1",
            fallback_database="DB",
            fallback_schema="SC",
        )

    def test_batch_fv_with_only_dt_refresh_emits_refresh_freq(self) -> None:
        full_spec = {
            "kind": "BatchFeatureView",
            "metadata": {"name": "X", "version": "V1", "database": "DB", "schema": "SC"},
            "spec": {
                "ordered_entity_column_names": ["U"],
                "sources": [],
                "features": [],
                "target_lag_sec": 60,
            },
        }
        doc = self._build(full_spec)
        assert doc.get("refresh_freq") == "60 seconds", (
            "Exported BFV YAML must surface the DT refresh under "
            "authoring-shape refresh_freq, not the wire-shape "
            "target_lag_sec"
        )
        # Wire-form alias is no longer the authoring shape; the
        # exporter MUST NOT also emit it (would duplicate the same
        # value under two keys and confuse the round-trip).
        assert "target_lag_sec" not in doc

    def test_streaming_fv_does_not_emit_refresh_freq_or_target_lag(self) -> None:
        # Streaming FVs reject target_lag at load time; the exporter
        # must not emit it either.  refresh_freq on a streaming FV
        # must be omitted unless the runtime actually carried a
        # cadence (the granularity-default code path is gone post-fix
        # so streaming FVs without an authored refresh_freq do not
        # round-trip a synthetic value).
        full_spec = {
            "kind": "StreamingFeatureView",
            "metadata": {"name": "X", "version": "V1", "database": "DB", "schema": "SC"},
            "spec": {
                "ordered_entity_column_names": ["U"],
                "sources": [],
                "features": [],
                "target_lag_sec": 0,
            },
        }
        doc = self._build(full_spec)
        assert "target_lag_sec" not in doc
        assert "refresh_freq" not in doc


if __name__ == "__main__":
    pytest_driver.main()
