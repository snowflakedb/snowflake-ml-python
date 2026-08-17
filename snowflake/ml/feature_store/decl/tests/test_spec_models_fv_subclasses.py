"""Pins for the Python authoring form's ``FeatureView`` subclasses.

Per Q2 of ``plans/python_form/python_authoring_form.md``, the Python
authoring path discriminates feature view kinds via Python class type,
not via a string ``kind:`` field.  Authors write
``StreamingFeatureView(...)``, ``BatchFeatureView(...)``, or
``RealtimeFeatureView(...)`` — three concrete subclasses of the existing
``FeatureView`` Pydantic model.

These tests pin three contracts:

1. Each subclass pins ``kind`` to its class name as a Pydantic class
   default, so authors never type the kind string by hand in the Python
   form.
2. The subclasses are *real* subclasses (``isinstance``-detectable) and
   continue to honour every inherited FV validator (backfill
   cross-kind checks, etc.).
3. The legacy YAML / JSON path is byte-stable: a dict carrying
   ``kind: "StreamingFeatureView"`` (etc.) still validates against the
   base ``FeatureView`` class, AND ``loader._dict_to_spec`` hands such
   dicts to the *subclass* so the resulting Python object's
   ``isinstance`` chain matches the Python-form path (Q8).

Together they guarantee one Python authoring path AND one byte-stable
wire shape across YAML / JSON / Python.
"""

from __future__ import annotations

import pytest

from snowflake.ml.feature_store.decl.loader import _dict_to_spec
from snowflake.ml.feature_store.decl.spec_models import (
    Backfill,
    BatchFeatureView,
    FeatureView,
    FSColumn,
    RealtimeFeatureView,
    SourceRef,
    StreamingFeatureView,
)


class TestFeatureViewSubclassDefaults:
    """Each subclass pins ``kind`` to its class name via a class default."""

    def test_streaming_feature_view_default_kind(self) -> None:
        fv = StreamingFeatureView(name="x")
        assert fv.kind == "StreamingFeatureView"

    def test_batch_feature_view_default_kind(self) -> None:
        fv = BatchFeatureView(name="x")
        assert fv.kind == "BatchFeatureView"

    def test_realtime_feature_view_default_kind(self) -> None:
        fv = RealtimeFeatureView(name="x")
        assert fv.kind == "RealtimeFeatureView"

    def test_subclass_kind_default_overrides_base(self) -> None:
        """The base ``FeatureView`` defaults to ``"StreamingFeatureView"``;
        the subclasses must shadow that default with their own class name."""
        assert FeatureView(name="x").kind == "StreamingFeatureView"
        assert BatchFeatureView(name="x").kind == "BatchFeatureView"
        assert RealtimeFeatureView(name="x").kind == "RealtimeFeatureView"


class TestFeatureViewSubclassIsInstance:
    """The subclasses are real subclasses of the base ``FeatureView``."""

    def test_streaming_is_feature_view(self) -> None:
        assert isinstance(StreamingFeatureView(name="x"), FeatureView)

    def test_batch_is_feature_view(self) -> None:
        assert isinstance(BatchFeatureView(name="x"), FeatureView)

    def test_realtime_is_feature_view(self) -> None:
        assert isinstance(RealtimeFeatureView(name="x"), FeatureView)

    def test_subclasses_are_distinguishable(self) -> None:
        s = StreamingFeatureView(name="x")
        b = BatchFeatureView(name="x")
        r = RealtimeFeatureView(name="x")
        assert isinstance(s, StreamingFeatureView)
        assert not isinstance(s, BatchFeatureView)
        assert not isinstance(s, RealtimeFeatureView)
        assert isinstance(b, BatchFeatureView)
        assert not isinstance(b, StreamingFeatureView)
        assert isinstance(r, RealtimeFeatureView)
        assert not isinstance(r, BatchFeatureView)


class TestFeatureViewSubclassValidators:
    """The subclasses inherit FV validators (backfill cross-kind checks).

    The base ``FeatureView`` carries a ``model_validator(mode="after")``
    that rejects batch-only backfill fields on streaming FVs and vice
    versa.  Because the subclasses fix ``kind``, the validator must fire
    against the *subclass* kind value, not the (unset) authoring string.
    """

    def test_streaming_rejects_batch_only_overwrite(self) -> None:
        with pytest.raises(ValueError) as exc:
            StreamingFeatureView(
                name="x",
                sources=[SourceRef(name="src", source_type="Stream")],
                backfill=Backfill(overwrite=True),
            )
        msg = str(exc.value)
        assert "backfill.overwrite" in msg
        assert "BatchFeatureView" in msg

    def test_streaming_rejects_batch_only_initialize(self) -> None:
        with pytest.raises(ValueError) as exc:
            StreamingFeatureView(
                name="x",
                sources=[SourceRef(name="src", source_type="Stream")],
                backfill=Backfill(initialize="ON_CREATE"),
            )
        assert "backfill.initialize" in str(exc.value)

    def test_batch_rejects_streaming_only_table(self) -> None:
        with pytest.raises(ValueError) as exc:
            BatchFeatureView(
                name="x",
                sources=[SourceRef(name="src", source_type="Batch")],
                backfill=Backfill(table="HISTORY"),
            )
        assert "backfill.table" in str(exc.value)

    def test_batch_rejects_streaming_only_start_time(self) -> None:
        with pytest.raises(ValueError) as exc:
            BatchFeatureView(
                name="x",
                sources=[SourceRef(name="src", source_type="Batch")],
                backfill=Backfill(start_time="2024-01-01T00:00:00Z"),
            )
        assert "backfill.start_time" in str(exc.value)

    def test_streaming_accepts_streaming_backfill_fields(self) -> None:
        fv = StreamingFeatureView(
            name="x",
            sources=[SourceRef(name="src", source_type="Stream")],
            backfill=Backfill(table="HISTORY", start_time="2024-01-01T00:00:00Z"),
        )
        assert fv.backfill is not None
        assert fv.backfill.table == "HISTORY"

    def test_batch_accepts_batch_backfill_fields(self) -> None:
        fv = BatchFeatureView(
            name="x",
            sources=[SourceRef(name="src", source_type="Batch")],
            backfill=Backfill(overwrite=True, initialize="ON_SCHEDULE"),
        )
        assert fv.backfill is not None
        assert fv.backfill.overwrite is True
        assert fv.backfill.initialize == "ON_SCHEDULE"


class TestFeatureViewModelDumpRoundTrip:
    """``model_dump()`` produces the same dict for subclasses as for the
    base ``FeatureView`` with an explicit ``kind`` string.

    This is the *load-bearing* invariant for the planner / executor /
    state path: every downstream consumer reads ``data["kind"]`` from a
    plain dict, so a subclass instance MUST serialize byte-identically
    to a base-class instance with the matching ``kind`` value.
    """

    def test_streaming_dump_matches_base_with_explicit_kind(self) -> None:
        sub = StreamingFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Stream")],
        )
        base = FeatureView(
            kind="StreamingFeatureView",
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Stream")],
        )
        assert sub.model_dump() == base.model_dump()

    def test_batch_dump_matches_base_with_explicit_kind(self) -> None:
        sub = BatchFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Batch")],
        )
        base = FeatureView(
            kind="BatchFeatureView",
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Batch")],
        )
        assert sub.model_dump() == base.model_dump()

    def test_realtime_dump_matches_base_with_explicit_kind(self) -> None:
        sub = RealtimeFeatureView(
            name="x",
            entities=["USER_ID"],
        )
        base = FeatureView(
            kind="RealtimeFeatureView",
            name="x",
            entities=["USER_ID"],
        )
        assert sub.model_dump() == base.model_dump()


class TestDictToSpecMapsKindToSubclass:
    """Q8 — YAML / JSON ``kind: <Streaming|Batch|Realtime>FeatureView`` is
    mapped to the matching subclass by ``loader._dict_to_spec`` so the
    YAML and Python paths converge on the same Python class.
    """

    def test_streaming_kind_maps_to_streaming_subclass(self) -> None:
        obj = _dict_to_spec(
            {
                "kind": "StreamingFeatureView",
                "name": "x",
                "entities": ["USER_ID"],
                "sources": [{"name": "src", "source_type": "Stream"}],
            }
        )
        assert isinstance(obj, StreamingFeatureView)

    def test_batch_kind_maps_to_batch_subclass(self) -> None:
        obj = _dict_to_spec(
            {
                "kind": "BatchFeatureView",
                "name": "x",
                "entities": ["USER_ID"],
                "sources": [{"name": "src", "source_type": "Batch"}],
            }
        )
        assert isinstance(obj, BatchFeatureView)

    def test_realtime_kind_maps_to_realtime_subclass(self) -> None:
        obj = _dict_to_spec(
            {
                "kind": "RealtimeFeatureView",
                "name": "x",
                "entities": ["USER_ID"],
            }
        )
        assert isinstance(obj, RealtimeFeatureView)


class TestFeatureViewBaseStillAccepts:
    """The base ``FeatureView`` keeps accepting ``model_validate`` calls
    with an explicit ``kind`` string — the YAML/JSON path that doesn't
    go through ``_dict_to_spec`` (e.g. direct ``FeatureView.model_validate``
    callsites in tests or external scripts) must remain byte-stable.
    """

    def test_base_accepts_streaming_kind_string(self) -> None:
        fv = FeatureView.model_validate(
            {
                "kind": "StreamingFeatureView",
                "name": "x",
                "entities": ["USER_ID"],
                "sources": [{"name": "src", "source_type": "Stream"}],
            }
        )
        assert fv.kind == "StreamingFeatureView"

    def test_base_accepts_batch_kind_string(self) -> None:
        fv = FeatureView.model_validate(
            {
                "kind": "BatchFeatureView",
                "name": "x",
                "entities": ["USER_ID"],
                "sources": [{"name": "src", "source_type": "Batch"}],
            }
        )
        assert fv.kind == "BatchFeatureView"


class TestSubclassesExposeFullFieldSurface:
    """The new subclasses must expose every authoring field the base FV
    exposes — they are *fewer-knobs* alternatives only to the extent that
    ``kind`` is fixed; every other field is still set-able.
    """

    def test_streaming_subclass_exposes_udf_and_features(self) -> None:
        from snowflake.ml.feature_store.decl.spec_models import UDF, Feature

        fv = StreamingFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Stream")],
            udf=UDF(
                name="udf",
                output_columns=[FSColumn(name="OUT", type="StringType")],
            ),
            features=[
                Feature(
                    source_column=FSColumn(name="A", type="StringType"),
                    output_column=FSColumn(name="B", type="StringType"),
                )
            ],
        )
        assert fv.udf is not None and fv.udf.name == "udf"
        assert len(fv.features) == 1

    def test_batch_subclass_exposes_advanced_fields(self) -> None:
        fv = BatchFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Batch")],
            warehouse="MY_WH",
            cluster_by=["USER_ID"],
            refresh_mode="INCREMENTAL",
            initialize="ON_CREATE",
        )
        assert fv.warehouse == "MY_WH"
        assert fv.cluster_by == ["USER_ID"]
        assert fv.refresh_mode == "INCREMENTAL"
        assert fv.initialize == "ON_CREATE"


class TestFeatureViewRejectsTargetLagOnStreamingAndRealtime:
    """Streaming and Realtime feature views always run at 0 seconds target
    lag — the Snowflake runtime enforces this and stamps
    ``target_lag_sec: 0`` onto the deployed ``DESCRIBE … TYPE = SPECIFICATION``
    payload regardless of what the author wrote.  Accepting an authored
    ``target_lag`` / ``target_lag_sec`` on these kinds is operator-confusing
    because the value is silently dropped at deploy time, so the declarative
    surface rejects any presence of either field at model-construction time
    (Python authoring path) and via :func:`loader._dict_to_spec` (YAML /
    JSON authoring path).  Strict semantics — even an explicit ``0`` is
    rejected; authors must remove the keys entirely.

    Pins the validator added to :class:`FeatureView`:

    * Python form: ``StreamingFeatureView(target_lag=...)`` /
      ``StreamingFeatureView(target_lag_sec=...)`` raise ``ValueError`` /
      ``ValidationError``.
    * Same for ``RealtimeFeatureView``.
    * Same for the base ``FeatureView(kind="StreamingFeatureView", ...)`` /
      ``FeatureView(kind="RealtimeFeatureView", ...)`` construction.
    * YAML / JSON path via ``_dict_to_spec`` raises the same error.
    * ``BatchFeatureView`` is unaffected — ``target_lag`` /
      ``target_lag_sec`` continue to be valid authoring fields for
      batch FVs (where they drive ``refresh_freq`` on the offline
      Dynamic Table).
    """

    # --- StreamingFeatureView (Python form) ---------------------------

    def test_streaming_fv_rejects_target_lag_string(self) -> None:
        with pytest.raises(ValueError) as exc:
            StreamingFeatureView(
                name="x",
                entities=["USER_ID"],
                sources=[SourceRef(name="src", source_type="Stream")],
                target_lag="1h",
            )
        msg = str(exc.value)
        assert "target_lag" in msg
        assert "StreamingFeatureView" in msg

    def test_streaming_fv_rejects_target_lag_int(self) -> None:
        with pytest.raises(ValueError) as exc:
            StreamingFeatureView(
                name="x",
                entities=["USER_ID"],
                sources=[SourceRef(name="src", source_type="Stream")],
                target_lag=3600,
            )
        assert "target_lag" in str(exc.value)

    def test_streaming_fv_rejects_target_lag_sec(self) -> None:
        with pytest.raises(ValueError) as exc:
            StreamingFeatureView(
                name="x",
                entities=["USER_ID"],
                sources=[SourceRef(name="src", source_type="Stream")],
                target_lag_sec=3600,
            )
        msg = str(exc.value)
        assert "target_lag_sec" in msg
        assert "StreamingFeatureView" in msg

    def test_streaming_fv_rejects_target_lag_zero(self) -> None:
        """Strict mode: explicit ``0`` is rejected so authors must remove
        the field entirely.  An exported YAML carrying the runtime-stamped
        ``target_lag_sec: 0`` would round-trip-fail without the exporter
        strip (covered in a later step), but the validator itself stays
        strict so a hand-authored ``target_lag: 0`` doesn't slip through.
        """
        with pytest.raises(ValueError) as exc:
            StreamingFeatureView(
                name="x",
                entities=["USER_ID"],
                sources=[SourceRef(name="src", source_type="Stream")],
                target_lag=0,
            )
        assert "target_lag" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            StreamingFeatureView(
                name="x",
                entities=["USER_ID"],
                sources=[SourceRef(name="src", source_type="Stream")],
                target_lag_sec=0,
            )
        assert "target_lag_sec" in str(exc.value)

    # --- RealtimeFeatureView (Python form) ----------------------------

    def test_realtime_fv_rejects_target_lag_string(self) -> None:
        with pytest.raises(ValueError) as exc:
            RealtimeFeatureView(
                name="x",
                entities=["USER_ID"],
                target_lag="1h",
            )
        msg = str(exc.value)
        assert "target_lag" in msg
        assert "RealtimeFeatureView" in msg

    def test_realtime_fv_rejects_target_lag_sec(self) -> None:
        with pytest.raises(ValueError) as exc:
            RealtimeFeatureView(
                name="x",
                entities=["USER_ID"],
                target_lag_sec=3600,
            )
        msg = str(exc.value)
        assert "target_lag_sec" in msg
        assert "RealtimeFeatureView" in msg

    def test_realtime_fv_rejects_target_lag_zero(self) -> None:
        with pytest.raises(ValueError):
            RealtimeFeatureView(name="x", entities=["USER_ID"], target_lag_sec=0)
        with pytest.raises(ValueError):
            RealtimeFeatureView(name="x", entities=["USER_ID"], target_lag=0)

    # --- Base FeatureView with explicit kind --------------------------

    def test_base_feature_view_rejects_target_lag_on_streaming_kind(self) -> None:
        with pytest.raises(ValueError) as exc:
            FeatureView(
                kind="StreamingFeatureView",
                name="x",
                entities=["USER_ID"],
                sources=[SourceRef(name="src", source_type="Stream")],
                target_lag="5m",
            )
        assert "target_lag" in str(exc.value)
        assert "StreamingFeatureView" in str(exc.value)

    def test_base_feature_view_rejects_target_lag_on_realtime_kind(self) -> None:
        with pytest.raises(ValueError) as exc:
            FeatureView(
                kind="RealtimeFeatureView",
                name="x",
                entities=["USER_ID"],
                target_lag_sec=60,
            )
        assert "target_lag_sec" in str(exc.value)
        assert "RealtimeFeatureView" in str(exc.value)

    # --- YAML / JSON path ---------------------------------------------

    def test_yaml_loader_streaming_fv_rejects_target_lag_sec(self) -> None:
        """``_dict_to_spec`` must surface the validator error so the
        YAML/JSON authoring path stays in sync with the Python path.
        """
        with pytest.raises(ValueError) as exc:
            _dict_to_spec(
                {
                    "kind": "StreamingFeatureView",
                    "name": "x",
                    "entities": ["USER_ID"],
                    "sources": [{"name": "src", "source_type": "Stream"}],
                    "target_lag_sec": 3600,
                }
            )
        assert "target_lag_sec" in str(exc.value)

    def test_yaml_loader_streaming_fv_rejects_target_lag_zero(self) -> None:
        """Exported YAML carrying the runtime-stamped ``target_lag_sec: 0``
        must fail to load — the exporter is responsible for omitting it
        (Step 5 of the plan).  This pins the strict-mode invariant.
        """
        with pytest.raises(ValueError) as exc:
            _dict_to_spec(
                {
                    "kind": "StreamingFeatureView",
                    "name": "x",
                    "entities": ["USER_ID"],
                    "sources": [{"name": "src", "source_type": "Stream"}],
                    "target_lag_sec": 0,
                }
            )
        assert "target_lag_sec" in str(exc.value)

    def test_yaml_loader_realtime_fv_rejects_target_lag(self) -> None:
        with pytest.raises(ValueError) as exc:
            _dict_to_spec(
                {
                    "kind": "RealtimeFeatureView",
                    "name": "x",
                    "entities": ["USER_ID"],
                    "target_lag": "5m",
                }
            )
        assert "target_lag" in str(exc.value)

    # --- Regression: BatchFeatureView still accepts target_lag when online ---

    def test_online_batch_fv_still_accepts_target_lag_string(self) -> None:
        # Authoring ``target_lag`` is OFT staleness — valid on an
        # ``online: true`` BatchFeatureView only.  Offline-only BFVs
        # are rejected by the sibling
        # ``_reject_target_lag_on_offline_batch_fv`` validator.
        fv = BatchFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Batch")],
            online=True,
            target_lag="1h",
        )
        assert fv.target_lag == "1h"

    def test_online_batch_fv_still_accepts_target_lag_sec(self) -> None:
        fv = BatchFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Batch")],
            online=True,
            target_lag_sec=3600,
        )
        assert fv.target_lag_sec == 3600

    def test_online_batch_fv_still_accepts_target_lag_zero(self) -> None:
        # An explicit zero on an online BFV is unusual but not invalid
        # at the model layer (the planner's BATCH_FV_TILING_REFRESH
        # check owns its own validation).  The rejection is targeted:
        # it fires only for streaming / realtime kinds and offline-only
        # BFVs.
        fv = BatchFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Batch")],
            online=True,
            target_lag_sec=0,
        )
        assert fv.target_lag_sec == 0

    def test_streaming_fv_unset_target_lag_is_accepted(self) -> None:
        """The canonical authoring shape — no ``target_lag`` /
        ``target_lag_sec`` at all — must construct without error so
        existing streaming-FV specs continue to load.
        """
        fv = StreamingFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Stream")],
        )
        assert fv.target_lag is None
        assert fv.target_lag_sec is None


class TestFeatureViewRefreshFreqAuthoring:
    """``BatchFeatureView`` accepts ``refresh_freq`` via every authoring
    form — Python form, YAML loader (``_dict_to_spec``), and direct
    ``model_validate``. The field round-trips through ``model_dump()``.

    ``refresh_freq`` is the renamed-from-``refresh_freq`` authoring key
    that maps 1:1 to the imperative ``FeatureView(refresh_freq=...)``
    constructor kwarg. Owning the same name on both sides removes a
    rename layer and matches the imperative API.
    """

    def test_batch_fv_accepts_refresh_freq_python_form(self) -> None:
        fv = BatchFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Batch")],
            refresh_freq="5 minutes",
        )
        assert fv.refresh_freq == "5 minutes"

    def test_batch_fv_accepts_refresh_freq_yaml_loader(self) -> None:
        obj = _dict_to_spec(
            {
                "kind": "BatchFeatureView",
                "name": "x",
                "entities": ["USER_ID"],
                "sources": [{"name": "src", "source_type": "Batch"}],
                "refresh_freq": "5 minutes",
            }
        )
        assert isinstance(obj, BatchFeatureView)
        assert obj.refresh_freq == "5 minutes"

    def test_batch_fv_accepts_refresh_freq_model_validate(self) -> None:
        fv = FeatureView.model_validate(
            {
                "kind": "BatchFeatureView",
                "name": "x",
                "entities": ["USER_ID"],
                "sources": [{"name": "src", "source_type": "Batch"}],
                "refresh_freq": "10 minutes",
            }
        )
        assert fv.refresh_freq == "10 minutes"

    def test_refresh_freq_round_trips_through_model_dump(self) -> None:
        fv = BatchFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Batch")],
            refresh_freq="5 minutes",
        )
        dumped = fv.model_dump()
        assert dumped["refresh_freq"] == "5 minutes"
        # Round-trip back through model_validate.
        rebuilt = FeatureView.model_validate(dumped)
        assert rebuilt.refresh_freq == "5 minutes"

    def test_batch_fv_unset_refresh_freq_is_accepted(self) -> None:
        fv = BatchFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Batch")],
        )
        assert fv.refresh_freq is None


class TestFeatureViewRejectsRefreshFreqOnStreamingAndRealtime:
    """Streaming and Realtime feature views must not accept ``refresh_freq``.

    ``refresh_freq`` controls the offline Dynamic Table's refresh cadence
    (the ``CREATE DYNAMIC TABLE … TARGET_LAG`` / ``SCHEDULE`` clause).
    Streaming and realtime kinds run at zero target lag — Snowflake's
    runtime stamps ``target_lag_sec=0`` regardless of any authored
    cadence — so the field is silently dropped at deploy time on those
    kinds. The declarative surface rejects the field at load time
    instead, so authors get a clear error rather than a silently-dropped
    value.

    Pins the validator added to :class:`FeatureView` mirroring
    :meth:`_reject_target_lag_on_stream_or_realtime`:

    * Python form: ``StreamingFeatureView(refresh_freq=...)`` /
      ``RealtimeFeatureView(refresh_freq=...)`` raise ``ValueError``.
    * Same for the base ``FeatureView(kind="StreamingFeatureView", ...)``
      / ``FeatureView(kind="RealtimeFeatureView", ...)`` construction.
    * YAML / JSON path via ``_dict_to_spec`` raises the same error.
    * ``BatchFeatureView`` is unaffected — ``refresh_freq`` continues to
      be a valid authoring field for batch FVs (where it drives the
      offline DT refresh cadence).
    """

    def test_streaming_fv_rejects_refresh_freq_string(self) -> None:
        with pytest.raises(ValueError) as exc:
            StreamingFeatureView(
                name="x",
                entities=["USER_ID"],
                sources=[SourceRef(name="src", source_type="Stream")],
                refresh_freq="5 minutes",
            )
        msg = str(exc.value)
        assert "refresh_freq" in msg
        assert "is not valid on" in msg
        assert "StreamingFeatureView" in msg

    def test_streaming_fv_rejects_refresh_freq_zero(self) -> None:
        """Strict mode: even an explicit ``"0 seconds"`` is rejected so
        authors must remove the field entirely. Mirrors the strict
        rejection of ``target_lag=0`` on streaming kinds.
        """
        with pytest.raises(ValueError) as exc:
            StreamingFeatureView(
                name="x",
                entities=["USER_ID"],
                sources=[SourceRef(name="src", source_type="Stream")],
                refresh_freq="0 seconds",
            )
        assert "refresh_freq" in str(exc.value)

    def test_realtime_fv_rejects_refresh_freq_string(self) -> None:
        with pytest.raises(ValueError) as exc:
            RealtimeFeatureView(
                name="x",
                entities=["USER_ID"],
                refresh_freq="5 minutes",
            )
        msg = str(exc.value)
        assert "refresh_freq" in msg
        assert "is not valid on" in msg
        assert "RealtimeFeatureView" in msg

    def test_base_feature_view_rejects_refresh_freq_on_streaming_kind(self) -> None:
        with pytest.raises(ValueError) as exc:
            FeatureView(
                kind="StreamingFeatureView",
                name="x",
                entities=["USER_ID"],
                sources=[SourceRef(name="src", source_type="Stream")],
                refresh_freq="5 minutes",
            )
        msg = str(exc.value)
        assert "refresh_freq" in msg
        assert "StreamingFeatureView" in msg

    def test_base_feature_view_rejects_refresh_freq_on_realtime_kind(self) -> None:
        with pytest.raises(ValueError) as exc:
            FeatureView(
                kind="RealtimeFeatureView",
                name="x",
                entities=["USER_ID"],
                refresh_freq="5 minutes",
            )
        msg = str(exc.value)
        assert "refresh_freq" in msg
        assert "RealtimeFeatureView" in msg

    def test_yaml_loader_streaming_fv_rejects_refresh_freq(self) -> None:
        with pytest.raises(ValueError) as exc:
            _dict_to_spec(
                {
                    "kind": "StreamingFeatureView",
                    "name": "x",
                    "entities": ["USER_ID"],
                    "sources": [{"name": "src", "source_type": "Stream"}],
                    "refresh_freq": "5 minutes",
                }
            )
        assert "refresh_freq" in str(exc.value)

    def test_yaml_loader_realtime_fv_rejects_refresh_freq(self) -> None:
        with pytest.raises(ValueError) as exc:
            _dict_to_spec(
                {
                    "kind": "RealtimeFeatureView",
                    "name": "x",
                    "entities": ["USER_ID"],
                    "refresh_freq": "5 minutes",
                }
            )
        assert "refresh_freq" in str(exc.value)

    # --- Regression: BatchFeatureView still accepts refresh_freq ---

    def test_batch_fv_still_accepts_refresh_freq(self) -> None:
        fv = BatchFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Batch")],
            refresh_freq="5 minutes",
        )
        assert fv.refresh_freq == "5 minutes"

    def test_streaming_fv_unset_refresh_freq_is_accepted(self) -> None:
        """The canonical authoring shape — no ``refresh_freq`` at all —
        must construct without error so existing streaming-FV specs
        continue to load. After the rename + rejection, streaming FVs
        carry no cadence knob; the imperative side derives cadence from
        ``StreamConfig`` and the runtime stamp.
        """
        fv = StreamingFeatureView(
            name="x",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Stream")],
        )
        assert fv.refresh_freq is None


class TestFeatureViewBatchScheduleMigrationError:
    """The legacy authoring key ``batch_schedule`` was hard-renamed to
    ``refresh_freq`` to match the imperative ``FeatureView(refresh_freq=...)``
    constructor kwarg.

    The rename is a hard break with no back-compat alias — authoring
    ``batch_schedule:`` in YAML / Python raises a ``ValueError`` with the
    stable substring ``"has been renamed to"``.  This mirrors the existing
    migration errors for ``ordered_entity_column_names`` →
    ``entities`` and ``timestamp_field`` → ``timestamp_col`` raised by
    :meth:`FeatureView._reject_legacy_authoring_keys`.

    The substring ``"has been renamed to"`` is the same one
    :func:`decl.loader._dict_to_spec` greps for to decide that a migration
    error must propagate to the user instead of being silently degraded
    to a bare ``SpecBase``, so the YAML / JSON authoring path raises the
    same error operators see in the Python form.
    """

    def test_batch_fv_rejects_legacy_batch_schedule(self) -> None:
        with pytest.raises(ValueError) as exc:
            FeatureView.model_validate(
                {
                    "kind": "BatchFeatureView",
                    "name": "x",
                    "entities": ["USER_ID"],
                    "sources": [{"name": "src", "source_type": "Batch"}],
                    "batch_schedule": "5 minutes",
                }
            )
        msg = str(exc.value)
        assert "batch_schedule" in msg
        assert "has been renamed to" in msg
        assert "refresh_freq" in msg

    def test_streaming_fv_rejects_legacy_batch_schedule(self) -> None:
        """Even on a streaming FV — where authoring ``refresh_freq`` is
        rejected by the new validator — the migration error for the
        legacy ``batch_schedule`` key must still fire so authors get a
        clear pointer to the new name (and then can decide the field
        does not belong on the streaming kind at all).
        """
        with pytest.raises(ValueError) as exc:
            FeatureView.model_validate(
                {
                    "kind": "StreamingFeatureView",
                    "name": "x",
                    "entities": ["USER_ID"],
                    "sources": [{"name": "src", "source_type": "Stream"}],
                    "batch_schedule": "5 minutes",
                }
            )
        msg = str(exc.value)
        assert "batch_schedule" in msg
        assert "has been renamed to" in msg

    def test_yaml_loader_propagates_batch_schedule_migration_error(self) -> None:
        """The YAML / JSON authoring path via ``_dict_to_spec`` must
        surface the migration error so authors loading specs from disk
        see the same pointer.
        """
        with pytest.raises(ValueError) as exc:
            _dict_to_spec(
                {
                    "kind": "BatchFeatureView",
                    "name": "x",
                    "entities": ["USER_ID"],
                    "sources": [{"name": "src", "source_type": "Batch"}],
                    "batch_schedule": "5 minutes",
                }
            )
        msg = str(exc.value)
        assert "batch_schedule" in msg
        assert "has been renamed to" in msg

    def test_python_form_subclass_rejects_legacy_batch_schedule(self) -> None:
        """The Python authoring form via the subclass constructors must
        also raise — the migration error fires before the subclass
        accepts kwargs because it is registered as a ``model_validator(mode="before")``.
        """
        with pytest.raises(ValueError) as exc:
            BatchFeatureView.model_validate(
                {
                    "name": "x",
                    "entities": ["USER_ID"],
                    "sources": [{"name": "src", "source_type": "Batch"}],
                    "batch_schedule": "5 minutes",
                }
            )
        msg = str(exc.value)
        assert "batch_schedule" in msg
        assert "has been renamed to" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
