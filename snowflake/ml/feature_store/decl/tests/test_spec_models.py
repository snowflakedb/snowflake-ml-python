"""Tests for decl/spec_models.py — Pydantic v2 authoring models."""

from typing import Any

import pytest
from pydantic import ValidationError

from snowflake.ml.feature_store.decl import spec_models
from snowflake.ml.feature_store.decl.spec_models import (
    UDF,
    BatchSource,
    Entity,
    Feature,
    FeatureGroup,
    FeatureView,
    FeatureViewRef,
    FSColumn,
    SourceRef,
    SpecBase,
    StreamingSource,
)
from snowflake.ml.test_utils import pytest_driver


class TestFSColumn:
    def test_minimal_construction(self) -> None:
        col = FSColumn(name="user_id", type="StringType")
        assert col.name == "user_id"
        assert col.type == "StringType"

    def test_optional_fields_default_none(self) -> None:
        col = FSColumn(name="x", type="IntegerType")
        assert col.length is None
        assert col.precision is None
        assert col.scale is None
        assert col.tz is None
        assert col.default is None

    def test_all_optional_fields(self) -> None:
        col = FSColumn(
            name="amount",
            type="DecimalType",
            precision=10,
            scale=2,
            length=None,
            tz=None,
            default=0.0,
        )
        assert col.precision == 10
        assert col.scale == 2
        assert col.default == 0.0

    def test_model_dump_excludes_none_by_default(self) -> None:
        col = FSColumn(name="x", type="FloatType")
        d = col.model_dump(exclude_none=True)
        assert "length" not in d
        assert d["name"] == "x"
        assert d["type"] == "FloatType"

    def test_model_dump_is_plain_dict(self) -> None:
        col = FSColumn(name="ts", type="TimestampType", tz="UTC")
        d = col.model_dump()
        assert isinstance(d, dict)
        assert d["tz"] == "UTC"


class TestSpecBase:
    def test_defaults(self) -> None:
        s = SpecBase()
        assert s.kind == ""
        assert s.name == ""
        assert s.version is None
        assert s.database is None
        assert s.schema_ is None
        assert s.description is None

    def test_construction(self) -> None:
        s = SpecBase(kind="Entity", name="customer", version="v1")
        assert s.kind == "Entity"
        assert s.name == "customer"
        assert s.version == "v1"


class TestEntity:
    def test_default_kind(self) -> None:
        e = Entity(name="customer")
        assert e.kind == "Entity"

    def test_join_keys(self) -> None:
        e = Entity(
            name="customer",
            join_keys=[FSColumn(name="customer_id", type="StringType")],
        )
        assert len(e.join_keys) == 1
        assert e.join_keys[0].name == "customer_id"

    def test_empty_join_keys_by_default(self) -> None:
        e = Entity(name="customer")
        assert e.join_keys == []


class TestStreamingSource:
    def test_default_kind_and_type(self) -> None:
        s = StreamingSource(name="events")
        assert s.kind == "StreamingSource"
        assert s.type == "REST"

    def test_columns(self) -> None:
        s = StreamingSource(
            name="events",
            columns=[FSColumn(name="event_id", type="StringType")],
        )
        assert len(s.columns) == 1


class TestStreamingSourceNoBackfillTable:
    """``StreamingSource.backfill_table`` has been removed.

    The legacy per-source ``backfill_table`` knob diverged from the
    imperative API (which attaches backfill to the FeatureView via
    ``StreamConfig``).  It has been replaced by an FV-level
    ``FeatureView.backfill`` block and any YAML carrying the legacy field
    must surface a clear migration error pointing authors at the new shape.
    """

    def test_backfill_table_field_is_removed(self) -> None:
        """The Pydantic field no longer exists on the model."""
        ss = StreamingSource(
            name="CLICKSTREAM_EVENTS",
            type="REST",
            columns=[FSColumn(name="USER_ID", type="StringType")],
        )
        assert not hasattr(ss, "backfill_table"), (
            "StreamingSource.backfill_table must be removed from the model. "
            "Backfill is now an FV-level concern: use FeatureView.backfill: "
            "{ table: <FQN> } to wire StreamConfig.backfill_df."
        )

    def test_legacy_backfill_table_in_yaml_is_rejected_with_migration_message(self) -> None:
        """A YAML / dict that still carries ``backfill_table`` raises a
        ``ValidationError`` whose message names the new ``FeatureView.backfill``
        block.  The error must be actionable, not a generic "extra field".
        """
        with pytest.raises(ValidationError) as exc:
            StreamingSource.model_validate(
                {
                    "name": "CLICKSTREAM_EVENTS",
                    "kind": "StreamingSource",
                    "type": "REST",
                    "columns": [{"name": "USER_ID", "type": "StringType"}],
                    "backfill_table": "DB.SCHEMA.HIST",
                }
            )
        msg = str(exc.value)
        assert "backfill_table" in msg, f"Error must name the legacy field; got: {msg}"
        assert "FeatureView" in msg and "backfill" in msg, (
            "Migration error must point operators at the new FeatureView " f"backfill block; got: {msg}"
        )


class TestBatchSource:
    def test_default_kind(self) -> None:
        b = BatchSource(name="orders", table="ORDER_HISTORY")
        assert b.kind == "BatchSource"

    def test_source_location_fields(self) -> None:
        b = BatchSource(
            name="orders",
            source_database="PROD_DB",
            source_schema="PUBLIC",
            table="ORDER_HISTORY",
        )
        assert b.source_database == "PROD_DB"
        assert b.source_schema == "PUBLIC"
        assert b.table == "ORDER_HISTORY"

    def test_query_field(self) -> None:
        """``query`` is a first-class authoring field — a SQL string that the
        imperative executor turns into ``session.sql(query) -> feature_df``."""
        b = BatchSource(name="orders", query="SELECT * FROM ORDERS WHERE ts > '2024-01-01'")
        assert b.query == "SELECT * FROM ORDERS WHERE ts > '2024-01-01'"
        assert b.table is None
        assert b.query_file is None

    def test_query_file_field(self) -> None:
        """``query_file`` is the sidecar form of ``query`` — mirrors ``udf.file``.
        The compiler resolves it to inline ``query`` at compile time."""
        b = BatchSource(name="orders", query_file="orders_query.sql")
        assert b.query_file == "orders_query.sql"
        assert b.table is None
        assert b.query is None

    def test_table_query_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            BatchSource(name="orders", table="ORDERS", query="SELECT 1")
        msg = str(excinfo.value)
        assert "table" in msg
        assert "query" in msg

    def test_query_file_query_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            BatchSource(name="orders", query="SELECT 1", query_file="orders.sql")
        msg = str(excinfo.value)
        assert "query" in msg
        assert "query_file" in msg

    def test_table_query_file_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            BatchSource(name="orders", table="ORDERS", query_file="orders.sql")
        msg = str(excinfo.value)
        assert "table" in msg
        assert "query_file" in msg

    def test_at_least_one_required(self) -> None:
        """A ``BatchSource`` with neither ``table`` / ``query`` / ``query_file``
        has no addressable backing data and is rejected at construction time
        — this catches partially-edited YAML before it hits the planner."""
        with pytest.raises(ValidationError) as excinfo:
            BatchSource(name="orders")
        msg = str(excinfo.value)
        assert "table" in msg
        assert "query" in msg
        assert "query_file" in msg

    def test_query_round_trips_through_model_dump(self) -> None:
        b = BatchSource(name="orders", query="SELECT 1")
        dumped = b.model_dump()
        assert dumped["query"] == "SELECT 1"
        assert dumped["table"] is None
        assert dumped["query_file"] is None
        rebuilt = BatchSource.model_validate(dumped)
        assert rebuilt.query == "SELECT 1"

    def test_query_file_round_trips_through_model_dump(self) -> None:
        b = BatchSource(name="orders", query_file="orders.sql")
        dumped = b.model_dump()
        assert dumped["query_file"] == "orders.sql"
        rebuilt = BatchSource.model_validate(dumped)
        assert rebuilt.query_file == "orders.sql"


class TestSQLDataSourceRemoved:
    """``SQLDataSource`` was removed during the decl/spec consolidation.

    SQL-backed batch sources are now expressed via ``BatchSource.query`` (inline
    SQL string) or ``BatchSource.query_file`` (sidecar ``.sql`` file). The
    imperative executor turns either form into ``session.sql(query)`` →
    ``feature_df``. The ``spec.enums.SourceType`` enum is intentionally
    unchanged: ``query`` is decl-only and is normalized into the FV's
    ``sources[]`` payload at compile time, so the unified Source spec stays
    small.

    This guard stays in place: any future re-introduction of a separate
    ``SQLDataSource`` kind should land alongside an explicit ``SourceType``
    extension and a discussion of the serialization contract.
    """

    def test_sqldatasource_class_removed(self) -> None:
        assert not hasattr(spec_models, "SQLDataSource")


class TestSourceRef:
    def test_construction(self) -> None:
        ref = SourceRef(name="events_src", source_type="Stream")
        assert ref.name == "events_src"
        assert ref.source_type == "Stream"


class TestUDF:
    def test_defaults(self) -> None:
        udf = UDF(name="compute_fn")
        assert udf.engine == "pandas"
        assert udf.file is None
        assert udf.function_definition is None
        assert udf.output_columns == []

    def test_with_file(self) -> None:
        udf = UDF(name="fn", file="./compute.py")
        assert udf.file == "./compute.py"

    def test_with_callable(self) -> None:
        def my_fn(df: Any) -> Any:
            return df

        udf = UDF(name="my_fn", function_definition=my_fn)
        assert callable(udf.function_definition)

    def test_with_string_source(self) -> None:
        udf = UDF(name="fn", function_definition="def fn(df):\n    return df\n")
        assert isinstance(udf.function_definition, str)


class TestFeature:
    def test_defaults(self) -> None:
        f = Feature()
        assert f.function is None
        assert f.window is None
        assert f.offset is None
        assert f.function_params is None

    def test_construction(self) -> None:
        f = Feature(
            source_column=FSColumn(name="val", type="FloatType"),
            output_column=FSColumn(name="sum_val_7d", type="FloatType"),
            function="sum",
            window="7d",
        )
        assert f.function == "sum"
        assert f.window == "7d"

    def test_window_accepts_int(self) -> None:
        f = Feature(window=3600)
        assert f.window == 3600

    def test_offset_accepts_string(self) -> None:
        f = Feature(offset="1m")
        assert f.offset == "1m"


class TestFeatureView:
    def test_default_kind(self) -> None:
        fv = FeatureView(name="my_fv")
        assert fv.kind == "StreamingFeatureView"

    def test_online_offline_defaults(self) -> None:
        # ``FeatureView(name=...)`` defaults to ``kind="StreamingFeatureView"``
        # (the default discriminator), and streaming FVs are always
        # online by design — the
        # ``_enforce_always_online_for_stream_or_realtime`` validator
        # in ``spec_models.FeatureView`` defaults ``online=True`` for
        # the streaming / realtime kinds and rejects an explicit
        # ``online=False``.  ``BatchFeatureView`` keeps the legacy
        # ``online: bool = False`` default; ``offline`` is independent
        # of the always-online rule and stays ``False`` for both kinds.
        fv = FeatureView(name="my_fv")
        assert fv.online is True
        assert fv.offline is False
        batch = FeatureView(kind="BatchFeatureView", name="my_fv")
        assert batch.online is False
        assert batch.offline is False

    def test_sources_accepts_list(self) -> None:
        fv = FeatureView(
            name="my_fv",
            sources=[SourceRef(name="src", source_type="Stream")],
        )
        assert len(fv.sources) == 1

    def test_sources_accepts_source_objects(self) -> None:
        src = StreamingSource(name="events")
        fv = FeatureView(name="my_fv", sources=[src])
        assert fv.sources[0] is src

    def test_entities_list(self) -> None:
        fv = FeatureView(
            name="my_fv",
            entities=["customer_id"],
        )
        assert fv.entities == ["customer_id"]

    def test_entities_accepts_entity_objects(self) -> None:
        entity = Entity(
            name="customer",
            join_keys=[FSColumn(name="customer_id", type="StringType")],
        )
        fv = FeatureView(
            name="my_fv",
            entities=[entity],
        )
        assert isinstance(fv.entities[0], Entity)

    def test_udf_field(self) -> None:
        udf = UDF(name="compute")
        fv = FeatureView(name="my_fv", udf=udf)
        assert fv.udf is udf

    def test_features_list(self) -> None:
        fv = FeatureView(
            name="my_fv",
            features=[Feature(function="sum", window="1h")],
        )
        assert len(fv.features) == 1

    def test_duration_fields_accept_strings(self) -> None:
        # ``target_lag`` is only valid on online batch FVs (the
        # streaming / realtime kinds reject it via
        # ``_reject_target_lag_on_stream_or_realtime``; offline-only
        # batch kinds reject it via
        # ``_reject_target_lag_on_offline_batch_fv``).  Exercise the
        # online batch kind so this test continues to pin the duration-
        # string parser without tripping either kind-specific rejector.
        fv = FeatureView(
            kind="BatchFeatureView",
            name="my_fv",
            online=True,
            feature_granularity="5m",
            target_lag="1h",
        )
        assert fv.feature_granularity == "5m"
        assert fv.target_lag == "1h"

    def test_duration_fields_accept_ints(self) -> None:
        fv = FeatureView(name="my_fv", feature_granularity=300)
        assert fv.feature_granularity == 300


class TestFeatureViewRef:
    def test_construction(self) -> None:
        ref = FeatureViewRef(name="my_fv", version="V1")
        assert ref.name == "my_fv"
        assert ref.version == "V1"


class TestFeatureGroup:
    def test_default_kind(self) -> None:
        fg = FeatureGroup(
            name="my_fg",
            feature_views=[FeatureViewRef(name="fv1", version="V1")],
        )
        assert fg.kind == "FeatureGroup"

    def test_feature_views_accepts_refs(self) -> None:
        ref = FeatureViewRef(name="fv1", version="V1")
        fg = FeatureGroup(name="my_fg", feature_views=[ref])
        assert isinstance(fg.feature_views[0], FeatureViewRef)
        assert fg.feature_views[0].version == "V1"


class TestSerializability:
    """All models must serialize to plain dicts — no Snowpark types."""

    def test_entity_model_dump(self) -> None:
        e = Entity(
            name="customer",
            version="v1",
            join_keys=[FSColumn(name="customer_id", type="StringType")],
        )
        d = e.model_dump()
        assert d["kind"] == "Entity"
        assert d["join_keys"][0]["name"] == "customer_id"

    def test_feature_view_model_dump(self) -> None:
        fv = FeatureView(
            name="fv",
            online=True,
            sources=[SourceRef(name="src", source_type="Stream")],
            features=[Feature(function="sum", window="1h")],
        )
        d = fv.model_dump()
        assert d["kind"] == "StreamingFeatureView"
        assert d["online"] is True
        assert d["sources"][0]["name"] == "src"
        assert d["features"][0]["function"] == "sum"


class TestFeatureViewAggregationMethod:
    def test_default_is_none(self) -> None:
        fv = FeatureView(name="fv")
        assert fv.feature_aggregation_method is None

    def test_accepts_continuous(self) -> None:
        fv = FeatureView(name="fv", feature_aggregation_method="continuous")
        assert fv.feature_aggregation_method == "continuous"

    def test_accepts_tiles(self) -> None:
        fv = FeatureView(name="fv", feature_aggregation_method="tiles")
        assert fv.feature_aggregation_method == "tiles"

    def test_model_dump_includes_field_when_set(self) -> None:
        fv = FeatureView(name="fv", feature_aggregation_method="continuous")
        d = fv.model_dump()
        assert d["feature_aggregation_method"] == "continuous"

    def test_model_dump_field_is_none_when_not_set(self) -> None:
        fv = FeatureView(name="fv")
        d = fv.model_dump()
        assert d["feature_aggregation_method"] is None


class TestBatchFeatureViewRefreshModeRestriction:
    """``refresh_mode: AUTO`` must be rejected after restricting the Literal.

    ``AUTO`` is not a meaningful declarative authoring choice — Snowflake
    resolves it to ``INCREMENTAL`` or ``FULL`` and the operator cannot
    predict which.  Valid values are ``INCREMENTAL``, ``FULL``, or ``None``
    (omit the field entirely).
    """

    def test_refresh_mode_auto_raises_validation_error(self) -> None:
        """``refresh_mode: AUTO`` must raise ``ValidationError``."""
        with pytest.raises(ValidationError):
            FeatureView.model_validate({"kind": "BatchFeatureView", "name": "MY_BFV", "refresh_mode": "AUTO"})

    def test_refresh_mode_incremental_is_accepted(self) -> None:
        fv = FeatureView.model_validate({"kind": "BatchFeatureView", "name": "MY_BFV", "refresh_mode": "INCREMENTAL"})
        assert fv.refresh_mode == "INCREMENTAL"

    def test_refresh_mode_full_is_accepted(self) -> None:
        fv = FeatureView.model_validate({"kind": "BatchFeatureView", "name": "MY_BFV", "refresh_mode": "FULL"})
        assert fv.refresh_mode == "FULL"

    def test_refresh_mode_none_accepted(self) -> None:
        fv = FeatureView.model_validate({"kind": "BatchFeatureView", "name": "MY_BFV"})
        assert fv.refresh_mode is None


if __name__ == "__main__":
    pytest_driver.main()
