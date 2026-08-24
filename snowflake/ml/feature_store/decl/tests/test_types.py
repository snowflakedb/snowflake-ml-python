"""Tests for PlanFile model — serialization / deserialization round-trips."""

from __future__ import annotations

import json

import pytest

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.types import Plan, PlanFile, PlanOp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan() -> Plan:
    op = PlanOp(
        kind=OpKind.CREATE_ENTITY,
        name="user",
        reason="new entity",
    )
    return Plan(ops=[op], warnings=["some warning"])


# ---------------------------------------------------------------------------
# PlanFile construction
# ---------------------------------------------------------------------------


class TestPlanFileConstruction:
    def test_default_version(self) -> None:
        pf = PlanFile()
        assert pf.version == "1"

    def test_default_empty_plan(self) -> None:
        pf = PlanFile()
        assert pf.plan == Plan()

    def test_default_empty_summary(self) -> None:
        pf = PlanFile()
        assert pf.summary == {}

    def test_fields_stored(self) -> None:
        plan = _make_plan()
        pf = PlanFile(
            target_database="DB",
            target_schema="SCH",
            source_files=["a.yaml"],
            plan=plan,
        )
        assert pf.target_database == "DB"
        assert pf.target_schema == "SCH"
        assert pf.source_files == ["a.yaml"]
        assert pf.plan is plan


# ---------------------------------------------------------------------------
# serialize_plan
# ---------------------------------------------------------------------------


class TestSerializePlan:
    def test_returns_string(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        out = decl_api.serialize_plan(plan, "DB", "SCH", ["a.yaml"])
        assert isinstance(out, str)

    def test_valid_json(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        out = decl_api.serialize_plan(plan, "DB", "SCH", ["a.yaml"])
        parsed = json.loads(out)
        assert isinstance(parsed, dict)

    def test_version_field(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        out = decl_api.serialize_plan(plan, "DB", "SCH", [])
        parsed = json.loads(out)
        assert parsed["version"] == "1"

    def test_database_and_schema(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        out = decl_api.serialize_plan(plan, "MYDB", "MYSCH", [])
        parsed = json.loads(out)
        assert parsed["target_database"] == "MYDB"
        assert parsed["target_schema"] == "MYSCH"

    def test_source_files_preserved(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        out = decl_api.serialize_plan(plan, "DB", "SCH", ["a.yaml", "b.yaml"])
        parsed = json.loads(out)
        assert parsed["source_files"] == ["a.yaml", "b.yaml"]

    def test_plan_ops_serialized(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        out = decl_api.serialize_plan(plan, "DB", "SCH", [])
        parsed = json.loads(out)
        ops = parsed["plan"]["ops"]
        assert len(ops) == 1
        assert ops[0]["name"] == "user"

    def test_created_at_is_iso8601(self) -> None:
        from datetime import datetime

        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        out = decl_api.serialize_plan(plan, "DB", "SCH", [])
        parsed = json.loads(out)
        ts = parsed["created_at"]
        assert isinstance(ts, str)
        assert len(ts) > 0
        # Should parse without error
        datetime.fromisoformat(ts)

    def test_summary_computed(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        out = decl_api.serialize_plan(plan, "DB", "SCH", [])
        parsed = json.loads(out)
        assert "summary" in parsed
        assert isinstance(parsed["summary"], dict)

    def test_summary_counts_by_op_kind(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        out = decl_api.serialize_plan(plan, "DB", "SCH", [])
        parsed = json.loads(out)
        # CREATE_ENTITY op → "CREATE_ENTITY" key in summary with count 1
        summary = parsed["summary"]
        assert summary.get("CREATE_ENTITY") == 1


# ---------------------------------------------------------------------------
# deserialize_plan
# ---------------------------------------------------------------------------


class TestDeserializePlan:
    def test_returns_plan_file(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api
        from snowflake.ml.feature_store.decl.types import PlanFile

        plan = _make_plan()
        json_str = decl_api.serialize_plan(plan, "DB", "SCH", [])
        pf = decl_api.deserialize_plan(json_str)
        assert isinstance(pf, PlanFile)

    def test_round_trip_database(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        json_str = decl_api.serialize_plan(plan, "ROUND_DB", "ROUND_SCH", [])
        pf = decl_api.deserialize_plan(json_str)
        assert pf.target_database == "ROUND_DB"
        assert pf.target_schema == "ROUND_SCH"

    def test_round_trip_ops(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        json_str = decl_api.serialize_plan(plan, "DB", "SCH", [])
        pf = decl_api.deserialize_plan(json_str)
        assert len(pf.plan.ops) == 1
        assert pf.plan.ops[0].name == "user"

    def test_round_trip_warnings(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        json_str = decl_api.serialize_plan(plan, "DB", "SCH", [])
        pf = decl_api.deserialize_plan(json_str)
        assert "some warning" in pf.plan.warnings

    def test_round_trip_source_files(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        plan = _make_plan()
        json_str = decl_api.serialize_plan(plan, "DB", "SCH", ["x.yaml"])
        pf = decl_api.deserialize_plan(json_str)
        assert pf.source_files == ["x.yaml"]

    def test_invalid_json_raises(self) -> None:
        from snowflake.ml.feature_store.decl import api as decl_api

        with pytest.raises(ValueError):
            decl_api.deserialize_plan("not-json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
