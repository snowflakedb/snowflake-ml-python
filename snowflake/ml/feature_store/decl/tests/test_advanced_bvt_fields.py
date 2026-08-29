"""Failing-tests skeleton for the six "advanced" BatchFeatureView fields.

This file is the visible matrix that drives the per-phase TDD cycle described
in ``plans/advanced_bfv_fields_tdd_plan_*.plan.md``: one parametrised xfail
per field so anyone running the test suite can see at a glance which fields
have landed and which are still pending.  As each phase ships its
implementation, the xfail markers for that field's tests are removed and
the assertions become real (red → green).

The six fields and their planner routing are:

* ``warehouse``                  — operational, ``UPDATE_FV``
* ``cluster_by``                 — structural,  ``RECREATE_FV``
* ``refresh_mode``               — structural,  ``RECREATE_FV``
* ``initialize``                 — structural,  ``RECREATE_FV`` (promoted
                                   from ``backfill.initialize``)
* ``storage_config``             — structural,  ``RECREATE_FV``
* ``aggregation_secondary_keys`` — structural,  ``RECREATE_FV``
                                   (private preview, max length 1; valid on
                                   tiled and non-tiled BFVs)
"""

from __future__ import annotations

import pytest

from snowflake.ml.feature_store.decl.spec_models import FeatureView
from snowflake.ml.test_utils import pytest_driver


def _xfail(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=reason)


# Each row is xfail-marked until the corresponding TDD phase deletes the marker.
# Strict=True means a row flipping to PASS without the marker removal fails the
# suite, forcing the implementer to update this table when their field lands.
ADVANCED_FIELDS = (
    pytest.param("warehouse", "WH_OVERRIDE", id="warehouse"),
    pytest.param("cluster_by", ["USER_ID"], id="cluster_by"),
    pytest.param("refresh_mode", "FULL", id="refresh_mode"),
    pytest.param("initialize", "ON_SCHEDULE", id="initialize"),
    pytest.param("storage_config", {"format": "snowflake"}, id="storage_config"),
    pytest.param("aggregation_secondary_keys", ["SECONDARY_COL"], id="aggregation_secondary_keys"),
)


@pytest.mark.parametrize("field,value", ADVANCED_FIELDS)
def test_authoring_field_survives_model_validate(field: str, value: object) -> None:
    """The authoring YAML key must round-trip through ``FeatureView.model_validate``.

    Today Pydantic silently drops every unknown key (default ``extra='ignore'``),
    so this test fails until each field is added to ``spec_models.FeatureView``.
    Per Phase 0–6 of the plan, the xfail marker for the corresponding row is
    deleted when that field's spec-model field lands.

    Args:
        field: Authoring-side YAML key for the field under test.
        value: Sample authoring value used to seed the round-trip assertion.
    """
    payload = {
        "kind": "BatchFeatureView",
        "name": "BFV_ADV",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "refresh_freq": "5 minutes",
        field: value,
    }
    fv = FeatureView.model_validate(payload)
    dumped = fv.model_dump(exclude_none=True)
    assert dumped.get(field) == value, (
        f"FeatureView dropped advanced field {field!r}; got dumped keys={sorted(dumped)}; "
        "add the field to spec_models.FeatureView per the plan."
    )


if __name__ == "__main__":
    pytest_driver.main()
