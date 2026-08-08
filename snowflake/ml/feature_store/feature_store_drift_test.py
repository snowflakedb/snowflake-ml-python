"""Unit tests for ``_canonicalize_operational_fields`` (Plan section A4).

Closes Bug 2 (``UPDATE_FV`` operational drift on tiled online BFVs) by
canonicalizing the operational-field surface symmetrically on both the
local-compile path and the applied-state recovery path.

Two fields must round-trip in canonical form so the diff in
``decl/invariants.py`` and ``decl/planner.py`` does not falsely flag
edits:

* ``online_store_type`` — lowercase, whitespace-stripped (Snowflake
  ``DESCRIBE … TYPE = SPECIFICATION`` returns mixed-case enum values
  with surrounding whitespace; the local compile emits lowercase form).
* ``target_lag_sec`` — always an ``int`` computed from the deployed
  string form via :func:`interval_utils.interval_to_seconds` (the OFT
  sometimes returns ``"5 MINUTES"``, sometimes ``"300 SECONDS"`` —
  diff must compare seconds, not strings).
"""

from typing import Any

from absl.testing import absltest

from snowflake.ml.feature_store.feature_store import _canonicalize_operational_fields


class TestOperationalCanonicalization(absltest.TestCase):
    """A4: operational fields canonicalize to a single representation."""

    def test_online_store_type_canonical_form(self) -> None:
        """``online_store_type`` is lowercased + whitespace-stripped."""
        payload = {"online_store_type": "  Postgres  "}
        out = _canonicalize_operational_fields(payload)
        self.assertEqual(out["online_store_type"], "postgres")

    def test_online_store_type_already_canonical(self) -> None:
        """An already-canonical value passes through unchanged."""
        payload = {"online_store_type": "hybrid_table"}
        out = _canonicalize_operational_fields(payload)
        self.assertEqual(out["online_store_type"], "hybrid_table")

    def test_online_store_type_none_passthrough(self) -> None:
        """``None`` (offline-only FV) is preserved, not coerced to empty."""
        payload = {"online_store_type": None}
        out = _canonicalize_operational_fields(payload)
        self.assertIsNone(out["online_store_type"])

    def test_online_store_type_missing_key(self) -> None:
        """Absent key stays absent (no synthetic insertion)."""
        payload: dict[str, Any] = {}
        out = _canonicalize_operational_fields(payload)
        self.assertNotIn("online_store_type", out)

    def test_target_lag_sec_canonical_form(self) -> None:
        """``target_lag_sec`` is always an ``int`` computed from the string form."""
        payload = {"target_lag": "5 minutes"}
        out = _canonicalize_operational_fields(payload)
        self.assertEqual(out["target_lag_sec"], 300)
        self.assertIsInstance(out["target_lag_sec"], int)

    def test_target_lag_sec_from_seconds_form(self) -> None:
        """``"300 SECONDS"`` and ``"5 minutes"`` collapse to the same seconds value."""
        payload_a = {"target_lag": "5 minutes"}
        payload_b = {"target_lag": "300 SECONDS"}
        out_a = _canonicalize_operational_fields(payload_a)
        out_b = _canonicalize_operational_fields(payload_b)
        self.assertEqual(out_a["target_lag_sec"], out_b["target_lag_sec"])

    def test_target_lag_sec_already_present_recomputed(self) -> None:
        """If ``target_lag_sec`` is already on the payload but ``target_lag`` is too,
        the string form wins so the diff is computed on the deployed value."""
        payload = {"target_lag": "1 hour", "target_lag_sec": 999}
        out = _canonicalize_operational_fields(payload)
        self.assertEqual(out["target_lag_sec"], 3600)

    def test_target_lag_sec_missing_when_target_lag_absent(self) -> None:
        """No ``target_lag`` string on the payload — ``target_lag_sec`` is left alone."""
        payload: dict[str, Any] = {}
        out = _canonicalize_operational_fields(payload)
        self.assertNotIn("target_lag_sec", out)

    def test_target_lag_downstream_sentinel_handled(self) -> None:
        """``DOWNSTREAM`` is the CRON-task sentinel — leave ``target_lag_sec`` unset
        rather than crash; the planner recovers via the companion Task."""
        payload = {"target_lag": "DOWNSTREAM"}
        out = _canonicalize_operational_fields(payload)
        self.assertNotIn("target_lag_sec", out)

    def test_returns_new_dict_does_not_mutate_input(self) -> None:
        """The helper returns a new dict; the caller's payload is unchanged."""
        payload = {"online_store_type": "  Postgres  ", "target_lag": "5 minutes"}
        snapshot = dict(payload)
        out = _canonicalize_operational_fields(payload)
        self.assertEqual(payload, snapshot)
        # And the output reflects the canonicalization.
        self.assertEqual(out["online_store_type"], "postgres")
        self.assertEqual(out["target_lag_sec"], 300)


if __name__ == "__main__":
    absltest.main()
