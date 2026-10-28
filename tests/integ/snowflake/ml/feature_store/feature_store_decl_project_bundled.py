"""Declarative feature-store project plan/apply/re-plan idempotency integ test.

Runs inside the ``feature_store_spec_oft_bundle_test`` bundle so it reuses the
shared per-shard Online Service (``_module_state``). It drives the genuine
declarative orchestration -- ``load_project`` -> ``fetch_applied_state`` ->
``validate_specs`` -> ``generate_plan`` -> ``execute_plan`` -- via the in-process
``decl_cli_driver`` (the ``snow feature`` CLI is not importable here) against a
small checked-in project with one object of each kind authored in a mix of YAML
and Python (``decl_integ_project/``).

The flow is DDL-only (no ingestion / online reads):

    1. plan  -> apply   (every FV/FG is CREATE_*)
    2. plan  (re-plan)  (every op is NO_CHANGE -- idempotency)
    3. edit one FV's refresh_freq, plan (exactly one UPDATE_FV) -> apply

Every object name carries a run-unique ``{{ suffix }}`` so repeated runs against
a reused schema never collide (the entity name is NOT suffixed -- it must equal
its join-key column ``USER_ID`` to avoid replan drift). The planner always runs
with ``no_delete=True`` so objects other bundle classes created in the shared
schema are never flagged for deletion.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import textwrap
import time
import uuid
from pathlib import Path

import decl_cli_driver
from absl.testing import absltest
from feature_store_streaming_fv_integ_base import StreamingFeatureViewIntegTestBase

from snowflake.ml.feature_store.decl import api as decl_api

logger = logging.getLogger(__name__)

# Op kinds that are acceptable on the initial create plan (fresh schema) plus
# the idempotent re-emission of an already-registered entity on a reused schema.
_CREATE_OR_NOOP_KINDS = frozenset({"CREATE_ENTITY", "CREATE_SOURCE", "CREATE_FV", "CREATE_FG", "NO_CHANGE"})
# Non-destructive operational-edit kinds accepted for the refresh_freq bump.
_OPERATIONAL_UPDATE_KINDS = frozenset({"UPDATE_FV", "EVOLVE_FV"})


class FeatureStoreDeclProjectIntegTest(StreamingFeatureViewIntegTestBase, absltest.TestCase):
    """End-to-end declarative plan/apply/re-plan idempotency on a live schema."""

    def setUp(self) -> None:
        super().setUp()

        # Pin the session to the shared feature-store schema; execute_plan and
        # the backing-table DDL both resolve objects relative to it.
        self._session.sql(f"USE SCHEMA {self.test_db}.{self.test_schema}").collect()

        # Run-unique suffix (leading underscore keeps a valid identifier).
        self._suffix = "_" + uuid.uuid4().hex[:6].upper()

        # Object names as they resolve after templating. The entity is NOT
        # suffixed (name must equal its join-key column USER_ID).
        self._entity = "USER_ID"
        self._batch_source = f"EVENTS_BATCH{self._suffix}"
        self._stream_source = f"CLICKSTREAM{self._suffix}"
        self._online_bfv = f"USER_AMOUNT_BFV{self._suffix}"
        self._offline_bfv = f"USER_OFFLINE_BFV{self._suffix}"
        self._streaming_fv = f"USER_CLICK_STATS{self._suffix}"
        self._feature_group = f"USER_FRAUD_FG{self._suffix}"
        self._table = f"RAW_EVENTS_DECL_INTEG{self._suffix}"
        self._fv_version = "V1"

        self._create_backing_table()
        self._project_root = self._stage_project()

        manifest = decl_api.load_manifest(Path(self._project_root))
        self._target = decl_api.resolve_target(manifest)

        self._log_environment_context()

    def tearDown(self) -> None:
        """Best-effort, reverse-dependency cleanup of this run's own objects.

        execute_plan builds its own FeatureStore (bypassing the base class's
        register-tracking wrappers), so none of our applied objects are
        auto-cleaned; we must drop them here. Never ``fs._clear`` -- the schema
        is shared with other bundle classes.
        """
        fs = self.fs
        self._try(lambda: fs.delete_feature_group(self._feature_group, self._fv_version))
        for fv in (self._online_bfv, self._offline_bfv, self._streaming_fv):
            self._try(lambda fv=fv: fs.delete_feature_view(fv, self._fv_version))
        self._try(lambda: fs.delete_stream_source(self._stream_source))
        self._try(lambda: fs.delete_entity(self._entity))
        self._try(
            lambda: self._session.sql(f"DROP TABLE IF EXISTS {self.test_db}.{self.test_schema}.{self._table}").collect()
        )
        if getattr(self, "_project_root", None):
            self._try(lambda: shutil.rmtree(self._project_root, ignore_errors=True))
        super().tearDown()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_plan_apply_replan_modify_apply(self) -> None:
        # 1. Initial plan -> apply: every FV/FG is created.
        plan = self._plan(label="initial create plan")
        summary = decl_cli_driver.op_summary(plan)
        kinds = {name: kind for (kind, name, _destr) in summary}

        for kind, name, destr in summary:
            self.assertFalse(destr, f"unexpected destructive op on create: {kind} {name}")
            self.assertIn(
                kind,
                _CREATE_OR_NOOP_KINDS,
                f"unexpected op kind on create: {kind} for {name}",
            )
        self.assertEqual(kinds.get(self._online_bfv), "CREATE_FV")
        self.assertEqual(kinds.get(self._offline_bfv), "CREATE_FV")
        self.assertEqual(kinds.get(self._streaming_fv), "CREATE_FV")
        self.assertEqual(kinds.get(self._feature_group), "CREATE_FG")
        self.assertIn(kinds.get(self._entity), ("CREATE_ENTITY", "NO_CHANGE"))

        self._apply(plan)

        # Gate the re-plan on the streaming FV's OFT being recoverable with a
        # non-empty spec.  On Postgres-backed accounts the OFT can lag the apply
        # (hypotheses 1 and 2): until it appears in SHOW ONLINE FEATURE TABLES
        # with a parseable DESCRIBE ... TYPE = SPECIFICATION, applied-state
        # recovery falls back to an empty-features spec and the re-plan spuriously
        # emits a destructive RECREATE_FV.
        self._wait_streaming_fv_recoverable()

        # 2. Re-plan with no source edits: fully idempotent (bounded convergence).
        self._assert_replan_no_change(label="re-plan after apply")

        # 3. Edit one FV (refresh_freq) and re-plan: exactly one non-destructive
        #    update, everything else unchanged (incl. the FeatureGroup).
        self._bump_online_bfv_refresh_freq("5 minutes", "10 minutes")
        modified = self._plan(label="re-plan after refresh_freq edit")
        details3 = decl_cli_driver.op_details(modified)
        changed = [(kind, name, destr, reason) for (kind, name, destr, reason) in details3 if kind != "NO_CHANGE"]
        destructive_ops = [(kind, name) for (kind, name, destr, _reason) in details3 if destr]
        edit_ok = (
            len(changed) == 1
            and changed[0][1] == self._online_bfv
            and changed[0][0] in _OPERATIONAL_UPDATE_KINDS
            and not destructive_ops
        )
        if not edit_ok:
            self.fail(
                "refresh_freq edit expected exactly one non-destructive "
                f"{sorted(_OPERATIONAL_UPDATE_KINDS)} op on {self._online_bfv}; got "
                f"changed={[(k, n) for (k, n, _d, _r) in changed]} destructive={destructive_ops}.\n"
                f"{decl_cli_driver.format_plan(modified)}\n"
                f"{self._capture_drift(changed)}"
            )

        self._apply(modified)

        # Re-gate on OFT recoverability before the final re-plan (cheap no-op
        # once the streaming OFT already exists; re-asserts PG settling).
        self._wait_streaming_fv_recoverable()

        # 4. Re-plan after the update: should be fully converged (bounded convergence).
        self._assert_replan_no_change(label="re-plan after update")

        live = self.fs.get_feature_view(self._online_bfv, self._fv_version)
        self.assertEqual(live.refresh_freq, "10 minutes")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _plan(self, *, label: str = "plan"):
        plan = decl_cli_driver.compute_plan(
            self._session,
            self._project_root,
            self._target,
            self.warehouse,
        )
        logger.info("%s:\n%s", label, decl_cli_driver.format_plan(plan))
        return plan

    def _log_environment_context(self) -> None:
        """Best-effort log of the runtime image tag for CI triage (never raises).

        The online store backend (``online_store_type``) is logged later, from
        the recovered streaming FV specification in
        :meth:`_wait_streaming_fv_recoverable`, since it is not known until an
        OFT exists.
        """
        try:
            logger.info("decl-integ environment: feature_store_image_tag=%r", self._feature_store_image_tag())
        except Exception:  # noqa: BLE001 — diagnostic logging must never fail the test
            logger.debug("decl-integ: could not read feature store image tag.", exc_info=True)

    def _wait_streaming_fv_recoverable(self, *, timeout_s: float = 180.0) -> None:
        """Wait until the tiled streaming FV's OFT is recoverable with features.

        Only the streaming FV is gated here: it is the object that races the
        Postgres OFT catalog (hypotheses 1 and 2) and always deploys a non-empty
        ``spec.features``.  The offline BatchFV has no OFT (``online: false``) so
        it is recovered via the list path, not SHOW ONLINE FEATURE TABLES; the
        bounded convergence retry in :meth:`_assert_replan_no_change` absorbs any
        residual lag on the remaining objects.

        Args:
            timeout_s: Max seconds to poll for the OFT to become recoverable.
        """
        spec = decl_cli_driver.wait_fv_specification_recoverable(
            self._session,
            self._target.database,
            self._target.schema,
            self._streaming_fv,
            self._fv_version,
            timeout_s=timeout_s,
        )
        if isinstance(spec, dict):
            logger.info("decl-integ environment: online_store_type=%r", spec.get("online_store_type"))

    def _assert_replan_no_change(self, *, label: str, timeout_s: float = 120.0, poll_s: float = 10.0) -> None:
        """Re-plan until every op is NO_CHANGE, or fail with drift diagnostics.

        A clean re-plan must converge to NO_CHANGE.  To absorb Postgres
        eventual-consistency after apply (hypothesis 1) this retries within a
        bounded budget, but still fails hard if it never converges -- so a
        genuine idempotency regression surfaces, with the drifted spec sections
        named (via :func:`decl_cli_driver.diagnose_fv_drift`) instead of an
        opaque RECREATE_FV.

        Args:
            label: Human label for the plan step (used in logs / failure text).
            timeout_s: Max seconds to retry convergence.
            poll_s: Sleep between re-plans.
        """
        deadline = time.time() + timeout_s
        first_drift_report: str | None = None
        last_plan = None
        attempt = 0
        while True:
            attempt += 1
            last_plan = self._plan(label=f"{label} (attempt {attempt})")
            drift = [
                (kind, name, destr, reason)
                for (kind, name, destr, reason) in decl_cli_driver.op_details(last_plan)
                if kind != "NO_CHANGE"
            ]
            if not drift:
                return
            if first_drift_report is None:
                # Capture the field-level diagnostic on the FIRST non-converged
                # plan, before any later retry possibly settles it -- this is the
                # evidence we want even if convergence eventually succeeds.
                first_drift_report = self._capture_drift(drift)
            if time.time() >= deadline:
                self.fail(
                    f"{label}: re-plan did not converge to NO_CHANGE within {timeout_s:.0f}s "
                    f"({attempt} attempt(s)).\n"
                    f"last {decl_cli_driver.format_plan(last_plan)}\n"
                    f"{first_drift_report}"
                )
            time.sleep(poll_s)

    def _capture_drift(self, drift) -> str:
        """Build a drift report for every non-NO_CHANGE op (never raises)."""
        reports: list[str] = []
        for kind, name, destr, reason in drift:
            reports.append(f"unexpected op {kind} (destructive={destr}) for {name}: {reason}")
            try:
                reports.append(
                    decl_cli_driver.diagnose_fv_drift(
                        self._session,
                        self._project_root,
                        self._target,
                        self.warehouse,
                        name,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — diagnostics must never mask the failure
                reports.append(f"(diagnose_fv_drift failed for {name}: {type(exc).__name__}: {exc})")
        return "\n".join(reports)

    def _apply(self, plan) -> None:
        result = decl_cli_driver.apply_plan(self._session, plan, self._target, self.warehouse)
        self.assertEqual(
            result.status,
            "applied",
            f"apply did not succeed: status={result.status} errors={result.errors}",
        )
        failed = [op for op in result.ops if op.get("status") == "error"]
        self.assertEqual(failed, [], f"apply had failed ops: {failed}")

    def _create_backing_table(self) -> None:
        fqn = f"{self.test_db}.{self.test_schema}.{self._table}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {fqn} (
                USER_ID VARCHAR,
                EVENT_TS TIMESTAMP_NTZ,
                AMOUNT FLOAT,
                CLICK_COUNT NUMBER(38, 0)
            )
            """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {fqn} VALUES
                ('U1', DATEADD('hour', -1, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), 42.0, 3),
                ('U2', DATEADD('hour', -2, CURRENT_TIMESTAMP()::TIMESTAMP_NTZ), 7.0, 1)
            """
        ).collect()

    def _stage_project(self) -> str:
        """Copy the checked-in project to a temp dir and rewrite its manifest."""
        src = Path(__file__).resolve().parent / "decl_integ_project"
        dst = Path(tempfile.mkdtemp(prefix="decl_integ_project_"))
        shutil.copytree(src, dst, dirs_exist_ok=True)

        role = (self._session.get_current_role() or "").strip('"')
        manifest = textwrap.dedent(
            f"""\
            manifest_version: 1
            type: feature_store
            default_target: default
            targets:
              default:
                account_identifier: SNOWML-DECL-INTEG
                database: {self.test_db}
                schema: {self.test_schema}
                role: {role}
            templating:
              defaults:
                warehouse: {self.warehouse}
                suffix: "{self._suffix}"
            """
        )
        (dst / "manifest.yml").write_text(manifest)
        return str(dst)

    def _bump_online_bfv_refresh_freq(self, old: str, new: str) -> None:
        """Rewrite the online BFV YAML's refresh_freq in the staged project."""
        path = Path(self._project_root) / "sources" / "feature_views" / "user_amount_bfv.yaml"
        text = path.read_text()
        needle = f"refresh_freq: {old}"
        self.assertIn(needle, text, f"expected {needle!r} in {path}")
        path.write_text(text.replace(needle, f"refresh_freq: {new}", 1))

    def _try(self, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — teardown is best-effort
            logger.warning("decl-project teardown step failed (ignored): %s", exc)


if __name__ == "__main__":
    absltest.main()
