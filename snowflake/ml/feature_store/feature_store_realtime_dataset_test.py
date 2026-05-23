"""Unit tests for ``realtime_dataset`` -- the RTFV dataset-generation helpers.

Focused on the in-process pieces that don't need a live Snowpark session:

- :func:`partition_features` splits direct vs realtime refs in first-seen order.
- :func:`validate_rtfvs_request_context_contract` accepts agreement and rejects
  cross-RTFV datatype conflicts.
- :func:`validate_rtfv_dataset_inputs` enforces the spine / draft / slice-name
  contract.
- :func:`attach_synthetic_row_id` calls ``with_column(seq8())`` and
  ``cache_result()`` on the spine.
- :func:`_expected_output_cols_for_ref` predicts column names for direct and
  realtime refs with and without ``auto_prefix`` / ``with_name``.

The full Stage 2 ``map_in_pandas`` path is exercised by the integ tests; here
we test only the SDK-side behavior.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pandas as pd
from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import realtime_dataset
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.realtime_config import RealtimeConfig
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.snowpark.types import DoubleType, StringType, StructField, StructType

_RTFV_OUTPUT_SCHEMA = StructType(
    [
        StructField("risk_score", DoubleType()),
        StructField("risk_bucket", StringType()),
    ]
)


def rtfv_compute_fn(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "risk_score": req["amount"],
            "risk_bucket": ["x"] * len(req),
        }
    )


def _make_upstream(
    *,
    name: str = "TXN_FV",
    version: str = "v1",
    join_keys: Optional[list[str]] = None,
    feature_name: str = "avg_amount",
    status: FeatureViewStatus = FeatureViewStatus.ACTIVE,
) -> FeatureView:
    keys = join_keys or ["USER_ID"]
    schema = StructType(
        [
            StructField(keys[0], StringType()),
            StructField(feature_name, DoubleType()),
        ]
    )
    mock_df = MagicMock()
    mock_df.columns = [f.name for f in schema.fields]
    mock_df.schema = schema
    mock_df.queries = {"queries": [f"SELECT * FROM {name}"]}
    fv = FeatureView(
        name=name,
        entities=[Entity(name="USER", join_keys=keys)],
        feature_df=mock_df,
        online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
    )
    fv._version = FeatureViewVersion(version)
    fv._database = SqlIdentifier("DB")
    fv._schema = SqlIdentifier("SCH")
    fv._infer_schema_df = mock_df
    fv._status = status
    return fv


def _make_rtfv(
    *,
    name: str = "MY_RTFV",
    join_keys: Optional[list[str]] = None,
    request_field_name: str = "amount",
    request_field_type: Optional[Any] = None,
    upstream: Optional[FeatureView] = None,
    compute_fn: Callable[..., pd.DataFrame] = rtfv_compute_fn,
) -> FeatureView:
    keys = join_keys or ["USER_ID"]
    request_source = RequestSource(
        schema=StructType([StructField(request_field_name, request_field_type or DoubleType())])
    )
    upstream_fv = upstream or _make_upstream(join_keys=keys)
    rtc = RealtimeConfig(
        compute_fn=compute_fn,
        sources=[request_source, upstream_fv],
        output_schema=_RTFV_OUTPUT_SCHEMA,
    )
    rtfv = FeatureView(
        name=name,
        entities=[Entity(name="USER", join_keys=keys)],
        realtime_config=rtc,
    )
    rtfv._version = FeatureViewVersion("v1")
    rtfv._status = FeatureViewStatus.ACTIVE
    return rtfv


class PartitionFeaturesTest(absltest.TestCase):
    """Direct refs and realtime refs are split, first-seen order preserved."""

    def test_partitions_in_order(self) -> None:
        bfv = _make_upstream(name="BFV_A")
        rtfv1 = _make_rtfv(name="RTFV_A")
        rtfv2 = _make_rtfv(name="RTFV_B")
        direct, rtfvs = realtime_dataset.partition_features([rtfv1, bfv, rtfv2])
        # ``partition_features`` returns refs typed as
        # ``Union[FeatureView, FeatureViewSlice]``; unwrap before reading
        # ``.name`` (slices proxy to ``feature_view_ref``).
        from snowflake.ml.feature_store import feature_group as fg_mod

        self.assertEqual([fg_mod.unwrap_fv(f).name.resolved() for f in direct], ["BFV_A"])
        self.assertEqual([fg_mod.unwrap_fv(f).name.resolved() for f in rtfvs], ["RTFV_A", "RTFV_B"])

    def test_no_rtfvs_returns_all_direct(self) -> None:
        bfv = _make_upstream(name="BFV_A")
        direct, rtfvs = realtime_dataset.partition_features([bfv])
        self.assertEqual(len(direct), 1)
        self.assertEqual(rtfvs, [])


class ValidateContractTest(absltest.TestCase):
    """Cross-RTFV RequestSource datatype agreement."""

    def test_empty_list_passes(self) -> None:
        realtime_dataset.validate_rtfvs_request_context_contract([])

    def test_same_name_same_type_passes(self) -> None:
        rtfv1 = _make_rtfv(name="RTFV_A", request_field_name="amount")
        rtfv2 = _make_rtfv(name="RTFV_B", request_field_name="amount")
        realtime_dataset.validate_rtfvs_request_context_contract([rtfv1, rtfv2])

    def test_same_name_different_type_raises(self) -> None:
        rtfv1 = _make_rtfv(name="RTFV_A", request_field_name="amount", request_field_type=DoubleType())
        rtfv2 = _make_rtfv(name="RTFV_B", request_field_name="amount", request_field_type=StringType())
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            realtime_dataset.validate_rtfvs_request_context_contract([rtfv1, rtfv2])
        self.assertIn("datatype", str(cm.exception.original_exception))


class ValidateDatasetInputsTest(absltest.TestCase):
    """Spine / draft / slice contract enforcement."""

    def test_missing_request_column_raises(self) -> None:
        rtfv = _make_rtfv()
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            realtime_dataset.validate_rtfv_dataset_inputs([rtfv], ["USER_ID"])  # no AMOUNT
        msg = str(cm.exception.original_exception)
        self.assertIn("request-context", msg)
        # The display name is canonicalized to upper case, mirroring the FS-wide identifier convention.
        self.assertIn("AMOUNT", msg.upper())

    def test_missing_entity_join_key_raises(self) -> None:
        rtfv = _make_rtfv()
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            realtime_dataset.validate_rtfv_dataset_inputs([rtfv], ["AMOUNT"])  # no USER_ID
        msg = str(cm.exception.original_exception)
        self.assertIn("join key", msg)
        self.assertIn("USER_ID", msg)

    def test_passing_check_does_not_raise(self) -> None:
        rtfv = _make_rtfv()
        realtime_dataset.validate_rtfv_dataset_inputs([rtfv], ["USER_ID", "AMOUNT"])

    def test_draft_upstream_raises(self) -> None:
        upstream = _make_upstream(status=FeatureViewStatus.DRAFT)
        rtfv = _make_rtfv(upstream=upstream)
        with self.assertRaises(snowml_exceptions.SnowflakeMLException) as cm:
            realtime_dataset.validate_rtfv_dataset_inputs([rtfv], ["USER_ID", "AMOUNT"])
        self.assertIn("draft", str(cm.exception.original_exception))

    def test_slice_with_invalid_name_raises_at_slice_construction(self) -> None:
        # ``FeatureView.slice`` already validates slice names against the
        # feature_names set, which for an RTFV is the output_schema fields.
        # The dataset-time validator therefore never sees an invalid slice;
        # we lock the FV-level rejection here so the contract stays.
        rtfv = _make_rtfv()
        with self.assertRaisesRegex(ValueError, "not found"):
            rtfv.slice(["not_a_real_output"])

    def test_slice_with_valid_name_passes(self) -> None:
        rtfv = _make_rtfv()
        good_slice = rtfv.slice(["risk_score"])
        realtime_dataset.validate_rtfv_dataset_inputs([good_slice], ["USER_ID", "AMOUNT"])


class AttachSyntheticRowIdTest(absltest.TestCase):
    """Stage 0 augments the spine with a stable per-row id and caches it."""

    def test_with_column_and_cache_result_called(self) -> None:
        spine = MagicMock()
        with_col = MagicMock()
        spine.with_column.return_value = with_col
        with_col.cache_result.return_value = "CACHED"

        result = realtime_dataset.attach_synthetic_row_id(spine)

        self.assertEqual(result, "CACHED")
        spine.with_column.assert_called_once()
        added_col_name = spine.with_column.call_args[0][0]
        self.assertEqual(added_col_name, realtime_dataset._RTFV_SPINE_ROW_ID_COL)
        with_col.cache_result.assert_called_once()


class ExpectedOutputColsTest(absltest.TestCase):
    """Predicted column names line up with how the join engine prefixes."""

    def setUp(self) -> None:
        self.fs = MagicMock()
        # Mimic _get_feature_prefix's contract: returns prefix or None.
        from snowflake.ml.feature_store import feature_view as fv_mod

        self.fs._get_feature_prefix = lambda ref, ap: fv_mod.get_feature_prefix(ref, ap)

    def test_direct_fv_no_prefix(self) -> None:
        bfv = _make_upstream(feature_name="score")
        cols = realtime_dataset._expected_output_cols_for_ref(self.fs, bfv, auto_prefix=False)
        self.assertEqual(cols, ["SCORE"])

    def test_direct_fv_auto_prefix(self) -> None:
        bfv = _make_upstream(feature_name="score")
        cols = realtime_dataset._expected_output_cols_for_ref(self.fs, bfv, auto_prefix=True)
        # ``identifier.concat_names`` joins at the SQL identifier level.
        self.assertEqual(len(cols), 1)
        self.assertIn("SCORE", cols[0])
        self.assertIn("TXN_FV", cols[0])

    def test_rtfv_no_prefix_full_output_schema(self) -> None:
        rtfv = _make_rtfv()
        cols = realtime_dataset._expected_output_cols_for_ref(self.fs, rtfv, auto_prefix=False)
        self.assertEqual(set(cols), {"RISK_SCORE", "RISK_BUCKET"})

    def test_rtfv_slice_projects(self) -> None:
        rtfv = _make_rtfv()
        sl = rtfv.slice(["risk_score"])
        cols = realtime_dataset._expected_output_cols_for_ref(self.fs, sl, auto_prefix=False)
        self.assertEqual(cols, ["RISK_SCORE"])


if __name__ == "__main__":
    absltest.main()
