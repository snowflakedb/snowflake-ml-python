"""Unit tests for the ``generate_training_set`` FeatureGroup overload."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import feature_store as fs_mod
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_group import FeatureGroup, FeatureGroupVersion
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.snowpark.types import DoubleType, StringType, StructField, StructType


def _make_registered_fv(
    *,
    name: str,
    version: str,
    feature_columns: list[str],
    online: bool = True,
    store_type: OnlineStoreType = OnlineStoreType.POSTGRES,
    entity_name: str = "USER",
    entity_join_keys: Optional[list[str]] = None,
    schema_key_col: str = "USER_ID",
) -> FeatureView:
    join_keys = entity_join_keys or ["USER_ID"]
    schema = StructType(
        [StructField(schema_key_col, StringType())] + [StructField(c, DoubleType()) for c in feature_columns]
    )
    mock_df = MagicMock()
    mock_df.columns = [f.name for f in schema.fields]
    mock_df.schema = schema
    mock_df.queries = {"queries": [f"SELECT * FROM {name}"]}

    fv = FeatureView(
        name=name,
        entities=[Entity(name=entity_name, join_keys=join_keys)],
        feature_df=mock_df,
        online_config=OnlineConfig(enable=online, store_type=store_type),
    )
    fv._version = FeatureViewVersion(version)
    fv._database = SqlIdentifier("DB")
    fv._schema = SqlIdentifier("SCH")
    fv._infer_schema_df = mock_df
    fv._status = FeatureViewStatus.ACTIVE
    return fv


def _new_fs_with_mocks(
    *,
    metadata_manager: Optional[MagicMock] = None,
    session: Optional[MagicMock] = None,
) -> FeatureStore:
    fs = object.__new__(FeatureStore)
    sess = session or MagicMock()
    # ``switch_warehouse`` passes the result to ``SqlIdentifier``; pin a real string.
    sess.get_current_warehouse = MagicMock(return_value="WH")
    md = metadata_manager or MagicMock()
    object.__setattr__(fs, "_session", sess)
    object.__setattr__(
        fs,
        "_config",
        fs_mod._FeatureStoreConfig(database=SqlIdentifier("DB"), schema=SqlIdentifier("SCH")),
    )
    object.__setattr__(fs, "_telemetry_stmp", {})
    object.__setattr__(fs, "_metadata_manager", md)
    object.__setattr__(fs, "_default_warehouse", SqlIdentifier("WH"))
    return fs


def _make_registered_fg(
    *,
    name: str = "FG",
    version: str = "v1",
    fv: Optional[FeatureView] = None,
    auto_prefix: bool = True,
) -> FeatureGroup:
    fv = fv or _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
    fg = FeatureGroup(name=name, features=[fv], auto_prefix=auto_prefix)
    fg._version = FeatureGroupVersion(version)
    return fg


def _join_features_call_kwargs(call_args: Any) -> dict[str, Any]:
    """Map ``_join_features`` positional args back onto names so tests don't depend on arg order."""
    positional_names = (
        "spine_df",
        "features",
        "spine_timestamp_col",
        "include_feature_view_timestamp_col",
        "auto_prefix",
        "join_method",
    )
    bound: dict[str, Any] = {name: value for name, value in zip(positional_names, call_args.args)}
    bound.update(call_args.kwargs)
    return bound


class GenerateTrainingSetMutualExclusionTest(absltest.TestCase):
    """Exactly one of ``features`` / ``feature_group`` must be set."""

    def test_neither_set_rejected(self) -> None:
        fs = _new_fs_with_mocks()
        spine = MagicMock()
        with self.assertRaisesRegex(ValueError, "exactly one of `features` or `feature_group`"):
            fs.generate_training_set(spine)  # type: ignore[call-overload]

    def test_both_set_rejected(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fg = _make_registered_fg(fv=fv)
        fs = _new_fs_with_mocks()
        spine = MagicMock()
        with self.assertRaisesRegex(ValueError, "exactly one of `features` or `feature_group`"):
            fs.generate_training_set(spine, [fv], feature_group=fg)  # type: ignore[call-overload]


class GenerateTrainingSetFgPathTest(absltest.TestCase):
    """The FG path forwards FG-derived params into the existing _join_features engine."""

    def test_fg_instance_forwards_features_and_overrides(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fg = _make_registered_fg(fv=fv, auto_prefix=True)
        fs = _new_fs_with_mocks()
        spine = MagicMock()
        joined = (MagicMock(), [SqlIdentifier("USER_ID")])
        join_mock = MagicMock(return_value=joined)
        object.__setattr__(fs, "_join_features", join_mock)

        result = fs.generate_training_set(spine, feature_group=fg)

        self.assertIs(result, joined[0])
        join_mock.assert_called_once()
        bound = _join_features_call_kwargs(join_mock.call_args)
        self.assertEqual(bound["features"], fg.features)
        self.assertIsNone(bound["spine_timestamp_col"])
        # FG forces include_feature_view_timestamp_col=False, auto_prefix=fg.auto_prefix, join_method='cte'.
        self.assertFalse(bound["include_feature_view_timestamp_col"])
        self.assertTrue(bound["auto_prefix"])
        self.assertEqual(bound["join_method"], "cte")
        self.assertEqual(bound["is_training"], True)

    def test_fg_tuple_resolves_via_get_feature_group(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fg = _make_registered_fg(fv=fv, auto_prefix=False)
        fs = _new_fs_with_mocks()
        spine = MagicMock()
        get_fg = MagicMock(return_value=fg)
        object.__setattr__(fs, "get_feature_group", get_fg)
        join_mock = MagicMock(return_value=(MagicMock(), []))
        object.__setattr__(fs, "_join_features", join_mock)

        fs.generate_training_set(spine, feature_group=("FG", "v1"))

        get_fg.assert_called_once_with("FG", "v1")
        bound = _join_features_call_kwargs(join_mock.call_args)
        # Mirrors fg.auto_prefix (False here), not the API default.
        self.assertFalse(bound["auto_prefix"])

    def test_fg_passes_spine_timestamp_for_pit(self) -> None:
        """spine_timestamp_col is forwarded so PIT correctness still works on the FG path."""
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fg = _make_registered_fg(fv=fv)
        fs = _new_fs_with_mocks()
        spine = MagicMock()
        join_mock = MagicMock(return_value=(MagicMock(), []))
        object.__setattr__(fs, "_join_features", join_mock)

        fs.generate_training_set(spine, feature_group=fg, spine_timestamp_col="EVENT_TS")

        bound = _join_features_call_kwargs(join_mock.call_args)
        self.assertEqual(str(bound["spine_timestamp_col"]), "EVENT_TS")

    def test_fg_unregistered_draft_rejected(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        draft = FeatureGroup(name="FG", features=[fv])
        fs = _new_fs_with_mocks()
        spine = MagicMock()
        with self.assertRaisesRegex(ValueError, "requires a registered FeatureGroup"):
            fs.generate_training_set(spine, feature_group=draft)


class GenerateTrainingSetFgIncompatibleParamsTest(absltest.TestCase):
    """Params that have no meaning with a FeatureGroup are rejected explicitly."""

    def setUp(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        self.fg = _make_registered_fg(fv=fv)
        self.fs = _new_fs_with_mocks()
        self.spine = MagicMock()
        object.__setattr__(self.fs, "_join_features", MagicMock(return_value=(MagicMock(), [])))

    # ``type: ignore[call-overload]`` is intentional below — these kwargs
    # are omitted from the FG overload's signature, but the runtime guards
    # still need coverage for callers that bypass type checking.

    def test_exclude_columns_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exclude_columns is not supported with `feature_group`"):
            self.fs.generate_training_set(  # type: ignore[call-overload]
                self.spine, feature_group=self.fg, exclude_columns=["X"]
            )

    def test_include_feature_view_timestamp_col_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "include_feature_view_timestamp_col is not supported with `feature_group`"
        ):
            self.fs.generate_training_set(  # type: ignore[call-overload]
                self.spine, feature_group=self.fg, include_feature_view_timestamp_col=True
            )

    def test_auto_prefix_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "auto_prefix is not supported with `feature_group`"):
            self.fs.generate_training_set(  # type: ignore[call-overload]
                self.spine, feature_group=self.fg, auto_prefix=True
            )

    def test_non_default_join_method_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "join_method is not supported with `feature_group`"):
            self.fs.generate_training_set(  # type: ignore[call-overload]
                self.spine, feature_group=self.fg, join_method="cte"
            )


class GenerateTrainingSetFvPathTest(absltest.TestCase):
    """Regression: the FV path still threads its kwargs through ``_join_features`` unchanged."""

    def test_fv_path_preserves_user_provided_join_method_and_auto_prefix(self) -> None:
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fs = _new_fs_with_mocks()
        spine = MagicMock()
        join_mock = MagicMock(return_value=(MagicMock(), []))
        object.__setattr__(fs, "_join_features", join_mock)

        fs.generate_training_set(spine, [fv], auto_prefix=True, join_method="cte")

        bound = _join_features_call_kwargs(join_mock.call_args)
        self.assertEqual(bound["features"], [fv])
        self.assertTrue(bound["auto_prefix"])
        self.assertEqual(bound["join_method"], "cte")

    def test_fv_path_forwards_include_feature_view_timestamp_col(self) -> None:
        """``include_feature_view_timestamp_col=True`` is FG-incompatible but valid on the FV path."""
        fv = _make_registered_fv(name="USER_FV", version="v1", feature_columns=["F1"])
        fs = _new_fs_with_mocks()
        spine = MagicMock()
        join_mock = MagicMock(return_value=(MagicMock(), []))
        object.__setattr__(fs, "_join_features", join_mock)

        fs.generate_training_set(spine, [fv], include_feature_view_timestamp_col=True)

        bound = _join_features_call_kwargs(join_mock.call_args)
        self.assertTrue(bound["include_feature_view_timestamp_col"])


if __name__ == "__main__":
    absltest.main()
