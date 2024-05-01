import json

from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.feature_store import (  # type: ignore[attr-defined]
    Entity,
    FeatureView,
    FeatureViewSlice,
    FeatureViewStatus,
    FeatureViewVersion,
)
from snowflake.ml.feature_store.feature_view import (
    _FEATURE_OBJ_TYPE,
    _TIMESTAMP_COL_PLACEHOLDER,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


class FeatureViewTest(absltest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    @classmethod
    def tearDownClass(self) -> None:
        self._session.close()

    def test_invalid_feature_view_name(self) -> None:
        df = self._session.create_dataframe([1, 2, 3], schema=["a"])
        e = Entity(name="foo", join_keys=["a"])
        with self.assertRaisesRegex(ValueError, "FeatureView name.* contains invalid character .*"):
            FeatureView(name="my$fv", entities=[e], feature_df=df)

    def test_invalid_entity_join_keys(self) -> None:
        df = self._session.create_dataframe([1, 2, 3], schema=["a"])
        e = Entity(name="foo", join_keys=["b"])
        with self.assertRaisesRegex(ValueError, "join_key.*is not found in input dataframe.*"):
            FeatureView(name="my_fv", entities=[e], feature_df=df)

    def test_happy_path(self) -> None:
        df = self._session.create_dataframe([1, 2, 3], schema=["a"])
        e = Entity(name="foo", join_keys=["a"])
        fv = FeatureView(name="my_fv", entities=[e], feature_df=df)
        self.assertEqual(FeatureViewStatus.DRAFT, fv.status)
        self.assertIsNone(fv.version)

    def test_slice(self) -> None:
        df = self._session.create_dataframe([[1, 2, 3]], schema=["a", "b", "c"])
        e = Entity(name="foo", join_keys=["a"])
        fv = FeatureView(name="my_fv", entities=[e], feature_df=df)
        fv_slice = fv.slice(names=["b"])
        self.assertEqual(fv_slice.feature_view_ref, fv)
        self.assertEqual(fv_slice.names, ["B"])

        # join_keys is not part of feature names
        with self.assertRaisesRegex(ValueError, "Feature name.*not found in FeatureView.*"):
            fv.slice(names=["a"])

        with self.assertRaisesRegex(ValueError, "Feature name.*not found in FeatureView.*"):
            fv.slice(names=["d"])

    def test_feature_descs(self) -> None:
        df = self._session.create_dataframe([[1, 2, 3, 4]], schema=["a", "b", "c", "d"])
        e = Entity(name="foo", join_keys=["a"])
        fv = FeatureView(name="my_fv", entities=[e], feature_df=df)

        with self.assertRaisesRegex(ValueError, "Feature name .* is not found .*"):
            fv.attach_feature_desc({"e": "foo"})

        fv.attach_feature_desc({"b": "foo", "d": "bar"})
        self.assertEqual(fv.feature_descs, {"B": "foo", "C": "", "D": "bar"})

    def test_invalid_timestamp_col(self) -> None:
        df = self._session.create_dataframe([[1, "bar", 3]], schema=["a", "b", "c"])
        e = Entity(name="foo", join_keys=["a"])

        with self.assertRaisesRegex(ValueError, "Invalid timestamp_col name.*"):
            FeatureView(name="my_fv", entities=[e], feature_df=df, timestamp_col=_TIMESTAMP_COL_PLACEHOLDER)

        with self.assertRaisesRegex(ValueError, "timestamp_col.*is not found in input dataframe.*"):
            FeatureView(name="my_fv", entities=[e], feature_df=df, timestamp_col="d")

        with self.assertRaisesRegex(ValueError, "Invalid data type for timestamp_col.*"):
            FeatureView(name="my_fv", entities=[e], feature_df=df, timestamp_col="b")

    def test_invalid_fully_qualified_name(self) -> None:
        df = self._session.create_dataframe([[1, "bar", 3]], schema=["a", "b", "c"])
        e = Entity(name="foo", join_keys=["a"])
        fv = FeatureView(name="my_fv", entities=[e], feature_df=df)
        with self.assertRaisesRegex(RuntimeError, ".*has not been registered."):
            fv.fully_qualified_name()

    def test_feature_view_serde(self) -> None:
        df = self._session.create_dataframe([[1, 2, 3]], schema=["a", "b", "c"])
        e = Entity(name="foo", join_keys=["a"])
        fv = FeatureView(name="my_fv", entities=[e], feature_df=df)
        serialized = fv.to_json()
        self.assertEqual(fv, FeatureView.from_json(serialized, self._session))

        malformed = json.dumps({_FEATURE_OBJ_TYPE: "foobar"})
        with self.assertRaisesRegex(ValueError, "Invalid json str for FeatureView.*"):
            FeatureView.from_json(malformed, self._session)

    def test_feature_view_slice_serde(self) -> None:
        df = self._session.create_dataframe([[1, 2, 3]], schema=["a", "b", "c"])
        e = Entity(name="foo", join_keys=["a"])
        fv = FeatureView(name="my_fv", entities=[e], feature_df=df)
        fv_slice = fv.slice(names=["b"])
        serialized = fv_slice.to_json()
        self.assertEqual(fv_slice, FeatureViewSlice.from_json(serialized, self._session))

        malformed = json.dumps({_FEATURE_OBJ_TYPE: "foobar"})
        with self.assertRaisesRegex(ValueError, "Invalid json str for FeatureViewSlice.*"):
            FeatureViewSlice.from_json(malformed, self._session)

    def test_feature_view_versions(self) -> None:
        valid_versions = [
            "v2",  # start with letter
            "3x",  # start with digit
            "1",  # single digit
            "2.1",  # digit with period
            "3_1",  # digit with underscore
            "4-1",  # digit with hyphen
            "4-1_2.3",  # digit with period, underscore and hyphen
            "x",  # single letter
            "4x_1",  # digit, letter and underscore
            "latest",  # pure lowercase letters
            "OLD",  # pure uppercase letters
            "OLap",  # pure uppercase letters
            "a" * 128,  # within maximum allowed length
        ]

        invalid_dataset_versions = [
            "",  # empty
            "_v1",  # start with underscore
            ".2",  # start with period
            "3/1",  # digit with slash
            "-4",  # start with hyphen
            "v1$",  # start with letter, contains invalid character
            "9^",  # start with digit, contains invalid character
            "a" * 129,  # exceed maximum allowed length
        ]

        for version in valid_versions:
            FeatureViewVersion(version)

        for version in invalid_dataset_versions:
            with self.assertRaisesRegex(
                snowml_exceptions.SnowflakeMLException, ".* is not a valid feature view version.*"
            ):
                FeatureViewVersion(version)


class EntityTest(absltest.TestCase):
    def test_invalid_entity_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Entity name .* exceeds maximum length.*"):
            Entity(name="foo" * 11, join_keys=["foo"])

        with self.assertRaisesRegex(ValueError, "Entity name contains invalid char.*"):
            Entity(name="my,entity", join_keys=["foo"])

        with self.assertRaisesRegex(ValueError, "Duplicate join keys detected in.*"):
            Entity(name="my_entity", join_keys=["foo", "foo"])

    def test_join_keys_exceed_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "Join key: .* exceeds length limit 256."):
            Entity(name="foo", join_keys=["f" * 257])
        with self.assertRaisesRegex(ValueError, "Maximum number of join keys are 300, but .* is provided."):
            Entity(name="foo", join_keys=["foo"] * 301)

    def test_equality_check(self) -> None:
        self.assertTrue(Entity(name="foo", join_keys=["a"]) == Entity(name="foo", join_keys=["a"]))
        self.assertTrue(
            Entity(name="foo", join_keys=["a"], desc="bar") == Entity(name="foo", join_keys=["a"], desc="bar")
        )
        self.assertFalse(Entity(name="foo", join_keys=["a"]) == Entity(name="bar", join_keys=["a"]))
        self.assertFalse(Entity(name="foo", join_keys=["a"]) == Entity(name="foo", join_keys=["b"]))
        self.assertFalse(Entity(name="foo", join_keys=["a"], desc="bar") == Entity(name="bar", join_keys=["a"]))
        self.assertFalse(
            Entity(name="foo", join_keys=["a"], desc="bar") == Entity(name="bar", join_keys=["a"], desc="baz")
        )

        self.assertTrue(
            [Entity(name="foo", join_keys=["a"]), Entity(name="bar", join_keys=["b"])]
            == [Entity(name="foo", join_keys=["a"]), Entity(name="bar", join_keys=["b"])]
        )
        self.assertFalse(
            [Entity(name="foo", join_keys=["a"]), Entity(name="bar", join_keys=["b"])]
            == [Entity(name="foo", join_keys=["c"]), Entity(name="bar", join_keys=["d"])]
        )


if __name__ == "__main__":
    absltest.main()
