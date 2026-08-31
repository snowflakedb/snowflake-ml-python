"""Phase 1 RED tests — enriched ``FeatureViewRef`` / ``FeatureGroup`` models.

Pin the canonical authoring shape from the FeatureGroup plan, including:

* ``FeatureViewRef`` carries ``name``, ``version``, optional ``slice_columns``,
  optional ``alias``.  ``alias = ""`` is preserved (semantics: "no prefix");
  ``alias = None`` means "use ``FeatureGroup.auto_prefix``".
* ``FeatureGroup`` exposes ``desc`` and ``auto_prefix``; rejects empty
  ``feature_views``, duplicate ``(name, version)`` refs, missing ``version``,
  and ``$`` in the FG name.

These mirror the imperative ``register_feature_group`` preflight so
``snow feature plan`` surfaces the same shape errors at compile time.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from snowflake.ml.feature_store.decl.spec_models import FeatureGroup, FeatureViewRef
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# FeatureViewRef
# ---------------------------------------------------------------------------


class TestFeatureViewRefVersion:
    def test_requires_version(self) -> None:
        with pytest.raises(ValidationError):
            FeatureViewRef(name="my_fv")  # type: ignore[call-arg]

    def test_construction_with_version(self) -> None:
        ref = FeatureViewRef(name="my_fv", version="V1")
        assert ref.name == "my_fv"
        assert ref.version == "V1"
        assert ref.slice_columns is None
        assert ref.alias is None


class TestFeatureViewRefSliceColumns:
    def test_slice_columns_default_none(self) -> None:
        ref = FeatureViewRef(name="my_fv", version="V1")
        assert ref.slice_columns is None

    def test_slice_columns_list(self) -> None:
        ref = FeatureViewRef(
            name="my_fv",
            version="V1",
            slice_columns=["A", "B"],
        )
        assert ref.slice_columns == ["A", "B"]


class TestFeatureViewRefAlias:
    def test_alias_default_none_means_auto_prefix(self) -> None:
        ref = FeatureViewRef(name="my_fv", version="V1")
        assert ref.alias is None

    def test_alias_empty_string_preserved(self) -> None:
        # alias="" semantically means "no prefix"; must NOT be coerced to None.
        ref = FeatureViewRef(name="my_fv", version="V1", alias="")
        assert ref.alias == ""

    def test_alias_value(self) -> None:
        ref = FeatureViewRef(name="my_fv", version="V1", alias="txn")
        assert ref.alias == "txn"


# ---------------------------------------------------------------------------
# FeatureGroup
# ---------------------------------------------------------------------------


def _ref(name: str = "FV1", version: str = "V1", **kwargs: Any) -> FeatureViewRef:
    return FeatureViewRef(name=name, version=version, **kwargs)


class TestFeatureGroupBasics:
    def test_default_kind(self) -> None:
        fg = FeatureGroup(name="MY_FG", feature_views=[_ref()])
        assert fg.kind == "FeatureGroup"

    def test_desc_default_empty_string(self) -> None:
        fg = FeatureGroup(name="MY_FG", feature_views=[_ref()])
        assert fg.desc == ""

    def test_auto_prefix_default_true(self) -> None:
        fg = FeatureGroup(name="MY_FG", feature_views=[_ref()])
        assert fg.auto_prefix is True

    def test_full_construction(self) -> None:
        fg = FeatureGroup(
            name="USER_FRAUD_FG",
            version="V1",
            desc="Combined user signals.",
            auto_prefix=True,
            feature_views=[
                _ref("USER_CLICK_STATS_DECL", "V1"),
                _ref("USER_TXN_STATS", "V1", slice_columns=["TOTAL_SPEND_30D"], alias="txn"),
            ],
        )
        assert len(fg.feature_views) == 2
        assert fg.feature_views[1].slice_columns == ["TOTAL_SPEND_30D"]
        assert fg.feature_views[1].alias == "txn"


class TestFeatureGroupValidatorRejects:
    def test_empty_feature_views_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureGroup(name="MY_FG")

    def test_duplicate_name_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureGroup(
                name="MY_FG",
                feature_views=[_ref("FV1", "V1"), _ref("FV1", "V1")],
            )

    def test_dollar_sign_in_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureGroup(name="MY$FG", feature_views=[_ref()])

    def test_same_name_different_version_allowed(self) -> None:
        # Two refs to the same FV at different versions are a legitimate FG shape.
        fg = FeatureGroup(
            name="MY_FG",
            feature_views=[_ref("FV1", "V1"), _ref("FV1", "V2")],
        )
        assert len(fg.feature_views) == 2


class TestFeatureGroupAcceptsRawDicts:
    """Round-trip check: ``model_validate`` accepts the YAML shape directly."""

    def test_loads_canonical_yaml_shape(self) -> None:
        data = {
            "kind": "FeatureGroup",
            "name": "USER_FRAUD_FG",
            "version": "V1",
            "desc": "Combined user signals.",
            "auto_prefix": True,
            "feature_views": [
                {"name": "FV_A", "version": "V1"},
                {
                    "name": "FV_B",
                    "version": "V1",
                    "slice_columns": ["X", "Y"],
                    "alias": "b",
                },
            ],
        }
        fg = FeatureGroup.model_validate(data)
        assert fg.feature_views[0].version == "V1"
        assert fg.feature_views[1].slice_columns == ["X", "Y"]
        assert fg.feature_views[1].alias == "b"


if __name__ == "__main__":
    pytest_driver.main()
