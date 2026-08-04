"""Identity-parity tests pinning the consolidation of ``decl/enums.py``.

After the consolidation, the ``FeatureViewKind`` / ``SourceType`` /
``FeatureAggregationMethod`` / ``FSBaseType`` / ``TYPE_ALIASES`` /
``normalize_type`` names exposed by ``decl/enums.py`` are the SAME
objects (``is``) as the ones exposed by ``spec/enums.py``. ``OpKind``
remains decl-local.
"""

import pytest

from snowflake.ml.feature_store.decl import enums as decl_enums
from snowflake.ml.feature_store.spec import enums as spec_enums


class TestEnumIdentityParity:
    """The consolidated names re-exported by decl point to the spec objects."""

    def test_feature_view_kind_identity(self) -> None:
        assert decl_enums.FeatureViewKind is spec_enums.FeatureViewKind

    def test_source_type_identity(self) -> None:
        assert decl_enums.SourceType is spec_enums.SourceType

    def test_feature_aggregation_method_identity(self) -> None:
        assert decl_enums.FeatureAggregationMethod is spec_enums.FeatureAggregationMethod

    def test_fs_base_type_identity(self) -> None:
        assert decl_enums.FSBaseType is spec_enums.FSBaseType

    def test_type_aliases_identity(self) -> None:
        assert decl_enums.TYPE_ALIASES is spec_enums.TYPE_ALIASES

    def test_normalize_type_identity(self) -> None:
        assert decl_enums.normalize_type is spec_enums.normalize_type


class TestOpKindStaysDeclLocal:
    """``OpKind`` is about plan operations and is NOT promoted to spec.enums."""

    def test_op_kind_defined_on_decl(self) -> None:
        assert decl_enums.OpKind is not None

    def test_op_kind_not_on_spec(self) -> None:
        assert not hasattr(spec_enums, "OpKind")


class TestSpecEnumsHostsConsolidatedNames:
    """spec/enums.py is the canonical home for FS enums + helpers."""

    def test_fs_base_type_on_spec(self) -> None:
        assert hasattr(spec_enums, "FSBaseType")
        assert spec_enums.FSBaseType.StringType == "StringType"

    def test_type_aliases_on_spec(self) -> None:
        assert hasattr(spec_enums, "TYPE_ALIASES")
        assert spec_enums.TYPE_ALIASES["str"] == "StringType"

    def test_normalize_type_on_spec(self) -> None:
        assert hasattr(spec_enums, "normalize_type")
        assert spec_enums.normalize_type("str") == "StringType"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
