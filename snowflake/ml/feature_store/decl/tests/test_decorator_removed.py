"""Pins for the removal of the legacy ``@feature_view`` decorator.

Q6 of ``plans/python_form/python_authoring_form.md`` retires the
function-decorator Python authoring style in favour of the new
class-based form (``StreamingFeatureView(...)`` /
``BatchFeatureView(...)`` / ``RealtimeFeatureView(...)``).  These
tests pin three contracts:

1. The ``decorator`` submodule no longer exists — both
   ``from snowflake.ml.feature_store.decl.decorator import feature_view``
   and ``from snowflake.ml.feature_store.decl import feature_view``
   raise :class:`ImportError`.
2. The loader's ``load_python_file`` no longer carries any reference
   to the ``_feature_view`` attribute it used to consult when a
   module's top-level symbol was a decorated function.
3. The loader's ``__init__`` does not re-export a ``feature_view``
   symbol.
"""

from __future__ import annotations

import ast
import importlib

import pytest


class TestDecoratorModuleRemoved:
    """The legacy decorator submodule is gone."""

    def test_import_decorator_module_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            importlib.import_module("snowflake.ml.feature_store.decl.decorator")

    def test_from_decorator_import_feature_view_raises(self) -> None:
        # Hammer the precise import path users would have written.
        with pytest.raises(ImportError):
            exec(
                "from snowflake.ml.feature_store.decl.decorator import feature_view",
                {},
            )


class TestDecoratorNotReexported:
    """The ``decl`` package's top-level ``__init__`` does not re-export
    ``feature_view``."""

    def test_decl_package_does_not_re_export_feature_view(self) -> None:
        import snowflake.ml.feature_store.decl as decl

        assert not hasattr(decl, "feature_view")
        assert "feature_view" not in getattr(decl, "__all__", [])


class TestLoaderDoesNotReadFeatureViewAttribute:
    """``loader.load_python_file`` no longer reads the ``_feature_view``
    attribute off a module's top-level symbols.

    The AST scan below is intentional: it is much more durable than
    a runtime smoke test (the legacy code path was triggered only
    when a decorated function was the spec source) and it survives
    refactors that move helpers without changing intent.  If a future
    change ever re-introduces a ``_feature_view`` attribute read into
    this file the test will fail and force the author to either
    rewrite the test or explain why the decorator path is back.
    """

    def test_loader_source_has_no_feature_view_attribute_access(self) -> None:
        import snowflake.ml.feature_store.decl.loader as loader

        with open(loader.__file__, encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "_feature_view":
                pytest.fail(
                    f"loader.py still reads the legacy ``_feature_view`` "
                    f"attribute at line {node.lineno}; the @feature_view "
                    f"decorator path was supposed to be removed per Q6 "
                    f"of plans/python_form/python_authoring_form.md."
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
