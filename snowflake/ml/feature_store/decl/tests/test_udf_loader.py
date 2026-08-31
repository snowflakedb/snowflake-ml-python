"""Tests for decl/udf_loader.py — compile UDF source string into a named callable.

The compiled callable must satisfy snowml-core's ``StreamConfig`` validators:

1. ``callable(fn)`` is True.
2. ``fn.__name__`` matches the requested name (lambdas rejected).
3. ``inspect.getsource(fn)`` returns the original source — this is the
   precondition that gates the AST-walk import guard inside
   ``StreamConfig.__post_init__``.
4. AST imports stay within ``{numpy, pandas, re, copy}``; everything else
   raises ``ValueError`` from ``StreamConfig._validate_imports``.
"""

from __future__ import annotations

import inspect

import pytest

from snowflake.ml.test_utils import pytest_driver

_VALID_PANDAS_UDF = """
import pandas as pd

def compute_engagement_metrics(clickstream: pd.DataFrame) -> pd.DataFrame:
    df = clickstream.copy()
    weights = {"page_view": 1.0, "click": 2.0}
    df["ENGAGEMENT_SCORE"] = df["EVENT_TYPE"].map(weights).fillna(1.0)
    return df
"""

_DISALLOWED_OS_IMPORT = """
def fn(df):
    import os
    os.getenv("X")
    return df
"""

_NO_MATCHING_FUNCTION = """
import pandas as pd

def some_other_fn(df):
    return df
"""


class TestCompileUdfCallable:
    def test_compile_udf_returns_named_callable(self) -> None:
        from snowflake.ml.feature_store.decl.udf_loader import compile_udf_callable

        fn = compile_udf_callable(_VALID_PANDAS_UDF, "compute_engagement_metrics")

        assert callable(fn), "compile_udf_callable must return a callable."
        assert fn.__name__ == "compute_engagement_metrics", (
            "fn.__name__ must match the requested function name so " "StreamConfig's named-function check passes."
        )

    def test_inspect_getsource_succeeds_on_compiled_callable(self) -> None:
        """``StreamConfig.__post_init__`` calls ``inspect.getsource(fn)`` on the
        transformation function before walking its AST.  This test pins that the
        compiled callable's source is registered (e.g. via ``linecache``) so
        ``inspect.getsource`` can recover it — otherwise ``StreamConfig`` raises
        ``ValueError("Cannot extract source code from transformation_fn ...")``.
        """
        from snowflake.ml.feature_store.decl.udf_loader import compile_udf_callable

        fn = compile_udf_callable(_VALID_PANDAS_UDF, "compute_engagement_metrics")

        source = inspect.getsource(fn)
        assert "def compute_engagement_metrics" in source, (
            "inspect.getsource must recover the compiled function's source — "
            "without linecache registration StreamConfig will reject it."
        )

    def test_pandas_only_imports_pass_stream_config_ast_guard(self) -> None:
        """End-to-end: feed the compiled callable to snowml-core's ``StreamConfig``
        and assert it constructs without raising.  This is the strongest possible
        check that the bridge from source-string to ``transformation_fn`` is
        wired correctly.
        """
        from snowflake.ml.feature_store.decl.udf_loader import compile_udf_callable
        from snowflake.ml.feature_store.stream_config import StreamConfig

        fn = compile_udf_callable(_VALID_PANDAS_UDF, "compute_engagement_metrics")

        class _SentinelDF:
            pass

        sc = StreamConfig(
            stream_source="CLICKSTREAM_EVENTS",
            transformation_fn=fn,
            backfill_df=_SentinelDF(),
        )

        assert sc.transformation_fn is fn
        assert sc.get_function_name() == "compute_engagement_metrics"

    def test_disallowed_import_raises_through_stream_config(self) -> None:
        """A UDF that imports outside the allowed set must compile fine on its
        own (Python permits it) but be rejected by ``StreamConfig`` when the
        callable is wired in.  This guarantees the guard fires end-to-end.
        """
        from snowflake.ml.feature_store.decl.udf_loader import compile_udf_callable
        from snowflake.ml.feature_store.stream_config import StreamConfig

        fn = compile_udf_callable(_DISALLOWED_OS_IMPORT, "fn")

        class _SentinelDF:
            pass

        with pytest.raises(ValueError, match="not allowed"):
            StreamConfig(
                stream_source="X",
                transformation_fn=fn,
                backfill_df=_SentinelDF(),
            )

    def test_missing_function_raises_clear_error(self) -> None:
        from snowflake.ml.feature_store.decl.udf_loader import compile_udf_callable

        with pytest.raises(ValueError, match="compute_engagement_metrics"):
            compile_udf_callable(_NO_MATCHING_FUNCTION, "compute_engagement_metrics")


if __name__ == "__main__":
    pytest_driver.main()
