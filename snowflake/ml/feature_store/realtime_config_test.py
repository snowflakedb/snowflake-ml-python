"""Unit tests for :mod:`realtime_config`.

Tests are organized by which validator layer owns each rejection so
reviewers can verify each layer is honestly enforced and gaps surface
immediately.

Layers (now shared with SFV via
:mod:`snowflake.ml.feature_store._compute_fn_validation`):
  - (a) Source / AST layer (``validate_compute_fn_source``) — imports,
        forbidden builtins, single top-level ``def``, no nested ``def`` /
        ``async def`` / ``lambda``.
  - (b) Callable layer (``validate_compute_fn_callable``) — closures,
        unbound names, free names outside the runtime namespace.
  - (c) Round-trip exec + arity check (RTFV-only).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.feature_store._compute_fn_validation import (
    ALLOWED_NAMESPACE_KEYS,
    FORBIDDEN_BUILTINS,
    validate_compute_fn_callable,
)
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
    OnlineConfig,
    OnlineStoreType,
)
from snowflake.ml.feature_store.realtime_config import (
    _RTFV_RUNTIME_NAMESPACE,
    RealtimeConfig,
    _rehydrate_realtime_compute_fn,
)
from snowflake.ml.feature_store.request_source import RequestSource
from snowflake.snowpark.types import DoubleType, StringType, StructField, StructType

# ============================================================================
# Module-level fixtures
# ============================================================================


def _make_request_source(columns: Optional[list[tuple[str, str]]] = None) -> RequestSource:
    fields = columns or [("amount", "double")]
    return RequestSource(
        schema=StructType([StructField(n, DoubleType() if t == "double" else StringType()) for n, t in fields])
    )


def _make_upstream_fv(name: str = "TXN_FV", feature_cols: Optional[list[str]] = None) -> FeatureView:
    """Build a registered-looking FeatureView for source tests."""
    feature_cols = feature_cols or ["avg_amount"]
    schema = StructType([StructField("USER_ID", StringType())] + [StructField(c, DoubleType()) for c in feature_cols])
    mock_df = MagicMock()
    mock_df.columns = [f.name for f in schema.fields]
    mock_df.schema = schema
    mock_df.queries = {"queries": [f"SELECT * FROM {name}"]}
    fv = FeatureView(
        name=name,
        entities=[Entity(name="USER", join_keys=["USER_ID"])],
        feature_df=mock_df,
        online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
    )
    fv._version = FeatureViewVersion("v1")
    fv._infer_schema_df = mock_df
    fv._status = FeatureViewStatus.ACTIVE
    return fv


_DEFAULT_OUTPUT_SCHEMA = StructType(
    [
        StructField("risk_score", DoubleType()),
        StructField("risk_bucket", StringType()),
    ]
)


# Module-level compute_fns. These MUST be module-level so ``inspect.getsource``
# can find them.


def good_compute_fn(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    """Single namespace symbol (pd) — happy path."""
    return pd.DataFrame(
        {
            "risk_score": req["amount"] / (txn["avg_amount"] + 1),
            "risk_bucket": ["high"] * len(req),
        }
    )


def compute_fn_with_in_body_pandas_import(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd_local  # noqa: F401 — exercise import allowlist
    import pandas as pd

    return pd.DataFrame(
        {
            "risk_score": req["amount"],
            "risk_bucket": ["x"] * len(req),
        }
    )


def compute_fn_with_in_body_numpy(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    return pd.DataFrame(
        {
            "risk_score": np.log1p(req["amount"]),
            "risk_bucket": ["x"] * len(req),
        }
    )


def compute_fn_with_in_body_re(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    import re

    return pd.DataFrame(
        {
            "risk_score": req["amount"],
            "risk_bucket": [re.sub(r"\W", "_", "x")] * len(req),
        }
    )


def compute_fn_with_in_body_copy(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    import copy

    return pd.DataFrame(
        {
            "risk_score": copy.deepcopy(req["amount"]),
            "risk_bucket": ["x"] * len(req),
        }
    )


def compute_fn_with_in_body_dataclasses(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    import dataclasses

    return pd.DataFrame(
        {
            "risk_score": req["amount"],
            "risk_bucket": [str(dataclasses.is_dataclass(int))] * len(req),
        }
    )


def bad_import_sklearn(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    import sklearn  # noqa: F401

    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_from_import_sklearn(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    from sklearn import preprocessing  # noqa: F401

    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_dunder_import(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    mod = __import__("os")  # noqa: F841
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_eval(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    v = eval("1 + 1")  # noqa: F841 — testing eval rejection
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_exec(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    exec("x = 1")  # testing exec rejection
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_compile(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    c = compile("1+1", "<x>", "eval")  # noqa: F841 — testing compile rejection
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_open(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    with open("/tmp/x") as f:  # noqa: F841 — testing open rejection
        pass
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_globals(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    g = globals()  # noqa: F841
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_getattr(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    fn = getattr(pd, "DataFrame")  # noqa: B009,F841
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_breakpoint(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    breakpoint()  # testing breakpoint rejection
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_input(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    v = input()  # noqa: F841
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def bad_nested_def(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    def helper(x: float) -> float:
        return x * 2

    return pd.DataFrame({"risk_score": [helper(1.0)] * len(req), "risk_bucket": ["x"] * len(req)})


def bad_lambda(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    sq = lambda x: x * x  # noqa: E731 — testing lambda rejection
    return pd.DataFrame(
        {"risk_score": [sq(1.0)] * len(req), "risk_bucket": ["x"] * len(req)}  # type: ignore[no-untyped-call]
    )


_MODULE_LEVEL_CONST = 42


def bad_undefined_name(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    risk = [SOME_UNDEFINED_NAME] * len(req)  # type: ignore[name-defined] # noqa: F821
    return pd.DataFrame({"risk_score": risk, "risk_bucket": ["x"] * len(req)})


def bad_module_level_helper(x: float) -> float:
    return x * 2


def bad_uses_module_helper(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "risk_score": [bad_module_level_helper(1.0)] * len(req),
            "risk_bucket": ["x"] * len(req),
        }
    )


def compute_fn_one_arg(req: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": req["amount"], "risk_bucket": ["x"] * len(req)})


def compute_fn_two_args_no_req(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": [0.0] * len(a), "risk_bucket": ["x"] * len(a)})


def compute_fn_three_args(req: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": req["amount"], "risk_bucket": ["x"] * len(req)})


def compute_fn_with_args(req: pd.DataFrame, *args: Any) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def compute_fn_with_kwargs(req: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


def compute_fn_with_kw_only(req: pd.DataFrame, txn: pd.DataFrame, *, flag: bool = False) -> pd.DataFrame:
    return pd.DataFrame({"risk_score": [0.0], "risk_bucket": ["x"]})


# Used to fabricate the closure rejection in symbol-layer tests. Returning
# the inner function from a factory is the canonical way to create a Python
# closure that `inspect.getclosurevars` can see as a `nonlocals` entry.
def _make_closure_compute_fn() -> Callable[..., pd.DataFrame]:
    enclosing_constant = 7

    def closure_compute_fn(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "risk_score": [enclosing_constant] * len(req),
                "risk_bucket": ["x"] * len(req),
            }
        )

    return closure_compute_fn


_CLOSURE_COMPUTE_FN = _make_closure_compute_fn()


# ============================================================================
# Happy path — every namespace symbol resolves cleanly.
# ============================================================================


class HappyPathTest(parameterized.TestCase):
    """Each namespace key + each in-body import variant exercises the full stack."""

    @parameterized.named_parameters(  # type: ignore[misc]
        ("module_level_pd_only", good_compute_fn),
        ("in_body_pandas_import", compute_fn_with_in_body_pandas_import),
        ("in_body_numpy_import", compute_fn_with_in_body_numpy),
        ("in_body_re_import", compute_fn_with_in_body_re),
        ("in_body_copy_import", compute_fn_with_in_body_copy),
        ("in_body_dataclasses_import", compute_fn_with_in_body_dataclasses),
    )
    def test_constructs_successfully(self, fn: Callable[..., pd.DataFrame]) -> None:
        rt = RealtimeConfig(
            compute_fn=fn,
            sources=[_make_request_source(), _make_upstream_fv()],
            output_schema=_DEFAULT_OUTPUT_SCHEMA,
        )
        self.assertEqual(rt.get_function_name(), fn.__name__)
        self.assertIn("def ", rt.get_function_source())
        # compute_fn is replaced by the rehydrated callable (object identity check).
        self.assertIsNot(rt.compute_fn, fn)
        self.assertEqual(rt.compute_fn.__name__, fn.__name__)


# ============================================================================
# Step 1 — author-shape rejections.
# ============================================================================


class AuthorShapeRejectionTest(absltest.TestCase):
    def test_rejects_lambda(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a named function"):
            RealtimeConfig(
                compute_fn=lambda req, txn: pd.DataFrame(),
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_rejects_callable_class_instance(self) -> None:
        class CallableInstance:
            def __call__(self, req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame()

        # A callable class instance has no __name__ attribute.
        with self.assertRaisesRegex(ValueError, "must be a named function"):
            RealtimeConfig(
                compute_fn=CallableInstance(),
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_rejects_non_callable(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be callable"):
            RealtimeConfig(
                compute_fn=42,  # type: ignore[arg-type]
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )


# ============================================================================
# Source layer — bad imports + dangerous builtins (__import__/eval/exec/compile).
# ============================================================================


class ImportLayerRejectionTest(parameterized.TestCase):
    @parameterized.named_parameters(  # type: ignore[misc]
        ("import_sklearn", bad_import_sklearn, "sklearn"),
        ("from_sklearn_import", bad_from_import_sklearn, "sklearn"),
        ("dunder_import", bad_dunder_import, "__import__"),
        ("eval", bad_eval, "eval"),
        ("exec", bad_exec, "exec"),
        ("compile", bad_compile, "compile"),
    )
    def test_rejects(self, fn: Callable[..., pd.DataFrame], offender: str) -> None:
        with self.assertRaises(ValueError) as cm:
            RealtimeConfig(
                compute_fn=fn,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )
        self.assertIn(offender, str(cm.exception))


# ============================================================================
# Source layer — nested defs / lambdas + extended forbidden builtins
# (open / globals / locals / vars / getattr / setattr / delattr /
# breakpoint / input).
# ============================================================================


class AstLayerRejectionTest(parameterized.TestCase):
    @parameterized.named_parameters(  # type: ignore[misc]
        ("open", bad_open, "open"),
        ("globals", bad_globals, "globals"),
        ("getattr", bad_getattr, "getattr"),
        ("breakpoint", bad_breakpoint, "breakpoint"),
        ("input", bad_input, "input"),
    )
    def test_rejects_forbidden_builtin(self, fn: Callable[..., pd.DataFrame], builtin_name: str) -> None:
        with self.assertRaises(ValueError) as cm:
            RealtimeConfig(
                compute_fn=fn,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )
        self.assertIn(builtin_name, str(cm.exception))

    def test_rejects_nested_def(self) -> None:
        with self.assertRaises(ValueError) as cm:
            RealtimeConfig(
                compute_fn=bad_nested_def,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )
        self.assertIn("nested function", str(cm.exception))
        # Error names the offending helper.
        self.assertIn("'helper'", str(cm.exception))

    def test_rejects_lambda_in_body(self) -> None:
        with self.assertRaises(ValueError) as cm:
            RealtimeConfig(
                compute_fn=bad_lambda,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )
        self.assertIn("lambda", str(cm.exception))

    def test_nested_def_passes_callable_layer_alone(self) -> None:
        """Regression guard: ``validate_compute_fn_callable`` alone would PASS a
        nested-def source. Without the AST layer the gap goes unnoticed; this
        test fails loudly if a future refactor drops the AST check.
        """
        # If this raises, the callable layer is incorrectly catching nested
        # defs that it shouldn't see.
        validate_compute_fn_callable(bad_nested_def, kind="realtime feature view")

    def test_does_not_reject_attribute_call_on_namespace_module(self) -> None:
        """``pd.read_csv(...)`` etc. must not be matched as forbidden. The forbidden
        set is checked by attribute name; ``read_csv`` is not in it."""

        def safe_uses_pd_read_csv(req: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
            # We don't actually call read_csv (no file), but the AST layer must
            # not reject the textual reference.
            _ = pd.read_csv
            return pd.DataFrame({"risk_score": [0.0] * len(req), "risk_bucket": ["x"] * len(req)})

        # AST layer alone must not reject this; full RealtimeConfig will
        # because the constructed-fn never executes ``read_csv``. Construction
        # succeeds.
        rt = RealtimeConfig(
            compute_fn=safe_uses_pd_read_csv,
            sources=[_make_request_source(), _make_upstream_fv()],
            output_schema=_DEFAULT_OUTPUT_SCHEMA,
        )
        self.assertEqual(rt.get_function_name(), "safe_uses_pd_read_csv")


# ============================================================================
# Callable layer — closure / unbound / free-name rejections.
# ============================================================================


class SymbolLayerRejectionTest(absltest.TestCase):
    def test_rejects_undefined_name(self) -> None:
        with self.assertRaises(ValueError) as cm:
            RealtimeConfig(
                compute_fn=bad_undefined_name,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )
        self.assertIn("SOME_UNDEFINED_NAME", str(cm.exception))

    def test_rejects_closure(self) -> None:
        with self.assertRaises(ValueError) as cm:
            RealtimeConfig(
                compute_fn=_CLOSURE_COMPUTE_FN,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )
        self.assertIn("closures", str(cm.exception))
        self.assertIn("enclosing_constant", str(cm.exception))

    def test_rejects_module_level_helper(self) -> None:
        """A module-level helper referenced from compute_fn lands in
        ``globals`` outside the runtime namespace."""
        with self.assertRaises(ValueError) as cm:
            RealtimeConfig(
                compute_fn=bad_uses_module_helper,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )
        self.assertIn("bad_module_level_helper", str(cm.exception))

    def test_dunder_builtins_is_allowed(self) -> None:
        """``__builtins__`` is allowed even though it appears in
        ``getclosurevars(fn).globals`` for all Python functions."""
        # The happy-path test above already exercises this; this is the
        # explicit regression guard.
        rt = RealtimeConfig(
            compute_fn=good_compute_fn,
            sources=[_make_request_source(), _make_upstream_fv()],
            output_schema=_DEFAULT_OUTPUT_SCHEMA,
        )
        self.assertEqual(rt.get_function_name(), "good_compute_fn")


# ============================================================================
# Step 5 — sources rules.
# ============================================================================


class SourceRejectionTest(absltest.TestCase):
    def test_rejects_empty_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "sources must be non-empty"):
            RealtimeConfig(
                compute_fn=compute_fn_one_arg,
                sources=[],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_rejects_request_source_not_first(self) -> None:
        with self.assertRaisesRegex(ValueError, "RequestSource must be at sources\\[0\\]"):
            RealtimeConfig(
                compute_fn=good_compute_fn,
                sources=[_make_upstream_fv(), _make_request_source()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_rejects_multiple_request_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "more than one RequestSource"):
            RealtimeConfig(
                compute_fn=good_compute_fn,
                sources=[_make_request_source(), _make_request_source()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_rejects_request_source_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one upstream"):
            RealtimeConfig(
                compute_fn=compute_fn_one_arg,
                sources=[_make_request_source()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_accepts_no_request_source(self) -> None:
        """RTFV without a RequestSource: a single upstream FV is the only source."""
        rt = RealtimeConfig(
            compute_fn=compute_fn_one_arg,
            sources=[_make_upstream_fv()],
            output_schema=_DEFAULT_OUTPUT_SCHEMA,
        )
        self.assertIsNone(rt.request_source)
        self.assertEqual(len(rt.feature_view_sources), 1)

    def test_accepts_multiple_upstreams_without_request_source(self) -> None:
        """RTFV without a RequestSource: multiple upstream FVs are valid."""
        rt = RealtimeConfig(
            compute_fn=compute_fn_two_args_no_req,
            sources=[_make_upstream_fv("A"), _make_upstream_fv("B")],
            output_schema=_DEFAULT_OUTPUT_SCHEMA,
        )
        self.assertIsNone(rt.request_source)
        self.assertEqual(len(rt.feature_view_sources), 2)

    def test_rejects_feature_group_source(self) -> None:
        from snowflake.ml.feature_store.feature_group import FeatureGroup

        upstream = _make_upstream_fv()
        fg = FeatureGroup(name="USER_GROUP", features=[upstream])
        with self.assertRaisesRegex(ValueError, "FeatureGroup"):
            RealtimeConfig(
                compute_fn=good_compute_fn,
                sources=[_make_request_source(), fg],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_rejects_arbitrary_object_as_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a FeatureView or FeatureViewSlice"):
            RealtimeConfig(
                compute_fn=good_compute_fn,
                sources=[_make_request_source(), "not an FV"],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_rejects_arity_mismatch(self) -> None:
        """compute_fn declares 2 positional args but sources has 3."""
        with self.assertRaisesRegex(ValueError, "positional argument"):
            RealtimeConfig(
                compute_fn=good_compute_fn,
                sources=[_make_request_source(), _make_upstream_fv("A"), _make_upstream_fv("B")],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_rejects_args(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not use \\*args"):
            RealtimeConfig(
                compute_fn=compute_fn_with_args,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_rejects_kwargs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not use \\*\\*kwargs"):
            RealtimeConfig(
                compute_fn=compute_fn_with_kwargs,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )

    def test_rejects_keyword_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "keyword-only"):
            RealtimeConfig(
                compute_fn=compute_fn_with_kw_only,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=_DEFAULT_OUTPUT_SCHEMA,
            )


# ============================================================================
# Step 6 — output_schema rules.
# ============================================================================


class OutputSchemaRejectionTest(absltest.TestCase):
    def test_rejects_empty_output_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_schema must be non-empty"):
            RealtimeConfig(
                compute_fn=good_compute_fn,
                sources=[_make_request_source(), _make_upstream_fv()],
                output_schema=StructType([]),
            )


# ============================================================================
# Step 7-9 — round-trip canary + arity.
# ============================================================================


class RoundTripTest(absltest.TestCase):
    def test_rehydrated_callable_matches_original_on_fixed_input(self) -> None:
        """Original and rehydrated compute_fn produce identical output on the
        same input — proves the namespace-only contract is enforced."""
        rt = RealtimeConfig(
            compute_fn=good_compute_fn,
            sources=[_make_request_source(), _make_upstream_fv()],
            output_schema=_DEFAULT_OUTPUT_SCHEMA,
        )
        req = pd.DataFrame({"amount": [100.0, 50.0]})
        txn = pd.DataFrame({"avg_amount": [10.0, 20.0]})
        original_out = good_compute_fn(req, txn)
        rehydrated_out = rt.compute_fn(req, txn)
        pd.testing.assert_frame_equal(original_out, rehydrated_out)

    def test_compute_fn_is_replaced_with_rehydrated_version(self) -> None:
        rt = RealtimeConfig(
            compute_fn=good_compute_fn,
            sources=[_make_request_source(), _make_upstream_fv()],
            output_schema=_DEFAULT_OUTPUT_SCHEMA,
        )
        self.assertIsNot(rt.compute_fn, good_compute_fn)
        self.assertEqual(rt.compute_fn.__name__, "good_compute_fn")

    def test_request_source_property(self) -> None:
        req = _make_request_source()
        fv = _make_upstream_fv()
        rt = RealtimeConfig(
            compute_fn=good_compute_fn,
            sources=[req, fv],
            output_schema=_DEFAULT_OUTPUT_SCHEMA,
        )
        self.assertIs(rt.request_source, req)
        self.assertEqual(len(rt.feature_view_sources), 1)
        self.assertIs(rt.feature_view_sources[0], fv)

    def test_request_source_property_returns_none_when_omitted(self) -> None:
        fv = _make_upstream_fv()
        rt = RealtimeConfig(
            compute_fn=compute_fn_one_arg,
            sources=[fv],
            output_schema=_DEFAULT_OUTPUT_SCHEMA,
        )
        self.assertIsNone(rt.request_source)
        self.assertEqual(rt.feature_view_sources, [fv])


# ============================================================================
# _rehydrate_realtime_compute_fn standalone tests
# ============================================================================


class RehydrateHelperTest(absltest.TestCase):
    def test_rehydrates_simple_function(self) -> None:
        source = (
            "def my_fn(req, txn):\n"
            "    return pd.DataFrame({'risk_score': req['amount'], 'risk_bucket': ['x'] * len(req)})\n"
        )
        fn = _rehydrate_realtime_compute_fn(source, "my_fn")
        out = fn(pd.DataFrame({"amount": [1.0]}), pd.DataFrame({"avg_amount": [2.0]}))
        self.assertEqual(list(out.columns), ["risk_score", "risk_bucket"])

    def test_rejects_missing_function(self) -> None:
        source = "def wrong_name(req, txn):\n    return None\n"
        with self.assertRaisesRegex(ValueError, "does not define a top-level function"):
            _rehydrate_realtime_compute_fn(source, "expected_name")

    def test_rejects_unparsable_source(self) -> None:
        with self.assertRaises(ValueError):
            _rehydrate_realtime_compute_fn("def f(:\n", "f")

    def test_rehydrated_fn_is_inspect_getsource_able(self) -> None:
        """inspect.getsource() must work on the rehydrated callable.

        Read-time reconstruction (compose_rtfv_from_metadata) feeds the
        rehydrated callable back into RealtimeConfig(...), whose __post_init__
        re-runs inspect.getsource() for validation. Without the linecache
        entry the second call raises OSError("could not get source code").
        """
        source = (
            "def round_trip_fn(req, txn):\n"
            "    return pd.DataFrame({'risk_score': req['amount'], 'risk_bucket': ['x'] * len(req)})\n"
        )
        fn = _rehydrate_realtime_compute_fn(source, "round_trip_fn")
        recovered = inspect.getsource(fn)
        self.assertIn("def round_trip_fn(req, txn):", recovered)


# ============================================================================
# _RTFV_RUNTIME_NAMESPACE sanity checks
# ============================================================================


class RuntimeNamespaceTest(absltest.TestCase):
    def test_runtime_namespace_has_documented_keys_only(self) -> None:
        self.assertEqual(
            set(_RTFV_RUNTIME_NAMESPACE.keys()),
            {"pd", "pandas", "np", "numpy", "re", "copy"},
        )

    def test_runtime_namespace_keys_match_validation_allowlist(self) -> None:
        """The runtime namespace dict and the validation allow-list must
        agree: every namespace key is permitted as a free global, and the
        only permitted keys are namespace keys."""
        self.assertEqual(
            set(_RTFV_RUNTIME_NAMESPACE.keys()),
            set(ALLOWED_NAMESPACE_KEYS),
        )

    def test_forbidden_builtins_covers_both_layers(self) -> None:
        """The unified forbidden-builtin set must contain both the
        ``__import__``/``eval``/``exec``/``compile`` group and the
        ``open``/``getattr``/``input``/etc. group — strict by
        construction. Regression guard against an accidental policy
        narrowing that would let either group slip through."""
        self.assertTrue(
            {"__import__", "eval", "exec", "compile"}.issubset(FORBIDDEN_BUILTINS),
        )
        self.assertTrue(
            {
                "open",
                "globals",
                "locals",
                "vars",
                "getattr",
                "setattr",
                "delattr",
                "breakpoint",
                "input",
            }.issubset(FORBIDDEN_BUILTINS),
        )


if __name__ == "__main__":
    absltest.main()
