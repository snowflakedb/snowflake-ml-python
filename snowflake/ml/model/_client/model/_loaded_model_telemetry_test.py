import pickle
from typing import Any
from unittest import mock

from absl.testing import absltest

from snowflake.ml.model._client.model import _loaded_model_telemetry as lmt

_SEND_CUSTOM_USAGE = "snowflake.ml.model._client.model._loaded_model_telemetry.telemetry.send_custom_usage"


class _FakeEstimator:
    def predict(self, x: Any) -> Any:
        return x * 2


class LoadedModelTelemetryTest(absltest.TestCase):
    @mock.patch(_SEND_CUSTOM_USAGE)
    def test_predict_emits_telemetry_on_every_call(self, mock_send: mock.MagicMock) -> None:
        model = _FakeEstimator()
        instrumented = lmt.instrument_for_telemetry(model, model_name="M", version_name="V")

        self.assertIs(instrumented, model)
        self.assertEqual(model.predict(3), 6)
        self.assertEqual(model.predict(4), 8)

        self.assertEqual(mock_send.call_count, 2)
        kwargs = mock_send.call_args.kwargs
        self.assertEqual(kwargs["project"], "MLOps")
        self.assertEqual(kwargs["subproject"], "ModelManagement")
        self.assertEqual(kwargs["telemetry_type"], "snowml_loaded_model_usage")
        data = kwargs["data"]
        self.assertEqual(data["method_name"], "predict")
        self.assertEqual(data["model_name"], "M")
        self.assertEqual(data["version_name"], "V")
        self.assertEqual(data["model_class_name"], "_FakeEstimator")


class _MultiMethodModel:
    def predict(self, x: Any) -> Any:
        return x

    def predict_proba(self, x: Any) -> Any:
        return [x]

    def score(self, y: Any) -> float:
        return 1.0

    def transform(self, x: Any) -> Any:
        return x

    def fit(self, x: Any) -> Any:
        return self

    def partial_fit(self, x: Any) -> Any:
        return self

    def get_params(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def set_params(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def forward(self, x: Any) -> Any:
        return x

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def train(self, *args: Any, **kwargs: Any) -> Any:
        return self

    def encode(self, x: Any) -> Any:
        return [x]


class AllowlistTest(absltest.TestCase):
    @mock.patch(_SEND_CUSTOM_USAGE)
    def test_each_allowlisted_method_emits_on_every_call(self, mock_send: mock.MagicMock) -> None:
        model = _MultiMethodModel()
        lmt.instrument_for_telemetry(model, model_name="M", version_name="V")

        names = (
            "predict",
            "predict_proba",
            "score",
            "transform",
            "fit",
            "partial_fit",
            "get_params",
            "set_params",
            "forward",
            "eval",
            "train",
            "encode",
        )
        for name in names:
            getattr(model, name)(1)
            getattr(model, name)(2)

        method_names = [c.kwargs["data"]["method_name"] for c in mock_send.call_args_list]
        self.assertEqual(method_names, [name for name in names for _ in range(2)])

    @mock.patch(_SEND_CUSTOM_USAGE)
    def test_object_without_allowlisted_methods_is_no_op(self, mock_send: mock.MagicMock) -> None:
        class _Empty:
            pass

        model = _Empty()
        result = lmt.instrument_for_telemetry(model, model_name="M", version_name="V")
        self.assertIs(result, model)
        self.assertIs(type(result), _Empty)
        mock_send.assert_not_called()

    @mock.patch(_SEND_CUSTOM_USAGE)
    def test_private_method_is_not_instrumented(self, mock_send: mock.MagicMock) -> None:
        class _M:
            def predict(self, x: Any) -> Any:
                return x

            def _predict(self, x: Any) -> Any:
                return x

        model = _M()
        lmt.instrument_for_telemetry(model, model_name="M", version_name="V")
        self.assertEqual(model._predict(1), 1)
        mock_send.assert_not_called()


class _PipelineLike:
    def __call__(self, text: str) -> str:
        return text.upper()


class CallSupportTest(absltest.TestCase):
    @mock.patch(_SEND_CUSTOM_USAGE)
    def test_call_dunder_emits_telemetry(self, mock_send: mock.MagicMock) -> None:
        model = _PipelineLike()
        lmt.instrument_for_telemetry(model, model_name="M", version_name="V")
        self.assertEqual(model("hi"), "HI")
        self.assertEqual(model("bye"), "BYE")
        self.assertEqual(mock_send.call_count, 2)
        self.assertEqual(mock_send.call_args.kwargs["data"]["method_name"], "__call__")

    @mock.patch(_SEND_CUSTOM_USAGE)
    def test_object_call_is_not_instrumented(self, mock_send: mock.MagicMock) -> None:
        class _NotCallable:
            pass

        model = _NotCallable()
        lmt.instrument_for_telemetry(model, model_name="M", version_name="V")
        with self.assertRaises(TypeError):
            model()  # type: ignore[operator]
        mock_send.assert_not_called()


class _Picklable:
    def __init__(self, n: int) -> None:
        self.n = n

    def predict(self, x: Any) -> Any:
        return x + self.n


class PickleCompatibilityTest(absltest.TestCase):
    @mock.patch(_SEND_CUSTOM_USAGE)
    def test_pickle_round_trip_yields_original_class_uninstrumented(self, mock_send: mock.MagicMock) -> None:
        model = _Picklable(10)
        lmt.instrument_for_telemetry(model, model_name="M", version_name="V")

        blob = pickle.dumps(model)
        restored = pickle.loads(blob)
        self.assertIs(type(restored), _Picklable)
        self.assertEqual(restored.n, 10)
        self.assertEqual(restored.predict(5), 15)

        mock_send.reset_mock()
        restored.predict(5)
        mock_send.assert_not_called()


class RobustnessTest(absltest.TestCase):
    @mock.patch(_SEND_CUSTOM_USAGE, side_effect=RuntimeError("boom"))
    def test_telemetry_failure_does_not_break_user_call(self, _mock: mock.MagicMock) -> None:
        class _M:
            def predict(self, x: Any) -> Any:
                return x + 1

        model = _M()
        lmt.instrument_for_telemetry(model, model_name="M", version_name="V")
        self.assertEqual(model.predict(1), 2)

    @mock.patch(_SEND_CUSTOM_USAGE)
    def test_user_exception_propagates(self, _mock: mock.MagicMock) -> None:
        class _M:
            def predict(self, x: Any) -> Any:
                raise ValueError("user bug")

        model = _M()
        lmt.instrument_for_telemetry(model, model_name="M", version_name="V")
        with self.assertRaises(ValueError):
            model.predict(1)

    @mock.patch(_SEND_CUSTOM_USAGE)
    def test_disable_env_var_returns_uninstrumented(self, mock_send: mock.MagicMock) -> None:
        class _M:
            def predict(self, x: Any) -> Any:
                return x

        model = _M()
        with mock.patch.dict("os.environ", {lmt._DISABLE_ENV_VAR: "1"}):
            result = lmt.instrument_for_telemetry(model, model_name="M", version_name="V")
        self.assertIs(result, model)
        self.assertIs(type(result), _M)
        model.predict(1)
        mock_send.assert_not_called()


if __name__ == "__main__":
    absltest.main()
