import json
import os
import tempfile
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import yaml
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta,
    model_sample_input_data,
)
from snowflake.ml.model._signatures import core


def _make_meta(
    *,
    tmpdir: str,
    signatures: dict[str, model_signature.ModelSignature],
) -> model_meta.ModelMetadata:
    with model_meta.create_model_metadata(
        model_dir_path=tmpdir,
        name="m",
        model_type="custom",
        signatures=signatures,
    ) as meta:
        meta.models["m"] = model_blob_meta.ModelBlobMeta(
            name="m", model_type="custom", path="mock", handler_version="version_0"
        )
    return meta


def _read_payload(tmpdir: str) -> dict[str, Any]:
    with open(os.path.join(tmpdir, model_sample_input_data.SAMPLE_INPUT_DATA_FILENAME)) as f:
        return cast(dict[str, Any], json.load(f))


_PREDICT_SIG = model_signature.ModelSignature(
    inputs=[
        model_signature.FeatureSpec(name="a", dtype=model_signature.DataType.FLOAT),
        model_signature.FeatureSpec(name="b", dtype=model_signature.DataType.INT64),
    ],
    outputs=[model_signature.FeatureSpec(name="y", dtype=model_signature.DataType.FLOAT)],
)
_PREDICT_PROBA_SIG = model_signature.ModelSignature(
    inputs=[
        model_signature.FeatureSpec(name="a", dtype=model_signature.DataType.FLOAT),
        model_signature.FeatureSpec(name="b", dtype=model_signature.DataType.INT64),
    ],
    outputs=[model_signature.FeatureSpec(name="p", dtype=model_signature.DataType.FLOAT)],
)
_OTHER_SCHEMA_SIG = model_signature.ModelSignature(
    inputs=[
        model_signature.FeatureSpec(name="text", dtype=model_signature.DataType.STRING),
    ],
    outputs=[model_signature.FeatureSpec(name="label", dtype=model_signature.DataType.STRING)],
)


class SampleInputDataTest(parameterized.TestCase):
    def test_no_sample_input_does_not_write_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": _PREDICT_SIG})
            model_sample_input_data.persist_sample_input_data(
                sample_input_data=None,
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            self.assertFalse(os.path.exists(os.path.join(tmpdir, model_sample_input_data.SAMPLE_INPUT_DATA_FILENAME)))
            self.assertEqual(meta.sample_input_file_paths, {})

    def test_sample_input_writes_file_and_records_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": _PREDICT_SIG})
            df = pd.DataFrame({"a": [1.5, 2.5, 3.5], "b": [10, 20, 30]})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=df,
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            payload = _read_payload(tmpdir)
            self.assertEqual(payload["dataframe_split"]["columns"], ["a", "b"])
            self.assertEqual(payload["dataframe_split"]["data"], [[1.5, 10]])
            self.assertNotIn("params", payload)
            self.assertEqual(meta.sample_input_file_paths, {"predict": "sample_input_data.json"})

    @parameterized.parameters(  # type: ignore[misc]
        {
            "name": "fully_populated_row_wins_over_earlier_null_heavy_row",
            "data": {"a": [None, 1.5, 2.5], "b": [10, 20, 30]},
            "expected_row": [1.5, 20],
        },
        {
            "name": "fewest_nulls_wins_when_no_fully_populated_row",
            "data": {"a": [None, None, None], "b": [None, 20, 30]},
            "expected_row": [None, 20],
        },
        {
            "name": "nan_serialized_as_null",
            "data": {"a": [float("nan"), 1.5], "b": [5, 10]},
            "expected_row": [1.5, 10],
        },
    )
    def test_row_selection(self, name: str, data: dict[str, list[Any]], expected_row: list[Any]) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": _PREDICT_SIG})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame(data),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            self.assertEqual(_read_payload(tmpdir)["dataframe_split"]["data"], [expected_row])

    @parameterized.parameters(  # type: ignore[misc]
        {"name": "large_tensor_skipped", "shape": (224, 224), "should_capture": False},
        {"name": "small_tensor_captured", "shape": (3,), "should_capture": True},
        {"name": "variable_length_captured", "shape": (-1,), "should_capture": True},
    )
    def test_tensor_shape_threshold(self, name: str, shape: tuple[int, ...], should_capture: bool) -> None:
        sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="x", dtype=model_signature.DataType.FLOAT, shape=shape)],
            outputs=[model_signature.FeatureSpec(name="y", dtype=model_signature.DataType.FLOAT)],
        )
        sample_value: Any = np.zeros(tuple(d if d != -1 else 2 for d in shape)).tolist()
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": sig})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"x": [sample_value]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            file_exists = os.path.exists(os.path.join(tmpdir, model_sample_input_data.SAMPLE_INPUT_DATA_FILENAME))
            self.assertEqual(file_exists, should_capture)
            self.assertEqual(bool(meta.sample_input_file_paths), should_capture)

    def test_long_string_truncated(self) -> None:
        sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="prompt", dtype=model_signature.DataType.STRING)],
            outputs=[model_signature.FeatureSpec(name="completion", dtype=model_signature.DataType.STRING)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": sig})
            big_string = "x" * (model_sample_input_data._MAX_STRING_LENGTH + 500)

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"prompt": [big_string]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            stored = _read_payload(tmpdir)["dataframe_split"]["data"][0][0]
            self.assertTrue(stored.endswith(model_sample_input_data._STRING_TRUNCATION_MARKER))
            self.assertEqual(
                len(stored),
                model_sample_input_data._MAX_STRING_LENGTH + len(model_sample_input_data._STRING_TRUNCATION_MARKER),
            )

    def test_methods_with_shared_schema_share_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(
                tmpdir=tmpdir,
                signatures={"predict": _PREDICT_SIG, "predict_proba": _PREDICT_PROBA_SIG},
            )

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"a": [0.25], "b": [7]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            self.assertEqual(
                meta.sample_input_file_paths,
                {"predict": "sample_input_data.json", "predict_proba": "sample_input_data.json"},
            )

    def test_method_with_mismatched_schema_not_referenced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(
                tmpdir=tmpdir,
                signatures={"predict": _PREDICT_SIG, "summarize": _OTHER_SCHEMA_SIG},
            )

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"a": [0.25], "b": [7]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            self.assertEqual(meta.sample_input_file_paths, {"predict": "sample_input_data.json"})

    def test_param_defaults_serialized(self) -> None:
        sig_with_params = model_signature.ModelSignature(
            inputs=_PREDICT_SIG.inputs,
            outputs=_PREDICT_SIG.outputs,
            params=[
                model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.DOUBLE, default_value=0.5),
                model_signature.ParamSpec(name="max_tokens", dtype=model_signature.DataType.INT64, default_value=128),
                model_signature.ParamSpec(name="seed", dtype=model_signature.DataType.INT64, default_value=None),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": sig_with_params})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"a": [0.25], "b": [7]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            self.assertEqual(_read_payload(tmpdir)["params"], {"temperature": 0.5, "max_tokens": 128, "seed": None})

    def test_param_group_default_serialized(self) -> None:
        sig_with_group_param = model_signature.ModelSignature(
            inputs=_PREDICT_SIG.inputs,
            outputs=_PREDICT_SIG.outputs,
            params=[
                core.ParamGroupSpec(
                    name="config",
                    specs=[
                        model_signature.ParamSpec(
                            name="temperature", dtype=model_signature.DataType.DOUBLE, default_value=0.5
                        ),
                        model_signature.ParamSpec(
                            name="max_tokens", dtype=model_signature.DataType.INT64, default_value=128
                        ),
                    ],
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": sig_with_group_param})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"a": [0.25], "b": [7]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            self.assertEqual(_read_payload(tmpdir)["params"], {"config": {"temperature": 0.5, "max_tokens": 128}})

    def test_deeply_nested_param_group_serialized(self) -> None:
        sig = model_signature.ModelSignature(
            inputs=_PREDICT_SIG.inputs,
            outputs=_PREDICT_SIG.outputs,
            params=[
                core.ParamGroupSpec(
                    name="outer",
                    specs=[
                        core.ParamGroupSpec(
                            name="middle",
                            specs=[
                                core.ParamGroupSpec(
                                    name="inner",
                                    specs=[
                                        model_signature.ParamSpec(
                                            name="temperature", dtype=model_signature.DataType.DOUBLE, default_value=0.5
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": sig})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"a": [0.25], "b": [7]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            self.assertEqual(_read_payload(tmpdir)["params"], {"outer": {"middle": {"inner": {"temperature": 0.5}}}})

    def test_numpy_array_aligned_to_signature_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": _PREDICT_SIG})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=np.array([[1.5, 10], [2.5, 20]]),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            payload = _read_payload(tmpdir)
            self.assertEqual(payload["dataframe_split"]["columns"], ["a", "b"])
            self.assertEqual(len(payload["dataframe_split"]["data"]), 1)

    def test_no_signature_alignment_skips_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": _PREDICT_SIG})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            self.assertFalse(os.path.exists(os.path.join(tmpdir, model_sample_input_data.SAMPLE_INPUT_DATA_FILENAME)))
            self.assertEqual(meta.sample_input_file_paths, {})

    def test_unsupported_data_type_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": _PREDICT_SIG})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=object(),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            self.assertFalse(os.path.exists(os.path.join(tmpdir, model_sample_input_data.SAMPLE_INPUT_DATA_FILENAME)))
            self.assertEqual(meta.sample_input_file_paths, {})

    def test_feature_group_input_captured_with_nested_value(self) -> None:
        feature_group_sig = model_signature.ModelSignature(
            inputs=[
                core.FeatureGroupSpec(
                    name="messages",
                    specs=[
                        model_signature.FeatureSpec(name="role", dtype=model_signature.DataType.STRING),
                        model_signature.FeatureSpec(name="content", dtype=model_signature.DataType.STRING),
                    ],
                    shape=(-1,),
                ),
            ],
            outputs=[model_signature.FeatureSpec(name="completion", dtype=model_signature.DataType.STRING)],
        )
        nested_value = [{"role": "user", "content": "Hello"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(
                tmpdir=tmpdir,
                signatures={"predict": _PREDICT_SIG, "chat": feature_group_sig},
            )

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"messages": [nested_value]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            payload = _read_payload(tmpdir)
            self.assertEqual(payload["dataframe_split"]["columns"], ["messages"])
            self.assertEqual(payload["dataframe_split"]["data"], [[nested_value]])
            self.assertEqual(meta.sample_input_file_paths, {"chat": "sample_input_data.json"})

    def test_feature_group_large_nested_shape_skipped(self) -> None:
        # The group's child carries the oversized shape; capture must skip even though
        # the group itself is not a leaf FeatureSpec.
        sig = model_signature.ModelSignature(
            inputs=[
                core.FeatureGroupSpec(
                    name="embeddings",
                    specs=[
                        model_signature.FeatureSpec(name="vector", dtype=model_signature.DataType.FLOAT, shape=(200,)),
                    ],
                    shape=(-1,),
                ),
            ],
            outputs=[model_signature.FeatureSpec(name="y", dtype=model_signature.DataType.FLOAT)],
        )
        nested_value = [{"vector": [0.0] * 200}]
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": sig})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"embeddings": [nested_value]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            self.assertFalse(os.path.exists(os.path.join(tmpdir, model_sample_input_data.SAMPLE_INPUT_DATA_FILENAME)))
            self.assertEqual(meta.sample_input_file_paths, {})

    def test_large_param_drops_params_but_keeps_inputs(self) -> None:
        # An oversized param drops the entire params block (all-or-nothing) but inputs are
        # still captured; the server falls back to signature defaults for the omitted params.
        sig = model_signature.ModelSignature(
            inputs=_PREDICT_SIG.inputs,
            outputs=_PREDICT_SIG.outputs,
            params=[
                model_signature.ParamSpec(name="temperature", dtype=model_signature.DataType.DOUBLE, default_value=0.5),
                model_signature.ParamSpec(
                    name="weights",
                    dtype=model_signature.DataType.DOUBLE,
                    default_value=[0.0] * 200,
                    shape=(200,),
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = _make_meta(tmpdir=tmpdir, signatures={"predict": sig})

            model_sample_input_data.persist_sample_input_data(
                sample_input_data=pd.DataFrame({"a": [1.0], "b": [2]}),
                model_meta=meta,
                model_dir_path=tmpdir,
            )

            payload = _read_payload(tmpdir)
            self.assertEqual(payload["dataframe_split"]["columns"], ["a", "b"])
            self.assertNotIn("params", payload)
            self.assertEqual(meta.sample_input_file_paths, {"predict": "sample_input_data.json"})

    @parameterized.parameters(  # type: ignore[misc]
        {"name": "default_captures", "options": {}, "should_capture": True},
        {"name": "explicit_true_captures", "options": {"capture_sample_input_data": True}, "should_capture": True},
        {"name": "explicit_false_skips", "options": {"capture_sample_input_data": False}, "should_capture": False},
    )
    def test_capture_sample_input_data_option(self, name: str, options: dict[str, Any], should_capture: bool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = model_packager.ModelPackager(tmpdir)
            mp.save(
                name="m",
                model=_SimpleAdditionModel(custom_model.ModelContext()),
                sample_input_data=pd.DataFrame({"a": [1.0], "b": [2.0]}),
                options=cast(model_types.ModelSaveOption, options),
            )

            file_exists = os.path.exists(os.path.join(tmpdir, model_sample_input_data.SAMPLE_INPUT_DATA_FILENAME))
            with open(os.path.join(tmpdir, "model.yaml")) as f:
                raw = yaml.safe_load(f)
            field: Optional[str] = raw["signatures"]["predict"].get("sample_input_file_path")

            self.assertEqual(file_exists, should_capture)
            if should_capture:
                self.assertEqual(field, model_sample_input_data.SAMPLE_INPUT_DATA_FILENAME)
            else:
                self.assertIsNone(field)

    @parameterized.named_parameters(  # type: ignore[misc]
        {
            "testcase_name": "scalar_feature",
            "spec": model_signature.FeatureSpec(name="x", dtype=model_signature.DataType.FLOAT),
            "expected": 1,
        },
        {
            "testcase_name": "shaped_feature",
            "spec": model_signature.FeatureSpec(name="x", dtype=model_signature.DataType.FLOAT, shape=(3, 4)),
            "expected": 12,
        },
        {
            "testcase_name": "variable_dim_ignored",
            "spec": model_signature.FeatureSpec(name="x", dtype=model_signature.DataType.FLOAT, shape=(-1, 5)),
            "expected": 5,
        },
        {
            "testcase_name": "feature_group_sums_children",
            "spec": core.FeatureGroupSpec(
                name="g",
                specs=[
                    model_signature.FeatureSpec(name="a", dtype=model_signature.DataType.STRING),
                    model_signature.FeatureSpec(name="b", dtype=model_signature.DataType.STRING),
                ],
                shape=(-1,),
            ),
            "expected": 2,
        },
        {
            "testcase_name": "feature_group_with_shaped_child",
            "spec": core.FeatureGroupSpec(
                name="g",
                specs=[model_signature.FeatureSpec(name="v", dtype=model_signature.DataType.FLOAT, shape=(200,))],
                shape=(-1,),
            ),
            "expected": 200,
        },
        {
            "testcase_name": "param_group_sums_children",
            "spec": core.ParamGroupSpec(
                name="cfg",
                specs=[
                    model_signature.ParamSpec(name="t", dtype=model_signature.DataType.DOUBLE, default_value=0.5),
                    model_signature.ParamSpec(name="m", dtype=model_signature.DataType.INT64, default_value=1),
                ],
            ),
            "expected": 2,
        },
    )
    def test_static_element_count(self, spec: Any, expected: int) -> None:
        self.assertEqual(model_sample_input_data._static_element_count(spec), expected)


class _SimpleAdditionModel(custom_model.CustomModel):
    @custom_model.inference_api
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"y": X["a"] + X["b"]})


if __name__ == "__main__":
    absltest.main()
