from pathlib import Path
from typing import Any, Optional
from unittest import mock

import yaml
from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import spec_utils, types
from snowflake.ml.jobs._utils.test_file_helper import TestAsset


def _get_dict_difference(expected: dict[str, Any], actual: dict[str, Any], prefix: str = "") -> str:
    diff = []
    for key, value in expected.items():
        actual_value = actual.get(key)
        if prefix:
            key = f"{prefix}.{key}"
        if actual_value is None:
            diff.append(f"{key} not found in actual")
        elif type(value) is not type(actual_value):
            diff.append(f"{key} has type {type(actual_value)} instead of expected {type(value)}")
        elif isinstance(value, dict):
            diff.append(_get_dict_difference(value, actual_value, prefix=key))
        elif isinstance(value, list) and all(isinstance(v, dict) for v in value):
            for i, (expected_item, actual_item) in enumerate(zip(value, actual_value)):
                diff.append(_get_dict_difference(expected_item, actual_item, prefix=f"{key}[{i}]"))
        elif value != actual_value:
            diff.append(f"{key} has value {actual_value} instead of expected {value}")

    return "\n".join(d for d in diff if d)


class SpecUtilsTests(parameterized.TestCase):
    @parameterized.named_parameters(  # type: ignore[misc]
        (
            "basic_properties",
            {"key1": "str_value", "key2": 1, "key3": 1.5},
            {"key2": 10, "key3": 20},
            {"key1": "str_value", "key2": 10, "key3": 20},
        ),
        (
            "nested_dictionaries",
            {"dict_prop": {"key1": "str_value", "key2": 1, "key3": 1.5, "key4": {"nested": "value"}}},
            {
                "key1": "distinct from dict_prop subkey",
                "dict_prop": {
                    "key1": "overridden key1",
                    "key4": {"nested": "overridden nested", "nested2": "new value"},
                    "key5": "new key",
                },
            },
            {
                "key1": "distinct from dict_prop subkey",
                "dict_prop": {
                    "key1": "overridden key1",
                    "key2": 1,
                    "key3": 1.5,
                    "key4": {"nested": "overridden nested", "nested2": "new value"},
                    "key5": "new key",
                },
            },
        ),
        (
            # List merging behavior:
            # - Lists of primitives should be overwritten
            # - Lists of dictionaries should be matched based on a match key, default "name"
            # - Lists of mixed types should be treated as lists of primitives
            "merge_lists",
            {
                "plain_list": [1, 2, 3],
                "dict_list": [
                    {"name": "unmatched", "key": "value"},
                    {"name": "match_me", "key": "value", "key2": "some other property"},
                    {"name": "match_noop", "key": "orig_value"},
                ],
                "nomerge_dicts": [
                    {"id": 1, "key": "value"},
                    {"id": 2, "key": "value"},
                ],
                "mixed_list": [1, "str", {"key": "value"}],
            },
            {
                "plain_list": [4, 5],
                "dict_list": [
                    {"name": "new entry", "key": "some property value"},
                    {"name": "match_me", "key": "value override", "key3": "some new property"},
                    {"name": "match_noop"},
                ],
                "nomerge_dicts": [
                    {"id": 2, "key": "new value"},
                    {"id": 3, "key": "new value"},
                ],
                "mixed_list": ["new str", {"key": "override"}],
            },
            {
                "plain_list": [4, 5],
                "dict_list": [
                    {"name": "unmatched", "key": "value"},
                    {
                        "name": "match_me",
                        "key": "value override",
                        "key2": "some other property",
                        "key3": "some new property",
                    },
                    {"name": "match_noop", "key": "orig_value"},
                    {"name": "new entry", "key": "some property value"},
                ],
                "nomerge_dicts": [
                    {"id": 2, "key": "new value"},
                    {"id": 3, "key": "new value"},
                ],
                "mixed_list": ["new str", {"key": "override"}],
            },
        ),
        (
            "drop_none_values",
            {"key1": 1, "key2": 0, "key3": False, "key4": "", "key5": "non-null"},
            {"key1": 1, "key2": 0, "key3": False, "key4": "", "key5": None},
            {"key1": 1, "key2": 0, "key3": False, "key4": ""},
        ),
        ("drop_empty_nested_dicts", {"nested": {"key1": 1}}, {"nested": {}}, {}),
        (
            "prune_deeply_nested_dicts",
            {
                "nested": {
                    "key1": 0,
                    "key2": "Hello key2",
                    "key3": "Hello key3",
                    "key4": {"key1": "Hello key4.key1", "key2": 0},
                    "key5": {"key1": {"key1": {"hello": "world"}}},
                    "key6": "retained",
                },
            },
            {
                "nested": {
                    "key1": 1,
                    "key2": "",
                    "key3": None,
                    "key4": {"key1": None, "key2": 0},
                    "key5": {"key1": {"key1": {}}},
                },
            },
            {
                "nested": {
                    "key1": 1,
                    "key2": "",
                    "key4": {"key2": 0},
                    "key6": "retained",
                },
            },
        ),
        ("preserve_empty_lists", {"nested_list": [1, 2, 3]}, {"nested_list": []}, {"nested_list": []}),
        (
            "drop_empty_dicts_in_list",
            {
                "list1": [
                    {
                        "name": "entry1",
                        "key1": 1,
                    },
                    {
                        "name": "entry2",
                        "key1": 3,
                    },
                ],
            },
            {
                "list1": [
                    {
                        "name": "entry1",
                        None: None,
                    },
                ],
            },
            {
                "list1": [
                    {
                        "name": "entry2",
                        "key1": 3,
                    },
                ],
            },
        ),
    )
    def test_merge_patch(self, base: dict[str, Any], override: dict[str, Any], expected: dict[str, Any]) -> None:
        actual = spec_utils.merge_patch(base, override)
        self.assertEqual(expected, actual)

    @parameterized.parameters(  # type: ignore[misc]
        {
            "expected": {},
        },
        {
            "environment_vars": {},
            "expected": {},
        },
        {
            "custom_overrides": {},
            "expected": {},
        },
        {
            "environment_vars": {"ENV_VAR1": "VALUE1"},
            "custom_overrides": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "env": {
                                "ENV_VAR1": None,
                            },
                        }
                    ]
                }
            },
            "expected": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                        }
                    ]
                }
            },
        },
        {
            "environment_vars": {
                "ENV_VAR1": "VALUE1",
                "ENV_VAR2": 1,
            },
            "expected": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "env": {
                                "ENV_VAR1": "VALUE1",
                                "ENV_VAR2": 1,
                            },
                        }
                    ]
                }
            },
        },
        {
            "environment_vars": {
                "ENV_VAR1": "VALUE1",
                "ENV_VAR2": 1,
            },
            "custom_overrides": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "env": {
                                "ENV_VAR1": "OVERRIDE_VALUE",
                            },
                        },
                        {
                            "name": "sidecar",
                            "image": "sidecar_image:latest",
                        },
                    ]
                },
                "capabilities": {"securityContext": {"executeAsCaller": True}},
            },
            "expected": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "env": {
                                "ENV_VAR1": "OVERRIDE_VALUE",
                                "ENV_VAR2": 1,
                            },
                        },
                        {
                            "name": "sidecar",
                            "image": "sidecar_image:latest",
                        },
                    ]
                },
                "capabilities": {"securityContext": {"executeAsCaller": True}},
            },
        },
    )
    def test_generate_spec_overrides(self, *, expected: dict[str, Any], **kwargs: Any) -> None:
        actual = spec_utils.generate_spec_overrides(**kwargs)
        self.assertEqual(expected, actual)

    @parameterized.parameters(  # type: ignore[misc]
        (TestAsset("specs/cpu_python.yaml"), types.ComputeResources(cpu=1, memory=1), Path("main.py")),
        (
            TestAsset("specs/cpu_python_subdir.yaml"),
            types.ComputeResources(cpu=2, memory=8),
            Path("src/main.py"),
        ),
        (
            TestAsset("specs/cpu_args.yaml"),
            types.ComputeResources(cpu=1, memory=4),
            Path("main.py"),
            ["pos_arg", "--named_arg", "value"],
        ),
        (
            TestAsset("specs/gpu_python.yaml"),
            types.ComputeResources(cpu=4, memory=16, gpu=1),
            Path("main.py"),
        ),
    )
    def test_prepare_spec(
        self,
        expected: TestAsset,
        resources: types.ComputeResources,
        entrypoint: Path,
        args: Optional[list[str]] = None,
    ) -> None:
        with mock.patch("snowflake.ml.jobs._utils.spec_utils._get_image_spec") as mock_get_image_spec:
            mock_get_image_spec.return_value = types.ImageSpec(
                repo="dummy_repo",
                image_name="dummy_image",
                image_tag="latest",
                resource_requests=resources,
                resource_limits=resources,
            )
            payload = types.UploadedPayload(
                Path("@dummy_stage"),
                [entrypoint],
            )
            spec = spec_utils.generate_service_spec(
                None,  # type: ignore[arg-type] # (Don't need session since we mock out _get_image_spec)
                compute_pool="dummy_pool",
                payload=payload,
                args=args,
            )

        with open(expected.path) as f:
            expected_spec = yaml.safe_load(f)
        self.assertEmpty(_get_dict_difference(expected_spec, spec))

    def test_prepare_spec_with_metrics(self) -> None:
        resources = types.ComputeResources(cpu=1, memory=4)
        entrypoint = Path("src/main.py")
        with mock.patch("snowflake.ml.jobs._utils.spec_utils._get_image_spec") as mock_get_image_spec:
            mock_get_image_spec.return_value = types.ImageSpec(
                repo="dummy_repo",
                image_name="dummy_image",
                image_tag="latest",
                resource_requests=resources,
                resource_limits=resources,
            )
            payload = types.UploadedPayload(
                Path("@dummy_stage"),
                [entrypoint],
            )
            spec = spec_utils.generate_service_spec(
                None,  # type: ignore[arg-type] # (Don't need session since we mock out _get_image_spec)
                compute_pool="dummy_pool",
                payload=payload,
                enable_metrics=True,
            )
            self.assertIn("platformMonitor", spec["spec"])
            self.assertIn("metricConfig", spec["spec"]["platformMonitor"])
            self.assertIn("groups", spec["spec"]["platformMonitor"]["metricConfig"])
            self.assertGreater(len(spec["spec"]["platformMonitor"]["metricConfig"]["groups"]), 0)


if __name__ == "__main__":
    absltest.main()
