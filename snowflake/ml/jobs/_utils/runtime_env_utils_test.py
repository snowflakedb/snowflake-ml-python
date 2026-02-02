import json
from typing import Optional
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import runtime_env_utils
from snowflake.snowpark.row import Row


class RuntimeEnvUtilsTests(parameterized.TestCase):
    @parameterized.named_parameters(  # type: ignore[misc]
        {
            "testcase_name": "basic_spcs_runtime",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "runtime:spcs": {
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11.10",
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/runtime_image/snowbooks:1.5.0"
                                    ),
                                },
                                "createdOn": "2025-01-16T10:30:45.123Z",
                                "description": "First ML Job Runtime with Python 3.11",
                                "id": "nre-3.11-runtime-a",
                            }
                        }
                    )
                )
            ],
            "expected": "/snowflake/images/snowflake_images/st_plat/runtime/x86/runtime_image/snowbooks:1.5.0",
        },
        {
            "testcase_name": "spcs_runtime_with_extra_fields",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "runtime:spcs": {
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11.11",
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/runtime_image/snowbooks:1.5.0"
                                    ),
                                    "randomKey1": "randomValue1",
                                    "extraField": "test123",
                                },
                                "randomOuterKey": "outerValue",
                                "extraOuterField": 42,
                                "createdOn": "2025-01-16T10:30:45.123Z",
                                "description": "First ML Job Runtime with Python 3.11",
                                "id": "nre-3.11-runtime-a",
                            },
                            "random_key": "random_value",
                        }
                    )
                )
            ],
            "expected": "/snowflake/images/snowflake_images/st_plat/runtime/x86/runtime_image/snowbooks:1.5.0",
        },
        {
            "testcase_name": "gpu_hardware_no_match",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "runtime:spcs": {
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11.0",
                                    "hardwareType": "GPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/runtime_image/snowbooks:1.5.0"
                                    ),
                                },
                                "createdOn": "2025-01-16T10:30:45.123Z",
                                "description": "First ML Job Runtime with Python 3.11",
                                "id": "nre-3.11-runtime-a",
                            }
                        }
                    )
                )
            ],
            "expected": None,
        },
        {
            "testcase_name": "image_without_tag",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "runtime:spcs": {
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11.19",
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/x86/runtime_image/snowbooks"
                                    ),
                                },
                                "createdOn": "2025-01-16T10:30:45.123Z",
                                "description": "First ML Job Runtime with Python 3.11",
                                "id": "nre-3.11-runtime-a",
                            }
                        }
                    )
                )
            ],
            "expected": "/snowflake/images/snowflake_images/st_plat/runtime/x86/runtime_image/snowbooks",
        },
        {
            "testcase_name": "mismatched_python_version",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "runtime:spcs": {
                                "spcsContainerImage": {
                                    "pythonVersion": "3.10.19",
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/runtime_image/snowbooks:1.5.0"
                                    ),
                                },
                                "createdOn": "2025-01-16T10:30:45.123Z",
                                "description": "First ML Job Runtime with Python 3.11",
                                "id": "nre-3.11-runtime-a",
                            }
                        }
                    )
                )
            ],
            "expected": None,
        },
        {
            "testcase_name": "multiple_same_python_version",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "MLJOB-RUNTIME-A:spcs": {
                                "createdOn": "2025-01-16T10:30:45.123Z",
                                "description": "First ML Job Runtime with Python 3.11",
                                "id": "nre-3.11-runtime-a",
                                "title": "ML Job Runtime A",
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11.5",
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/runtime_image/snowbooks:1.5.0"
                                    ),
                                },
                            },
                            "MLJOB-RUNTIME-B:spcs": {
                                "createdOn": "2025-01-15T11:30:45.123Z",
                                "description": "Second ML Job Runtime with Python 3.11",
                                "id": "nre-3.11-runtime-b",
                                "title": "ML Job Runtime B",
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11.5",
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/runtime_image/snowbooks:1.6.0"
                                    ),
                                },
                            },
                        }
                    )
                )
            ],
            "expected": "/snowflake/images/snowflake_images/st_plat/runtime/x86/runtime_image/snowbooks:1.5.0",
        },
        {
            "testcase_name": "gpu_and_cpu_runtimes",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "MLJOB-GPU-RUNTIME:spcs": {
                                "createdOn": "2025-01-15T10:30:45.123Z",
                                "description": "GPU ML Job Runtime with Python 3.11",
                                "id": "nre-3.11-gpu-runtime",
                                "title": "ML Job GPU Runtime",
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11.5",
                                    "hardwareType": "GPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/gpu_image/snowbooks:1.5.0"
                                    ),
                                },
                            },
                            "MLJOB-CPU-RUNTIME:spcs": {
                                "createdOn": "2025-01-15T11:30:45.123Z",
                                "description": "CPU ML Job Runtime with Python 3.11",
                                "id": "nre-3.11-cpu-runtime",
                                "title": "ML Job CPU Runtime",
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11.5",
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/runtime_image/snowbooks:1.5.0"
                                    ),
                                },
                            },
                        }
                    )
                )
            ],
            "expected": "/snowflake/images/snowflake_images/st_plat/runtime/x86/runtime_image/snowbooks:1.5.0",
        },
        {
            "testcase_name": "runtime_without_spcs_suffix",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "MLJOB-RUNTIME-NO-SUFFIX": {
                                "createdOn": "2025-01-15T10:30:45.123Z",
                                "description": "Runtime without spcs suffix",
                                "id": "nre-3.11-no-suffix",
                                "title": "Runtime No Suffix",
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11.5",
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/runtime_image/snowbooks:1.5.0"
                                    ),
                                },
                            },
                        }
                    )
                )
            ],
            "expected": "/snowflake/images/snowflake_images/st_plat/runtime/x86/runtime_image/snowbooks:1.5.0",
        },
        {
            "testcase_name": "python_version_without_patch",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "MLJOB-RUNTIME-3.11:spcs": {
                                "createdOn": "2025-01-15T10:30:45.123Z",
                                "description": "ML Job Runtime with Python 3.11",
                                "id": "nre-3.11-runtime",
                                "title": "ML Job Runtime 3.11",
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11",
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/runtime_image/snowbooks:1.5.0"
                                    ),
                                },
                            }
                        }
                    )
                )
            ],
            "expected": "/snowflake/images/snowflake_images/st_plat/runtime/x86/runtime_image/snowbooks:1.5.0",
        },
        {
            "testcase_name": "mixed_warehouse_and_spcs",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "WH-RUNTIME-2.0:warehouse": {
                                "createdOn": "2025-04-11T20:03:52.569Z",
                                "description": "includes Python 3.11",
                                "id": "nre-3.10-2.0",
                                "title": "Snowflake Warehouse Runtime 2.0",
                                "warehouseRuntime": {"pythonEnvironmentId": "3.11.2"},
                            },
                            "MLJOB-CORRECT:spcs": {
                                "createdOn": "2025-07-28T21:49:39.685Z",
                                "description": "Correct Python 3.11 runtime",
                                "id": "nre-3.11-mljob-test",
                                "title": "Correct Runtime Test",
                                "spcsContainerRuntime": {
                                    "hardwareType": "CPU",
                                    "pythonVersion": "3.11.2",
                                    "runtimeContainerImage": (
                                        "/snowflake/images/snowflake_images/st_plat/runtime/"
                                        "x86/runtime_image/snowbooks:1.5.0"
                                    ),
                                },
                            },
                        }
                    )
                )
            ],
            "expected": "/snowflake/images/snowflake_images/st_plat/runtime/x86/runtime_image/snowbooks:1.5.0",
        },
        {
            "testcase_name": "only_warehouse_runtimes",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "WH-RUNTIME-2.0:warehouse": {
                                "createdOn": "2025-04-11T20:03:52.569Z",
                                "description": "includes Python 3.11",
                                "id": "nre-3.11-2.0",
                                "title": "Snowflake Warehouse Runtime 2.0",
                                "warehouseRuntime": {"pythonEnvironmentId": "3.11-2.0"},
                            },
                            "WH-RUNTIME-1.0:warehouse": {
                                "createdOn": "2025-04-11T20:03:51.363Z",
                                "description": "includes Python 3.9",
                                "id": "nre-3.9-1.0",
                                "title": "Snowflake Warehouse Runtime 1.0",
                                "warehouseRuntime": {"pythonEnvironmentId": "3.9-1.0"},
                            },
                        }
                    )
                )
            ],
            "expected": None,
        },
        {
            "testcase_name": "invalid_python_version_format",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "INVALID-BAD-PYVER:spcs": {
                                "createdOn": "2025-01-15T10:30:45.123Z",
                                "description": "Invalid runtime with bad pythonVersion",
                                "id": "nre-invalid-1",
                                "title": "Invalid Runtime 1",
                                "spcsContainerRuntime": {
                                    "pythonVersion": "invalid.version.format",
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": "/snowflake/images/image:1.5.0",
                                },
                            }
                        }
                    )
                )
            ],
            "expected": None,
        },
        {
            "testcase_name": "missing_python_version",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "INVALID-MISSING-PYVER:spcs": {
                                "createdOn": "2025-01-15T10:30:45.123Z",
                                "description": "Invalid runtime missing pythonVersion",
                                "id": "nre-invalid-2",
                                "title": "Invalid Runtime 2",
                                "spcsContainerRuntime": {
                                    "hardwareType": "CPU",
                                    "runtimeContainerImage": "/snowflake/images/image:1.5.0",
                                },
                            }
                        }
                    )
                )
            ],
            "expected": None,
        },
        {
            "testcase_name": "missing_hardware_type",
            "query_result": [
                Row(
                    RESULT=json.dumps(
                        {
                            "INVALID-MISSING-HW:spcs": {
                                "createdOn": "2025-01-15T10:30:45.123Z",
                                "description": "Invalid runtime missing hardwareType",
                                "id": "nre-invalid-3",
                                "title": "Invalid Runtime 3",
                                "spcsContainerRuntime": {
                                    "pythonVersion": "3.11.5",
                                    "runtimeContainerImage": "/snowflake/images/image:1.5.0",
                                },
                            }
                        }
                    )
                )
            ],
            "expected": None,
        },
    )
    def test_get_runtime_image(
        self,
        query_result: list[Row],
        expected: Optional[str],
    ) -> None:
        """Test _get_runtime_image function core scenarios."""
        with mock.patch("snowflake.ml.jobs._utils.query_helper.run_query") as mock_query, mock.patch(
            "sys.version_info", new=mock.Mock(major=3, minor=11)
        ):

            mock_query.return_value = query_result

            result = runtime_env_utils.find_runtime_image(
                mock.Mock(),
                "CPU",
                "3.11",
            )
            self.assertEqual(expected, result)


if __name__ == "__main__":
    absltest.main()
