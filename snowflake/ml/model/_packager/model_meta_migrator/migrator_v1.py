from typing import Any

from packaging import requirements, version

from snowflake.ml import version as snowml_version
from snowflake.ml.model._packager.model_meta_migrator import base_migrator


class MetaMigrator_v1(base_migrator.BaseModelMetaMigrator):
    source_version = "1"
    target_version = "2023-12-01"

    @staticmethod
    def upgrade(original_meta_dict: dict[str, Any]) -> dict[str, Any]:
        loaded_python_version = version.parse(original_meta_dict["python_version"])
        if original_meta_dict.get("local_ml_library_version", None):
            loaded_lib_version = str(version.parse(original_meta_dict["local_ml_library_version"]))
        else:
            lib_spec_str = next(
                filter(
                    lambda x: requirements.Requirement(x).name == "snowflake-ml-python",
                    original_meta_dict["conda_dependencies"],
                ),
                None,
            )
            if lib_spec_str is None:
                loaded_lib_version = snowml_version.VERSION
            loaded_lib_version = list(requirements.Requirement(str(lib_spec_str)).specifier)[0].version

        return dict(
            creation_timestamp=original_meta_dict["creation_timestamp"],
            env=dict(
                conda="env/conda.yaml",
                pip="env/requirements.txt",
                python_version=f"{loaded_python_version.major}.{loaded_python_version.minor}",
                cuda_version=original_meta_dict.get("cuda_version", None),
                snowpark_ml_version=loaded_lib_version,
            ),
            metadata=original_meta_dict.get("metadata", None),
            model_type=original_meta_dict["model_type"],
            models={
                name: {**value, "handler_version": "2023-12-01"} for name, value in original_meta_dict["models"].items()
            },
            name=original_meta_dict["name"],
            signatures=original_meta_dict["signatures"],
            version=MetaMigrator_v1.target_version,
            min_snowpark_ml_version="1.0.12",
        )
