"""Unit tests for resolve_prefer_pip_for_automatic_dependencies (core-only target)."""

from typing import Optional
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml._internal import env_utils
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env


class ResolvePreferPipForAutomaticDependenciesTest(parameterized.TestCase):
    """Matrix for :func:`model_env.resolve_prefer_pip_for_automatic_dependencies`."""

    @parameterized.parameters(  # type: ignore[misc]
        # SPCS-only always prefers pip for automatic deps (independent of flag, conda list, local env).
        {
            "target_platforms": [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
            "conda_dependencies": None,
            "pip_flag": False,
            "is_local_conda": True,
            "force_conda_defaults": False,
            "expected": True,
        },
        {
            "target_platforms": [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
            "conda_dependencies": ["numpy"],
            "pip_flag": False,
            "is_local_conda": False,
            "force_conda_defaults": False,
            "expected": True,
        },
        # Non-SPCS: pip-only flag + no conda deps + not local conda.
        {
            "target_platforms": [model_types.TargetPlatform.WAREHOUSE],
            "conda_dependencies": None,
            "pip_flag": True,
            "is_local_conda": False,
            "force_conda_defaults": False,
            "expected": True,
        },
        {
            "target_platforms": [model_types.TargetPlatform.WAREHOUSE],
            "conda_dependencies": [],
            "pip_flag": True,
            "is_local_conda": False,
            "force_conda_defaults": False,
            "expected": True,
        },
        {
            "target_platforms": None,
            "conda_dependencies": None,
            "pip_flag": True,
            "is_local_conda": False,
            "force_conda_defaults": False,
            "expected": True,
        },
        # Mixed targets: not the SPCS-only shortcut; follows pip-flag path.
        {
            "target_platforms": [
                model_types.TargetPlatform.WAREHOUSE,
                model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
            ],
            "conda_dependencies": None,
            "pip_flag": True,
            "is_local_conda": False,
            "force_conda_defaults": False,
            "expected": True,
        },
        {
            "target_platforms": [
                model_types.TargetPlatform.WAREHOUSE,
                model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES,
            ],
            "conda_dependencies": None,
            "pip_flag": False,
            "is_local_conda": False,
            "force_conda_defaults": False,
            "expected": False,
        },
        # User conda deps block the pip-flag path (non-SPCS).
        {
            "target_platforms": [model_types.TargetPlatform.WAREHOUSE],
            "conda_dependencies": ["numpy"],
            "pip_flag": True,
            "is_local_conda": False,
            "force_conda_defaults": False,
            "expected": False,
        },
        # Local conda environment blocks the pip-flag path.
        {
            "target_platforms": [model_types.TargetPlatform.WAREHOUSE],
            "conda_dependencies": None,
            "pip_flag": True,
            "is_local_conda": True,
            "force_conda_defaults": False,
            "expected": False,
        },
        # Pip-only flag off.
        {
            "target_platforms": [model_types.TargetPlatform.WAREHOUSE],
            "conda_dependencies": None,
            "pip_flag": False,
            "is_local_conda": False,
            "force_conda_defaults": False,
            "expected": False,
        },
        # Warehouse reconciler forces conda defaults after shared PyPI is unavailable.
        {
            "target_platforms": [model_types.TargetPlatform.WAREHOUSE],
            "conda_dependencies": None,
            "pip_flag": True,
            "is_local_conda": False,
            "force_conda_defaults": True,
            "expected": False,
        },
    )
    def test_resolve_prefer_pip_for_automatic_dependencies(
        self,
        *,
        target_platforms: Optional[list[model_types.TargetPlatform]],
        conda_dependencies: Optional[list[str]],
        pip_flag: bool,
        is_local_conda: bool,
        force_conda_defaults: bool,
        expected: bool,
    ) -> None:
        with mock.patch.object(model_env, "is_pip_only_packaging_enabled", return_value=pip_flag):
            with mock.patch.object(env_utils, "is_local_conda_environment", return_value=is_local_conda):
                got = model_env.resolve_prefer_pip_for_automatic_dependencies(
                    target_platforms=target_platforms,
                    conda_dependencies=conda_dependencies,
                    force_conda_defaults=force_conda_defaults,
                )
        self.assertEqual(got, expected)


if __name__ == "__main__":
    absltest.main()
