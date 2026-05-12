"""Module extension for the conda-based Python toolchain.

Replaces the WORKSPACE-based conda toolchain setup with a bzlmod module extension.
Reads BAZEL_CONDA_ENV_NAME and BAZEL_CONDA_PYTHON_VERSION from environment variables
to configure the conda environment and Python toolchain.
"""

load("//bazel/platforms:optional_dependency_groups.bzl", "OPTIONAL_DEPENDENCY_GROUPS")
load(":conda.bzl", "load_conda_rule")
load(":env.bzl", "conda_create_rule")
load(":toolchain.bzl", "toolchain_rule")

_CONDA_DIR = "conda"
_CONDA_ENV_NAME = "env"

def _conda_toolchain_impl(module_ctx):
    env_name = module_ctx.os.environ.get("BAZEL_CONDA_ENV_NAME", "core").lower()
    python_ver = module_ctx.os.environ.get("BAZEL_CONDA_PYTHON_VERSION", "3.11").lower()

    conda_env_map = {
        "all": {
            "compatible_target": [
                Label("//bazel/platforms:core_conda_channel"),
            ] + [
                Label("//bazel/platforms:{}_conda_channel".format(group_name))
                for group_name in OPTIONAL_DEPENDENCY_GROUPS.keys()
            ],
            "environment": Label("//bazel/environments:conda-env-all.yml"),
        },
        "build": {
            "compatible_target": [Label("//bazel/platforms:core_conda_channel")],
            "environment": Label("//bazel/environments:conda-env-build.yml"),
        },
        "core": {
            "compatible_target": [Label("//bazel/platforms:core_conda_channel")],
            "environment": Label("//bazel/environments:conda-env-core.yml"),
        },
    }

    conda_env_map.update({
        group_name: {
            "compatible_target": [
                Label("//bazel/platforms:core_conda_channel"),
                Label("//bazel/platforms:{}_conda_channel".format(group_name)),
            ],
            "environment": Label("//bazel/environments:conda-env-{}.yml".format(group_name)),
        }
        for group_name in OPTIONAL_DEPENDENCY_GROUPS.keys()
    })

    if env_name not in conda_env_map:
        fail("Unsupported conda env: {}. Supported: {}".format(
            env_name,
            ", ".join(conda_env_map.keys()),
        ))

    cfg = conda_env_map[env_name]

    load_conda_rule(
        name = "snowml_conda",
        conda_dir = _CONDA_DIR,
        quiet = True,
    )

    conda_create_rule(
        name = "snowml_conda_env",
        timeout = 3600,
        clean = False,
        conda_repo = "snowml_conda",
        conda_dir = _CONDA_DIR,
        conda_env_name = _CONDA_ENV_NAME,
        coverage_tool = Label("//bazel/coverage_tool:coverage_tool.py"),
        environment = cfg["environment"],
        python_version = python_ver,
        quiet = True,
    )

    toolchain_rule(
        name = "snowml_toolchain",
        runtime = "@snowml_conda_env//:python_runtime",
        toolchain_name = "py_toolchain",
        target_compatible_with = cfg["compatible_target"],
    )

conda_toolchain = module_extension(
    implementation = _conda_toolchain_impl,
    environ = ["BAZEL_CONDA_ENV_NAME", "BAZEL_CONDA_PYTHON_VERSION"],
)
