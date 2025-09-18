load("//bazel/platforms:optional_dependency_groups.bzl", "OPTIONAL_DEPENDENCY_GROUPS")

def _fetch_conda_env_config_impl(rctx):
    # read the particular environment variable we are interested in
    env_name = rctx.os.environ.get("BAZEL_CONDA_ENV_NAME", "core").lower()
    python_ver = rctx.os.environ.get("BAZEL_CONDA_PYTHON_VERSION", "3.10").lower()

    # necessary to create empty BUILD file for this rule
    # which will be located somewhere in the Bazel build files
    rctx.file("BUILD")

    conda_env_map = {
        "all": {
            "compatible_target": [
                "@SnowML//bazel/platforms:core_conda_channel",
            ] + [
                "@SnowML//bazel/platforms:{}_conda_channel".format(group_name)
                for group_name in OPTIONAL_DEPENDENCY_GROUPS.keys()
            ],
            "environment": "@//bazel/environments:conda-env-all.yml",
        },
        "build": {
            "compatible_target": [
                "@SnowML//bazel/platforms:core_conda_channel",
            ],
            "environment": "@//bazel/environments:conda-env-build.yml",
        },
        "core": {
            "compatible_target": [
                "@SnowML//bazel/platforms:core_conda_channel",
            ],
            "environment": "@//bazel/environments:conda-env-core.yml",
        },
    }

    conda_env_map.update({
        group_name: {
            "compatible_target": [
                "@SnowML//bazel/platforms:core_conda_channel",
                "@SnowML//bazel/platforms:{}_conda_channel".format(group_name),
            ],
            "environment": "@//bazel/environments:conda-env-{}.yml".format(group_name),
        }
        for group_name in OPTIONAL_DEPENDENCY_GROUPS.keys()
    })

    if env_name not in conda_env_map.keys():
        fail("Unsupported conda env {} specified. Only {} is supported.".format(env_name, repr(conda_env_map.keys())))

    # create a temporary file called config.bzl to be loaded into WORKSPACE
    # passing in any desired information from this rule implementation
    rctx.file(
        "config.bzl",
        content = """
NAME = {}
ENVIRONMENT = {}
COMPATIBLE_TARGET = {}
PYTHON_VERSION = {}
""".format(repr(env_name), repr(conda_env_map[env_name]["environment"]), repr(conda_env_map[env_name]["compatible_target"]), repr(python_ver)),
    )

fetch_conda_env_config = repository_rule(
    implementation = _fetch_conda_env_config_impl,
    environ = ["BAZEL_CONDA_ENV_NAME", "BAZEL_CONDA_PYTHON_VERSION"],
)
