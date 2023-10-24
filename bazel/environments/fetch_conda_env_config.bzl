def _fetch_conda_env_config_impl(rctx):
    # read the particular environment variable we are interested in
    env_name = rctx.os.environ.get("BAZEL_CONDA_ENV_NAME", "extended").lower()
    python_ver = rctx.os.environ.get("BAZEL_CONDA_PYTHON_VERSION", "3.8").lower()

    # necessary to create empty BUILD file for this rule
    # which will be located somewhere in the Bazel build files
    rctx.file("BUILD")

    conda_env_map = {
        "build": {
            "compatible_target": ["@SnowML//bazel/platforms:snowflake_conda_channel"],
            "environment": "@//bazel/environments:conda-env-build.yml",
        },
        "extended": {
            "compatible_target": ["@SnowML//bazel/platforms:extended_conda_channels"],
            "environment": "@//bazel/environments:conda-env.yml",
        },
        # `extended_oss` is the extended env for OSS repo which is a  strict subset of `extended`.
        # It's intended for development without dev VPN.
        "extended_oss": {
            "compatible_target": ["@SnowML//bazel/platforms:extended_conda_channels"],
            "environment": "@//bazel/environments:conda-env.yml",
        },
        "sf_only": {
            "compatible_target": ["@SnowML//bazel/platforms:snowflake_conda_channel"],
            "environment": "@//bazel/environments:conda-env-snowflake.yml",
        },
    }

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
