def _fetch_conda_env_config_impl(rctx):
    # read the particular environment variable we are interested in
    config = rctx.os.environ.get("BUILD_CONDA_ENV", "extended").lower()

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
        "sf_only": {
            "compatible_target": ["@SnowML//bazel/platforms:snowflake_conda_channel"],
            "environment": "@//bazel/environments:conda-env-snowflake.yml",
        },
    }

    if config not in conda_env_map.keys():
        fail("Unsupported conda env {} specified. Only {} is supported.".format(config, repr(conda_env_map.keys())))

    # create a temporary file called config.bzl to be loaded into WORKSPACE
    # passing in any desired information from this rule implementation
    rctx.file(
        "config.bzl",
        content = """
NAME = {}
ENVIRONMENT = {}
COMPATIBLE_TARGET = {}
""".format(repr(config), repr(conda_env_map[config]["environment"]), repr(conda_env_map[config]["compatible_target"])),
    )

fetch_conda_env_config = repository_rule(
    implementation = _fetch_conda_env_config_impl,
    environ = ["BUILD_CONDA_ENV"],
)
