def _fetch_conda_env_config_impl(rctx):
    # read the particular environment variable we are interested in
    config = rctx.os.environ.get("BUILD_CONDA_ENV", "extended").lower()

    # necessary to create empty BUILD file for this rule
    # which will be located somewhere in the Bazel build files
    rctx.file("BUILD")

    conda_env_map = {
        "build":{
            "environment": "@//bazel/environments:conda-env-build.yml",
            "compatible_target": ["@SnowML//bazel/platforms:extended_conda_channels"]
        },
        "sf_only":{
            "environment": "@//bazel/environments:conda-env-snowflake.yml",
            "compatible_target": ["@SnowML//bazel/platforms:snowflake_conda_channel"]
        },
        "extended":{
            "environment": "@//bazel/environments:conda-env.yml",
            "compatible_target": ["@SnowML//bazel/platforms:extended_conda_channels"]
        },
    }

    if config not in conda_env_map.keys():
        fail("Unsupported conda env {} specified. Only {} is supported.".format(config, repr(conda_env_map.keys())))

    # create a temporary file called config.bzl to be loaded into WORKSPACE
    # passing in any desired information from this rule implementation
    rctx.file("config.bzl", content = """
NAME = {}
ENVIRONMENT = {}
COMPATIBLE_TARGET = {}
""".format(repr(config), repr(conda_env_map[config]["environment"]), repr(conda_env_map[config]["compatible_target"]))
    )


fetch_conda_env_config = repository_rule(
    implementation=_fetch_conda_env_config_impl,
    environ = ["BUILD_CONDA_ENV"]
)
