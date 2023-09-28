load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(":conda.bzl", "load_conda_rule")
load(":env.bzl", "conda_create_rule")
load(":toolchain.bzl", "toolchain_rule")

CONDA_DIR = "conda"
CONDA_ENV_NAME = "env"

# download and install conda
def load_conda(conda_repo_name, **kwargs):
    maybe(
        load_conda_rule,
        conda_repo_name,
        conda_dir = CONDA_DIR,
        **kwargs
    )

# create conda environment
def conda_create(name, conda_repo_name, **kwargs):
    maybe(
        conda_create_rule,
        name,
        conda_repo = conda_repo_name,
        conda_dir = CONDA_DIR,
        conda_env_name = CONDA_ENV_NAME,
        **kwargs
    )

# register python toolchain from environments
def register_toolchain(env, name, toolchain_name, **kwargs):
    runtime = "@{}//:python_runtime".format(env)

    maybe(
        toolchain_rule,
        name,
        runtime = runtime,
        toolchain_name = toolchain_name,
        **kwargs
    )

    native.register_toolchains("@{}//:{}".format(name, toolchain_name))
