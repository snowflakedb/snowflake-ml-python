# Common Default

# Wrapper to make sure tests are run.
# Allow at most 4 hours for eternal tests.
test --run_under='//bazel:test_wrapper' --test_timeout=-1,-1,-1,14400

# Since integration tests are located in different packages than code under test,
# the default instrumentation filter would exclude the code under test. This
# makes bazel consider all the source code in our repo for coverage.
coverage --instrumentation_filter="-//tests[/:]"

# Internal definitions

# Make the target platform and the host platform the same
build:_build --platforms //bazel/platforms:core_conda_env --host_platform //bazel/platforms:core_conda_env --repo_env=BAZEL_CONDA_ENV_NAME=build
build:_core --platforms //bazel/platforms:core_conda_env --host_platform //bazel/platforms:core_conda_env --repo_env=BAZEL_CONDA_ENV_NAME=core
build:_all --platforms //bazel/platforms:all_conda_env --host_platform //bazel/platforms:all_conda_env --repo_env=BAZEL_CONDA_ENV_NAME=all

# Public definitions

# Python environment flag, should use in combination with other configs

build:py3.9 --repo_env=BAZEL_CONDA_PYTHON_VERSION=3.9
build:py3.10 --repo_env=BAZEL_CONDA_PYTHON_VERSION=3.10
build:py3.11 --repo_env=BAZEL_CONDA_PYTHON_VERSION=3.11
build:py3.12 --repo_env=BAZEL_CONDA_PYTHON_VERSION=3.12
build:py3.13 --repo_env=BAZEL_CONDA_PYTHON_VERSION=3.13
build:py3.14 --repo_env=BAZEL_CONDA_PYTHON_VERSION=3.14

build:build --config=_build

# Config to sync files
run:pre_build --config=_build --config=py3.11

# Config to run type check
build:typecheck --aspects @rules_mypy//:mypy.bzl%mypy_aspect --output_groups=mypy --config=_all --config=py3.11

# Config to build the doc
# Note: docs build uses py3.10 due to Sphinx module resolution issues with py3.11
build:docs --config=_all --config=py3.10

# Public the extended setting

cquery:core --config=_core
test:core --config=_core
run:core --config=_core
cquery:all --config=_all
test:all --config=_all
run:all --config=_all

# Environment variables for Hugging Face
build --action_env=HF_HUB_ETAG_TIMEOUT=86400
build --action_env=HF_HUB_DOWNLOAD_TIMEOUT=86400
build --action_env=HF_ENDPOINT=https://artifactory.ci1.us-west-2.aws-dev.app.snowflake.com/artifactory/api/huggingfaceml/huggingface-remote
build --action_env=HF_TOKEN

# Below are auto-generated settings, do not modify them directly
