#!/bin/bash

# Usage
# build_and_run_tests.sh <workspace> [-b <bazel path>] [--env pip|conda] [--mode merge_gate|continuous_run] [--with-snowpark] [--with-spcs-image] [--report <report_path>]
#
# Args
# workspace: path to the workspace, SnowML code should be in snowml directory.
#
# Optional Args
# b: specify path to bazel
# env: Set the environment, choose from pip and conda
# mode: Set the tests set to be run.
#   merge_gate: run affected tests only.
#   continuous_run (default): run all tests. (For nightly run. Alias: release)
#   quarantined: run all quarantined tests.
# with-snowpark: Build and test with snowpark in snowpark-python directory in the workspace.
# with-spcs-image: Build and test with spcs-image in spcs-image directory in the workspace.
# snowflake-env: The environment of the snowflake, use to determine the test quarantine list
# report: Path to xml test report
#
# Action
#   - Copy the integration tests from workspace folder and execute them in testing Python env using pytest.
#   - This is to mimic the behavior of using snowml wheel package in user land.

set -o pipefail
set -u
set -e

PROG=$0

help() {
    local exit_code=$1
    echo "Usage: ${PROG} <workspace> [-b <bazel path>] [--env pip|conda] [--mode merge_gate|continuous_run|quarantined] [--with-snowpark] [--with-spcs-image] [--snowflake-env <sf_env>] [--report <report_path>]"
    exit "${exit_code}"
}

WORKSPACE=$1 && shift || help 1
BAZEL="bazel"
ENV="pip"
WITH_SNOWPARK=false
WITH_SPCS_IMAGE=false
MODE="continuous_run"
PYTHON_VERSION=3.9
PYTHON_ENABLE_SCRIPT="bin/activate"
SNOWML_DIR="snowml"
SNOWPARK_DIR="snowpark-python"
IS_NT=false
JUNIT_REPORT_PATH=""
SF_ENV="prod3"

while (($#)); do
    case $1 in
    -b | --bazel_path)
        shift
        BAZEL=$1
        ;;
    -e | --env)
        shift
        if [[ $1 = "pip" || $1 = "conda" ]]; then
            ENV=$1
        else
            help 1
        fi
        ;;
    --with-snowpark)
        WITH_SNOWPARK=true
        ;;
    --mode)
        shift
        if [[ $1 = "merge_gate" || $1 = "continuous_run" || $1 = "quarantined" || $1 = "release" ]]; then
            MODE=$1
            if [[ $MODE = "release" ]]; then
                MODE="continuous_run"
            fi
        else
            help 1
        fi
        ;;
    --snowflake-env)
        shift
        SF_ENV=$1
        ;;
    --report)
        shift
        JUNIT_REPORT_PATH=$1
        ;;
    --python-version)
        shift
        PYTHON_VERSION=$1
        ;;
    --with-spcs-image)
        WITH_SPCS_IMAGE=true
        ;;
    -h | --help)
        help 0
        ;;
    *)
        help 1
        ;;
    esac
    shift
done

echo "Running build_and_run_tests with PYTHON_VERSION ${PYTHON_VERSION}"

EXT=""
BAZEL_ADDITIONAL_BUILD_FLAGS=()
BAZEL_ADDITIONAL_STARTUP_FLAGS=()

# Computing artifact location
# Detect the platform, also update some platform specific bazel settings
case "$(uname)" in
  Linux)
    PLATFORM="linux" ;;
  Darwin)
    PLATFORM="darwin" ;;
  *NT*)
    PLATFORM="windows"
    IS_NT=true ;;
esac

# Detect the architecture
ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|arm64)
    ARCH="arm64"
    MICROMAMBA_ARCH="aarch64" ;;
  ppc64le)
    ARCH="ppc64le"
    MICROMAMBA_ARCH="ppc64le" ;;
  *)
    ARCH="amd64"
    MICROMAMBA_ARCH="64" ;;
esac

# Compute the platform-arch string used to download yq.
case "${PLATFORM}_${ARCH}" in
  linux_arm64|linux_amd64|darwin_arm64|darwin_amd64|windows_amd64)
      ;;  # pass
  *)
    echo "Platform / Architecture is not supported by yq." >&2
    exit 1
    ;;
esac

if [ ${IS_NT} = true ]; then
    EXT=".exe"
    PYTHON_ENABLE_SCRIPT="Scripts/activate"
    BAZEL_ADDITIONAL_BUILD_FLAGS+=(--nobuild_python_zip)
    BAZEL_ADDITIONAL_BUILD_FLAGS+=(--enable_runfiles)
    BAZEL_ADDITIONAL_BUILD_FLAGS+=(--action_env="USERPROFILE=${USERPROFILE}")
    BAZEL_ADDITIONAL_BUILD_FLAGS+=(--host_action_env="USERPROFILE=${USERPROFILE}")
    BAZEL_ADDITIONAL_STARTUP_FLAGS+=(--output_user_root=C:/broot)
fi

case ${PYTHON_VERSION} in
  3.9)
    if [ ${IS_NT} = true ]; then
        PYTHON_EXECUTABLE="py -3.9"
    else
        PYTHON_EXECUTABLE="python3.9"
    fi
    ;;
  3.10)
    if [ ${IS_NT} = true ]; then
        PYTHON_EXECUTABLE="py -3.10"
    else
        PYTHON_EXECUTABLE="python3.10"
    fi
    ;;
  3.11)
    if [ ${IS_NT} = true ]; then
        PYTHON_EXECUTABLE="py -3.11"
    else
        PYTHON_EXECUTABLE="python3.11"
    fi
    ;;
  3.12)
    if [ ${IS_NT} = true ]; then
        PYTHON_EXECUTABLE="py -3.12"
    else
        PYTHON_EXECUTABLE="python3.12"
    fi
    ;;
esac

cd "${WORKSPACE}"

# Check and download yq if not presented.
TEMP_BIN=$(mktemp -d "${WORKSPACE}/tmp_bin_XXXXX")
trap 'rm -rf "${TEMP_BIN}"' EXIT

_YQ_BIN="yq${EXT}"
if ! command -v "${_YQ_BIN}" &>/dev/null; then
    curl -Lsv https://github.com/mikefarah/yq/releases/latest/download/yq_${PLATFORM}_${ARCH}${EXT} -o "${TEMP_BIN}/yq${EXT}" && chmod +x "${TEMP_BIN}/yq${EXT}"
    _YQ_BIN="${TEMP_BIN}/yq${EXT}"
fi

# Install micromamba
_MICROMAMBA_BIN="micromamba${EXT}"
if [ "${ENV}" = "conda" ]; then
    if ! command -v "${_MICROMAMBA_BIN}" &>/dev/null; then
        curl -Lsv "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-${PLATFORM}-${MICROMAMBA_ARCH}" -o "${TEMP_BIN}/micromamba${EXT}" && chmod +x "${TEMP_BIN}/micromamba${EXT}"
        _MICROMAMBA_BIN="${TEMP_BIN}/micromamba${EXT}"
        export MAMBA_ROOT_PREFIX="${WORKSPACE}/micromamba"
    fi
fi

# Create temp release folder
TEMP_TEST_DIR=$(mktemp -d "${WORKSPACE}/tmp_XXXXX")
trap 'rm -rf "${TEMP_TEST_DIR}"' EXIT

pushd ${SNOWML_DIR}
# Get the version from snowflake/ml/version.bzl
VERSION=$(grep -oE "VERSION = \"[0-9]+\\.[0-9]+\\.[0-9]+.*\"" snowflake/ml/version.bzl | cut -d'"' -f2)
echo "Extracted Package Version from code: ${VERSION}"

# Get optional requirements from snowflake/ml/requirements.bzl
OPTIONAL_REQUIREMENTS=()
while IFS='' read -r line; do OPTIONAL_REQUIREMENTS+=("$line"); done < <("${_YQ_BIN}" '.requirements.run_constrained.[] | ... style=""' ci/conda_recipe/meta.yaml)

# Compare test required dependencies with wheel pkg dependencies and exclude tests if necessary
EXCLUDE_TESTS=$(mktemp "${TEMP_TEST_DIR}/exclude_tests_XXXXX")
./ci/get_excluded_tests.sh -f "${EXCLUDE_TESTS}" -m "${MODE}" -b "${BAZEL}" -e "${SF_ENV}"

# Generate and copy auto-gen tests.
"${BAZEL}" "${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]+"${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]}"}" build "${BAZEL_ADDITIONAL_BUILD_FLAGS[@]+"${BAZEL_ADDITIONAL_BUILD_FLAGS[@]}"}" //tests/integ/...

# Rsync cannot work well with path that has drive letter in Windows,
# Thus, these two rsync has to use relative path instead of absolute ones.

rsync -av --exclude '*.runfiles_manifest' --exclude '*.runfiles/**' "bazel-bin/tests" .

# Copy tests into temp directory
pushd "${TEMP_TEST_DIR}"
rsync -av --exclude-from "${EXCLUDE_TESTS}" "../${SNOWML_DIR}/tests" .
popd

"${BAZEL}" "${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]+"${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]}"}" clean --expunge
popd

# Build snowml package
if [ "${ENV}" = "pip" ]; then
    # Clean build workspace
    rm -f "${WORKSPACE}"/*.whl

    # Build Snowpark
    if [ "${WITH_SNOWPARK}" = true ]; then
        pushd ${SNOWPARK_DIR}
        rm -rf venv
        ${PYTHON_EXECUTABLE} -m venv venv
        # shellcheck disable=SC1090
        source "venv/${PYTHON_ENABLE_SCRIPT}"
        python --version
        python -m pip install -U pip setuptools wheel
        echo "Building snowpark wheel from main:$(git rev-parse HEAD)."
        python -m pip wheel . --no-deps
        cp "$(find . -maxdepth 1 -iname 'snowflake_snowpark_python-*.whl')" "${WORKSPACE}"
        deactivate
        popd
    fi

    # Build SnowML
    pushd ${SNOWML_DIR}
    "${BAZEL}" "${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]+"${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]}"}" build "${BAZEL_ADDITIONAL_BUILD_FLAGS[@]+"${BAZEL_ADDITIONAL_BUILD_FLAGS[@]}"}" //:wheel
    cp "$("${BAZEL}" "${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]+"${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]}"}" info bazel-bin)/dist/snowflake_ml_python-${VERSION}-py3-none-any.whl" "${WORKSPACE}"
    popd
else
    # Clean conda cache
    conda clean --all --force-pkgs-dirs -y

    # Clean conda build workspace
    rm -rf "${WORKSPACE}/conda-bld"

    # Build Snowpark
    if [ "${WITH_SNOWPARK}" = true ]; then
        pushd ${SNOWPARK_DIR}
        conda build recipe/ --python=${PYTHON_VERSION} --numpy=1.16 --croot "${WORKSPACE}/conda-bld"
        popd
    fi

    # Build SnowML
    pushd ${SNOWML_DIR}
    # Build conda package
    conda build -c conda-forge --override-channels --prefix-length 50 --python=${PYTHON_VERSION} --croot "${WORKSPACE}/conda-bld" ci/conda_recipe
    conda build purge
    popd
fi

if [[ "${WITH_SPCS_IMAGE}" = true ]]; then
    pushd ${SNOWML_DIR}
    # Build SPCS Image
    source model_container_services_deployment/ci/build_and_push_images.sh
    popd
fi

# Start testing
pushd "${TEMP_TEST_DIR}"

# Set up common pytest flag
COMMON_PYTEST_FLAG=()
COMMON_PYTEST_FLAG+=(--strict-markers) # Strict the pytest markers to avoid typo in markers
COMMON_PYTEST_FLAG+=(--import-mode=append)
COMMON_PYTEST_FLAG+=(--log-cli-level=INFO)
COMMON_PYTEST_FLAG+=(-n logical)
COMMON_PYTEST_FLAG+=(--reruns 1)

if [[ -n "${JUNIT_REPORT_PATH}" ]]; then
    COMMON_PYTEST_FLAG+=(--junitxml "${JUNIT_REPORT_PATH}")
fi

if [ "${ENV}" = "pip" ]; then
    if [ "${WITH_SPCS_IMAGE}" = true ]; then
        COMMON_PYTEST_FLAG+=(-m "spcs_deployment_image and not pip_incompatible")
    else
        COMMON_PYTEST_FLAG+=(-m "not pip_incompatible")
    fi
    # Copy wheel package
    cp "${WORKSPACE}/snowflake_ml_python-${VERSION}-py3-none-any.whl" "${TEMP_TEST_DIR}"

    # Create testing env
    ${PYTHON_EXECUTABLE} -m venv testenv
    # shellcheck disable=SC1090
    source "testenv/${PYTHON_ENABLE_SCRIPT}"
    # Install all of the packages in single line,
    # otherwise it will fail in dependency resolution.
    python --version
    python -m pip install --upgrade pip
    python -m pip list
    python -m pip install "snowflake_ml_python-${VERSION}-py3-none-any.whl[all]" -r "${WORKSPACE}/${SNOWML_DIR}/requirements.txt" --no-cache-dir --force-reinstall
    if [ "${WITH_SNOWPARK}" = true ]; then
        cp "$(find "${WORKSPACE}" -maxdepth 1 -iname 'snowflake_snowpark_python-*.whl')" "${TEMP_TEST_DIR}"
        python -m pip install "$(find . -maxdepth 1 -iname 'snowflake_snowpark_python-*.whl')" --no-deps --force-reinstall
    fi
    python -m pip list

    # Run the tests
    set +e
    TEST_SRCDIR="${TEMP_TEST_DIR}" python -m pytest "${COMMON_PYTEST_FLAG[@]}" tests/integ/
    TEST_RETCODE=$?
    set -e
else
    if [ "${WITH_SPCS_IMAGE}" = true ]; then
        COMMON_PYTEST_FLAG+=(-m "spcs_deployment_image and not conda_incompatible")
    else
        COMMON_PYTEST_FLAG+=(-m "not conda_incompatible")
    fi
    # Create local conda channel
    conda index "${WORKSPACE}/conda-bld"

    # Clean conda cache
    "${_MICROMAMBA_BIN}" clean --all --force-pkgs-dirs -y

    # Create testing env
    "${_MICROMAMBA_BIN}" create -y -p testenv -c "${WORKSPACE}/conda-bld" -c "https://repo.anaconda.com/pkgs/snowflake/" --override-channels "python=${PYTHON_VERSION}" snowflake-ml-python "${OPTIONAL_REQUIREMENTS[@]}"
    "${_MICROMAMBA_BIN}" env update -p testenv -f "${WORKSPACE}/${SNOWML_DIR}/bazel/environments/conda-env-build-test.yml"
    "${_MICROMAMBA_BIN}" list -p testenv

    # Run integration tests
    set +e
    TEST_SRCDIR="${TEMP_TEST_DIR}" conda run -p testenv --no-capture-output python -m pytest "${COMMON_PYTEST_FLAG[@]}" tests/integ/
    TEST_RETCODE=$?
    set -e

    # Clean the conda environment
    "${_MICROMAMBA_BIN}" env remove -p testenv
fi

popd

echo "Done running ${PROG}"
# Pytest exit code
#   0: Success;
#   5: no tests found
# See https://docs.pytest.org/en/7.1.x/reference/exit-codes.html
if [[ (${MODE} = "merge_gate" || ${MODE} = "quarantined") && ${TEST_RETCODE} -eq 5 ]]; then
    exit 0
fi
exit ${TEST_RETCODE}
