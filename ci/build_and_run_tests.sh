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
    MICROMAMBA_PLATFORM="linux" ;;
  Darwin)
    MICROMAMBA_PLATFORM="osx" ;;
  *NT*)
    MICROMAMBA_PLATFORM="win"
    IS_NT=true ;;
esac

# Detect the architecture
ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|arm64)
    MICROMAMBA_ARCH="aarch64" ;;
  ppc64le)
    MICROMAMBA_ARCH="ppc64le" ;;
  *)
    MICROMAMBA_ARCH="64" ;;
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

# Install micromamba
_MICROMAMBA_BIN="micromamba${EXT}"
if [ "${ENV}" = "conda" ]; then
    if ! command -v "${_MICROMAMBA_BIN}" &>/dev/null; then
        curl -Lsv "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-${MICROMAMBA_PLATFORM}-${MICROMAMBA_ARCH}" -o "${TEMP_BIN}/micromamba${EXT}" && chmod +x "${TEMP_BIN}/micromamba${EXT}"
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

# Generate and copy auto-gen tests.
"${BAZEL}" "${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]+"${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]}"}" build --config=build "${BAZEL_ADDITIONAL_BUILD_FLAGS[@]+"${BAZEL_ADDITIONAL_BUILD_FLAGS[@]}"}" //tests/integ/...

# Rsync cannot work well with path that has drive letter in Windows,
# Thus, rsync has to use relative path instead of absolute ones.

rsync -av --exclude '*.runfiles_manifest' --exclude '*.runfiles/**' "bazel-bin/tests" .

# Read environments from optional_dependency_groups.bzl
groups=()
while IFS= read -r line; do
    groups+=("$line")
done < <(python3 -c '
import ast
with open("bazel/platforms/optional_dependency_groups.bzl", "r") as f:
    tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if type(node) == ast.Assign and node.targets[0].id == "OPTIONAL_DEPENDENCY_GROUPS":
            groups = ast.literal_eval(node.value)
            for group in groups.keys():
                print(group)
')

groups+=("core")

for i in "${!groups[@]}"; do
    group="${groups[$i]}"

    # Compare test required dependencies with wheel pkg dependencies and exclude tests if necessary
    EXCLUDE_TESTS=$(mktemp "${TEMP_TEST_DIR}/exclude_tests_${group}_XXXXX")
    ./ci/get_excluded_tests.sh -f "${EXCLUDE_TESTS}" -m "${MODE}" -b "${BAZEL}" -e "${SF_ENV}" -g "${group}"

    # Copy tests into temp directory
    pushd "${TEMP_TEST_DIR}"
    rsync -av --exclude-from "${EXCLUDE_TESTS}" "../${SNOWML_DIR}/tests" "${group}"
    popd
done

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
    "${BAZEL}" "${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]+"${BAZEL_ADDITIONAL_STARTUP_FLAGS[@]}"}" build --config=build "${BAZEL_ADDITIONAL_BUILD_FLAGS[@]+"${BAZEL_ADDITIONAL_BUILD_FLAGS[@]}"}" //:wheel
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
COMMON_PYTEST_FLAG+=(--timeout=3600)

group_exit_codes=()
group_coverage_report_files=()

for i in "${!groups[@]}"; do
    group="${groups[$i]}"

    if [[ -n "${JUNIT_REPORT_PATH}" ]]; then
        group_coverage_report_files[$i]=$(mktemp "${TEMP_TEST_DIR}/junit_report_${group}_XXXXX")
        COMMON_PYTEST_FLAG+=(--junitxml "${group_coverage_report_files[$i]}")
    fi

    pushd "${group}"
    if [ "${ENV}" = "pip" ]; then
        if [ "${WITH_SPCS_IMAGE}" = true ]; then
            COMMON_PYTEST_FLAG+=(-m "spcs_deployment_image and not pip_incompatible")
        else
            COMMON_PYTEST_FLAG+=(-m "not pip_incompatible")
        fi
        # Copy wheel package
        cp "${WORKSPACE}/snowflake_ml_python-${VERSION}-py3-none-any.whl" "${TEMP_TEST_DIR}/${group}"

        # Create testing env
        ${PYTHON_EXECUTABLE} -m venv testenv
        # shellcheck disable=SC1090
        source "testenv/${PYTHON_ENABLE_SCRIPT}"
        # Install all of the packages in single line,
        # otherwise it will fail in dependency resolution.
        if [ -f /opt/rh/devtoolset-10/root/usr/bin/gcc ]; then
            export CXX=/opt/rh/devtoolset-10/root/usr/bin/g++
            export CC=/opt/rh/devtoolset-10/root/usr/bin/gcc
        fi
        python --version
        python -m pip install --upgrade pip
        python -m pip list
        python -m pip install "snowflake_ml_python-${VERSION}-py3-none-any.whl" -r "${WORKSPACE}/${SNOWML_DIR}/bazel/environments/requirements_${group}.txt" --no-cache-dir --force-reinstall
        if [ "${WITH_SNOWPARK}" = true ]; then
            cp "$(find "${WORKSPACE}" -maxdepth 1 -iname 'snowflake_snowpark_python-*.whl')" "${TEMP_TEST_DIR}"
            python -m pip install "$(find "${TEMP_TEST_DIR}" -maxdepth 1 -iname 'snowflake_snowpark_python-*.whl')" --no-deps --force-reinstall
        fi
        python -m pip list

        # Run the tests
        set +e
        TEST_SRCDIR="${TEMP_TEST_DIR}" python -m pytest "${COMMON_PYTEST_FLAG[@]}" tests/integ/
        group_exit_codes[$i]=$?
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
        "${_MICROMAMBA_BIN}" create -y -p ./testenv -c "${WORKSPACE}/conda-bld" -c "https://repo.anaconda.com/pkgs/snowflake/" --override-channels "python=${PYTHON_VERSION}" snowflake-ml-python
        if [[ "${group}" != "core" ]]; then
            "${_MICROMAMBA_BIN}" env update -p ./testenv -f "${WORKSPACE}/${SNOWML_DIR}/bazel/environments/conda-optional-dependency-${group}.yml"
        fi
        "${_MICROMAMBA_BIN}" env update -p ./testenv -f "${WORKSPACE}/${SNOWML_DIR}/bazel/environments/conda-env-build-test.yml"
        "${_MICROMAMBA_BIN}" list -p ./testenv

        # Run integration tests
        set +e
        TEST_SRCDIR="${TEMP_TEST_DIR}" conda run -p ./testenv --no-capture-output python -m pytest "${COMMON_PYTEST_FLAG[@]}" tests/integ/
        group_exit_codes[$i]=$?
        set -e

        # Clean the conda environment
        "${_MICROMAMBA_BIN}" env remove -p ./testenv
    fi

    popd

done

popd


# Pytest exit code
#   0: Success;
#   5: no tests found
# See https://docs.pytest.org/en/7.1.x/reference/exit-codes.html
# Initialize final exit code as 0
final_exit_code=0

# Merge all junit test report files
if [[ -n "${JUNIT_REPORT_PATH}" ]]; then
    # Merge all JUnit report files into one
    JUNIT_REPORT_MERGED="${TEMP_TEST_DIR}/junit_report_merged.xml"

    # Python script to merge JUnit XML files
    ${PYTHON_EXECUTABLE} -c "
import xml.etree.ElementTree as ET
import os

def merge_junit_reports(report_files, output_file):
    root = ET.Element('testsuites')
    for report_file in report_files:
        if os.path.exists(report_file):
            try:
                tree = ET.parse(report_file)
                testsuite = tree.getroot()
                if testsuite.tag == 'testsuite':
                    root.append(testsuite)
                elif testsuite.tag == 'testsuites':
                    for suite in testsuite:
                        root.append(suite)
            except ET.ParseError as e:
                print(f'Error parsing XML file {report_file}: {e}')
        else:
            print(f'Warning: Report file not found: {report_file}')
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

report_files_str = '${group_coverage_report_files[*]}'
report_files = report_files_str.split()
output_file = '${JUNIT_REPORT_MERGED}'
merge_junit_reports(report_files, output_file)
"
    # Copy the merged JUnit report to the specified path
    cp "${JUNIT_REPORT_MERGED}" "${JUNIT_REPORT_PATH}"
fi

# Check all group exit codes
for exit_code in "${group_exit_codes[@]}"; do
    if [[ (${MODE} = "merge_gate" || ${MODE} = "quarantined" || ${WITH_SPCS_IMAGE} = "true" ) && ${exit_code} -eq 5 ]]; then
        continue
    fi
    if [[ ${exit_code} -ne 0 ]]; then
        final_exit_code=${exit_code}
    fi
done

echo "Done running ${PROG}"
exit ${final_exit_code}
