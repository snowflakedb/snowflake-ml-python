#!/bin/bash

# Usage
# copy_and_run_tests.sh <workspace> [--env pip|conda] [--with-snowpark]
#
# Args
# workspace: path to the workspace, SnowML code should be in snowml directory.
#
# Optional Args
# env: Set the environment, choose from pip and conda
# with-snowpark: Build and test with snowpark in snowpark-python directory in the workspace.
#
# Action
#   - Copy the integration tests from workspace folder and execute them in testing Python env using pytest.
#   - This is to mimic the behavior of using snowml wheel package in user land.

set -o pipefail
set -eu

PROG=$0

help()
{
    exit_code=$1
    echo "Invalid usage, must provide argument for workspace"
    echo "Usage: ${PROG} <workspace> [--env pip|conda] [--with-snowpark]"
    exit ${exit_code}
}

WORKSPACE=$1 && shift || help 1
ENV="pip"
WITH_SNOWPARK=false
SNOWML_DIR="snowml"
SNOWPARK_DIR="snowpark-python"

while (($#)); do
    case $1 in
        -e|--env)
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
        -h|--help)
            help 0
            ;;
        *)
            help 1
            ;;
    esac
    shift
done

# Check Python3.8 exist
# TODO(SNOW-845592): ideally we should download py3.8 from conda if not exist. Currently we just fail.
set +eu
source /opt/rh/rh-python38/enable
PYTHON38_EXIST=$?
if [ $PYTHON38_EXIST -ne 0 ]; then
    echo "Failed to execute tests: Python3.8 is not installed."
    rm -rf "${TEMP_TEST_DIR}"
    exit ${PYTHON38_EXIST}
fi
set -eu

cd "${WORKSPACE}"

# Create temp release folder
TEMP_TEST_DIR=$(mktemp -d "${WORKSPACE}/tmp_XXXXX")

pushd ${SNOWML_DIR}
# Get the version from snowflake/ml/version.bzl
VERSION=$(grep -oE "VERSION = \"[0-9]+\\.[0-9]+\\.[0-9]+.*\"" snowflake/ml/version.bzl | cut -d'"' -f2)
echo "Extracted Package Version from code: ${VERSION}"

# Get optional requirements from snowflake/ml/requirements.bzl
OPTIONAL_REQUIREMENTS=$(cat snowflake/ml/requirements.bzl | python3 -c "import sys; exec(sys.stdin.read()); print(' '.join(map(lambda x: '\"'+x+'\"', EXTRA_REQUIREMENTS['all'])))")

# Compare test required dependencies with wheel pkg dependencies and exclude tests if necessary
EXCLUDE_TESTS=$(mktemp "${TEMP_TEST_DIR}/exclude_tests_XXXXX")
./ci/get_excluded_tests.sh -f "${EXCLUDE_TESTS}"
# Copy tests into temp directory
pushd "${TEMP_TEST_DIR}"
rsync -av --exclude-from "${EXCLUDE_TESTS}" "${WORKSPACE}/${SNOWML_DIR}/tests" .
ls  tests/integ/snowflake/ml
popd
popd

# Build snowml package
if [ ${ENV} = "pip" ]; then
    # Clean build workspace
    rm -f ${WORKSPACE}/*.whl

    # Build Snowpark
    if [ "${WITH_SNOWPARK}" = true ]; then
        pushd ${SNOWPARK_DIR}
        rm -rf venv
        python3.8 -m venv venv
        source venv/bin/activate
        python3.8 -m pip install -U pip setuptools wheel
        echo "Building snowpark wheel from main:$(git rev-parse HEAD)."
        pip wheel . --no-deps
        cp snowflake_snowpark_python-*.whl ${WORKSPACE}
        deactivate
        popd
    fi

    # Build SnowML
    pushd ${SNOWML_DIR}
    bazel build //snowflake/ml:wheel
    cp bazel-bin/snowflake/ml/snowflake_ml_python-*.whl ${WORKSPACE}
    popd
else
    which conda

    # Clean conda build workspace
    rm -rf ${WORKSPACE}/conda-bld

    # Build Snowpark
    if [ "${WITH_SNOWPARK}" = true ]; then
        pushd ${SNOWPARK_DIR}
        conda build recipe/ --python=3.8 --numpy=1.16 --croot "${WORKSPACE}/conda-bld"
        popd
    fi

    # Build SnowML
    pushd ${SNOWML_DIR}
    # Build conda package
    conda build --channel=conda-forge --prefix-length 50 --croot "${WORKSPACE}/conda-bld" ci/conda_recipe
    conda build purge
    popd
fi

# Start testing
pushd "${TEMP_TEST_DIR}"

# Set up common pytest flag
COMMON_PYTEST_FLAG=()
COMMON_PYTEST_FLAG+=(--strict-markers) # Strict the pytest markers to avoid typo in markers
COMMON_PYTEST_FLAG+=(--import-mode=append)
COMMON_PYTEST_FLAG+=(-n 10)


if [ ${ENV} = "pip" ]; then
    # Copy wheel package
    cp "${WORKSPACE}/snowflake_ml_python-${VERSION}-py3-none-any.whl" "${TEMP_TEST_DIR}"

    # Create testing env
    python3.8 -m venv testenv
    source testenv/bin/activate
    # Install all of the packages in single line,
    # otherwise it will fail in dependency resolution.
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip list
    python3.8 -m pip install "snowflake_ml_python-${VERSION}-py3-none-any.whl[all]" pytest-xdist inflection --no-cache-dir --force-reinstall
    if [ "${WITH_SNOWPARK}" = true ]; then
        cp ${WORKSPACE}/snowflake_snowpark_python-*.whl "${TEMP_TEST_DIR}"
        python3.8 -m pip install $(find . -maxdepth 1 -iname 'snowflake_snowpark_python-*.whl') --force-reinstall
    fi
    python3.8 -m pip list

    # Set up pip specific pytest flags
    PIP_PYTEST_FLAG=()
    PIP_PYTEST_FLAG+=(-m "not pip_incompatible")  # Filter out those pip incompatible tests.

    # Run the tests
    set +e
    TEST_SRCDIR="${TEMP_TEST_DIR}" python3.8 -m pytest "${COMMON_PYTEST_FLAG[@]}" "${PIP_PYTEST_FLAG[@]}" tests/
    TEST_RETCODE=$?
    set -e
else
    # Create local conda channel
    conda index ${WORKSPACE}/conda-bld

    # Clean conda cache
    conda clean --all --force-pkgs-dirs -y

    # Create testing env
    conda create -y -p testenv -c "file://${WORKSPACE}/conda-bld" -c "https://repo.anaconda.com/pkgs/snowflake/" --override-channel "python=3.8" snowflake-ml-python pytest-xdist inflection ${OPTIONAL_REQUIREMENTS}
    conda list -p testenv

    # Run the tests
    set +e
    TEST_SRCDIR="${TEMP_TEST_DIR}" conda run -p testenv --no-capture-output python3.8 -m pytest "${COMMON_PYTEST_FLAG[@]}" tests/
    TEST_RETCODE=$?
    set -e

    # Clean the conda environment
    conda env remove -p testenv
fi

popd

# clean up temp dir
rm -rf "${TEMP_TEST_DIR}"

echo "Done running ${PROG}"
exit ${TEST_RETCODE}
