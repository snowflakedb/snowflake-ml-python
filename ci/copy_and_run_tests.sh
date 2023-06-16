#!/bin/bash

# Usage
# copy_and_run_tests.sh <workspace>
#
# Args
# workspace: path to the SnowML Github Repo
#
# Variables
# SNOWPARK_WHEEL (optional): if defined, the script will reinstall snowpark using the provided wheel file.
#
# Action
#   - Copy the integration tests from workspace folder and execute them in testing Python env using pytest.
#   - This is to mimic the behavior of using snowml wheel package in user land.

set -o pipefail
set -eu

if [ $# -lt 1 ]; then
    echo "Invalid usage, must provide argument for workspace"
    echo "Usage: $0 <workspace>"
    exit 1
fi

WORKSPACE=$1


cd "${WORKSPACE}"

# Get the version from snowflake/ml/version.bzl
VERSION=$(grep -oE "VERSION = \"[0-9]+\\.[0-9]+\\.[0-9]+.*\"" snowflake/ml/version.bzl | cut -d'"' -f2)
echo "Extracted Package Version from code: ${VERSION}"

#TODO: SNOW-830849 have a better way to provide these during pip install
CONDA_PKG_VERSIONS="xgboost==1.7.3"

# Create temp release folder and copy wheel package
TEMP_TEST_DIR=$(mktemp -d "${WORKSPACE}/tmp_XXXXX")
cp "${WORKSPACE}/snowflake_ml_python-${VERSION}-py3-none-any.whl" "${TEMP_TEST_DIR}"

# Compare test required dependencies with wheel pkg dependencies and exclude tests if necessary
EXCLUDE_TESTS=$(mktemp "${TEMP_TEST_DIR}/exclude_tests_XXXXX")
${WORKSPACE}/ci/get_excluded_tests.sh -f "${EXCLUDE_TESTS}"
# Copy tests into temp directory
pushd "${TEMP_TEST_DIR}"
rsync -av --exclude-from "${EXCLUDE_TESTS}" "${WORKSPACE}/tests" .
ls  tests/integ/snowflake/ml

# Check Python3.8 exist
# TODO: ideally we should download py3.8 from conda if not exist. Currently we just fail.
set +eu
source /opt/rh/rh-python38/enable
PYTHON38_EXIST=$?
if [ $PYTHON38_EXIST -ne 0 ]; then
    echo "Failed to execute tests: Python3.8 is not installed."
    rm -rf "${TEMP_TEST_DIR}"
    exit ${PYTHON38_EXIST}
fi
set -eu

# Create testing env
python3.8 -m venv testenv
source testenv/bin/activate
# Install all of the packages in single line,
# otherwise it will fail in dependency resolution.
python3.8 -m pip install --upgrade pip
python3.8 -m pip list
python3.8 -m pip install "snowflake_ml_python-${VERSION}-py3-none-any.whl[all]" "${CONDA_PKG_VERSIONS}" pytest-xdist inflection --no-cache-dir --force-reinstall

# If there's a snowpark wheel provided, install it here to replace the version installed above.
if [ "${SNOWPARK_WHEEL-}" ]; then
    echo "Installing specified SNOWPARK_WHEEL."
    cp "${WORKSPACE}/${SNOWPARK_WHEEL}" "${TEMP_TEST_DIR}"
    python3.8 -m pip install "$SNOWPARK_WHEEL" --force-reinstall
fi

# Run the tests
python3.8 -m pip list
TEST_SRCDIR="${TEMP_TEST_DIR}" python3.8 -m pytest --import-mode=append -n 10 tests/
popd

# clean up temp dir
rm -rf "${TEMP_TEST_DIR}"

echo "Done running "$0
