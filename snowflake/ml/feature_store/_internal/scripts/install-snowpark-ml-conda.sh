#!/bin/bash

# Setup a conda environment & installs snowpark ML.
#
# Usage
# install-snowpark-ml-conda.sh [-d <output dir>] [-n <conda env name>] [-p 3.8|3.9|3.10] [-h]

set -o pipefail
set -eu

PROG=$0

function help_pkg() {
    pkg_name=$1
    echo "## ${pkg_name} could not be found."
    echo "## To install this package on macOS, run:"
    echo "      brew install ${pkg_name}"
    echo "## To install this package on Debian/Ubuntu, run:"
    echo "      sudo apt install ${pkg_name}"
    echo "## To install this package on RHEL/CentOS, run:"
    echo "      sudo yum install ${pkg_name}"
    exit 1
}

if ! command -v conda &> /dev/null
then
    echo "## conda could not be found. This script is only useful for conda."
    exit 1
fi

CONDA_ENV_BASE=$(conda run -n base conda info --json | python3 -c "import sys, json; print(json.load(sys.stdin)['envs_dirs'][0])" | tr -d '"')
CHANNEL_HOME="${HOME}/snowpark-ml-local-channel"
PY_VERSION="3.8"
# Needs to be updated every release. Can be moved to snowml repo once that is open sourced.
DEFAULT_FILENAME=$(dirname "$PROG")/snowflake-ml-python-1.0.12-fs-0.2.0-conda.zip

function help() {
    exitcode=$1 && shift
    echo "Usage: ${PROG} [-d <output dir>] [-n <conda env name>] [-p 3.8|3.9|3.10] [-h]"
    echo "  -d OUTPUT_DIR: Optional, default is ${CHANNEL_HOME}"
    echo "  -p PY_VERSION: Optional, default is 3.8. Options are 3.9, 3.10."
    if [ "${CONDA_DEFAULT_ENV-}" ]; then
        echo "  -n CONDA_ENV_NAME: Optional, default is \`${CONDA_DEFAULT_ENV}\` (current environment). If an existing env provided, it will reuse. It will create otherwise."
    else
        echo "  -n CONDA_ENV_NAME: If an existing env provided, it will reuse. It will create otherwise."
    fi
    exit ${exitcode}
}

while (($#)); do
    case $1 in
        -d)
            shift
            CHANNEL_HOME=$1
            ;;
        -n)
            shift
            TARGET_CONDA_ENV=$1
            ;;
        -p)
            shift
            if [[ $1 = "3.8" || $1 = "3.9" || $1 == "3.10" ]]; then
                PY_VERSION=$1
            else
                echo "Invalid python version: $1"
                help 1
            fi
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

if [ -z ${TARGET_CONDA_ENV+x} ]; then
    if [ "${CONDA_DEFAULT_ENV-}" ]; then
        TARGET_CONDA_ENV="${CONDA_DEFAULT_ENV}"
    else
        help 1
    fi
fi

echo "## Target conda channel is ${TARGET_CONDA_ENV}"
CONDA_ENV_PATH="${CONDA_ENV_BASE}/${TARGET_CONDA_ENV}"
CONDA_PLATFORM=$(conda info --json | python3 -c "import sys, json; print(json.load(sys.stdin)['platform'])")
CONDA_ALL_ENVS=$(conda info --json | python3 -c "import sys, json; print(json.load(sys.stdin)['envs'])")

unzip "${DEFAULT_FILENAME}" -d ${CHANNEL_HOME}

if [[ "$CONDA_ALL_ENVS" == *"$CONDA_ENV_PATH"* ]]; then
    echo "## Conda env ${TARGET_CONDA_ENV} exists. Assuming setup correctly."
else
    echo "## Creating conda env ${TARGET_CONDA_ENV}"
    if [[ "$CONDA_PLATFORM" == 'osx-arm64' ]]; then
        echo "## Mac M1 detected. Following special conda treatment as per https://docs.snowflake.com/en/developer-guide/snowpark/python/setup"
        CONDA_SUBDIR=osx-64 conda create -p "${CONDA_ENV_PATH}" -y python=${PY_VERSION} numpy pandas --override-channels -c https://repo.anaconda.com/pkgs/snowflake
        conda run -p "${CONDA_ENV_PATH}" conda config --env --set subdir osx-64
    else
        conda create -p "${CONDA_ENV_PATH}" -y --override-channels -c https://repo.anaconda.com/pkgs/snowflake python=${PY_VERSION} numpy pandas
    fi
fi

if [[ "$CONDA_PLATFORM" == 'osx-arm64' ]]; then
    CONDA_SUBDIR=osx-64 conda install -p "${CONDA_ENV_PATH}" -y -c "file://${CHANNEL_HOME}/snowpark-ml-local-channel" -c "https://repo.anaconda.com/pkgs/snowflake/" --override-channels snowflake-ml-python
else
    conda install -p "${CONDA_ENV_PATH}" -y -c "file://${CHANNEL_HOME}/snowpark-ml-local-channel" -c "https://repo.anaconda.com/pkgs/snowflake/" --override-channels snowflake-ml-python
fi

echo "## ALL DONE. Please activate the env by executing \`conda activate ${TARGET_CONDA_ENV}\`"
