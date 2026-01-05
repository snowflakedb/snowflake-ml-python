#!/bin/bash

set -e # exit if a command fails

# Get and change to system scripts directory
SYSTEM_DIR=$(cd "$(dirname "$0")" && pwd)

# Change directory to user payload directory
if [ -n "${MLRS_PAYLOAD_DIR}" ]; then
    cd ${MLRS_STAGE_MOUNT_PATH}/${MLRS_PAYLOAD_DIR}
fi

##### Set up Python environment #####
export PYTHONPATH=/opt/env/site-packages/
MLRS_SYSTEM_REQUIREMENTS_FILE=${MLRS_SYSTEM_REQUIREMENTS_FILE:-"${SYSTEM_DIR}/requirements.txt"}
if [ -f "${MLRS_SYSTEM_REQUIREMENTS_FILE}" ]; then
    echo "Installing packages from $MLRS_SYSTEM_REQUIREMENTS_FILE"
    if ! pip install --no-index -r $MLRS_SYSTEM_REQUIREMENTS_FILE; then
        echo "Offline install failed, falling back to regular pip install"
        pip install -r $MLRS_SYSTEM_REQUIREMENTS_FILE
    fi
fi

MLRS_REQUIREMENTS_FILE=${MLRS_REQUIREMENTS_FILE:-"requirements.txt"}
if [ -f "${MLRS_REQUIREMENTS_FILE}" ]; then
    # TODO: Prevent collisions with MLRS packages using virtualenvs
    echo "Installing packages from $MLRS_REQUIREMENTS_FILE"
    pip install -r $MLRS_REQUIREMENTS_FILE
fi

MLRS_CONDA_ENV_FILE=${MLRS_CONDA_ENV_FILE:-"environment.yml"}
if [ -f "${MLRS_CONDA_ENV_FILE}" ]; then
    # TODO: Handle conda environment
    echo "Custom conda environments not currently supported"
    exit 1
fi
##### End Python environment setup #####

##### Set up multi-node configuration #####
# Configure IP address
if [ -f "${SYSTEM_DIR}/get_instance_ip.py" ]; then
    eth0Ip=$(python3 "${SYSTEM_DIR}/get_instance_ip.py" \
        "$SNOWFLAKE_SERVICE_NAME" --instance-index=-1)
else
    eth0Ip=$(ifconfig eth0 2>/dev/null | sed -En -e 's/.*inet ([0-9.]+).*/\1/p')
fi

# Check if eth0Ip is a valid IP address and fall back to default if necessary
if [[ ! $eth0Ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    eth0Ip="127.0.0.1"
fi

# Set default values for job environment variables if they don't exist
# (e.g. some are only populate by SPCS for batch jobs, others just may not be set at all)
export SNOWFLAKE_JOBS_COUNT=${SNOWFLAKE_JOBS_COUNT:-1}
export SNOWFLAKE_JOB_INDEX=${SNOWFLAKE_JOB_INDEX:-0}
export SERVICE_NAME="${SERVICE_NAME:-$SNOWFLAKE_SERVICE_NAME}"

##### Ray configuration #####

# Determine if it should be a worker or a head node for batch jobs
if [[ "$SNOWFLAKE_JOBS_COUNT" -gt 1 ]]; then
    head_info=$(python3 "${SYSTEM_DIR}/get_instance_ip.py" "$SNOWFLAKE_SERVICE_NAME" --head)
    if [ $? -eq 0 ]; then
        # Parse the output using read
        read head_index head_ip head_status<<< "$head_info"

        if [ "$SNOWFLAKE_JOB_INDEX" -ne "$head_index" ]; then
            NODE_TYPE="worker"
        fi

        # Use the parsed variables
        echo "Head Instance Index: $head_index"
        echo "Head Instance IP: $head_ip"
        echo "Head Instance Status: $head_status"

        # If the head status is not "READY" or "PENDING", exit early
        if [ "$head_status" != "READY" ] && [ "$head_status" != "PENDING" ]; then
            echo "Head instance status is not READY or PENDING. Exiting."
            exit 0
        fi

    else
        echo "Error: Failed to get head instance information."
        echo "$head_info" # Print the error message
        exit 1
    fi
fi

# Start ML Runtime (non-blocking call)
NODE_TYPE=$NODE_TYPE RAY_HEAD_ADDRESS="$head_ip" bash ${SYSTEM_DIR}/start_mlruntime.sh

if [ "$NODE_TYPE" = "worker" ]; then
    echo "Worker node started on address $eth0Ip. See more logs in the head node."

    # Start the worker shutdown listener in the background
    echo "Starting worker shutdown listener..."
    python "${SYSTEM_DIR}/worker_shutdown_listener.py"
    WORKER_EXIT_CODE=$?

    echo "Worker shutdown listener exited with code $WORKER_EXIT_CODE"
    exit $WORKER_EXIT_CODE
else
    # Run user's Python entrypoint via mljob_launcher
    echo Running command: python "${SYSTEM_DIR}/mljob_launcher.py" "$@"
    python "${SYSTEM_DIR}/mljob_launcher.py" "$@"

    # After the user's job completes, signal workers to shut down
    echo "User job completed. Signaling workers to shut down..."
    python "${SYSTEM_DIR}/signal_workers.py" --wait-time 15
    echo "Head node job completed. Exiting."
fi
