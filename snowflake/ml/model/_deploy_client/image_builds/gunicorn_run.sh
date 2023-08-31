#!/bin/bash
set -eu

OS=$(uname)

if [[ ${OS} = "Linux" ]]; then
    NUM_CORES=$(nproc)
elif [[ ${OS} = "Darwin" ]]; then
    # macOS
    NUM_CORES=$(sysctl -n hw.ncpu)
elif [[ ${OS} = "Windows" ]]; then
    NUM_CORES=$(wmic cpu get NumberOfCores | grep -Eo '[0-9]+')
else
    echo "Unsupported operating system: ${OS}"
    exit 1
fi

# Check if the "NUM_WORKERS" variable is set by the user
if [[ -n "${NUM_WORKERS-}" && "${NUM_WORKERS}" != "None" ]]; then
    # If the user has set the "num_workers" variable, use it to overwrite the default value
    FINAL_NUM_WORKERS=${NUM_WORKERS}
else
    # Based on the Gunicorn documentation, set the number of workers to number_of_cores * 2 + 1. This assumption is
    # based on an ideal scenario where one core is handling two processes simultaneously, while one process is dedicated to
    # IO operations and the other process is performing compute tasks.
    # However, in case when the model is large, we will run into OOM error as each process will need to load the model
    # into memory. In such cases, we require the user to pass in "num_workers" to overwrite the default.
    FINAL_NUM_WORKERS=$((NUM_CORES * 2 + 1))
fi

echo "Number of CPU cores: $NUM_CORES"
echo "Setting number of workers to $FINAL_NUM_WORKERS"

# Exclude preload option as it won't work with non-thread-safe model, and no easy way to detect whether model is
# thread-safe or not. Defer the optimization later.
exec /opt/conda/bin/gunicorn -w "$FINAL_NUM_WORKERS" -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 --timeout 600 inference_server.main:app
