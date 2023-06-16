#!/bin/sh
set -eu

OS=$(uname)

if [ "${OS}" = "Linux" ]; then
    NUM_CORES=$(nproc)
elif [ "${OS}" = "Darwin" ]; then
    # macOS
    NUM_CORES=$(sysctl -n hw.ncpu)
elif [ "${OS}" = "Windows" ]; then
    NUM_CORES=$(wmic cpu get NumberOfCores | grep -Eo '[0-9]+')
else
    echo "Unsupported operating system: ${OS}"
    exit 1
fi

# Based on the Gunicorn documentation, set the number of workers to number_of_cores * 2 + 1. This assumption is
# based on an ideal scenario where one core is handling two processes simultaneously, while one process is dedicated to
# IO operations and the other process is performing compute tasks.
NUM_WORKERS=$((NUM_CORES * 2 + 1))
echo "Number of CPU cores: $NUM_CORES"
echo "Setting number of workers to $NUM_WORKERS"
exec /opt/conda/bin/gunicorn --preload -w "$NUM_WORKERS" -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 inference_server.main:app
