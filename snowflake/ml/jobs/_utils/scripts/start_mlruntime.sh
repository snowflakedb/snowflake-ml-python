#!/bin/bash

set -e # exit if a command fails

echo "Creating log directories..."
mkdir -p /var/log/managedservices/user/mlrs
mkdir -p /var/log/managedservices/system/mlrs
mkdir -p /var/log/managedservices/system/ray

echo "*/1 * * * * root /etc/ray_copy_cron.sh" >> /etc/cron.d/ray_copy_cron
echo "" >> /etc/cron.d/ray_copy_cron
chmod 744 /etc/cron.d/ray_copy_cron

service cron start

mkdir -p /tmp/prometheus-multi-dir

# Configure IP address and logging directory
eth0Ip=$(ifconfig eth0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p')
log_dir="/tmp/ray"

# Check if eth0Ip is empty and set default if necessary
if [ -z "$eth0Ip" ]; then
    # This should never happen, but just in case ethOIp is not set, we should default to localhost
    eth0Ip="127.0.0.1"
fi

shm_size=$(df --output=size --block-size=1 /dev/shm | tail -n 1)
total_memory_size=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)

# Determine if dashboard should be enabled based on total memory size
# Enable dashboard only if total memory size >= 8GB (i.e. not on XS compute pool)
# TODO (SNOW-2860029): use a environment variable to determine the node type
total_memory_threshold=8192
if [ "$total_memory_size" -ge "$total_memory_threshold" ]; then
    enable_dashboard="true"
else
    enable_dashboard="false"
fi

echo "Shared memory size: $shm_size bytes"
echo "Dashboard enabled: $enable_dashboard"

# Common parameters for both head and worker nodes
common_params=(
    "--node-ip-address=$eth0Ip"
    "--object-manager-port=${RAY_OBJECT_MANAGER_PORT:-12011}"
    "--node-manager-port=${RAY_NODE_MANAGER_PORT:-12012}"
    "--runtime-env-agent-port=${RAY_RUNTIME_ENV_AGENT_PORT:-12013}"
    "--dashboard-agent-grpc-port=${RAY_DASHBOARD_AGENT_GRPC_PORT:-12014}"
    "--dashboard-agent-listen-port=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-12015}"
    "--min-worker-port=${RAY_MIN_WORKER_PORT:-12031}"
    "--max-worker-port=${RAY_MAX_WORKER_PORT:-13000}"
    "--metrics-export-port=11502"
    "--temp-dir=$log_dir"
    "--disable-usage-stats"
)

# Specific parameters for head and worker nodes
if [ "$NODE_TYPE" = "worker" ]; then
    # Check mandatory environment variables for worker
    if [ -z "$RAY_HEAD_ADDRESS" ] || [ -z "$SERVICE_NAME" ]; then
        echo "Error: RAY_HEAD_ADDRESS and SERVICE_NAME must be set."
        exit 1
    fi

    # Additional worker-specific parameters
    worker_params=(
        "--address=${RAY_HEAD_ADDRESS}:${RAY_HEAD_GCS_PORT:-12001}"   # Connect to head node
        "--resources={\"${SERVICE_NAME}\":1, \"node_tag:worker\":1}"  # Custom resource for node identification
        "--object-store-memory=${shm_size}"
    )

    # Start Ray on a worker node
    ray start "${common_params[@]}" "${worker_params[@]}" "$@" -v
else
    # Additional head-specific parameters
   head_params=(
        "--head"
        "--include-dashboard=$enable_dashboard"
        "--disable-usage-stats"
        "--port=${RAY_HEAD_GCS_PORT:-12001}"                                  # Port of Ray (GCS server)
        "--ray-client-server-port=${RAY_HEAD_CLIENT_SERVER_PORT:-10001}"      # Listening port for Ray Client Server
        "--dashboard-host=${NODE_IP_ADDRESS}"                                            # Host to bind the dashboard server
        "--dashboard-grpc-port=${RAY_HEAD_DASHBOARD_GRPC_PORT:-12002}"        # Dashboard head to listen for grpc on
        "--dashboard-port=${DASHBOARD_PORT}"                  # Port to bind the dashboard server for local debugging
        "--resources={\"node_tag:head\":1}"                   # Resource tag for selecting head as coordinator
    )

    # Start Ray
    ray start "${common_params[@]}" "${head_params[@]}" "$@"
fi
