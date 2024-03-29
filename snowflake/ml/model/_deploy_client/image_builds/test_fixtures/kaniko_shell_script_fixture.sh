#!/bin/sh

# Set the file path to monitor
REGISTRY_CRED_PATH="/kaniko/.docker/config.json"
SESSION_TOKEN_PATH="/snowflake/session/token"

# Function to gracefully terminate the file monitoring job
cleanup() {
  echo "Stopping file monitoring job..."
  trap - INT TERM # Remove the signal handlers
  kill -- -$$ # Kill the entire process group. Extra $ to escape, the generated shell script should have two $.
}

# SNOW-990976, This is an additional safety check to ensure token file exists, on top of the token file check upon
# launching SPCS job. This additional check could provide value in cases things go wrong with token refresh that result
# in token file to disappear.
wait_till_token_file_exists() {
  timeout=60  # 1 minute timeout
  elapsed_time=0

  while [ ! -f "${SESSION_TOKEN_PATH}" ] && [ "$elapsed_time" -lt "$timeout" ]; do
    sleep 1
    elapsed_time=$((elapsed_time + 1))
    remaining_time=$((timeout - elapsed_time))
    echo "Waiting for token file to exist. Wait time remaining: ${remaining_time} seconds."
  done

  if [ ! -f "${SESSION_TOKEN_PATH}" ]; then
    echo "Error: Token file '${SESSION_TOKEN_PATH}' does not show up within the ${timeout} seconds timeout period."
    exit 1
  fi
}

generate_registry_cred() {
  wait_till_token_file_exists
  AUTH_TOKEN=$(printf '0auth2accesstoken:%s' "$(cat ${SESSION_TOKEN_PATH})" | base64);
  echo '{"auths":{"mock_image_repo":{"auth":"'"$AUTH_TOKEN"'"}}}' | tr -d '\n' > $REGISTRY_CRED_PATH;
}

on_session_token_change() {
  wait_till_token_file_exists
  # Get the initial checksum of the file
  CHECKSUM=$(md5sum "${SESSION_TOKEN_PATH}" | awk '{ print $1 }')
  # Run the command once before the loop
  echo "Monitoring session token changes in the background..."
  (
    while true; do
      wait_till_token_file_exists
      # Get the current checksum of the file
      CURRENT_CHECKSUM=$(md5sum "${SESSION_TOKEN_PATH}" | awk '{ print $1 }')
      if [ "${CURRENT_CHECKSUM}" != "${CHECKSUM}" ]; then
        # Session token file has changed, regenerate registry credential.
        echo "Session token has changed. Regenerating registry auth credentials."
        generate_registry_cred
        CHECKSUM="${CURRENT_CHECKSUM}"
      fi
      # Wait for a short period of time before checking again
      sleep 1
    done
  )
}

run_kaniko() {
  # Run the Kaniko command in the foreground
  echo "Starting Kaniko command..."

  # Set cache ttl to a large value as snowservice registry doesn't support deleting cache anyway.
  # Compression level set to 1 for fastest compression/decompression speed at the cost of compression ration.
  /kaniko/executor \
    --dockerfile Dockerfile \
    --context dir:///stage/models/id/context \
    --destination=org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest \
    --cache=true \
    --compressed-caching=false \
    --cache-copy-layers=false \
    --use-new-run \
    --snapshot-mode=redo \
    --cache-repo=mock_image_repo/cache \
    --cache-run-layers=true \
    --cache-ttl=8760h \
    --push-retry=3 \
    --image-fs-extract-retry=5 \
    --compression=zstd \
    --compression-level=1 \
    --log-timestamp
}

setup() {
  tar -C "/stage/models/id" -xf "/stage/models/id/context.tar.gz";
  generate_registry_cred
  # Set up the signal handlers
  trap cleanup TERM
}

setup

# Running kaniko job on the foreground and session token monitoring on the background. When session token changes,
# overwrite the existing registry cred file with the new session token.
on_session_token_change &
run_kaniko

# Capture the exit code from the previous kaniko command.
KANIKO_EXIT_CODE=$?
# Exit with the same exit code as the Kaniko command. This then triggers the cleanup function.
exit $KANIKO_EXIT_CODE
